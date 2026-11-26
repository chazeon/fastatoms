// SPDX-License-Identifier: MIT
#include "fastatoms/io/cif.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gemmi/cif.hpp"
#include "gemmi/smcif.hpp"

#include "fastatoms/core/cell.hpp"
#include "fastatoms/core/elements.hpp"
#include "fastatoms/io/format.hpp"

namespace fastatoms::io {
namespace {

namespace gm = gemmi;

/// Mass of deuterium, as ase.io.cif assigns it to a "D" site.
constexpr double kDeuteriumMass = 2.01355;
/// Mass of tritium (ase.io.cif rejects "T" sites; this is the nuclear mass).
constexpr double kTritiumMass = 3.0160492;

/// One expanded atom, before it becomes part of `Atoms`.
struct Entry {
    int number;
    std::string label;
    double occupancy;
    Vec3 fract;       ///< wrapped into [0, 1)
    double mass;      ///< isotope mass for D and T sites, else 0 (use the element mass)
};

/// Gram-Schmidt orthogonalisation of three vectors, as used by LLL.
struct GramSchmidt {
    std::array<Vec3, 3> ortho{};
    std::array<std::array<double, 3>, 3> mu{};
    std::array<double, 3> norm2{};
};

GramSchmidt gram_schmidt(const std::array<Vec3, 3>& b) {
    GramSchmidt g;
    for (std::size_t i = 0; i < 3; ++i) {
        g.ortho[i] = b[i];
        for (std::size_t j = 0; j < i; ++j) {
            g.mu[i][j] = dot(b[i], g.ortho[j]) / g.norm2[j];
            g.ortho[i] = g.ortho[i] - g.ortho[j] * g.mu[i][j];
        }
        g.norm2[i] = dot(g.ortho[i], g.ortho[i]);
    }
    return g;
}

/// LLL reduction of three lattice vectors. Only the basis changes, so the
/// lattice is the same; the result has short, nearly orthogonal vectors.
std::array<Vec3, 3> lll_reduce(std::array<Vec3, 3> b) {
    constexpr double kDelta = 0.99;
    std::size_t k = 1;
    for (int step = 0; step < 10000 && k < 3; ++step) {
        GramSchmidt g = gram_schmidt(b);
        for (std::size_t j = k; j-- > 0;) {  // size reduction against b[j]
            const double q = std::round(g.mu[k][j]);
            if (q != 0.0) {
                b[k] = b[k] - b[j] * q;
                g = gram_schmidt(b);
            }
        }
        const double m = g.mu[k][k - 1];
        if (g.norm2[k] < (kDelta - m * m) * g.norm2[k - 1]) {  // Lovasz condition
            std::swap(b[k], b[k - 1]);
            k = std::max<std::size_t>(k - 1, 1);
        } else {
            ++k;
        }
    }
    return b;
}

/// Fold a fractional coordinate into [0, 1).
Vec3 wrap_unit(const Vec3& f) {
    Vec3 w;
    for (std::size_t k = 0; k < 3; ++k) {
        w[k] = f[k] - std::floor(f[k]);
        if (w[k] >= 1.0) {  // x = -1e-17 gives 1.0 in floating point
            w[k] = 0.0;
        }
    }
    return w;
}

/// Spatial hash of fractional points in a periodic cell. Two points are the
/// same when their minimum-image Cartesian distance is at most the tolerance.
///
/// The cell is cut into buckets per lattice direction, sized so that any two
/// points within the tolerance fall into neighbouring buckets (a Cartesian
/// displacement r changes fractional coordinate k by at most |b_k| |r|, with
/// b_k the reciprocal vector). A query therefore visits at most 27 buckets.
class PointIndex {
public:
    PointIndex(const Cell& cell, double tolerance)
        : cell_(cell), tolerance2_(tolerance * tolerance) {
        const Mat3& m = cell.matrix();
        const std::array<Vec3, 3> basis = lll_reduce({m.row(0), m.row(1), m.row(2)});
        reduced_ = Mat3::from_rows(basis[0], basis[1], basis[2]);
        reduced_recip_ = reduced_.inverse().transposed();
        const Mat3 reciprocal = cell.reciprocal();
        for (std::size_t k = 0; k < 3; ++k) {
            const double reach = tolerance * reciprocal.row(k).norm();
            const double count = reach > 0.0 ? std::floor(1.0 / reach) : kMaxBuckets;
            buckets_[k] = static_cast<std::size_t>(std::clamp(count, 1.0, kMaxBuckets));
        }
    }

    /// Store a point; returns its id (ids count up from 0 in insertion order).
    std::size_t add(const Vec3& f) {
        const std::size_t id = points_.size();
        points_.push_back(f);
        cells_[key(f)].push_back(id);
        return id;
    }

    /// Replace the position of point `id`.
    void move(std::size_t id, const Vec3& f) {
        const std::uint64_t old_key = key(points_[id]);
        const std::uint64_t new_key = key(f);
        points_[id] = f;
        if (old_key == new_key) {
            return;
        }
        std::vector<std::size_t>& old_cell = cells_[old_key];
        old_cell.erase(std::find(old_cell.begin(), old_cell.end(), id));
        cells_[new_key].push_back(id);
    }

    /// Smallest id of a stored point within the tolerance of `f`, if any.
    [[nodiscard]] std::optional<std::size_t> first_within(const Vec3& f) const {
        std::optional<std::size_t> best;
        for_each_neighbour_cell(f, [&](const std::vector<std::size_t>& ids) {
            for (const std::size_t id : ids) {
                if ((!best || id < *best) && distance2(f, points_[id]) <= tolerance2_) {
                    best = id;
                }
            }
        });
        return best;
    }

private:
    static constexpr double kMaxBuckets = 65536.0;

    /// Squared minimum-image Cartesian distance: the shortest of the lattice
    /// images of a - b. Coordinates are taken in the LLL-reduced basis, where
    /// the nearest lattice point lies within two steps of the rounded
    /// coordinates. This holds for skewed cells whose nearest image has a
    /// large fractional offset, which a search around the fractional difference
    /// misses.
    [[nodiscard]] double distance2(const Vec3& a, const Vec3& b) const {
        const Vec3 r = cell_.cartesian(a - b);
        std::array<double, 3> centre{};
        for (std::size_t k = 0; k < 3; ++k) {
            centre[k] = std::round(dot(r, reduced_recip_.row(k)));
        }
        double best = std::numeric_limits<double>::infinity();
        for (int i = -2; i <= 2; ++i) {
            for (int j = -2; j <= 2; ++j) {
                for (int k = -2; k <= 2; ++k) {
                    const Vec3 image = reduced_.row(0) * (centre[0] + i) +
                                       reduced_.row(1) * (centre[1] + j) +
                                       reduced_.row(2) * (centre[2] + k);
                    best = std::min(best, (r - image).norm2());
                }
            }
        }
        return best;
    }

    [[nodiscard]] std::size_t bucket(const Vec3& f, std::size_t k) const {
        const double scaled = std::floor(f[k] * static_cast<double>(buckets_[k]));
        const double clamped = std::clamp(scaled, 0.0, static_cast<double>(buckets_[k] - 1));
        return static_cast<std::size_t>(clamped);
    }

    [[nodiscard]] std::uint64_t key(const Vec3& f) const {
        std::uint64_t h = 0;
        for (std::size_t k = 0; k < 3; ++k) {
            h = h * buckets_[k] + bucket(f, k);
        }
        return h;
    }

    /// Calls `visit` once for each bucket that can hold a point within the
    /// tolerance of `f`. Neighbours wrap around, and with few buckets per axis
    /// the same bucket is not visited twice.
    template <typename Visit>
    void for_each_neighbour_cell(const Vec3& f, Visit&& visit) const {
        std::array<std::array<std::size_t, 3>, 3> axes{};
        std::array<std::size_t, 3> counts{};
        for (std::size_t k = 0; k < 3; ++k) {
            const std::size_t n = buckets_[k];
            const std::size_t base = bucket(f, k);
            for (std::size_t d = 0; d < 3; ++d) {
                const std::size_t c = (base + n - 1 + d) % n;  // base - 1, base, base + 1
                bool seen = false;
                for (std::size_t i = 0; i < counts[k]; ++i) {
                    seen = seen || axes[k][i] == c;
                }
                if (!seen) {
                    axes[k][counts[k]++] = c;
                }
            }
        }
        for (std::size_t i = 0; i < counts[0]; ++i) {
            for (std::size_t j = 0; j < counts[1]; ++j) {
                for (std::size_t k = 0; k < counts[2]; ++k) {
                    const std::uint64_t h =
                        (axes[0][i] * buckets_[1] + axes[1][j]) * buckets_[2] + axes[2][k];
                    const auto it = cells_.find(h);
                    if (it != cells_.end()) {
                        visit(it->second);
                    }
                }
            }
        }
    }

    Cell cell_;
    double tolerance2_;
    Mat3 reduced_;        ///< LLL-reduced lattice rows, for distance2
    Mat3 reduced_recip_;  ///< their reciprocal rows (coordinates of a Cartesian vector)
    std::array<std::size_t, 3> buckets_{};
    std::vector<Vec3> points_;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> cells_;
};

bool is_missing(const gm::cif::Block& block, const char* tag) {
    const std::string* value = block.find_value(tag);
    return value == nullptr || gm::cif::is_null(*value);
}

/// Atomic number from the type symbol (charge stripped, e.g. "O2-" -> O),
/// else from the type symbol or the label as a label.
/// The type symbol without its charge or digits ("O2-" -> "O", "D2+" -> "D").
std::string bare_type_symbol(const gm::SmallStructure::Site& site) {
    std::string bare = site.type_symbol;
    while (!bare.empty() && (bare.back() == '+' || bare.back() == '-' ||
                             (bare.back() >= '0' && bare.back() <= '9'))) {
        bare.pop_back();
    }
    return bare;
}

int atomic_number(const gm::SmallStructure::Site& site) {
    const std::string bare = bare_type_symbol(site);
    int z = try_number_from_symbol(bare);
    if (z < 0) {
        z = try_number_from_label(site.type_symbol);
    }
    if (z < 0) {
        z = try_number_from_label(site.label);
    }
    if (z < 0) {
        throw FormatError("cif: cannot determine the element of site '" + site.label + "'");
    }
    return z;
}

/// Isotope mass of a site labelled D or T (deuterium, tritium), else 0. The
/// symbol is read as atomic_number reads it (charge and digits stripped).
double isotope_mass(const gm::SmallStructure::Site& site) {
    const std::string bare = bare_type_symbol(site);
    if (bare == "D") {
        return kDeuteriumMass;
    }
    if (bare == "T") {
        return kTritiumMass;
    }
    return 0.0;
}

/// True when the block gives any symmetry: a symop loop, a Hall or H-M
/// symbol, or an IT number. Without one the structure is P1.
bool has_symmetry_source(const gm::SmallStructure& st) {
    return !st.symops.empty() || !st.spacegroup_hall.empty() || !st.spacegroup_hm.empty() ||
           st.spacegroup_number != 0;
}

/// The space group operations as a full list (symmetry ops times centring
/// vectors). Order: the symop loop if complete, else the Hall symbol, the
/// H-M symbol, then the IT number, as gemmi's tables provide them.
gm::GroupOps space_group_ops(const gm::SmallStructure& st) {
    if (!has_symmetry_source(st)) {
        return gm::split_centering_vectors({gm::Op::identity()});
    }
    if (!st.symops.empty()) {
        std::vector<gm::Op> ops;
        for (const std::string& triplet : st.symops) {
            try {
                const std::vector<gm::Op> parsed = gm::triplets_to_ops({triplet});
                ops.insert(ops.end(), parsed.begin(), parsed.end());
            } catch (const std::exception&) {
                throw FormatError("cif: block '" + st.name +
                                  "': cannot parse symmetry operation '" + triplet + "'");
            }
        }
        gm::GroupOps gops = gm::split_centering_vectors(ops);
        if (gm::is_complete(gops)) {
            return gops;
        }
        // An incomplete symop loop is not a group: use the symbols instead.
    }
    // The Hall, H-M and IT-number entries describe the same group in
    // different ways, so an unparsable one falls through to the next.
    if (!st.spacegroup_hall.empty()) {
        try {
            return gm::symops_from_hall(st.spacegroup_hall.c_str());
        } catch (const std::exception&) {
        }
    }
    if (!st.spacegroup_hm.empty()) {
        if (const gm::SpaceGroup* sg =
                gm::find_spacegroup_by_name(st.spacegroup_hm, st.cell.alpha, st.cell.gamma)) {
            return sg->operations();
        }
    }
    if (st.spacegroup_number != 0) {
        if (const gm::SpaceGroup* sg = gm::find_spacegroup_by_number(st.spacegroup_number)) {
            return sg->operations();
        }
    }
    std::string name = !st.spacegroup_hm.empty() ? st.spacegroup_hm : st.spacegroup_hall;
    throw FormatError("cif: no symmetry operations for block '" + st.name + "'" +
                      (name.empty() ? std::string() : " (space group '" + name + "' unknown)"));
}

/// All images of the sites of one block, in site order then op order, with
/// images closer than `tolerance` (Cartesian, Å) merged per site.
std::vector<Entry> expand_block(const gm::SmallStructure& st, const gm::GroupOps& gops,
                                const Cell& cell, double tolerance) {
    std::vector<Entry> entries;
    for (const gm::SmallStructure::Site& site : st.sites) {
        const int number = atomic_number(site);
        const double occupancy = site.occ;
        const double mass = isotope_mass(site);
        PointIndex seen(cell, tolerance);
        for (const gm::Op& sym : gops.sym_ops) {
            for (const gm::Op::Tran& cen : gops.cen_ops) {
                const gm::Op op = sym.add_centering(cen);
                const gm::FTransform image{
                    gm::Transform{gm::rot_as_mat33(op), gm::tran_as_vec3(op)}};
                const gm::Fractional g = image.apply(site.fract);
                const Vec3 f = wrap_unit(Vec3(g.x, g.y, g.z));
                if (seen.first_within(f)) {
                    continue;
                }
                seen.add(f);
                entries.push_back({number, site.label, occupancy, f, mass});
            }
        }
    }
    return entries;
}

/// Apply the disorder policy. MaxOccupancy keeps one entry per position
/// (positions closer than `tolerance`), the most occupied one, ties to the first.
std::vector<Entry> apply_disorder(std::vector<Entry> entries, CifDisorder disorder,
                                  const Cell& cell, double tolerance) {
    if (disorder != CifDisorder::MaxOccupancy) {
        return entries;
    }
    PointIndex index(cell, tolerance);
    std::vector<Entry> kept;
    for (Entry& e : entries) {
        const std::optional<std::size_t> same = index.first_within(e.fract);
        if (!same) {
            index.add(e.fract);
            kept.push_back(std::move(e));
        } else if (e.occupancy > kept[*same].occupancy) {
            index.move(*same, e.fract);
            kept[*same] = std::move(e);
        }
    }
    return kept;
}

Atoms build_atoms(const std::vector<Entry>& entries, const Cell& cell) {
    std::vector<int> numbers;
    std::vector<double> positions;
    std::vector<std::string> labels;
    std::vector<double> occupancy;
    numbers.reserve(entries.size());
    positions.reserve(3 * entries.size());
    labels.reserve(entries.size());
    occupancy.reserve(entries.size());
    for (const Entry& e : entries) {
        const Vec3 r = cell.cartesian(e.fract);
        numbers.push_back(e.number);
        positions.push_back(r.x);
        positions.push_back(r.y);
        positions.push_back(r.z);
        labels.push_back(e.label);
        occupancy.push_back(e.occupancy);
    }
    // Per-atom masses only when some site is an isotope, as ase.io.cif does;
    // the other atoms keep their element masses.
    const bool isotopes = std::any_of(entries.begin(), entries.end(),
                                      [](const Entry& e) { return e.mass > 0.0; });
    std::vector<double> masses;
    if (isotopes) {
        masses.reserve(entries.size());
        for (const Entry& e : entries) {
            masses.push_back(e.mass > 0.0 ? e.mass : atomic_mass(e.number));
        }
    }
    Atoms atoms(std::move(numbers), std::move(positions), cell);
    atoms.set_string_array("labels", labels);
    atoms.set_array("occupancy", occupancy, 1);
    if (isotopes) {
        atoms.set_masses(masses);
    }
    return atoms;
}

/// Reject atom-site layouts this reader does not handle, with a message that
/// says which one it is. Called for a block without `_atom_site_fract_x`.
void reject_unsupported_sites(gm::cif::Block& block) {
    if (block.find_values("_atom_site.label_atom_id") || block.find_values("_atom_site.Cartn_x") ||
        block.find_values("_atom_site.fract_x")) {
        throw FormatError("cif: block '" + block.name +
                          "': this looks like an mmCIF (macromolecular) file; not supported");
    }
    if (block.find_values("_atom_site_Cartn_x")) {
        throw FormatError("cif: block '" + block.name +
                          "': Cartesian-only atom sites are not supported "
                          "(needs _atom_site_fract_x, _atom_site_fract_y, _atom_site_fract_z)");
    }
}

/// True when a label must be quoted to read back as the same string: empty,
/// containing whitespace or quotes, starting with a character CIF reserves,
/// or equal to a reserved word (CIF keywords are case-insensitive).
bool needs_quotes(const std::string& s) {
    if (s.empty() || s.find_first_of(" \t\r\n'\"") != std::string::npos) {
        return true;
    }
    if (s == "?" || s == ".") {  // bare ? and . read back as missing values
        return true;
    }
    if (std::string_view("_#$[];").find(s[0]) != std::string_view::npos) {
        return true;
    }
    std::string lower = s;
    for (char& c : lower) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return lower == "loop_" || lower == "global_" || lower == "stop_" ||
           lower.rfind("data_", 0) == 0 || lower.rfind("save_", 0) == 0;
}

/// Write a label as a CIF token that reads back unchanged. Line breaks become
/// '_'. A label holding both quote characters cannot be quoted, so every
/// quote character becomes '_' (the label is then single-quoted).
std::string cif_token(std::string s) {
    std::replace(s.begin(), s.end(), '\n', '_');
    std::replace(s.begin(), s.end(), '\r', '_');
    if (!needs_quotes(s)) {
        return s;
    }
    if (s.find('\'') == std::string::npos) {
        return "'" + s + "'";
    }
    if (s.find('"') == std::string::npos) {
        return "\"" + s + "\"";
    }
    std::replace(s.begin(), s.end(), '\'', '_');
    std::replace(s.begin(), s.end(), '"', '_');
    return "'" + s + "'";
}

std::string number(double v) {
    char buf[64];
    const auto [end, ec] = std::to_chars(buf, buf + sizeof buf, v);
    (void)ec;
    return std::string(buf, end);
}

}  // namespace

std::vector<Atoms> read_cif(std::string_view text, const CifReadOptions& options) {
    if (!(options.merge_tolerance >= 0.0)) {
        throw FormatError("cif: 'merge_tolerance' must be a non-negative number");
    }
    try {
        gm::cif::Document doc = gm::cif::read_memory(text.data(), text.size(), "<cif>");

        std::vector<Atoms> structures;
        for (gm::cif::Block& block : doc.blocks) {
            if (!block.find_values("_atom_site_fract_x")) {
                reject_unsupported_sites(block);
                continue;
            }
            for (const char* tag : {"_cell_length_a", "_cell_length_b", "_cell_length_c",
                                    "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"}) {
                if (is_missing(block, tag)) {
                    throw FormatError("cif: block '" + block.name + "' has no cell parameters (" +
                                      tag + ")");
                }
            }
            const gm::SmallStructure st = gm::make_small_structure_from_block(block);
            if (st.sites.empty()) {
                continue;
            }
            for (const char* tag : {"_atom_site_fract_y", "_atom_site_fract_z"}) {
                if (!block.find_values(tag)) {
                    throw FormatError("cif: block '" + block.name + "' has atom sites but no " +
                                      tag);
                }
            }
            for (const gm::SmallStructure::Site& site : st.sites) {
                if (!std::isfinite(site.fract.x) || !std::isfinite(site.fract.y) ||
                    !std::isfinite(site.fract.z)) {
                    throw FormatError("cif: site '" + site.label + "' in block '" + block.name +
                                      "' has no finite fractional coordinates ('?', '.' or "
                                      "missing)");
                }
            }
            if (options.disorder == CifDisorder::Error) {
                for (const gm::SmallStructure::Site& site : st.sites) {
                    if (site.occ < 1.0) {
                        throw FormatError("cif: site '" + site.label + "' in block '" +
                                          block.name + "' has occupancy " +
                                          number(site.occ) + " (disorder = error)");
                    }
                }
            }
            const Cell cell = Cell::from_parameters(st.cell.a, st.cell.b, st.cell.c,
                                                    st.cell.alpha, st.cell.beta, st.cell.gamma);
            if (!cell.is_invertible()) {
                throw FormatError("cif: block '" + block.name + "' has a singular cell");
            }
            const gm::GroupOps gops = space_group_ops(st);
            std::vector<Entry> entries =
                apply_disorder(expand_block(st, gops, cell, options.merge_tolerance),
                               options.disorder, cell, options.merge_tolerance);

            Atoms atoms = build_atoms(entries, cell);
            if (!st.spacegroup_hm.empty()) {
                atoms.info()["spacegroup"] = Value(st.spacegroup_hm);
            } else if (!has_symmetry_source(st)) {
                atoms.info()["spacegroup"] = Value(std::string("P 1"));
            }
            structures.push_back(std::move(atoms));
        }

        if (structures.empty()) {
            throw FormatError("cif: no atoms in any data block");
        }
        return structures;
    } catch (const FormatError&) {
        throw;
    } catch (const std::exception& error) {
        throw FormatError(std::string("cif: ") + error.what());
    }
}

std::string write_cif(const Atoms& atoms) {
    const Cell& cell = atoms.cell();
    if (cell.is_zero() || !cell.is_invertible()) {
        throw FormatError("cif: writing needs a cell with three lattice vectors");
    }
    const Vec3 lengths = cell.lengths();
    const Vec3 angles = cell.angles();
    const std::size_t n = atoms.size();
    const std::vector<double> scaled = atoms.scaled_positions(false);
    const std::vector<std::string> labels = atoms.labels();
    const std::vector<std::string> symbols = atoms.symbols();
    const bool has_occupancy = atoms.has_array("occupancy") &&
                               atoms.array("occupancy").ncols == 1 &&
                               atoms.array("occupancy").data.size() == n;

    std::string out;
    out.reserve(256 + n * 96);
    out += "data_";
    const std::string formula = atoms.formula();
    out += formula.empty() ? std::string("structure") : formula;
    out += "\n_cell_length_a ";
    out += number(lengths[0]);
    out += "\n_cell_length_b ";
    out += number(lengths[1]);
    out += "\n_cell_length_c ";
    out += number(lengths[2]);
    out += "\n_cell_angle_alpha ";
    out += number(angles[0]);
    out += "\n_cell_angle_beta ";
    out += number(angles[1]);
    out += "\n_cell_angle_gamma ";
    out += number(angles[2]);
    out += "\n_symmetry_space_group_name_H-M 'P 1'\n";
    out += "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
           "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n";
    if (has_occupancy) {
        out += "_atom_site_occupancy\n";
    }
    for (std::size_t i = 0; i < n; ++i) {
        out += cif_token(labels[i]);
        out += ' ';
        out += cif_token(symbols[i]);
        for (std::size_t k = 0; k < 3; ++k) {
            out += ' ';
            out += number(scaled[3 * i + k]);
        }
        if (has_occupancy) {
            out += ' ';
            out += number(atoms.array("occupancy").data[i]);
        }
        out += '\n';
    }
    return out;
}

}  // namespace fastatoms::io
