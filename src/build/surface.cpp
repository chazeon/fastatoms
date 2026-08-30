// SPDX-License-Identifier: MIT
#include "fastatoms/build/surface.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "fastatoms/core/elements.hpp"
#include "fastatoms/phonon/supercell.hpp"

namespace fastatoms::build {
namespace {

/// Solve `a x + b y = gcd(a, b)`, returning (x, y).
std::pair<int, int> extended_gcd(int a, int b) {
    if (b == 0) {
        return {1, 0};
    }
    if (a % b == 0) {
        return {0, 1};
    }
    const auto [x, y] = extended_gcd(b, a % b);
    return {y, x - y * (a / b)};
}

Vec3 row(const Mat3& matrix, std::size_t index) {
    return Vec3{matrix.m[3 * index], matrix.m[3 * index + 1], matrix.m[3 * index + 2]};
}

Mat3 rows(const Vec3& a, const Vec3& b, const Vec3& c) { return Mat3::from_rows(a, b, c); }

/// The slab, given three integer vectors that re-express the bulk cell with
/// `c3` along the surface normal.
Atoms build_slab(const Atoms& crystal, const Mat3& basis, int layers, bool periodic) {
    constexpr double tol = 1e-10;

    // Re-express every atom in the new basis, then wrap into it. Doing this
    // before the cell changes keeps the atoms where they were in space.
    const Mat3 inverse = basis.inverse();
    std::vector<double> scaled = crystal.scaled_positions();
    for (std::size_t atom = 0; atom < crystal.size(); ++atom) {
        const Vec3 old{scaled[3 * atom], scaled[3 * atom + 1], scaled[3 * atom + 2]};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            double value = old.x * inverse.m[axis] + old.y * inverse.m[3 + axis] +
                           old.z * inverse.m[6 + axis];
            value -= std::floor(value + tol);
            scaled[3 * atom + axis] = value;
        }
    }

    Atoms slab = crystal;
    const Mat3& cell = crystal.cell().matrix();
    Mat3 new_cell{};
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            double sum = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                sum += basis.m[3 * r + k] * cell.m[3 * k + c];
            }
            new_cell.m[3 * r + c] = sum;
        }
    }
    slab.set_cell(new_cell);
    slab.set_scaled_positions(scaled);

    slab = slab.repeat(1, 1, layers);

    // The third vector need not be perpendicular to the surface; project it so
    // that it is, which is what makes "add vacuum along z" mean anything.
    Vec3 a1 = row(slab.cell().matrix(), 0);
    Vec3 a2 = row(slab.cell().matrix(), 1);
    Vec3 a3 = row(slab.cell().matrix(), 2);
    const Vec3 normal = cross(a1, a2);
    const double normal_squared = dot(normal, normal);
    slab.set_cell(rows(a1, a2, normal * (dot(a3, normal) / normal_squared)));

    // Reduce the in-plane pair to the shortest, most nearly orthogonal basis of
    // the same two-dimensional lattice (Gauss reduction). The cut can hand back
    // a badly skewed pair - fcc(110) comes out at 125 degrees - which describes
    // the right surface in an awkward box, and every later repeat or adsorbate
    // placement inherits the awkwardness.
    {
        Vec3 b1 = row(slab.cell().matrix(), 0);
        Vec3 b2 = row(slab.cell().matrix(), 1);
        const Vec3 c = row(slab.cell().matrix(), 2);
        for (int guard = 0; guard < 100; ++guard) {
            if (dot(b1, b1) > dot(b2, b2)) {
                std::swap(b1, b2);
            }
            const long steps = std::lround(dot(b1, b2) / dot(b1, b1));
            if (steps == 0) {
                break;
            }
            b2 = b2 - b1 * static_cast<double>(steps);
        }
        if (dot(b1, b1) > dot(b2, b2)) {
            std::swap(b1, b2);
        }
        // Swapping can flip the handedness; negating one vector spans the same
        // lattice and puts it back.
        if (dot(cross(b1, b2), c) < 0.0) {
            b2 = b2 * -1.0;
        }
        slab.set_cell(rows(b1, b2, c));
    }

    // Turn the cell so a1 runs along x and the normal along z. `scale_atoms`
    // carries the atoms with it, so this is a rotation, not a distortion.
    a1 = row(slab.cell().matrix(), 0);
    a2 = row(slab.cell().matrix(), 1);
    a3 = row(slab.cell().matrix(), 2);
    const double length1 = a1.norm();
    const double projection = dot(a1, a2) / length1;
    const double height = std::sqrt(std::max(0.0, dot(a2, a2) - projection * projection));
    slab.set_cell(rows(Vec3{length1, 0.0, 0.0}, Vec3{projection, height, 0.0},
                       Vec3{0.0, 0.0, a3.norm()}),
                  /*scale_atoms=*/true);

    slab.set_pbc({true, true, periodic});

    // Wrap in the two surface directions only; wrapping along the normal would
    // fold the slab through itself.
    std::vector<double> final_scaled = slab.scaled_positions();
    for (std::size_t atom = 0; atom < slab.size(); ++atom) {
        for (std::size_t axis = 0; axis < 2; ++axis) {
            double& value = final_scaled[3 * atom + axis];
            value -= std::floor(value);
        }
    }
    slab.set_scaled_positions(final_scaled);
    return slab;
}

/// Look up a lattice constant, insisting the element is in the phase asked for.
double lattice_constant(const std::string& symbol, const char* structure, double given) {
    if (given > 0.0) {
        return given;
    }
    const ReferenceState& state = reference_state(number_from_symbol(symbol));
    if (state.symmetry != structure) {
        throw BuildError(std::string(symbol) + " is " +
                         (state.symmetry.empty() ? "not tabulated" : std::string(state.symmetry)) +
                         " at ambient conditions, not " + structure +
                         ". Pass `a` to build the " + structure +
                         " phase anyway, which is a real question about a phase this element is "
                         "not in.");
    }
    if (state.a <= 0.0) {
        throw BuildError("no lattice constant is tabulated for " + symbol + "; pass one");
    }
    return state.a;
}

void require_size(const std::array<int, 3>& size) {
    for (const int value : size) {
        if (value < 1) {
            throw BuildError("a slab size is three positive numbers: (nx, ny, layers)");
        }
    }
}

/// The common tail: cut the face, repeat in plane, add vacuum, record where
/// the adsorption sites are.
Atoms finish(Atoms slab, const std::array<int, 3>& size, double vacuum,
             const std::vector<std::pair<std::string, std::array<double, 2>>>& sites) {
    slab = slab.repeat(size[0], size[1], 1);
    if (vacuum > 0.0) {
        slab.center(vacuum, {false, false, true});
    }

    // Sites are recorded in fractions of the *surface* cell, so a repeat does
    // not move them and `add_adsorbate` needs no knowledge of the face.
    Value::Map map;
    for (const auto& [name, position] : sites) {
        Value::List pair;
        pair.push_back(Value(position[0] / size[0]));
        pair.push_back(Value(position[1] / size[1]));
        map.emplace_back(name, Value(std::move(pair)));
    }
    Value::Map info = slab.info().is_map() ? slab.info().as_map() : Value::Map{};
    info.emplace_back("adsorbate_sites", Value(std::move(map)));
    slab.set_info(Value(std::move(info)));
    return slab;
}

}  // namespace

Atoms surface(const Atoms& crystal, const std::array<int, 3>& indices, int layers, double vacuum,
              bool periodic) {
    if (layers < 1) {
        throw BuildError("a slab needs at least one layer");
    }
    if (indices[0] == 0 && indices[1] == 0 && indices[2] == 0) {
        throw BuildError("(0 0 0) does not name a plane");
    }
    if (std::abs(crystal.cell().matrix().determinant()) < 1e-12) {
        throw BuildError("surface(): the crystal has no cell to cut");
    }

    const int h = indices[0];
    const int k = indices[1];
    const int l = indices[2];
    const bool h0 = h == 0;
    const bool k0 = k == 0;
    const bool l0 = l == 0;

    Mat3 basis{};
    if ((h0 && k0) || (h0 && l0) || (k0 && l0)) {
        // Two indices zero: the plane is a face of the cell, so the basis is a
        // permutation that puts the odd axis last.
        if (!h0) {
            basis = Mat3{0, 1, 0, 0, 0, 1, 1, 0, 0};
        } else if (!k0) {
            basis = Mat3{0, 0, 1, 1, 0, 0, 0, 1, 0};
        } else {
            basis = Mat3{1, 0, 0, 0, 1, 0, 0, 0, 1};
        }
    } else {
        auto [p, q] = extended_gcd(k, l);
        const Mat3& cell = crystal.cell().matrix();
        const Vec3 a1 = row(cell, 0);
        const Vec3 a2 = row(cell, 1);
        const Vec3 a3 = row(cell, 2);

        // Of the family of in-plane vectors, pick the pair that is closest to
        // orthogonal - otherwise a high-index face comes out as a needle.
        const Vec3 reference = a2 * static_cast<double>(l) - a3 * static_cast<double>(k);
        const double k1 = dot((a1 * static_cast<double>(k) - a2 * static_cast<double>(h)) *
                                      static_cast<double>(p) +
                                  (a1 * static_cast<double>(l) - a3 * static_cast<double>(h)) *
                                      static_cast<double>(q),
                              reference);
        const double k2 = dot((a1 * static_cast<double>(k) - a2 * static_cast<double>(h)) *
                                      static_cast<double>(l) -
                                  (a1 * static_cast<double>(l) - a3 * static_cast<double>(h)) *
                                      static_cast<double>(k),
                              reference);
        if (std::abs(k2) > 1e-10) {
            const int shift = -static_cast<int>(std::lround(k1 / k2));
            p += shift * l;
            q -= shift * k;
        }

        const auto [alpha, beta] = extended_gcd(p * k + q * l, h);
        const int divisor = std::abs(std::gcd(l, k));
        basis = Mat3{static_cast<double>(p * k + q * l),
                     static_cast<double>(-p * h),
                     static_cast<double>(-q * h),
                     0.0,
                     static_cast<double>(l / divisor),
                     static_cast<double>(-k / divisor),
                     static_cast<double>(beta),
                     static_cast<double>(alpha * p),
                     static_cast<double>(alpha * q)};
    }

    Atoms slab = build_slab(crystal, basis, layers, periodic);
    if (vacuum > 0.0) {
        slab.center(vacuum, {false, false, true});
    }
    return slab;
}

// ---------------------------------------------------------------------------
// The named faces
// ---------------------------------------------------------------------------

namespace {


/// Re-express conventional Miller indices in another cell's basis.
///
/// `fcc111` and friends name planes of the *conventional cubic* cell, which is
/// what everyone means by "the (100) face". Cutting those same numbers out of
/// the primitive cell gives a different plane entirely - primitive (100) is
/// conventional (111) - so the indices have to be converted before the cut.
///
/// Cutting the primitive cell rather than the conventional one is what keeps
/// the surface cell primitive too, so `(2, 2, 4)` means four atoms a layer and
/// not eight.
std::array<int, 3> miller_in_cell(const Mat3& conventional, const std::array<int, 3>& indices,
                                  const Mat3& target) {
    // The plane's reciprocal vector, in Cartesian coordinates.
    const Mat3 reciprocal = conventional.inverse().transposed();
    Vec3 g{0.0, 0.0, 0.0};
    for (std::size_t k = 0; k < 3; ++k) {
        g = g + row(reciprocal, k) * static_cast<double>(indices[k]);
    }

    // Its components on the target lattice vectors are the indices there, and
    // are integers whenever the two cells describe the same lattice.
    std::array<double, 3> exact{};
    for (std::size_t k = 0; k < 3; ++k) {
        exact[k] = dot(g, row(target, k));
    }

    // Clear any halves, then reduce: (0, 0.5, 0.5) is the plane (0, 1, 1).
    for (int scale : {1, 2, 3, 4, 6}) {
        std::array<int, 3> candidate{};
        bool integral = true;
        for (std::size_t k = 0; k < 3; ++k) {
            const double scaled = exact[k] * scale;
            candidate[k] = static_cast<int>(std::lround(scaled));
            if (std::abs(scaled - static_cast<double>(candidate[k])) > 1e-8) {
                integral = false;
                break;
            }
        }
        if (!integral) {
            continue;
        }
        const int divisor = std::gcd(std::gcd(std::abs(candidate[0]), std::abs(candidate[1])),
                                     std::abs(candidate[2]));
        if (divisor > 1) {
            for (int& value : candidate) {
                value /= divisor;
            }
        }
        return candidate;
    }
    throw BuildError("surface(): these Miller indices do not name a plane of that lattice");
}

/// How many distinct atomic planes a slab has, and where they sit.
std::vector<double> layer_heights(const Atoms& slab, double tolerance = 0.1) {
    std::vector<double> heights;
    for (std::size_t atom = 0; atom < slab.size(); ++atom) {
        const double z = slab.positions()[3 * atom + 2];
        const auto near = std::find_if(heights.begin(), heights.end(), [&](double existing) {
            return std::abs(existing - z) < tolerance;
        });
        if (near == heights.end()) {
            heights.push_back(z);
        }
    }
    std::sort(heights.begin(), heights.end());
    return heights;
}

/// A named face is the general cut plus in-plane repeats and the site map.
///
/// `size[2]` counts *atomic layers*, not repeats of the surface cell - the two
/// differ wherever the cell is more than one plane thick, as hcp(0001) and
/// diamond(111) are, and "five layers" meaning ten planes for one face and
/// five for another would be a trap.
Atoms named_face(const Atoms& crystal, const std::array<int, 3>& conventional_indices,
                 const std::array<int, 3>& size, double vacuum,
                 const std::vector<std::pair<std::string, std::array<double, 2>>>& sites,
                 const Mat3& conventional) {
    require_size(size);
    const std::array<int, 3> indices =
        miller_in_cell(conventional, conventional_indices, crystal.cell().matrix());

    // One repeat, to find out how many planes it holds.
    const Atoms single = surface(crystal, indices, 1, 0.0, false);
    const std::size_t per_repeat = std::max<std::size_t>(1, layer_heights(single).size());
    const int repeats =
        static_cast<int>((static_cast<std::size_t>(size[2]) + per_repeat - 1) / per_repeat);

    Atoms slab = surface(crystal, indices, repeats, 0.0, false);
    const std::vector<double> heights = layer_heights(slab);
    if (heights.size() > static_cast<std::size_t>(size[2])) {
        // Keep the topmost `size[2]` planes: the surface is what is being
        // asked about, so the bottom of the block is what goes.
        const double cut = heights[heights.size() - static_cast<std::size_t>(size[2])] - 0.1;
        std::vector<std::size_t> keep;
        for (std::size_t atom = 0; atom < slab.size(); ++atom) {
            if (slab.positions()[3 * atom + 2] >= cut) {
                keep.push_back(atom);
            }
        }
        slab = slab.subset(keep);
    }
    return finish(std::move(slab), size, vacuum, sites);
}

/// Sites of a four-fold face: one atom per surface cell.
const std::vector<std::pair<std::string, std::array<double, 2>>>& square_sites() {
    static const std::vector<std::pair<std::string, std::array<double, 2>>> sites{
        {"ontop", {0.0, 0.0}}, {"bridge", {0.5, 0.0}}, {"hollow", {0.5, 0.5}}};
    return sites;
}

/// Sites of a close-packed face. `fcc` sits over a third-layer atom, `hcp`
/// over a second-layer one, which is the distinction that matters for binding.
const std::vector<std::pair<std::string, std::array<double, 2>>>& hexagonal_sites() {
    static const std::vector<std::pair<std::string, std::array<double, 2>>> sites{
        {"ontop", {0.0, 0.0}},
        {"bridge", {0.5, 0.0}},
        {"fcc", {1.0 / 3.0, 1.0 / 3.0}},
        {"hcp", {2.0 / 3.0, 2.0 / 3.0}},
        {"hollow", {1.0 / 3.0, 1.0 / 3.0}}};
    return sites;
}

/// Re-express a slanted surface cell as a rectangular one.
///
/// The second vector is replaced by the shortest `n a1 + m a2` that is
/// perpendicular to `a1`. For the 60- and 120-degree cells of a close-packed
/// face that is `2 a2 -+ a1`, so the cell holds twice the atoms and has right
/// angles - which is what makes a slab easy to stack, cut or hand to a code
/// that assumes an orthogonal box.
Atoms maybe_orthogonal(Atoms slab, bool orthogonal) {
    if (!orthogonal) {
        return slab;
    }
    const Vec3 a1 = row(slab.cell().matrix(), 0);
    const Vec3 a2 = row(slab.cell().matrix(), 1);
    const double ratio = dot(a1, a2) / dot(a1, a1);
    if (std::abs(ratio) < 1e-10) {
        return slab;   // already rectangular
    }

    // `ratio` has to be a half-integer for an integer combination to be
    // perpendicular; anything else is a cell no repeat can square off.
    const double doubled = 2.0 * ratio;
    const long rounded = std::lround(doubled);
    if (std::abs(doubled - static_cast<double>(rounded)) > 1e-8) {
        throw BuildError("orthogonal=True: this surface cell cannot be made rectangular by "
                         "repeating it; use the natural cell, or cut the face with surface()");
    }

    const Mat3 transformation{1, 0, 0, static_cast<double>(-rounded), 2, 0, 0, 0, 1};
    Atoms squared = phonon::make_supercell(slab, transformation).atoms;
    squared.set_pbc(slab.cell().pbc());
    squared.set_info(slab.info());
    return squared;
}

}  // namespace

Atoms fcc100(const std::string& symbol, const std::array<int, 3>& size, double a, double vacuum,
             bool orthogonal) {
    const double edge = lattice_constant(symbol, "fcc", a);
    const Atoms crystal = fcc(symbol, edge);
    return maybe_orthogonal(named_face(crystal, {1, 0, 0}, size, vacuum, square_sites(), Cell::cubic(edge).matrix()),
                            orthogonal);
}

Atoms fcc110(const std::string& symbol, const std::array<int, 3>& size, double a, double vacuum,
             bool orthogonal) {
    const double edge = lattice_constant(symbol, "fcc", a);
    const Atoms crystal = fcc(symbol, edge);
    return maybe_orthogonal(named_face(crystal, {1, 1, 0}, size, vacuum, square_sites(), Cell::cubic(edge).matrix()),
                            orthogonal);
}

Atoms fcc111(const std::string& symbol, const std::array<int, 3>& size, double a, double vacuum,
             bool orthogonal) {
    const double edge = lattice_constant(symbol, "fcc", a);
    const Atoms crystal = fcc(symbol, edge);
    return maybe_orthogonal(named_face(crystal, {1, 1, 1}, size, vacuum, hexagonal_sites(), Cell::cubic(edge).matrix()),
                            orthogonal);
}

Atoms fcc211(const std::string& symbol, const std::array<int, 3>& size, double a, double vacuum,
             bool orthogonal) {
    const double edge = lattice_constant(symbol, "fcc", a);
    const Atoms crystal = fcc(symbol, edge);
    return maybe_orthogonal(named_face(crystal, {2, 1, 1}, size, vacuum, square_sites(), Cell::cubic(edge).matrix()),
                            orthogonal);
}

Atoms bcc100(const std::string& symbol, const std::array<int, 3>& size, double a, double vacuum,
             bool orthogonal) {
    const double edge = lattice_constant(symbol, "bcc", a);
    const Atoms crystal = bcc(symbol, edge);
    return maybe_orthogonal(named_face(crystal, {1, 0, 0}, size, vacuum, square_sites(), Cell::cubic(edge).matrix()),
                            orthogonal);
}

Atoms bcc110(const std::string& symbol, const std::array<int, 3>& size, double a, double vacuum,
             bool orthogonal) {
    const double edge = lattice_constant(symbol, "bcc", a);
    const Atoms crystal = bcc(symbol, edge);
    return maybe_orthogonal(named_face(crystal, {1, 1, 0}, size, vacuum, square_sites(), Cell::cubic(edge).matrix()),
                            orthogonal);
}

Atoms bcc111(const std::string& symbol, const std::array<int, 3>& size, double a, double vacuum,
             bool orthogonal) {
    const double edge = lattice_constant(symbol, "bcc", a);
    const Atoms crystal = bcc(symbol, edge);
    return maybe_orthogonal(named_face(crystal, {1, 1, 1}, size, vacuum, hexagonal_sites(), Cell::cubic(edge).matrix()),
                            orthogonal);
}

Atoms hcp0001(const std::string& symbol, const std::array<int, 3>& size, double a, double c,
              double vacuum, bool orthogonal) {
    const double edge = lattice_constant(symbol, "hcp", a);
    if (c <= 0.0) {
        const ReferenceState& state = reference_state(number_from_symbol(symbol));
        c = state.covera > 0.0 ? edge * state.covera : edge * std::sqrt(8.0 / 3.0);
    }
    return maybe_orthogonal(named_face(hcp(symbol, edge, c), {0, 0, 1}, size, vacuum, hexagonal_sites(),
                                       hcp(symbol, edge, c).cell().matrix()),
                            orthogonal);
}

Atoms diamond100(const std::string& symbol, const std::array<int, 3>& size, double a,
                 double vacuum) {
    const double edge = lattice_constant(symbol, "diamond", a);
    const Atoms crystal = diamond(symbol, edge);
    return named_face(crystal, {1, 0, 0}, size, vacuum, square_sites(), Cell::cubic(edge).matrix());
}

Atoms diamond111(const std::string& symbol, const std::array<int, 3>& size, double a,
                 double vacuum) {
    const double edge = lattice_constant(symbol, "diamond", a);
    const Atoms crystal = diamond(symbol, edge);
    return named_face(crystal, {1, 1, 1}, size, vacuum, hexagonal_sites(), Cell::cubic(edge).matrix());
}

// ---------------------------------------------------------------------------
// Vacuum and adsorbates
// ---------------------------------------------------------------------------

void add_vacuum(Atoms& atoms, double vacuum) {
    if (vacuum < 0.0) {
        throw BuildError("vacuum is a distance, so it cannot be negative");
    }
    Mat3 cell = atoms.cell().matrix();
    const Vec3 c = row(cell, 2);
    const double length = c.norm();
    if (length < 1e-12) {
        throw BuildError("add_vacuum(): the cell has no third vector to extend");
    }
    const Vec3 stretched = c * ((length + vacuum) / length);
    cell.m[6] = stretched.x;
    cell.m[7] = stretched.y;
    cell.m[8] = stretched.z;
    atoms.set_cell(cell);
}

std::string_view site_name(Site site) {
    switch (site) {
        case Site::Ontop:  return "ontop";
        case Site::Bridge: return "bridge";
        case Site::Hollow: return "hollow";
        case Site::FCC:    return "fcc";
        case Site::HCP:    return "hcp";
    }
    return "ontop";
}

Site site_from_string(std::string_view name) {
    std::string key(name);
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (key == "ontop" || key == "top")  return Site::Ontop;
    if (key == "bridge")                 return Site::Bridge;
    if (key == "hollow")                 return Site::Hollow;
    if (key == "fcc")                    return Site::FCC;
    if (key == "hcp")                    return Site::HCP;
    throw BuildError("'" + std::string(name) +
                     "' is not an adsorption site; try ontop, bridge, hollow, fcc or hcp");
}

namespace {

/// Place the adsorbate with `mol_index` at (x, y, top + height).
void place(Atoms& slab, const Atoms& adsorbate, double height, double x, double y,
           std::size_t mol_index) {
    if (adsorbate.size() == 0) {
        throw BuildError("add_adsorbate(): the adsorbate has no atoms");
    }
    if (mol_index >= adsorbate.size()) {
        throw BuildError("add_adsorbate(): atom " + std::to_string(mol_index) +
                         " of an adsorbate with " + std::to_string(adsorbate.size()) +
                         " atoms does not exist");
    }
    if (slab.size() == 0) {
        throw BuildError("add_adsorbate(): the slab has no atoms");
    }

    // The reference is the top of the *slab*, recorded the first time and
    // reused: measuring from whatever is currently highest would stack a
    // second adsorbate on top of the first.
    Value::Map info = slab.info().is_map() ? slab.info().as_map() : Value::Map{};
    double top = 0.0;
    const Value recorded = slab.info().is_map() ? slab.info()["surface_height"] : Value{};
    if (recorded.is_number()) {
        top = recorded.as_double();
    } else {
        top = slab.positions()[2];
        for (std::size_t atom = 1; atom < slab.size(); ++atom) {
            top = std::max(top, slab.positions()[3 * atom + 2]);
        }
        info.emplace_back("surface_height", Value(top));
        slab.set_info(Value(info));
    }

    // The named atom goes at the site; the rest keep their arrangement, which
    // is what lets an OH be bound through its oxygen without reorienting it.
    Atoms placed = adsorbate;
    const std::vector<double>& source = adsorbate.positions();
    const Vec3 anchor{source[3 * mol_index], source[3 * mol_index + 1],
                      source[3 * mol_index + 2]};
    std::vector<double> moved = source;
    for (std::size_t atom = 0; atom < placed.size(); ++atom) {
        moved[3 * atom + 0] += x - anchor.x;
        moved[3 * atom + 1] += y - anchor.y;
        moved[3 * atom + 2] += top + height - anchor.z;
    }
    placed.set_positions(moved);
    slab.extend(placed);
}

}  // namespace

void add_adsorbate(Atoms& slab, const Atoms& adsorbate, double height, Site site,
                   const std::array<double, 2>& offset, std::size_t mol_index) {
    const Value& info = slab.info();
    const Value sites = info.is_map() ? info["adsorbate_sites"] : Value{};
    if (!sites.is_map()) {
        throw BuildError(
            "add_adsorbate(): this slab does not know where its sites are. The named face "
            "builders (fcc111 and friends) record them; a slab from surface() or read from a "
            "file does not, so give an explicit (x, y) instead - or use AdsorptionFinder, which "
            "works out the sites of any structure.");
    }

    const Value entry = sites[std::string(site_name(site))];
    if (!entry.is_list()) {
        throw BuildError("add_adsorbate(): this face has no '" + std::string(site_name(site)) +
                         "' site");
    }
    const std::vector<double> fraction = entry.as_doubles();

    const Mat3& cell = slab.cell().matrix();
    const double u = fraction[0] + offset[0];
    const double v = fraction[1] + offset[1];
    const double x = u * cell.m[0] + v * cell.m[3];
    const double y = u * cell.m[1] + v * cell.m[4];
    place(slab, adsorbate, height, x, y, mol_index);
}

void add_adsorbate(Atoms& slab, const Atoms& adsorbate, double height,
                   const std::array<double, 2>& position, std::size_t mol_index) {
    place(slab, adsorbate, height, position[0], position[1], mol_index);
}

}  // namespace fastatoms::build
