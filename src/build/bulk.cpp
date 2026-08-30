// SPDX-License-Identifier: MIT

#include "fastatoms/build/bulk.hpp"

#include "fastatoms/core/elements.hpp"
#include "fastatoms/phonon/supercell.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace fastatoms::build {
namespace {

/// Assemble a structure from a cell and fractional coordinates, which is how
/// every lattice below is most readably written.
Atoms from_scaled(const std::vector<std::string>& symbols, const std::vector<double>& scaled,
                  const Cell& cell) {
    Atoms atoms(symbols, std::vector<double>(3 * symbols.size(), 0.0), cell);
    atoms.set_scaled_positions(scaled);
    atoms.set_pbc({true, true, true});
    return atoms;
}

void require_positive(double value, const char* what) {
    if (!(value > 0.0)) {
        throw BuildError(std::string(what) + " must be positive");
    }
}

/// The four sites of the conventional fcc cell, in fractional coordinates.
const std::vector<double>& fcc_sites() {
    static const std::vector<double> sites{0.0, 0.0, 0.0, 0.0, 0.5, 0.5,
                                           0.5, 0.0, 0.5, 0.5, 0.5, 0.0};
    return sites;
}

std::vector<std::string> repeated(const std::string& symbol, std::size_t n) {
    return std::vector<std::string>(n, symbol);
}

/// Alternating species, for the binary lattices.
std::vector<std::string> alternating(const std::string& first, const std::string& second,
                                     std::size_t pairs) {
    std::vector<std::string> out;
    out.reserve(2 * pairs);
    for (std::size_t k = 0; k < pairs; ++k) {
        out.push_back(first);
        out.push_back(second);
    }
    return out;
}

}  // namespace

std::string_view lattice_name(Lattice lattice) {
    switch (lattice) {
        case Lattice::SC:         return "sc";
        case Lattice::FCC:        return "fcc";
        case Lattice::BCC:        return "bcc";
        case Lattice::HCP:        return "hcp";
        case Lattice::Diamond:    return "diamond";
        case Lattice::RockSalt:   return "rocksalt";
        case Lattice::ZincBlende: return "zincblende";
        case Lattice::CsCl:       return "cscl";
        case Lattice::Fluorite:   return "fluorite";
        case Lattice::Wurtzite:   return "wurtzite";
        case Lattice::Hexagonal:  return "hexagonal";
        case Lattice::Tetragonal: return "tetragonal";
        case Lattice::Orthorhombic: return "orthorhombic";
        case Lattice::Rhombohedral: return "rhombohedral";
    }
    return "sc";
}

Lattice lattice_from_string(std::string_view name) {
    std::string key(name);
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    key.erase(std::remove_if(key.begin(), key.end(),
                             [](unsigned char c) { return c == '_' || c == '-' || c == ' '; }),
              key.end());

    if (key == "sc" || key == "simplecubic" || key == "cubic")   return Lattice::SC;
    if (key == "fcc" || key == "facecentredcubic")               return Lattice::FCC;
    if (key == "bcc" || key == "bodycentredcubic")               return Lattice::BCC;
    if (key == "hcp")                                            return Lattice::HCP;
    if (key == "hexagonal" || key == "hex")                      return Lattice::Hexagonal;
    if (key == "diamond" || key == "diamondcubic")               return Lattice::Diamond;
    if (key == "rocksalt" || key == "nacl" || key == "b1")       return Lattice::RockSalt;
    if (key == "zincblende" || key == "zns" || key == "b3")      return Lattice::ZincBlende;
    if (key == "cscl" || key == "b2")                            return Lattice::CsCl;
    if (key == "fluorite" || key == "caf2" || key == "c1")       return Lattice::Fluorite;
    if (key == "wurtzite" || key == "b4")                        return Lattice::Wurtzite;
    if (key == "tetragonal" || key == "bct")                     return Lattice::Tetragonal;
    if (key == "orthorhombic" || key == "ortho")                 return Lattice::Orthorhombic;
    if (key == "rhombohedral" || key == "rhomb" || key == "trigonal")
        return Lattice::Rhombohedral;
    throw BuildError("'" + std::string(name) + "' is not a lattice this builder knows");
}

Atoms sc(const std::string& symbol, double a) {
    require_positive(a, "the lattice constant");
    return from_scaled({symbol}, {0.0, 0.0, 0.0}, Cell::cubic(a));
}

Atoms fcc(const std::string& symbol, double a, bool cubic) {
    require_positive(a, "the lattice constant");
    if (cubic) {
        return from_scaled(repeated(symbol, 4), fcc_sites(), Cell::cubic(a));
    }
    // The primitive cell: half-face diagonals, one atom, a quarter the volume.
    const Cell cell(Vec3{0.0, a / 2, a / 2}, Vec3{a / 2, 0.0, a / 2}, Vec3{a / 2, a / 2, 0.0},
                    {true, true, true});
    return from_scaled({symbol}, {0.0, 0.0, 0.0}, cell);
}

Atoms bcc(const std::string& symbol, double a, bool cubic) {
    require_positive(a, "the lattice constant");
    if (cubic) {
        return from_scaled(repeated(symbol, 2), {0.0, 0.0, 0.0, 0.5, 0.5, 0.5}, Cell::cubic(a));
    }
    const Cell cell(Vec3{-a / 2, a / 2, a / 2}, Vec3{a / 2, -a / 2, a / 2},
                    Vec3{a / 2, a / 2, -a / 2}, {true, true, true});
    return from_scaled({symbol}, {0.0, 0.0, 0.0}, cell);
}

Atoms hcp(const std::string& symbol, double a, double c) {
    require_positive(a, "the lattice constant");
    if (c == 0.0) {
        c = a * std::sqrt(8.0 / 3.0);   // the ideal ratio, 1.633
    }
    require_positive(c, "the c axis");

    const Cell cell(Vec3{a, 0.0, 0.0}, Vec3{-a / 2, a * std::sqrt(3.0) / 2, 0.0},
                    Vec3{0.0, 0.0, c}, {true, true, true});
    return from_scaled(repeated(symbol, 2),
                       {0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 0.5}, cell);
}

Atoms diamond(const std::string& symbol, double a, bool cubic) {
    require_positive(a, "the lattice constant");
    if (cubic) {
        std::vector<double> scaled = fcc_sites();
        // The same four sites again, shifted along the body diagonal by a
        // quarter: that offset is the whole of the diamond structure.
        for (std::size_t k = 0; k < 12; ++k) {
            scaled.push_back(fcc_sites()[k] + 0.25);
        }
        return from_scaled(repeated(symbol, 8), scaled, Cell::cubic(a));
    }
    const Cell cell(Vec3{0.0, a / 2, a / 2}, Vec3{a / 2, 0.0, a / 2}, Vec3{a / 2, a / 2, 0.0},
                    {true, true, true});
    return from_scaled(repeated(symbol, 2),
                       {0.0, 0.0, 0.0, 0.25, 0.25, 0.25}, cell);
}

Atoms rocksalt(const std::string& first, const std::string& second, double a, bool cubic) {
    require_positive(a, "the lattice constant");
    if (cubic) {
        std::vector<double> scaled;
        std::vector<std::string> symbols;
        for (std::size_t k = 0; k < 4; ++k) {
            for (std::size_t c = 0; c < 3; ++c) {
                scaled.push_back(fcc_sites()[3 * k + c]);
            }
            symbols.push_back(first);
        }
        for (std::size_t k = 0; k < 4; ++k) {
            // The second sublattice sits half a cell along x from the first.
            scaled.push_back(fcc_sites()[3 * k] + 0.5);
            scaled.push_back(fcc_sites()[3 * k + 1]);
            scaled.push_back(fcc_sites()[3 * k + 2]);
            symbols.push_back(second);
        }
        return from_scaled(symbols, scaled, Cell::cubic(a));
    }
    const Cell cell(Vec3{0.0, a / 2, a / 2}, Vec3{a / 2, 0.0, a / 2}, Vec3{a / 2, a / 2, 0.0},
                    {true, true, true});
    return from_scaled(alternating(first, second, 1), {0.0, 0.0, 0.0, 0.5, 0.5, 0.5}, cell);
}

Atoms zincblende(const std::string& first, const std::string& second, double a, bool cubic) {
    require_positive(a, "the lattice constant");
    if (cubic) {
        std::vector<double> scaled;
        std::vector<std::string> symbols;
        for (std::size_t k = 0; k < 4; ++k) {
            for (std::size_t c = 0; c < 3; ++c) {
                scaled.push_back(fcc_sites()[3 * k + c]);
            }
            symbols.push_back(first);
        }
        for (std::size_t k = 0; k < 12; ++k) {
            scaled.push_back(fcc_sites()[k] + 0.25);
        }
        for (std::size_t k = 0; k < 4; ++k) {
            symbols.push_back(second);
        }
        return from_scaled(symbols, scaled, Cell::cubic(a));
    }
    const Cell cell(Vec3{0.0, a / 2, a / 2}, Vec3{a / 2, 0.0, a / 2}, Vec3{a / 2, a / 2, 0.0},
                    {true, true, true});
    return from_scaled(alternating(first, second, 1), {0.0, 0.0, 0.0, 0.25, 0.25, 0.25}, cell);
}

Atoms cscl(const std::string& first, const std::string& second, double a) {
    require_positive(a, "the lattice constant");
    return from_scaled({first, second}, {0.0, 0.0, 0.0, 0.5, 0.5, 0.5}, Cell::cubic(a));
}


Atoms fluorite(const std::string& first, const std::string& second, double a, bool cubic) {
    require_positive(a, "the lattice constant");
    if (cubic) {
        std::vector<std::string> symbols;
        std::vector<double> scaled;
        // Four cations on the fcc sites, eight anions on the tetrahedral ones.
        const double fcc_sites[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
        for (const auto& site : fcc_sites) {
            symbols.push_back(first);
            scaled.insert(scaled.end(), {site[0], site[1], site[2]});
        }
        for (const auto& site : fcc_sites) {
            for (const double offset : {0.25, 0.75}) {
                symbols.push_back(second);
                scaled.insert(scaled.end(),
                              {site[0] + offset, site[1] + offset, site[2] + offset});
            }
        }
        return from_scaled(symbols, scaled, Cell::cubic(a));
    }
    const Cell cell(Vec3{0.0, a / 2, a / 2}, Vec3{a / 2, 0.0, a / 2}, Vec3{a / 2, a / 2, 0.0},
                    {true, true, true});
    return from_scaled({first, second, second},
                       {0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75}, cell);
}

Atoms wurtzite(const std::string& first, const std::string& second, double a, double c, double u) {
    require_positive(a, "the lattice constant");
    if (c == 0.0) {
        c = a * std::sqrt(8.0 / 3.0);
    }
    require_positive(c, "the c axis");
    if (u == 0.0) {
        u = 3.0 / 8.0;   // the ideal value, from perfect tetrahedra
    }

    const Cell cell(Vec3{a, 0.0, 0.0}, Vec3{-a / 2, a * std::sqrt(3.0) / 2, 0.0},
                    Vec3{0.0, 0.0, c}, {true, true, true});
    return from_scaled({first, first, second, second},
                       {0.0, 0.0, 0.0,
                        1.0 / 3.0, 2.0 / 3.0, 0.5,
                        0.0, 0.0, u,
                        1.0 / 3.0, 2.0 / 3.0, 0.5 + u},
                       cell);
}

Atoms hexagonal(const std::string& symbol, double a, double c) {
    require_positive(a, "the lattice constant");
    require_positive(c, "the c axis");
    const Cell cell(Vec3{a, 0.0, 0.0}, Vec3{-a / 2, a * std::sqrt(3.0) / 2, 0.0},
                    Vec3{0.0, 0.0, c}, {true, true, true});
    return from_scaled({symbol}, {0.0, 0.0, 0.0}, cell);
}

Atoms tetragonal(const std::string& symbol, double a, double c) {
    require_positive(a, "the lattice constant");
    require_positive(c, "the c axis");
    const Cell cell(Vec3{a, 0.0, 0.0}, Vec3{0.0, a, 0.0}, Vec3{0.0, 0.0, c}, {true, true, true});
    return from_scaled({symbol}, {0.0, 0.0, 0.0}, cell);
}

Atoms orthorhombic(const std::string& symbol, double a, double b, double c) {
    require_positive(a, "the a axis");
    require_positive(b, "the b axis");
    require_positive(c, "the c axis");
    const Cell cell(Vec3{a, 0.0, 0.0}, Vec3{0.0, b, 0.0}, Vec3{0.0, 0.0, c}, {true, true, true});
    return from_scaled({symbol}, {0.0, 0.0, 0.0}, cell);
}

Atoms rhombohedral(const std::string& symbol, double a, double alpha) {
    require_positive(a, "the lattice constant");
    if (alpha <= 0.0 || alpha >= 180.0) {
        throw BuildError("the rhombohedral angle is between 0 and 180 degrees, not " +
                         std::to_string(alpha));
    }
    // Three equal vectors at the same angle to each other, arranged with their
    // sum along z so the cell comes out in the standard orientation.
    const double radians = alpha * 3.14159265358979323846 / 180.0;
    const double cosine = std::cos(radians);
    const double tx = std::sqrt((1.0 - cosine) / 2.0);
    const double ty = std::sqrt((1.0 - cosine) / 6.0);
    const double tz = std::sqrt((1.0 + 2.0 * cosine) / 3.0);
    const Cell cell(Vec3{a * tx, -a * ty, a * tz}, Vec3{0.0, 2.0 * a * ty, a * tz},
                    Vec3{-a * tx, -a * ty, a * tz}, {true, true, true});
    return from_scaled({symbol}, {0.0, 0.0, 0.0}, cell);
}

Atoms bulk(const std::vector<std::string>& symbols, Lattice lattice, double a, double c,
           bool cubic) {
    const bool binary = lattice == Lattice::RockSalt || lattice == Lattice::ZincBlende ||
                        lattice == Lattice::CsCl || lattice == Lattice::Fluorite ||
                        lattice == Lattice::Wurtzite;
    if (symbols.empty()) {
        throw BuildError("a structure needs at least one chemical symbol");
    }
    if (binary && symbols.size() != 2) {
        throw BuildError(std::string(lattice_name(lattice)) + " needs exactly two symbols, got " +
                         std::to_string(symbols.size()));
    }
    if (!binary && symbols.size() != 1) {
        throw BuildError(std::string(lattice_name(lattice)) + " needs exactly one symbol, got " +
                         std::to_string(symbols.size()));
    }

    switch (lattice) {
        case Lattice::SC:         return sc(symbols[0], a);
        case Lattice::FCC:        return fcc(symbols[0], a, cubic);
        case Lattice::BCC:        return bcc(symbols[0], a, cubic);
        case Lattice::HCP:        return hcp(symbols[0], a, c);
        case Lattice::Diamond:    return diamond(symbols[0], a, cubic);
        case Lattice::RockSalt:   return rocksalt(symbols[0], symbols[1], a, cubic);
        case Lattice::ZincBlende: return zincblende(symbols[0], symbols[1], a, cubic);
        case Lattice::CsCl:       return cscl(symbols[0], symbols[1], a);
        case Lattice::Fluorite:   return fluorite(symbols[0], symbols[1], a, cubic);
        case Lattice::Wurtzite:   return wurtzite(symbols[0], symbols[1], a, c);
        case Lattice::Hexagonal:  return hexagonal(symbols[0], a, c);
        case Lattice::Tetragonal: return tetragonal(symbols[0], a, c);
        case Lattice::Orthorhombic:
            // Without a b axis of its own there is nothing to distinguish this
            // from tetragonal, so the three-argument form is the way in.
            return orthorhombic(symbols[0], a, a, c);
        case Lattice::Rhombohedral:
            return rhombohedral(symbols[0], a, c);
    }
    throw BuildError("unreachable");
}



/// The rectangular cell of a hexagonal lattice.
///
/// `b` is replaced by `a + 2b`, which is perpendicular to `a` and twice as
/// long as the hexagonal one - so the cell holds twice the atoms and has right
/// angles. This is what a slab or an interface wants, since a 120-degree cell
/// makes every downstream box awkward.
Atoms orthorhombic_of(const Atoms& hexagonal_cell) {
    const Mat3 transformation{1, 0, 0, 1, 2, 0, 0, 0, 1};
    return phonon::make_supercell(hexagonal_cell, transformation).atoms;
}

Atoms bulk(const std::string& symbol, const std::string& structure, double a, double c,
           bool cubic, bool orthorhombic_cell) {
    const int z = try_number_from_symbol(symbol);
    if (z <= 0) {
        throw BuildError("'" + symbol + "' is not a chemical symbol");
    }
    const ReferenceState& state = reference_state(z);

    // A named structure overrides the table, which is how a hypothetical phase
    // gets built; otherwise the element's own is what "bulk" means.
    std::string wanted = structure;
    if (wanted.empty()) {
        wanted = std::string(state.symmetry);
        if (wanted.empty()) {
            throw BuildError("no ambient crystal structure is tabulated for " + symbol +
                             "; name one, as in bulk(\"" + symbol + "\", \"fcc\", 4.0)");
        }
    }

    // Structures whose cell needs a basis that a lattice constant alone does
    // not give. Refusing beats returning a plausible-looking wrong crystal.
    if (wanted == "diatomic") {
        throw BuildError(symbol + " is a molecular solid at ambient conditions, not a lattice "
                         "this builds. Use build::molecule(\"" + symbol +
                         "2\") for the molecule, or name a structure to build a hypothetical "
                         "crystal.");
    }
    if (wanted == "cubic" || wanted == "monoclinic") {
        throw BuildError(symbol + " crystallises as a complex " + wanted +
                         " cell whose basis is not derivable from the lattice constant. Name a "
                         "simpler structure to build an approximation of it.");
    }

    const Lattice lattice = lattice_from_string(wanted);
    const bool from_table = structure.empty() || wanted == std::string(state.symmetry);

    if (a == 0.0) {
        a = from_table ? state.a : 0.0;
        if (a == 0.0) {
            throw BuildError("no lattice constant is tabulated for " + symbol + " as " + wanted +
                             "; pass one");
        }
    }
    if (c == 0.0 && from_table && state.covera > 0.0) {
        c = a * state.covera;
    }

    if (lattice == Lattice::Rhombohedral) {
        // `c` carries the angle for this one; the table has it in `alpha`.
        const double alpha = c > 0.0 ? c : (from_table ? state.alpha : 0.0);
        if (alpha == 0.0) {
            throw BuildError("a rhombohedral cell needs its angle; pass it as `c`");
        }
        return rhombohedral(symbol, a, alpha);
    }
    if (lattice == Lattice::Orthorhombic) {
        const double b = from_table && state.bovera > 0.0 ? a * state.bovera : a;
        return build::orthorhombic(symbol, a, b, c > 0.0 ? c : a);
    }

    // ASE's `orthorhombic=True`: the rectangular cell of a hexagonal lattice,
    // which is what a slab or an interface wants.
    if (orthorhombic_cell) {
        if (lattice != Lattice::HCP && lattice != Lattice::Hexagonal) {
            throw BuildError("orthorhombic=True applies to hexagonal lattices; " + wanted +
                             " already has right angles");
        }
        Atoms atoms = lattice == Lattice::HCP ? hcp(symbol, a, c) : hexagonal(symbol, a, c);
        return orthorhombic_of(atoms);
    }

    return bulk(std::vector<std::string>{symbol}, lattice, a, c, cubic);
}

}  // namespace fastatoms::build
