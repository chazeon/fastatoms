// SPDX-License-Identifier: MIT

#include "fastatoms/build/bulk.hpp"

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
    if (key == "hcp" || key == "hexagonal")                      return Lattice::HCP;
    if (key == "diamond" || key == "diamondcubic")               return Lattice::Diamond;
    if (key == "rocksalt" || key == "nacl" || key == "b1")       return Lattice::RockSalt;
    if (key == "zincblende" || key == "zns" || key == "b3")      return Lattice::ZincBlende;
    if (key == "cscl" || key == "b2")                            return Lattice::CsCl;
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

Atoms bulk(const std::vector<std::string>& symbols, Lattice lattice, double a, double c,
           bool cubic) {
    const bool binary = lattice == Lattice::RockSalt || lattice == Lattice::ZincBlende ||
                        lattice == Lattice::CsCl;
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
    }
    throw BuildError("unreachable");
}

}  // namespace fastatoms::build
