// SPDX-License-Identifier: MIT
#pragma once

/// \file bulk.hpp
/// \brief Standard crystal lattices.
///
/// Every generator here returns an `Atoms` with its cell already set and
/// periodicity already on, so the result is ready to hand to a calculator
/// without further arrangement.

#include <stdexcept>
#include <string>

#include "fastatoms/core/atoms.hpp"

namespace fastatoms::build {

/// Raised for a structure that cannot be built as asked.
class BuildError : public std::runtime_error {
public:
    explicit BuildError(const std::string& message) : std::runtime_error(message) {}
};

/// Which lattice a bulk crystal has.
enum class Lattice {
    SC,
    FCC,
    BCC,
    HCP,
    Diamond,
    RockSalt,
    ZincBlende,
    CsCl,
    Fluorite,       ///< CaF2: fcc cations with anions on both tetrahedral sites
    Wurtzite,       ///< the hexagonal counterpart of zinc blende
    Hexagonal,      ///< one atom, a = b, gamma = 120 degrees
    Tetragonal,     ///< one atom, a = b != c
    Orthorhombic,   ///< one atom, a != b != c
    Rhombohedral,   ///< one atom, a = b = c, alpha = beta = gamma
};

[[nodiscard]] std::string_view lattice_name(Lattice lattice);
[[nodiscard]] Lattice lattice_from_string(std::string_view name);

/// \name The lattices
///
/// `a` is the conventional cubic lattice constant in angstrom - the edge of
/// the cube, not the nearest-neighbour distance. `cubic` asks for the
/// conventional cell (4 atoms for fcc, 2 for bcc, 8 for diamond) instead of
/// the primitive one; the primitive cell is what you want for a band structure
/// and the conventional one for anything you intend to look at.
/// @{

/// Simple cubic: one atom, cell edge `a`.
[[nodiscard]] Atoms sc(const std::string& symbol, double a);

/// Face-centred cubic. Nearest-neighbour distance is `a / sqrt(2)`.
[[nodiscard]] Atoms fcc(const std::string& symbol, double a, bool cubic = false);

/// Body-centred cubic. Nearest-neighbour distance is `a * sqrt(3) / 2`.
[[nodiscard]] Atoms bcc(const std::string& symbol, double a, bool cubic = false);

/// Hexagonal close packed, two atoms.
///
/// \param c the axis length; zero asks for the ideal `a * sqrt(8/3)`, which is
///        what hard spheres would give and what most metals are within a few
///        per cent of.
[[nodiscard]] Atoms hcp(const std::string& symbol, double a, double c = 0.0);

/// Diamond cubic: fcc with a two-atom basis. Bond length is `a * sqrt(3) / 4`.
[[nodiscard]] Atoms diamond(const std::string& symbol, double a, bool cubic = false);

/// Rock salt (NaCl): two interpenetrating fcc lattices, offset by a/2.
[[nodiscard]] Atoms rocksalt(const std::string& first, const std::string& second, double a,
                             bool cubic = false);

/// Zinc blende (GaAs): diamond with two species.
[[nodiscard]] Atoms zincblende(const std::string& first, const std::string& second, double a,
                               bool cubic = false);

/// Caesium chloride: simple cubic with a body-centred second species.
[[nodiscard]] Atoms cscl(const std::string& first, const std::string& second, double a);

/// Fluorite (CaF2): fcc `first` with `second` on both tetrahedral sites.
[[nodiscard]] Atoms fluorite(const std::string& first, const std::string& second, double a,
                             bool cubic = false);

/// Wurtzite (ZnS, GaN): hexagonal, two of each species.
///
/// \param c zero asks for the ideal `a * sqrt(8/3)`.
/// \param u the internal parameter; zero asks for the ideal 3/8.
[[nodiscard]] Atoms wurtzite(const std::string& first, const std::string& second, double a,
                             double c = 0.0, double u = 0.0);

/// Simple hexagonal: one atom, `a = b`, 120 degrees between them.
[[nodiscard]] Atoms hexagonal(const std::string& symbol, double a, double c);

/// Simple tetragonal: one atom, `a = b != c`.
[[nodiscard]] Atoms tetragonal(const std::string& symbol, double a, double c);

/// Simple orthorhombic: one atom, three different edges, all right angles.
[[nodiscard]] Atoms orthorhombic(const std::string& symbol, double a, double b, double c);

/// Rhombohedral: one atom, three equal edges at `alpha` degrees to each other.
[[nodiscard]] Atoms rhombohedral(const std::string& symbol, double a, double alpha);
/// @}

/// Build any of the above by name, for the input deck and the Python layer.
///
/// \param symbols one symbol, or two for the binary lattices
/// \throws BuildError if the lattice needs a different number of species
[[nodiscard]] Atoms bulk(const std::vector<std::string>& symbols, Lattice lattice, double a,
                         double c = 0.0, bool cubic = false);

/// Build an element's own crystal structure, looked up from the periodic table.
///
///     auto copper  = build::bulk("Cu");             // fcc, a = 3.61
///     auto silicon = build::bulk("Si");             // diamond, a = 5.43
///     auto strained = build::bulk("Cu", "", 3.70);  // fcc, but say the constant
///
/// `structure` empty means "whatever the element crystallises as"; naming one
/// overrides the table, which is how a hypothetical phase gets built. `a` and
/// `c` of zero mean "the tabulated value", so only the parameter being varied
/// has to be given.
///
/// \throws BuildError when the element's ambient structure is one this cannot
///         build from a symbol alone - a molecular solid, or a complex cell
///         with a basis that is not derivable from the lattice constant. The
///         message says which, and naming a `structure` gets past it.
[[nodiscard]] Atoms bulk(const std::string& symbol, const std::string& structure = "",
                         double a = 0.0, double c = 0.0, bool cubic = false,
                         bool orthorhombic = false);

}  // namespace fastatoms::build
