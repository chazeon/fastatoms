// SPDX-License-Identifier: MIT
#pragma once

/// \file spacegroup.hpp
/// \brief Determining the space group of a structure.
///
/// The Hermann-Mauguin symbol is *derived* here rather than matched against a
/// stored list of operation sets: the crystal system fixes a set of symmetry
/// directions, each direction is asked which axis and which plane it carries,
/// and the symbol is assembled from the answers. That is how the symbol is
/// defined, and it means a group this code has never seen still produces the
/// right symbol. Only the last step - turning the symbol into the number
/// between 1 and 230 that International Tables gives it - is a lookup, because
/// that numbering is a convention and not something derivable.
///
/// Nothing here uses spglib or any other symmetry library.

#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/phonon/symmetry.hpp"
#include "fastatoms/symmetry/pointgroup.hpp"
#include "fastatoms/symmetry/standardize.hpp"

namespace fastatoms::symmetry {

/// The operation types, re-exported here so that using this module does not
/// require reaching into the phonon one, where the search happens to live.
using SymmetryOperation = phonon::SymmetryOperation;
using SymmetryDataset = phonon::SymmetryDataset;
using phonon::find_symmetry;
using phonon::lattice_point_group;

/// One Wyckoff orbit: a set of atoms that symmetry ties together.
struct WyckoffPosition {
    /// The letter, assigned by increasing multiplicity as International Tables
    /// does. Sites of equal multiplicity are ordered by site symmetry and then
    /// by position, which reproduces the standard lettering for most groups
    /// but is not looked up from the tables.
    char letter{'a'};

    /// How many atoms are in the orbit, in the conventional cell.
    int multiplicity{1};

    /// The point group of the site, as a Hermann-Mauguin symbol: "1" for a
    /// general position, "4mm" for an atom on a four-fold axis with mirrors.
    std::string site_symmetry{"1"};

    /// The atoms of the orbit, as indices into `SpaceGroup::primitive`.
    std::vector<std::size_t> atoms{};

    /// One member's position, in fractional coordinates of the primitive cell.
    Vec3 representative{};

    [[nodiscard]] std::string to_string() const;
};

/// Everything this module can say about a structure's symmetry.
struct SpaceGroup {
    /// The International Tables number, 1 to 230. Zero when the derived symbol
    /// is not one this build has a number for, which is a bug worth reporting
    /// rather than something to work around.
    int number{0};

    /// The short Hermann-Mauguin symbol, e.g. "Fd-3m".
    std::string hermann_mauguin{};

    /// The full symbol, with every direction written out: "P 4/m 2/m 2/m"
    /// rather than "P4/mmm".
    std::string hermann_mauguin_full{};

    /// The Schoenflies symbol with its superscript, e.g. "Oh-7".
    std::string schoenflies{};

    PointGroup point_group{};
    CrystalSystem system{CrystalSystem::Triclinic};
    Centring centring{Centring::P};

    /// The primitive cell the analysis was carried out on. When the structure
    /// handed in was already primitive this is a copy of it; when it was a
    /// conventional cell or a supercell it is the reduction. The operations
    /// and the Wyckoff orbits below index this cell.
    Atoms primitive{};

    /// Every operation, in the basis of `primitive`.
    std::vector<SymmetryOperation> operations{};

    /// The orbits, ordered by increasing multiplicity.
    std::vector<WyckoffPosition> wyckoff{};

    /// Atoms in the primitive cell, and in the conventional one.
    std::size_t primitive_atoms{0};
    std::size_t conventional_atoms{0};

    double tolerance{1e-5};

    [[nodiscard]] std::size_t order() const noexcept { return operations.size(); }
    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string summary() const;
};

/// Determine the space group of a structure.
///
/// \param tolerance distance in angstrom within which two atoms count as
///        coincident. Too tight and real symmetry is missed; too loose and
///        operations appear that are not there, which shows up as a point
///        group that is not one of the thirty-two.
/// \throws SymmetryError if the structure has no usable cell, or if the
///         operations found do not form a crystallographic group.
[[nodiscard]] SpaceGroup analyse(const Atoms& atoms, double tolerance = 1e-5);

/// The International Tables number for a short Hermann-Mauguin symbol, or zero
/// if it is not one of the 230.
[[nodiscard]] int space_group_number(const std::string& hermann_mauguin);

/// The short Hermann-Mauguin symbol for a number between 1 and 230.
/// \throws SymmetryError if the number is out of range.
[[nodiscard]] std::string space_group_symbol(int number);

/// All 230, as (number, symbol) in order. For tests and for listing.
[[nodiscard]] const std::vector<std::pair<int, std::string>>& space_group_table();

}  // namespace fastatoms::symmetry
