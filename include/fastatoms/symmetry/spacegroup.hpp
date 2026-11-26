// SPDX-License-Identifier: MIT
#pragma once

/// \file spacegroup.hpp
/// \brief The symmetry dataset of a structure and the space group it implies.
///
/// Everything here is computed by spglib. Conventions follow spglib:
///  - `rotations` and `translations` act on fractional coordinates of the input
///    cell as x' = R x + t, with R an integer matrix in that basis.
///  - `transformation_matrix` P and `origin_shift` p map input to standard
///    coordinates, x_s = P x + p, as spglib defines them.
/// Lattices are the exception: they are returned with rows as lattice vectors,
/// like every other cell in FastAtoms.
///
/// Labels: by default every atom is typed by its atomic number. With
/// `use_labels` the type is the label (`Fe1`, `Fe2`, ...), so atoms of one
/// element with different labels are treated as different sites.

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/core/vec3.hpp"
#include "fastatoms/symmetry/pointgroup.hpp"

namespace fastatoms::symmetry {

using IntMatrix3 = std::array<std::array<int, 3>, 3>;

/// Everything spglib knows about a structure, with the arrays laid out per
/// atom and the lattices as rows.
struct SymmetryDataset {
    int number{0};                  ///< International Tables number, 1..230
    std::string international;      ///< short Hermann-Mauguin symbol, e.g. "Fm-3m"
    std::string hall;               ///< Hall symbol, e.g. "-F 4 2 3"
    int hall_number{0};             ///< spglib's serial number, 1..530
    std::string choice;             ///< origin / axis choice, e.g. "1" or "2"
    PointGroup pointgroup;
    CrystalSystem crystal_system{CrystalSystem::Triclinic};

    std::vector<IntMatrix3> rotations;
    std::vector<Vec3> translations;

    std::vector<char> wyckoffs;     ///< one Wyckoff letter per input atom
    std::vector<std::string> site_symmetry_symbols;  ///< one per input atom
    std::vector<int> equivalent_atoms;               ///< input atom -> orbit representative
    std::vector<int> crystallographic_orbits;

    Mat3 std_lattice{};             ///< standardised conventional cell, rows
    std::vector<int> std_types;
    std::vector<Vec3> std_positions;  ///< fractional, in std_lattice
    Mat3 transformation_matrix{};   ///< P
    Vec3 origin_shift{};            ///< p

    Mat3 primitive_lattice{};       ///< primitive cell, rows, input frame
    std::vector<int> mapping_to_primitive;      ///< input atom -> primitive atom id
    std::vector<int> std_mapping_to_primitive;  ///< standard atom -> primitive atom id
};

/// The full symmetry dataset of a structure.
/// \param symprec  distance tolerance in angstrom
/// \param angle_tolerance  angle tolerance in degrees, or negative for spglib's default
/// \throws SymmetryError if the structure is empty, not periodic in all three
///         directions, or spglib cannot find its symmetry.
[[nodiscard]] SymmetryDataset get_symmetry_dataset(const Atoms& atoms, double symprec = 1e-5,
                                                   double angle_tolerance = -1.0,
                                                   bool use_labels = false);

/// The atoms that one Wyckoff letter collects.
struct WyckoffPosition {
    char letter{'a'};
    std::string site_symmetry{"1"};
    std::vector<std::size_t> atoms;  ///< indices into the input structure
};

/// The space group of a structure, as the summary most callers want.
struct SpaceGroup {
    int number{0};
    std::string international;
    PointGroup pointgroup;
    CrystalSystem crystal_system{CrystalSystem::Triclinic};
    std::vector<WyckoffPosition> wyckoff;  ///< ordered by letter
};

/// The space group of a structure. Same arguments as `get_symmetry_dataset`.
[[nodiscard]] SpaceGroup analyse(const Atoms& atoms, double tolerance = 1e-5,
                                 double angle_tolerance = -1.0, bool use_labels = false);

/// The International Tables number for a short Hermann-Mauguin symbol, or zero
/// if it is not one of the 230.
[[nodiscard]] int space_group_number(const std::string& hermann_mauguin);

/// The short Hermann-Mauguin symbol for a number between 1 and 230.
/// \throws SymmetryError if the number is out of range.
[[nodiscard]] std::string space_group_symbol(int number);

/// All 230 as (number, short symbol), in order of number.
[[nodiscard]] const std::vector<std::pair<int, std::string>>& space_group_table();

}  // namespace fastatoms::symmetry
