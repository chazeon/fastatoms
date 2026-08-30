// SPDX-License-Identifier: MIT
#pragma once

/// \file standardize.hpp
/// \brief Reducing a cell to its primitive form, and raising it to the
///        conventional one.
///
/// A structure can be given in any cell that tiles the crystal - a supercell,
/// a conventional cell with centring translations in it, or the primitive cell
/// itself. Everything downstream is easier if which one it is has been settled
/// first, so these routines answer that: what is the smallest cell, what is
/// the standard one, and what is the transformation between them.

#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/symmetry/pointgroup.hpp"

namespace fastatoms::symmetry {

/// The centring of a conventional cell.
enum class Centring { P, A, B, C, I, F, R };

[[nodiscard]] char centring_letter(Centring centring);
[[nodiscard]] int centring_multiplicity(Centring centring);

/// The centring vectors a lattice type adds to the primitive translations,
/// in fractional coordinates of the conventional cell.
[[nodiscard]] std::vector<Vec3> centring_vectors(Centring centring);

/// What a cell turned out to be.
struct CellAnalysis {
    /// The crystal's point group, which fixes the crystal system and so which
    /// conventional cell is the right one.
    PointGroup point_group{};
    CrystalSystem system{CrystalSystem::Triclinic};
    Centring centring{Centring::P};

    /// Atoms in the primitive cell, and in the conventional one.
    std::size_t primitive_atoms{0};
    std::size_t conventional_atoms{0};

    /// How many copies of the primitive cell the given cell contains. One when
    /// the input was already primitive.
    int input_multiplicity{1};

    /// Rows are the primitive vectors in fractional coordinates of the input
    /// cell, so `primitive_matrix * input_cell` is the primitive cell.
    Mat3 to_primitive{};
    /// The same for the conventional cell.
    Mat3 to_conventional{};
    /// The conventional lattice vectors themselves, in angstrom.
    Mat3 conventional_cell{};

    double tolerance{1e-5};

    [[nodiscard]] std::string to_string() const;
};

/// Work out what cell a structure is given in.
/// \throws SymmetryError if the structure has no usable cell.
[[nodiscard]] CellAnalysis analyse_cell(const Atoms& atoms, double tolerance = 1e-5);

/// The primitive cell: the smallest one that tiles the crystal.
///
/// A conventional fcc cell of four atoms comes back with one; a supercell
/// comes back as whatever it is a supercell of. The calculator, constraints
/// and velocities are not carried over - this is a different structure, not
/// the same one rearranged.
[[nodiscard]] Atoms primitive_cell(const Atoms& atoms, double tolerance = 1e-5);

/// The conventional cell in its standard setting.
///
/// The axes follow the crystal system's convention: the unique axis along z
/// for tetragonal, hexagonal and trigonal, the three orthogonal two-folds
/// along x, y and z for orthorhombic, and b unique for monoclinic. For
/// triclinic there is nothing to align, and the Niggli-reduced cell is used.
[[nodiscard]] Atoms conventional_cell(const Atoms& atoms, double tolerance = 1e-5);

/// The Niggli-reduced form of a cell: the unique shortest, most orthogonal
/// basis for the same lattice.
///
/// Used for the triclinic case above, and useful on its own for deciding
/// whether two cells describe the same lattice.
[[nodiscard]] Atoms niggli_reduce(const Atoms& atoms, double tolerance = 1e-5);

}  // namespace fastatoms::symmetry
