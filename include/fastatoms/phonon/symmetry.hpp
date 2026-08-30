// SPDX-License-Identifier: MIT
#pragma once

/// \file symmetry.hpp
/// \brief Finding the space group of a periodic structure.

#include <cstddef>
#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/phonon/supercell.hpp"

namespace fastatoms::phonon {

/// One space group operation, `s -> s W + t` on fractional coordinates.
///
/// Row-vector convention throughout, matching the rest of FastAtoms: a
/// fractional coordinate is a row vector multiplied from the left, so the
/// Cartesian rotation acts as `r -> r R` with `R = M^-1 W M`.
struct SymmetryOperation {
    /// The rotation in the lattice basis. Integer valued by construction.
    Mat3 rotation{};

    /// The fractional translation, folded into [0, 1).
    Vec3 translation{};

    /// The same rotation in Cartesian coordinates. Orthogonal.
    Mat3 cartesian_rotation{};

    /// Where each atom is sent. `atom_map[i]` is the index of the atom that
    /// atom `i` lands on.
    std::vector<std::size_t> atom_map{};

    /// True when the rotation is the identity, i.e. this is a pure
    /// translation.
    [[nodiscard]] bool is_pure_translation(double tolerance = 1e-8) const;

    [[nodiscard]] std::string to_string() const;
};

/// Everything a phonon calculation needs to know about a structure's symmetry.
struct SymmetryDataset {
    std::vector<SymmetryOperation> operations{};

    /// The distinct rotations, i.e. the point group of the crystal.
    std::vector<Mat3> point_group{};

    /// For each atom, the lowest index in its orbit. Atoms sharing a value are
    /// related by symmetry and only one of them needs displacing.
    std::vector<std::size_t> equivalent_atoms{};

    /// The orbit representatives, ascending. These are the atoms a
    /// finite-displacement calculation actually has to touch.
    std::vector<std::size_t> independent_atoms{};

    double tolerance{1e-5};

    [[nodiscard]] std::size_t order() const noexcept { return operations.size(); }
    [[nodiscard]] std::string summary() const;
};

/// The point group of the lattice alone, ignoring the atoms.
///
/// These are the integer matrices `W` that preserve the metric tensor,
/// `W G W^T = G` with `G = M M^T`. Every space group operation's rotation is
/// one of these, so finding them first is what makes the search tractable.
///
/// The search bound is derived from the cell itself, so it is exact for any
/// cell; a wildly skewed one simply costs more candidates.
[[nodiscard]] std::vector<Mat3> lattice_point_group(const Cell& cell, double tolerance = 1e-5);

/// Find the space group operations of a structure.
///
/// For each rotation of the lattice point group, the candidate translations
/// are those that carry one atom of the rarest species onto another; each is
/// then checked against every atom. `tolerance` is a distance in angstrom.
///
/// \throws PhononError if the structure has no usable cell.
[[nodiscard]] SymmetryDataset find_symmetry(const Atoms& atoms, double tolerance = 1e-5);

/// The operations that leave `atom` where it is - its site symmetry group.
[[nodiscard]] std::vector<SymmetryOperation> site_symmetry(const SymmetryDataset& symmetry,
                                                           std::size_t atom);

/// An operation that maps `from` onto `to`, or nullptr when none does.
[[nodiscard]] const SymmetryOperation* mapping_operation(const SymmetryDataset& symmetry,
                                                         std::size_t from, std::size_t to);

}  // namespace fastatoms::phonon
