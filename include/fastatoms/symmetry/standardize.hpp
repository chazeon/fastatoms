// SPDX-License-Identifier: MIT
#pragma once

/// \file standardize.hpp
/// \brief Standardised, primitive and reduced cells, as spglib produces them.
///
/// The functions mirror spglib's Python API (`standardize_cell`,
/// `find_primitive`, `niggli_reduce`). Results are new `Atoms` with periodic
/// boundary conditions on. Per-atom "labels" are carried over where the
/// spglib mapping identifies the source atom; other atoms get their element
/// symbol.

#include <cstddef>
#include <string>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/symmetry/pointgroup.hpp"

namespace fastatoms::symmetry {

/// What a cell turned out to be.
struct CellAnalysis {
    int number{0};
    std::string international;
    PointGroup pointgroup;
    CrystalSystem crystal_system{CrystalSystem::Triclinic};
    std::size_t primitive_atoms{0};
    /// Rows are the primitive vectors in fractional coordinates of the input
    /// cell, so `to_primitive * input_cell` is the primitive cell.
    Mat3 to_primitive{};
};

/// Space group, primitive size and the change of basis to the primitive cell.
/// \throws SymmetryError as for `get_symmetry_dataset`.
[[nodiscard]] CellAnalysis analyse_cell(const Atoms& atoms, double symprec = 1e-5,
                                        double angle_tolerance = -1.0, bool use_labels = false);

/// spglib's standardised cell.
///
/// With `to_primitive` the primitive cell is returned; with `no_idealize` the
/// atoms keep their input positions and only the basis is changed.
/// \throws SymmetryError as for `get_symmetry_dataset`.
[[nodiscard]] Atoms standardize_cell(const Atoms& atoms, bool to_primitive = false,
                                     bool no_idealize = false, double symprec = 1e-5,
                                     double angle_tolerance = -1.0, bool use_labels = false);

/// The primitive cell, idealised.
[[nodiscard]] Atoms find_primitive(const Atoms& atoms, double symprec = 1e-5,
                                   double angle_tolerance = -1.0, bool use_labels = false);

/// The primitive cell. Same as `find_primitive`, under the old name.
[[nodiscard]] inline Atoms primitive_cell(const Atoms& atoms, double symprec = 1e-5,
                                          double angle_tolerance = -1.0,
                                          bool use_labels = false) {
    return find_primitive(atoms, symprec, angle_tolerance, use_labels);
}

/// The conventional cell in the standard setting. Same as `standardize_cell`
/// with the defaults.
[[nodiscard]] inline Atoms conventional_cell(const Atoms& atoms, double symprec = 1e-5,
                                             double angle_tolerance = -1.0,
                                             bool use_labels = false) {
    return standardize_cell(atoms, false, false, symprec, angle_tolerance, use_labels);
}

/// The Niggli-reduced cell of the same lattice. Atoms keep their Cartesian
/// positions. The reduced basis may be left-handed, as spglib returns it.
/// \throws SymmetryError unless the cell is periodic in all directions.
[[nodiscard]] Atoms niggli_reduce(const Atoms& atoms, double symprec = 1e-5);

}  // namespace fastatoms::symmetry
