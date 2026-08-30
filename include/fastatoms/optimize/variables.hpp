// SPDX-License-Identifier: MIT
#pragma once

/// \file variables.hpp
/// \brief What a relaxation is allowed to change.
///
/// An optimizer only sees a flat vector of numbers. These classes say what
/// those numbers mean: either the atomic positions alone, or the positions
/// together with the cell.

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/optimize/types.hpp"
#include "fastatoms/optimize/vector_math.hpp"

namespace fastatoms::optimize {

/// The mapping between a structure and the optimizer's coordinate vector.
class Variables {
public:
    virtual ~Variables() = default;

    [[nodiscard]] virtual std::string name() const = 0;

    /// Length of the coordinate vector.
    [[nodiscard]] virtual std::size_t size() const = 0;

    /// Rows that describe atoms. Rows beyond this are cell degrees of freedom.
    [[nodiscard]] virtual std::size_t atom_rows() const = 0;

    /// Read the current coordinates out of a structure.
    [[nodiscard]] virtual std::vector<double> pack(const Atoms& atoms) const = 0;

    /// Write coordinates back into a structure.
    virtual void unpack(const std::vector<double>& x, Atoms& atoms) const = 0;

    /// Energy and gradient at the structure's current geometry, using its
    /// attached calculator.
    [[nodiscard]] virtual Evaluation evaluate(Atoms& atoms) const = 0;

    /// Largest gradient row norm - the quantity the force threshold tests.
    [[nodiscard]] double fmax(const std::vector<double>& gradient) const {
        return max_row_norm(gradient);
    }

    /// Largest gradient row norm over the atomic rows only.
    [[nodiscard]] double atomic_fmax(const std::vector<double>& gradient) const {
        return max_row_norm(gradient, 0, atom_rows());
    }
};

/// Relax the atomic positions, holding the cell fixed.
///
/// The coordinates are simply the Cartesian positions, so the gradient is the
/// negative of the forces and no stress is ever requested.
class AtomicVariables : public Variables {
public:
    explicit AtomicVariables(std::size_t natoms) : natoms_(natoms) {}

    [[nodiscard]] std::string name() const override { return "positions"; }
    [[nodiscard]] std::size_t size() const override { return 3 * natoms_; }
    [[nodiscard]] std::size_t atom_rows() const override { return natoms_; }

    [[nodiscard]] std::vector<double> pack(const Atoms& atoms) const override;
    void unpack(const std::vector<double>& x, Atoms& atoms) const override;
    [[nodiscard]] Evaluation evaluate(Atoms& atoms) const override;

private:
    std::size_t natoms_{0};
};

/// Settings for a combined position and cell relaxation.
struct CellVariablesOptions {
    /// Scale that puts the cell degrees of freedom on the same footing as the
    /// atomic ones. Zero means "use the number of atoms", which is what makes
    /// a single force threshold sensible for both.
    double cell_factor{0.0};

    /// External pressure in eV/angstrom^3. The quantity minimised becomes the
    /// enthalpy E + p V. Multiply by `units::GPa` to give a value in GPa.
    double pressure{0.0};

    /// Which strain components may vary, in Voigt order
    /// (xx, yy, zz, yz, xz, xy). Zeroing a component freezes it - for example
    /// {1, 1, 0, 0, 0, 0} relaxes only the a and b axes of a slab.
    std::array<double, 6> mask{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
};

/// Relax the atomic positions and the cell together.
///
/// The cell is parameterised by a deformation `S` relative to the cell the
/// structure had when the relaxation started: `cell = cell_0 S`. Atoms are
/// carried by that deformation, so the coordinates are the positions in the
/// undeformed frame, `r_i = y_i S`, plus the nine components of `S - I`.
///
/// The gradient with respect to the deformation follows from the definition of
/// the stress, `dE/deps_ab = V sigma_ab` for `S -> S(I + eps)`:
///
///     dE/dS = V S^-T sigma        and        dE/dy_i = -F_i S^T
///
/// Both are checked against finite differences in the test suite.
class CellVariables : public Variables {
public:
    /// \throws OptimizerError if the structure has no usable cell.
    CellVariables(const Atoms& atoms, CellVariablesOptions options = {});

    [[nodiscard]] std::string name() const override { return "positions+cell"; }
    [[nodiscard]] std::size_t size() const override { return 3 * natoms_ + 9; }
    [[nodiscard]] std::size_t atom_rows() const override { return natoms_; }

    [[nodiscard]] std::vector<double> pack(const Atoms& atoms) const override;
    void unpack(const std::vector<double>& x, Atoms& atoms) const override;
    [[nodiscard]] Evaluation evaluate(Atoms& atoms) const override;

    /// The deformation of `atoms` relative to the reference cell.
    [[nodiscard]] Mat3 deformation(const Atoms& atoms) const;

    [[nodiscard]] const Mat3& reference_cell() const noexcept { return reference_cell_; }
    [[nodiscard]] double cell_factor() const noexcept { return cell_factor_; }

private:
    std::size_t natoms_{0};
    Mat3 reference_cell_{};
    Mat3 reference_inverse_{};
    double cell_factor_{1.0};
    double pressure_{0.0};
    Mat3 mask_{};
};

}  // namespace fastatoms::optimize
