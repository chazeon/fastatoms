// SPDX-License-Identifier: MIT
#pragma once

/// \file constraints.hpp
/// \brief SHAKE and RATTLE: keeping rigid geometry rigid during dynamics.
///
/// A rigid water model has no O-H stretch and no H-O-H bend, which is the
/// point: those vibrations are the fastest motions in the system, and freezing
/// them lets the timestep grow from half a femtosecond to two. What replaces
/// them is a set of holonomic constraints, `|r_i - r_j| = d`, enforced by
/// adding forces along the constrained directions.
///
/// SHAKE solves for those forces one constraint at a time, sweeping until they
/// all hold at once - a Gauss-Seidel iteration on the Lagrange multipliers.
/// For water it converges in a handful of sweeps, because the three
/// constraints of a molecule are only weakly coupled to each other and not at
/// all to the rest of the box.
///
/// SHAKE alone constrains the positions. It leaves a velocity component along
/// each bond, which is small but does not average away: it shows up as a slow
/// drift in the reported temperature. RATTLE's second stage removes it by
/// projecting the velocities onto the constraint surface as well, so that
/// `d/dt |r_i - r_j|^2 = 0` holds exactly at every step.

#include <cstddef>
#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/core/constraint.hpp"

namespace fastatoms::md {

/// What one constraint solve did.
struct ConstraintReport {
    int iterations{0};
    double residual{0.0};   ///< largest `| |r_ij|^2 - d^2 | / d^2` left over
    bool converged{true};

    [[nodiscard]] std::string to_string() const;
};

/// A set of distance constraints and the machinery to enforce them.
///
/// The constraints are grouped into independent clusters - molecules, in
/// practice - so a sweep touches each one in turn and the whole solve is
/// linear in the number of molecules. Nothing here assumes water; a set of
/// rigid triangles is simply what water happens to give.
class Constraints {
public:
    Constraints() = default;

    /// \param bonds     the constraints to enforce
    /// \param tolerance relative tolerance on `|r_ij|^2 / d^2 - 1`
    /// \param max_iterations sweeps before giving up
    explicit Constraints(std::vector<DistanceConstraint> bonds, double tolerance = 1.0e-10,
                         int max_iterations = 500);

    [[nodiscard]] std::size_t size() const noexcept { return bonds_.size(); }
    [[nodiscard]] bool empty() const noexcept { return bonds_.empty(); }
    [[nodiscard]] const std::vector<DistanceConstraint>& bonds() const noexcept {
        return bonds_;
    }

    [[nodiscard]] double tolerance() const noexcept { return tolerance_; }
    void set_tolerance(double tolerance);

    [[nodiscard]] int max_iterations() const noexcept { return max_iterations_; }
    void set_max_iterations(int iterations);

    /// How many degrees of freedom the constraints remove - one per constraint,
    /// as long as they are independent, which they are for rigid triangles.
    ///
    /// The temperature of a constrained system is `2K / (dof kB)` with this
    /// already subtracted; forgetting it makes a rigid water box report a
    /// temperature about a third too low.
    [[nodiscard]] int removed_degrees_of_freedom() const noexcept {
        return static_cast<int>(bonds_.size());
    }

    /// SHAKE. Correct the positions in `atoms` so that every constraint holds,
    /// given where the atoms were before the unconstrained update.
    ///
    /// When `velocities` is given and `dt` is non-zero, the same displacement
    /// is added to the velocities as `delta_r / dt`, which is the impulse the
    /// constraint force delivered over the step. Skipping that would lose
    /// energy steadily.
    ConstraintReport shake(Atoms& atoms, const std::vector<double>& reference,
                           std::vector<double>* velocities = nullptr, double dt = 0.0) const;

    /// RATTLE's velocity stage. Project the velocities so that every
    /// constrained distance has zero rate of change.
    ConstraintReport rattle(const Atoms& atoms, std::vector<double>& velocities) const;

    /// Move the atoms onto the constraint surface without reference positions,
    /// using the current bond directions. Used once at the start of a run to
    /// clean up a structure whose geometry is only approximately right.
    ConstraintReport enforce(Atoms& atoms) const;

    /// Largest relative violation `| |r_ij| - d | / d` in the structure.
    [[nodiscard]] double max_deviation(const Atoms& atoms) const;

    /// Largest `|d/dt |r_ij||` left in the velocities, in angstrom per internal
    /// time unit. Zero when the velocities lie on the constraint surface.
    [[nodiscard]] double max_velocity_deviation(const Atoms& atoms,
                                                const std::vector<double>& velocities) const;

    [[nodiscard]] std::string to_string() const;

private:
    /// Minimum-image displacement `r_j - r_i`, so a molecule that straddles the
    /// periodic boundary is still one molecule.
    [[nodiscard]] Vec3 displacement(const Atoms& atoms, const double* positions, std::size_t i,
                                    std::size_t j) const;

    void cache_cell(const Atoms& atoms) const;

    std::vector<DistanceConstraint> bonds_{};
    double tolerance_{1.0e-10};
    int max_iterations_{500};

    mutable Mat3 cell_{};
    mutable Mat3 inverse_cell_{};
    mutable bool periodic_{false};
};

}  // namespace fastatoms::md
