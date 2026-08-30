// SPDX-License-Identifier: MIT
#pragma once

/// \file types.hpp
/// \brief The vocabulary of the molecular dynamics module.
///
/// Units follow the rest of FastAtoms: angstrom, eV, amu, kelvin, and an
/// internal time unit of angstrom*sqrt(amu/eV) ~= 10.18 fs. Timesteps are
/// written in femtoseconds everywhere a user sees them and converted on the
/// way in; pressures are written in GPa for the same reason.

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/core/vec3.hpp"

namespace fastatoms::md {

class Constraints;

/// Raised when a run cannot be set up or has gone unphysical.
class DynamicsError : public std::runtime_error {
public:
    explicit DynamicsError(const std::string& message) : std::runtime_error(message) {}
};

/// Settings the integrator, thermostat and barostat all need to see.
struct DynamicsContext {
    /// Timestep in the internal time unit.
    double timestep{0.0};

    /// Momentum degrees of freedom, `3N` minus three when the centre-of-mass
    /// motion is removed. The temperature is `2K / (dof * kB)`.
    int degrees_of_freedom{0};

    /// Target temperature in kelvin; zero when no thermostat is attached.
    double temperature{0.0};

    /// Target pressure in eV/angstrom^3; use `units::GPa` to convert.
    double pressure{0.0};

    /// Whether the net momentum is being held at zero. A stochastic
    /// thermostat has to know: its noise adds momentum that would otherwise
    /// accumulate into a drift the degree-of-freedom count does not allow for.
    bool fix_center_of_mass{true};

    /// Steps taken so far, for reporting.
    int step{0};

    /// Rigid-geometry constraints to enforce, or null for a flexible system.
    ///
    /// The integrator applies them: SHAKE after the positions move, RATTLE
    /// after the velocities are complete. They live here rather than inside
    /// the integrator so that every scheme gets them from one place, and so
    /// that a thermostat can see how many degrees of freedom they removed.
    const Constraints* constraints{nullptr};
};

/// How a thermostat changed the velocities: `v <- factor * v + shift`.
///
/// A rescaling thermostat leaves `shift` empty. A stochastic one cannot: its
/// update is affine, not multiplicative, and an integrator that keeps its own
/// copy of the velocities - leapfrog does - has to apply the same realisation
/// of the noise, not just the same damping.
struct VelocityUpdate {
    double factor{1.0};
    std::vector<double> shift{};  ///< 3N long, or empty for a pure rescaling

    [[nodiscard]] bool is_identity() const noexcept {
        return factor == 1.0 && shift.empty();
    }
};

/// One sampled frame of a run.
struct DynamicsStep {
    int step{0};
    double time{0.0};                ///< femtoseconds since the run started
    double potential_energy{0.0};    ///< eV
    double kinetic_energy{0.0};      ///< eV
    double total_energy{0.0};        ///< eV, potential + kinetic

    /// Potential + kinetic + the extended-system terms of any thermostat and
    /// barostat. This is the quantity that should stay flat: for plain NVE it
    /// equals `total_energy`, and under a Nose-Hoover thermostat it is the
    /// conserved Hamiltonian of the extended system.
    double conserved_energy{0.0};

    double temperature{0.0};         ///< K
    double pressure{0.0};            ///< eV/angstrom^3, including the kinetic term
    double volume{0.0};              ///< angstrom^3
};

/// \name Instantaneous observables
///
/// The stress sign convention is the one used throughout FastAtoms:
/// `sigma = (1/V) dE/d(strain)`, so the pressure is `-(1/3) trace(sigma)`.
/// @{

/// Kinetic contribution to the stress, `-(1/V) sum m v (x) v`.
///
/// Needs no calculator: it is the ideal-gas part of the pressure.
[[nodiscard]] Mat3 kinetic_stress(const Atoms& atoms);

/// Potential stress from the calculator plus the kinetic term.
[[nodiscard]] Mat3 total_stress(Atoms& atoms);

/// Instantaneous pressure in eV/angstrom^3, including the kinetic term.
[[nodiscard]] double instantaneous_pressure(Atoms& atoms);
/// @}

/// Called at the sampling interval. Returning false stops the run.
using DynamicsCallback = std::function<bool(const Atoms&, const DynamicsStep&)>;

/// Why a run stopped.
enum class DynamicsStopReason {
    Completed,    ///< the requested number of steps was taken
    Interrupted,  ///< a callback asked to stop
    /// A coordinate or a momentum stopped being a finite number.
    ///
    /// This catches arithmetic that has genuinely overflowed - a bonded or
    /// harmonic system past the stability limit, or a calculator returning
    /// NaN. It does not catch every physically ruined run: with a truncated
    /// pair potential, atoms blown apart simply fly off at large but finite
    /// speed, and the run completes. The drift of `conserved_energy` is the
    /// measure to watch for that.
    Unstable,
};

[[nodiscard]] std::string_view dynamics_stop_reason_name(DynamicsStopReason reason);

/// The outcome of a run.
struct DynamicsResult {
    DynamicsStopReason reason{DynamicsStopReason::Completed};
    int steps{0};
    int evaluations{0};              ///< force evaluations, the DFT-cost metric
    double time{0.0};                ///< femtoseconds simulated
    std::string message{};

    /// Frames at the sampling interval, starting with the initial state.
    std::vector<DynamicsStep> history{};

    /// Drift of the conserved quantity over the run, in eV per atom. The
    /// headline number for judging whether a timestep was small enough.
    [[nodiscard]] double energy_drift_per_atom(std::size_t natoms) const;

    [[nodiscard]] std::string to_string() const;
};

}  // namespace fastatoms::md
