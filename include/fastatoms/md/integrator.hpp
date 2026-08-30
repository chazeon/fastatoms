// SPDX-License-Identifier: MIT
#pragma once

/// \file integrator.hpp
/// \brief Time integration schemes.

#include <memory>
#include <string>
#include <vector>

#include "fastatoms/io/parameters.hpp"
#include "fastatoms/md/types.hpp"

namespace fastatoms::md {

/// Base class for the schemes that advance positions and velocities.
///
/// An integrator is responsible for one plain, unthermostatted step: it reads
/// the forces from the structure's calculator, moves the atoms, and updates
/// the velocities. Temperature and pressure control are applied around it by
/// `Dynamics`, so any thermostat composes with any integrator.
class Integrator {
public:
    virtual ~Integrator() = default;

    Integrator() = default;
    Integrator(const Integrator&) = delete;
    Integrator& operator=(const Integrator&) = delete;

    [[nodiscard]] virtual std::string name() const = 0;

    /// Prepare internal state for a run. Called once, before the first step.
    virtual void begin(Atoms& atoms, const DynamicsContext& context) {
        (void)atoms;
        (void)context;
    }

    /// Advance the structure by one timestep.
    virtual void step(Atoms& atoms, const DynamicsContext& context) = 0;

    /// Tell the integrator that a thermostat transformed every velocity as
    /// `v <- factor * v + shift`.
    ///
    /// Schemes that keep their own copy of the velocities - leapfrog holds
    /// them at half steps - have to follow along, or the thermostat's work
    /// would be silently undone on the next step.
    virtual void adjust_velocities(const VelocityUpdate& update) { (void)update; }

    /// As `scale_velocities`, for the anisotropic transform a barostat applies.
    virtual void transform_velocities(const Mat3& transform) { (void)transform; }

    /// Whether this scheme conserves energy for a well-chosen timestep.
    [[nodiscard]] virtual bool is_symplectic() const { return true; }

    [[nodiscard]] virtual std::string to_string() const { return name() + "()"; }
};

/// Explicit (forward) Euler.
///
///     r(t+dt) = r(t) + v(t) dt
///     v(t+dt) = v(t) + F(t)/m dt
///
/// First order and not symplectic: the energy of an oscillator grows by a
/// factor of roughly `1 + (omega dt)^2` every period, so a run heats up
/// without bound. It is here because it is the scheme every derivation starts
/// from, and because seeing that drift next to Verlet's is the clearest way to
/// understand why nobody uses it. Do not run production dynamics with it.
class EulerIntegrator : public Integrator {
public:
    [[nodiscard]] std::string name() const override { return "euler"; }
    void step(Atoms& atoms, const DynamicsContext& context) override;
    [[nodiscard]] bool is_symplectic() const override { return false; }
};

/// Velocity Verlet.
///
///     v(t+dt/2) = v(t) + F(t)/(2m) dt
///     r(t+dt)   = r(t) + v(t+dt/2) dt
///     v(t+dt)   = v(t+dt/2) + F(t+dt)/(2m) dt
///
/// Second order, symplectic and time reversible, with one force evaluation per
/// step and positions and velocities both defined at the same instant. The
/// default, and the right choice unless you have a reason otherwise.
class VelocityVerletIntegrator : public Integrator {
public:
    [[nodiscard]] std::string name() const override { return "velocity_verlet"; }
    void step(Atoms& atoms, const DynamicsContext& context) override;
};

/// Leapfrog.
///
///     v(t+dt/2) = v(t-dt/2) + F(t)/m dt
///     r(t+dt)   = r(t) + v(t+dt/2) dt
///
/// The velocities live half a step away from the positions, which is where the
/// name comes from. The positions it generates are identical to velocity
/// Verlet's - the two schemes are algebraically the same - so the difference is
/// only in where the velocities are defined and what that costs to report.
///
/// The structure is given the on-step estimate `v(t) = [v(t-dt/2) +
/// v(t+dt/2)]/2` so that kinetic energies and thermostats see a velocity that
/// matches the positions, and the half-step velocities are kept internally.
class LeapfrogIntegrator : public Integrator {
public:
    [[nodiscard]] std::string name() const override { return "leapfrog"; }

    void begin(Atoms& atoms, const DynamicsContext& context) override;
    void step(Atoms& atoms, const DynamicsContext& context) override;
    void adjust_velocities(const VelocityUpdate& update) override;
    void transform_velocities(const Mat3& transform) override;

    /// The half-step velocities, 3N long.
    [[nodiscard]] const std::vector<double>& half_step_velocities() const noexcept {
        return half_velocities_;
    }

private:
    std::vector<double> half_velocities_{};
    bool started_{false};
};

/// Build an integrator by name: `euler`, `velocity_verlet` (aliases `verlet`,
/// `vv`), or `leapfrog`. Matching ignores case, hyphens and underscores.
/// \throws DynamicsError if the name is not recognised.
[[nodiscard]] std::shared_ptr<Integrator> make_integrator(std::string_view name);

/// Every integrator name, in alphabetical order.
[[nodiscard]] std::vector<std::string> integrator_names();

}  // namespace fastatoms::md
