// SPDX-License-Identifier: MIT
#include "fastatoms/md/integrator.hpp"

#include <cctype>
#include <cmath>
#include <string>

#include "fastatoms/core/vec3.hpp"
#include "fastatoms/md/constraints.hpp"

namespace fastatoms::md {
namespace {

/// Half of `dt / m` for every atom, so a kick is one multiply per component.
std::vector<double> inverse_masses(const Atoms& atoms) {
    const std::vector<double> masses = atoms.masses();
    std::vector<double> inverse(masses.size());
    for (std::size_t i = 0; i < masses.size(); ++i) {
        if (!(masses[i] > 0.0)) {
            throw DynamicsError("atom " + std::to_string(i) +
                                " has a non-positive mass, so it cannot be integrated");
        }
        inverse[i] = 1.0 / masses[i];
    }
    return inverse;
}

/// `v += factor * F / m`, in place.
void kick(std::vector<double>& velocities, const std::vector<double>& forces,
          const std::vector<double>& inverse_mass, double factor) {
    for (std::size_t i = 0; i < inverse_mass.size(); ++i) {
        const double scale = factor * inverse_mass[i];
        velocities[3 * i] += scale * forces[3 * i];
        velocities[3 * i + 1] += scale * forces[3 * i + 1];
        velocities[3 * i + 2] += scale * forces[3 * i + 2];
    }
}

/// `r += factor * v`, in place, straight into the structure's own buffer.
void drift(Atoms& atoms, const std::vector<double>& velocities, double factor) {
    double* FASTATOMS_RESTRICT positions = atoms.positions_data();
    for (std::size_t k = 0; k < velocities.size(); ++k) {
        positions[k] += factor * velocities[k];
    }
}

std::string canonical(std::string_view name) {
    std::string key;
    for (const char c : name) {
        if (c != '-' && c != '_' && c != ' ') {
            key += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    return key;
}

}  // namespace

// ---------------------------------------------------------------------------
// Euler
// ---------------------------------------------------------------------------

void EulerIntegrator::step(Atoms& atoms, const DynamicsContext& context) {
    const double dt = context.timestep;
    const std::vector<double> inverse_mass = inverse_masses(atoms);
    const std::vector<double> forces = atoms.get_forces();
    std::vector<double> velocities = atoms.velocities();
    const bool constrained = context.constraints != nullptr && !context.constraints->empty();
    const std::vector<double> reference = constrained ? atoms.positions() : std::vector<double>{};

    // Positions move on the old velocities, velocities on the old forces:
    // both updates use information from the start of the step, which is
    // exactly why the scheme does not conserve energy.
    drift(atoms, velocities, dt);
    if (constrained) {
        context.constraints->shake(atoms, reference, &velocities, dt);
    }
    kick(velocities, forces, inverse_mass, dt);
    if (constrained) {
        context.constraints->rattle(atoms, velocities);
    }
    atoms.set_velocities(velocities);
}

// ---------------------------------------------------------------------------
// Velocity Verlet
// ---------------------------------------------------------------------------

void VelocityVerletIntegrator::step(Atoms& atoms, const DynamicsContext& context) {
    const double dt = context.timestep;
    const std::vector<double> inverse_mass = inverse_masses(atoms);
    std::vector<double> velocities = atoms.velocities();
    const bool constrained = context.constraints != nullptr && !context.constraints->empty();

    // Half kick with the forces at the start of the step. The calculator has
    // these cached from the end of the previous step, so this is free.
    kick(velocities, atoms.get_forces(), inverse_mass, 0.5 * dt);

    // SHAKE needs where the atoms were, because the constraint force acts
    // along the old bond directions - that is what keeps the scheme
    // time-reversible and symplectic.
    const std::vector<double> reference = constrained ? atoms.positions() : std::vector<double>{};
    drift(atoms, velocities, dt);
    if (constrained) {
        context.constraints->shake(atoms, reference, &velocities, dt);
    }

    // The only force evaluation of the step, at the new positions.
    kick(velocities, atoms.get_forces(), inverse_mass, 0.5 * dt);

    // RATTLE: the half kick just put velocity back along the bonds, so project
    // it out again. Without this the constrained distances stay right while
    // their rates of change drift away from zero.
    if (constrained) {
        context.constraints->rattle(atoms, velocities);
    }
    atoms.set_velocities(velocities);
}

// ---------------------------------------------------------------------------
// Leapfrog
// ---------------------------------------------------------------------------

void LeapfrogIntegrator::begin(Atoms& atoms, const DynamicsContext& context) {
    // The scheme wants v(-dt/2). Take half a step backwards from the given
    // on-step velocities so that the first update lands on v(+dt/2).
    const std::vector<double> inverse_mass = inverse_masses(atoms);
    half_velocities_ = atoms.velocities();
    kick(half_velocities_, atoms.get_forces(), inverse_mass, -0.5 * context.timestep);
    started_ = true;
}

void LeapfrogIntegrator::step(Atoms& atoms, const DynamicsContext& context) {
    if (!started_ || half_velocities_.size() != 3 * atoms.size()) {
        begin(atoms, context);
    }

    const double dt = context.timestep;
    const std::vector<double> inverse_mass = inverse_masses(atoms);
    const std::vector<double> previous = half_velocities_;
    const bool constrained = context.constraints != nullptr && !context.constraints->empty();
    const std::vector<double> reference = constrained ? atoms.positions() : std::vector<double>{};

    // v(t+dt/2) = v(t-dt/2) + F(t)/m dt, then the positions leap over it.
    kick(half_velocities_, atoms.get_forces(), inverse_mass, dt);
    drift(atoms, half_velocities_, dt);
    if (constrained) {
        // The half-step velocities pick up the constraint impulse, which is
        // exactly what makes them the finite difference of the constrained
        // positions on the next step.
        context.constraints->shake(atoms, reference, &half_velocities_, dt);
    }

    // The structure carries the on-step velocity, midway between the two half
    // steps, so kinetic energies and thermostats stay in step with positions.
    std::vector<double> on_step(half_velocities_.size());
    for (std::size_t k = 0; k < on_step.size(); ++k) {
        on_step[k] = 0.5 * (previous[k] + half_velocities_[k]);
    }
    if (constrained) {
        // The average of two half-step velocities is not exactly on the
        // constraint surface even when both endpoints are; project it, so the
        // reported temperature counts only the degrees of freedom that exist.
        context.constraints->rattle(atoms, on_step);
    }
    atoms.set_velocities(on_step);
}

void LeapfrogIntegrator::adjust_velocities(const VelocityUpdate& update) {
    if (update.is_identity()) {
        return;
    }
    const bool has_shift = update.shift.size() == half_velocities_.size();
    for (std::size_t k = 0; k < half_velocities_.size(); ++k) {
        half_velocities_[k] *= update.factor;
        if (has_shift) {
            half_velocities_[k] += update.shift[k];
        }
    }
}

void LeapfrogIntegrator::transform_velocities(const Mat3& transform) {
    for (std::size_t i = 0; i < half_velocities_.size() / 3; ++i) {
        double* v = half_velocities_.data() + 3 * i;
        (Vec3::load(v) * transform).store(v);
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::shared_ptr<Integrator> make_integrator(std::string_view name) {
    const std::string key = canonical(name);
    if (key.empty() || key == "velocityverlet" || key == "verlet" || key == "vv") {
        return std::make_shared<VelocityVerletIntegrator>();
    }
    if (key == "euler") {
        return std::make_shared<EulerIntegrator>();
    }
    if (key == "leapfrog" || key == "lf") {
        return std::make_shared<LeapfrogIntegrator>();
    }
    throw DynamicsError("unknown integrator '" + std::string(name) +
                        "'; use euler, velocity_verlet or leapfrog");
}

std::vector<std::string> integrator_names() {
    return {"euler", "leapfrog", "velocity_verlet"};
}

}  // namespace fastatoms::md
