// SPDX-License-Identifier: MIT
//
// Integrators and velocity setup, checked against problems with exact answers:
// a free particle, a harmonic oscillator, and the algebraic identity between
// leapfrog and velocity Verlet.

#include <cmath>
#include <memory>
#include <vector>

#include "fastatoms/calculators/calculator.hpp"
#include "fastatoms/calculators/lennard_jones.hpp"
#include "fastatoms/core/units.hpp"
#include "fastatoms/md/dynamics.hpp"
#include "fastatoms/md/velocity.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::md;

namespace {

Value settings(std::initializer_list<std::pair<const char*, Value>> entries) {
    Value map{Value::Map{}};
    for (const auto& [key, value] : entries) {
        map.insert(key, value);
    }
    return map;
}

/// E = (k/2) sum |r_i - r0_i|^2: a set of independent harmonic wells with an
/// exactly known solution, so an integrator can be compared with the truth
/// rather than with itself.
class HarmonicCalculator : public Calculator {
public:
    HarmonicCalculator(double stiffness, std::vector<double> centres)
        : Calculator(Value(Value::Map{})),
          stiffness_(stiffness),
          centres_(std::move(centres)) {}

    [[nodiscard]] std::string name() const override { return "harmonic"; }
    [[nodiscard]] std::vector<Property> implemented_properties() const override {
        return {Property::Energy, Property::FreeEnergy, Property::Forces};
    }

protected:
    void compute(const Atoms& atoms, const std::vector<Property>&) override {
        const std::vector<double>& positions = atoms.positions();
        double energy = 0.0;
        std::vector<double> forces(positions.size());
        for (std::size_t k = 0; k < positions.size(); ++k) {
            const double d = positions[k] - centres_[k];
            energy += 0.5 * stiffness_ * d * d;
            forces[k] = -stiffness_ * d;
        }
        results_.energy = energy;
        results_.free_energy = energy;
        results_.forces = std::move(forces);
    }

private:
    double stiffness_;
    std::vector<double> centres_;
};

/// A calculator with no forces at all, for free-flight tests.
class FreeCalculator : public Calculator {
public:
    FreeCalculator() : Calculator(Value(Value::Map{})) {}
    [[nodiscard]] std::string name() const override { return "free"; }
    [[nodiscard]] std::vector<Property> implemented_properties() const override {
        return {Property::Energy, Property::FreeEnergy, Property::Forces};
    }

protected:
    void compute(const Atoms& atoms, const std::vector<Property>&) override {
        results_.energy = 0.0;
        results_.free_energy = 0.0;
        results_.forces.assign(3 * atoms.size(), 0.0);
    }
};

Atoms fcc_argon(double a = 5.30, int repeat = 2) {
    const double basis[4][3] = {
        {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
    std::vector<double> scaled;
    for (const auto& site : basis) {
        scaled.insert(scaled.end(), {site[0], site[1], site[2]});
    }
    Atoms cell(std::vector<int>(4, 18), std::vector<double>(12, 0.0), Cell::cubic(a));
    cell.set_scaled_positions(scaled);
    Atoms atoms = repeat > 1 ? cell.repeat(repeat, repeat, repeat) : cell;
    atoms.set_calculator(std::make_shared<LennardJonesCalculator>(settings(
        {{"epsilon", Value(0.0103)}, {"sigma", Value(3.40)}, {"cutoff", Value(8.5)}})));
    return atoms;
}

}  // namespace

// ---------------------------------------------------------------------------
// Velocity setup
// ---------------------------------------------------------------------------

FA_TEST(maxwell_boltzmann_hits_the_requested_temperature) {
    Atoms atoms = fcc_argon();
    const int dof = momentum_degrees_of_freedom(atoms, true);
    FA_CHECK(dof == 3 * static_cast<int>(atoms.size()) - 3);

    set_maxwell_boltzmann_velocities(atoms, 300.0, 1234);
    FA_CHECK_CLOSE(atoms.temperature(dof), 300.0, 1e-10);
    // Removing the drift is what makes 3N-3 the right count.
    FA_CHECK_CLOSE(atoms.total_momentum().norm(), 0.0, 1e-10);

    // A seed makes the draw reproducible; a different one changes it.
    Atoms again = fcc_argon();
    set_maxwell_boltzmann_velocities(again, 300.0, 1234);
    FA_CHECK(again.momenta() == atoms.momenta());

    Atoms other = fcc_argon();
    set_maxwell_boltzmann_velocities(other, 300.0, 4321);
    FA_CHECK(other.momenta() != atoms.momenta());
}

FA_TEST(velocity_setup_edge_cases) {
    Atoms atoms = fcc_argon();
    set_maxwell_boltzmann_velocities(atoms, 0.0, 1);
    FA_CHECK_CLOSE(atoms.kinetic_energy(), 0.0, 1e-15);
    FA_CHECK_THROWS(set_maxwell_boltzmann_velocities(atoms, -1.0, 1), DynamicsError);

    // Without the exact rescaling the sample only lands near the target.
    Atoms sampled = fcc_argon();
    set_maxwell_boltzmann_velocities(sampled, 300.0, 99, true, false);
    const double t = sampled.temperature(momentum_degrees_of_freedom(sampled, true));
    FA_CHECK(t > 150.0 && t < 500.0);

    Atoms rescaled = sampled;
    rescale_to_temperature(rescaled, 300.0, momentum_degrees_of_freedom(rescaled, true));
    FA_CHECK_CLOSE(rescaled.temperature(momentum_degrees_of_freedom(rescaled, true)), 300.0,
                   1e-10);
}

// ---------------------------------------------------------------------------
// Integrators against exact answers
// ---------------------------------------------------------------------------

FA_TEST(a_free_particle_moves_in_a_straight_line) {
    for (const char* scheme : {"euler", "velocity_verlet", "leapfrog"}) {
        Atoms atoms(std::vector<std::string>{"H"}, {0.0, 0.0, 0.0});
        atoms.set_calculator(std::make_shared<FreeCalculator>());
        atoms.set_velocities({0.1, -0.2, 0.05});

        Dynamics dynamics(settings({{"timestep", Value(1.0)},
                                    {"integrator", Value(scheme)},
                                    {"fix_center_of_mass", Value(false)}}));
        const DynamicsResult result = dynamics.run(atoms, 100);
        FA_CHECK(result.steps == 100);

        // r = v t exactly, for every scheme, since there is no force.
        const double elapsed = 100.0 * units::fs;
        FA_CHECK_CLOSE(atoms.position(0).x, 0.1 * elapsed, 1e-12);
        FA_CHECK_CLOSE(atoms.position(0).y, -0.2 * elapsed, 1e-12);
        FA_CHECK_CLOSE(atoms.velocities()[0], 0.1, 1e-12);
    }
}

FA_TEST(velocity_verlet_tracks_a_harmonic_oscillator) {
    // One atom of unit mass in a well of unit stiffness: omega = 1 in internal
    // units, so the period is 2 pi ~= 64 fs.
    const double stiffness = 1.0;
    const double amplitude = 0.3;

    Atoms atoms(std::vector<int>{1}, {amplitude, 0.0, 0.0});
    atoms.set_masses({1.0});
    atoms.set_calculator(
        std::make_shared<HarmonicCalculator>(stiffness, std::vector<double>{0.0, 0.0, 0.0}));
    atoms.set_velocities({0.0, 0.0, 0.0});

    const double omega = std::sqrt(stiffness / 1.0);
    const double timestep_fs = 0.2;
    const int steps = 400;

    Dynamics dynamics(settings({{"timestep", Value(timestep_fs)},
                                {"integrator", Value("velocity_verlet")},
                                {"fix_center_of_mass", Value(false)}}));
    const DynamicsResult result = dynamics.run(atoms, steps);

    const double elapsed = steps * timestep_fs * units::fs;
    // x(t) = A cos(omega t). A second-order scheme at this step size should
    // track it to a few parts in ten thousand.
    FA_CHECK_CLOSE(atoms.position(0).x, amplitude * std::cos(omega * elapsed), 2e-3);
    FA_CHECK_CLOSE(atoms.velocities()[0], -amplitude * omega * std::sin(omega * elapsed), 2e-3);

    // The energy of the oscillator is fixed at A^2 k / 2.
    const double expected = 0.5 * stiffness * amplitude * amplitude;
    for (const DynamicsStep& frame : result.history) {
        FA_CHECK_CLOSE(frame.total_energy, expected, 1e-3);
    }
}

FA_TEST(leapfrog_and_velocity_verlet_generate_the_same_positions) {
    // The two schemes are algebraically the same; only the instant at which
    // the velocities are defined differs. The positions must agree exactly.
    const auto trajectory = [](const char* scheme) {
        Atoms atoms = fcc_argon(5.30, 2);
        set_maxwell_boltzmann_velocities(atoms, 60.0, 77);
        Dynamics dynamics(settings({{"timestep", Value(4.0)}, {"integrator", Value(scheme)}}));
        (void)dynamics.run(atoms, 60);
        return atoms.positions();
    };

    const std::vector<double> verlet = trajectory("velocity_verlet");
    const std::vector<double> leapfrog = trajectory("leapfrog");
    FA_CHECK(verlet.size() == leapfrog.size());
    for (std::size_t k = 0; k < verlet.size(); ++k) {
        FA_CHECK_CLOSE(leapfrog[k], verlet[k], 1e-10);
    }
}

FA_TEST(euler_drifts_where_verlet_does_not) {
    const auto drift_of = [](const char* scheme, double timestep) {
        Atoms atoms = fcc_argon(5.30, 2);
        set_maxwell_boltzmann_velocities(atoms, 40.0, 2024);
        Dynamics dynamics(
            settings({{"timestep", Value(timestep)}, {"integrator", Value(scheme)}}));
        const DynamicsResult result = dynamics.run(atoms, 400);
        FA_CHECK(result.reason == DynamicsStopReason::Completed);
        return std::abs(result.energy_drift_per_atom(atoms.size()));
    };

    const double euler = drift_of("euler", 2.0);
    const double verlet = drift_of("velocity_verlet", 2.0);
    const double leapfrog = drift_of("leapfrog", 2.0);

    // Symplectic schemes keep the energy bounded; explicit Euler pumps it in.
    FA_CHECK(verlet < 1e-5);
    FA_CHECK(leapfrog < 1e-5);
    FA_CHECK(euler > 50.0 * verlet);
}

FA_TEST(verlet_conserves_energy_and_momentum) {
    Atoms atoms = fcc_argon(5.30, 2);
    set_maxwell_boltzmann_velocities(atoms, 50.0, 31337);

    Dynamics dynamics(settings({{"timestep", Value(5.0)}, {"interval", Value(10)}}));
    const DynamicsResult result = dynamics.run(atoms, 500);

    FA_CHECK(result.reason == DynamicsStopReason::Completed);
    FA_CHECK(dynamics.ensemble() == "nve");
    // A microcanonical run has nothing but the physical energy to conserve.
    FA_CHECK_CLOSE(result.history.front().conserved_energy,
                   result.history.front().total_energy, 1e-12);
    FA_CHECK(std::abs(result.energy_drift_per_atom(atoms.size())) < 1e-5);
    // Newton's third law: the total momentum cannot change.
    FA_CHECK_CLOSE(atoms.total_momentum().norm(), 0.0, 1e-9);
    // One force evaluation per step, plus the one for the initial state.
    FA_CHECK(result.evaluations == 501);
}

FA_TEST(a_reckless_timestep_is_caught) {
    // Verlet is stable only while omega*dt < 2. In a harmonic well with
    // omega = 1 per internal time unit, a 500 fs step is fifty times past
    // that, and the amplitude grows without bound instead of oscillating.
    Atoms atoms(std::vector<int>{1}, {0.3, 0.0, 0.0});
    atoms.set_masses({1.0});
    atoms.set_calculator(
        std::make_shared<HarmonicCalculator>(1.0, std::vector<double>{0.0, 0.0, 0.0}));
    atoms.set_velocities({0.0, 0.0, 0.0});

    Dynamics dynamics(settings({{"timestep", Value(500.0)},
                                {"fix_center_of_mass", Value(false)},
                                {"interval", Value(1000)}}));
    const DynamicsResult result = dynamics.run(atoms, 400);
    FA_CHECK(result.reason == DynamicsStopReason::Unstable);
    FA_CHECK(result.message.find("timestep") != std::string::npos);
    FA_CHECK(result.steps < 400);
}

FA_TEST(integrator_names_and_errors) {
    FA_CHECK(make_integrator("")->name() == "velocity_verlet");
    FA_CHECK(make_integrator("VV")->name() == "velocity_verlet");
    FA_CHECK(make_integrator("Velocity-Verlet")->name() == "velocity_verlet");
    FA_CHECK(make_integrator("euler")->name() == "euler");
    FA_CHECK(make_integrator("leapfrog")->name() == "leapfrog");
    FA_CHECK(!make_integrator("euler")->is_symplectic());
    FA_CHECK(make_integrator("leapfrog")->is_symplectic());
    FA_CHECK_THROWS(make_integrator("runge_kutta"), DynamicsError);
    FA_CHECK(integrator_names().size() == 3);
}

FA_TEST(dynamics_reports_useful_failures) {
    Atoms atoms = fcc_argon(5.30, 1);
    atoms.set_calculator(nullptr);
    Dynamics dynamics;
    FA_CHECK_THROWS(dynamics.run(atoms, 1), DynamicsError);

    Atoms empty;
    empty.set_calculator(std::make_shared<FreeCalculator>());
    FA_CHECK_THROWS(Dynamics().run(empty, 1), DynamicsError);

    Atoms good = fcc_argon(5.30, 1);
    FA_CHECK_THROWS(Dynamics(settings({{"timestep", Value(-1.0)}})).run(good, 1), DynamicsError);
    // A thermostat with nothing to aim at.
    FA_CHECK_THROWS(Dynamics(settings({{"thermostat", Value("nose_hoover")}})).run(good, 1),
                    DynamicsError);
}

FA_TEST_MAIN()
