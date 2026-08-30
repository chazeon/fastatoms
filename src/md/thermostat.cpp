// SPDX-License-Identifier: MIT
#include "fastatoms/md/thermostat.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

#include "fastatoms/core/units.hpp"

namespace fastatoms::md {
namespace {

std::string canonical(std::string_view name) {
    std::string key;
    for (const char c : name) {
        if (c != '-' && c != '_' && c != ' ') {
            key += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    return key;
}

/// Multiply every velocity by `factor`, in place.
void scale_velocities(Atoms& atoms, double factor) {
    if (factor == 1.0) {
        return;
    }
    std::vector<double> momenta = atoms.momenta();
    for (double& value : momenta) {
        value *= factor;
    }
    atoms.set_momenta(momenta);
}

}  // namespace

// ---------------------------------------------------------------------------
// Base
// ---------------------------------------------------------------------------

Thermostat::Thermostat(Value parameters) : parameters_(std::move(parameters)) {
    if (parameters_.is_null()) {
        parameters_ = Value(Value::Map{});
    }
    if (!parameters_.is_map()) {
        throw DynamicsError("thermostat parameters must be a mapping, got " +
                            std::string(parameters_.type_name()));
    }
}

const Value& Thermostat::parameter(std::string_view key) const {
    if (const Value* found = parameters_.find(key)) {
        return *found;
    }
    return Value::null();
}

void Thermostat::fail(const std::string& message) const {
    throw DynamicsError(name() + " thermostat: " + message);
}

double Thermostat::time_constant_fs() const {
    double value = parameter("tau").value_or(0.0);
    if (value <= 0.0) {
        value = parameter("time_constant").value_or(0.0);
    }
    if (value <= 0.0) {
        // A hundred femtoseconds is slow enough not to disturb the dynamics
        // and fast enough to equilibrate in a few picoseconds.
        value = 100.0;
    }
    return value;
}

std::string Thermostat::to_string() const {
    std::ostringstream out;
    out.precision(4);
    out << name() << "(tau=" << time_constant_fs() << " fs)";
    return out.str();
}

// ---------------------------------------------------------------------------
// Berendsen
// ---------------------------------------------------------------------------

BerendsenThermostat::BerendsenThermostat(Value parameters)
    : Thermostat(std::move(parameters)) {}

VelocityUpdate BerendsenThermostat::apply(Atoms& atoms, double dt,
                                          const DynamicsContext& context) {
    const double target = context.temperature;
    if (!(target > 0.0)) {
        fail("a target temperature is required");
    }
    const double current = atoms.temperature(context.degrees_of_freedom);
    if (!(current > 0.0)) {
        // Nothing is moving; there is no velocity to rescale.
        return {};
    }

    const double tau = time_constant_fs() * units::fs;
    double factor = std::sqrt(1.0 + (dt / tau) * (target / current - 1.0));

    // A cold start can ask for an enormous factor; cap it so one step cannot
    // launch the structure.
    const double limit = parameter("max_scale").value_or(1.25);
    factor = std::clamp(factor, 1.0 / limit, limit);

    scale_velocities(atoms, factor);
    return VelocityUpdate{factor, {}};
}

// ---------------------------------------------------------------------------
// Nose
// ---------------------------------------------------------------------------

NoseThermostat::NoseThermostat(Value parameters) : Thermostat(std::move(parameters)) {}

void NoseThermostat::begin(const Atoms&, const DynamicsContext&) {
    s_ = 1.0;
    friction_ = 0.0;
}

double NoseThermostat::mass(const DynamicsContext& context) const {
    if (const double given = parameter("mass").value_or(0.0); given > 0.0) {
        return given;
    }
    // Q = g kB T tau^2 gives the thermostat a natural period of about tau.
    const double tau = time_constant_fs() * units::fs;
    const double g = static_cast<double>(context.degrees_of_freedom) + 1.0;
    return g * units::kB * context.temperature * tau * tau;
}

VelocityUpdate NoseThermostat::apply(Atoms& atoms, double dt,
                                     const DynamicsContext& context) {
    if (!(context.temperature > 0.0)) {
        fail("a target temperature is required");
    }
    const double q = mass(context);
    if (!(q > 0.0)) {
        fail("the thermostat mass must be positive");
    }

    // Nose counts the extended system's degrees of freedom, which is one more
    // than the physical N_f because of the time scaling in his derivation.
    const double g = static_cast<double>(context.degrees_of_freedom) + 1.0;
    const double target = g * units::kB * context.temperature;

    double kinetic = atoms.kinetic_energy();

    // Symmetric Trotter split: half a kick of the friction, the velocity
    // scaling over the whole interval, then the other half kick with the
    // kinetic energy that resulted.
    friction_ += 0.5 * dt * (2.0 * kinetic - target) / q;

    const double factor = std::exp(-friction_ * dt);
    scale_velocities(atoms, factor);
    kinetic *= factor * factor;

    // s obeys ds/dt = s xi, so ln s accumulates the friction.
    s_ *= std::exp(friction_ * dt);

    friction_ += 0.5 * dt * (2.0 * kinetic - target) / q;
    return VelocityUpdate{factor, {}};
}

double NoseThermostat::conserved_energy(const DynamicsContext& context) const {
    const double q = mass(context);
    const double g = static_cast<double>(context.degrees_of_freedom) + 1.0;
    // H_Nose = ... + p_s^2/(2Q) + g kB T ln s, with p_s = Q xi.
    return 0.5 * q * friction_ * friction_ +
           g * units::kB * context.temperature * std::log(s_);
}

// ---------------------------------------------------------------------------
// Nose-Hoover, with chains
// ---------------------------------------------------------------------------

NoseHooverThermostat::NoseHooverThermostat(Value parameters)
    : Thermostat(std::move(parameters)) {}

std::size_t NoseHooverThermostat::chain_length() const {
    const auto value = parameter("chain_length").value_or(std::int64_t{1});
    if (value < 1) {
        fail("'chain_length' must be at least 1");
    }
    return static_cast<std::size_t>(value);
}

std::vector<double> NoseHooverThermostat::masses(const DynamicsContext& context) const {
    const std::size_t length = chain_length();
    const double tau = time_constant_fs() * units::fs;
    const double kt = units::kB * context.temperature;

    std::vector<double> q(length, kt * tau * tau);
    // The first thermostat carries the whole system, the rest carry one
    // degree of freedom each.
    q[0] = static_cast<double>(context.degrees_of_freedom) * kt * tau * tau;

    if (const double given = parameter("mass").value_or(0.0); given > 0.0) {
        q.assign(length, given);
    }
    return q;
}

void NoseHooverThermostat::begin(const Atoms&, const DynamicsContext& context) {
    friction_.assign(chain_length(), 0.0);
    position_.assign(chain_length(), 0.0);
    (void)context;
}

VelocityUpdate NoseHooverThermostat::apply(Atoms& atoms, double dt,
                                           const DynamicsContext& context) {
    if (!(context.temperature > 0.0)) {
        fail("a target temperature is required");
    }
    const std::size_t m = chain_length();
    if (friction_.size() != m) {
        begin(atoms, context);
    }

    const std::vector<double> q = masses(context);
    for (const double value : q) {
        if (!(value > 0.0)) {
            fail("the thermostat masses must be positive");
        }
    }

    const double kt = units::kB * context.temperature;
    const double target = static_cast<double>(context.degrees_of_freedom) * kt;
    const double half = 0.5 * dt;

    double kinetic = atoms.kinetic_energy();

    // Forces on the chain: the first is driven by the physical kinetic energy,
    // each of the rest by the one before it.
    std::vector<double> g(m, 0.0);
    const auto refresh_forces = [&](double current_kinetic) {
        g[0] = (2.0 * current_kinetic - target) / q[0];
        for (std::size_t j = 1; j < m; ++j) {
            g[j] = (q[j - 1] * friction_[j - 1] * friction_[j - 1] - kt) / q[j];
        }
    };

    // --- inward sweep: update the chain from its far end towards the atoms --
    refresh_forces(kinetic);
    friction_[m - 1] += g[m - 1] * half;
    for (std::size_t j = m - 1; j-- > 0;) {
        const double damping = std::exp(-friction_[j + 1] * 0.25 * dt);
        friction_[j] = (friction_[j] * damping + g[j] * half) * damping;
    }

    // --- the atoms feel only the innermost friction ------------------------
    const double factor = std::exp(-friction_[0] * dt);
    scale_velocities(atoms, factor);
    kinetic *= factor * factor;
    for (std::size_t j = 0; j < m; ++j) {
        position_[j] += friction_[j] * dt;
    }

    // --- outward sweep with the kinetic energy that resulted ----------------
    refresh_forces(kinetic);
    for (std::size_t j = 0; j + 1 < m; ++j) {
        const double damping = std::exp(-friction_[j + 1] * 0.25 * dt);
        friction_[j] = (friction_[j] * damping + g[j] * half) * damping;
        g[j + 1] = (q[j] * friction_[j] * friction_[j] - kt) / q[j + 1];
    }
    friction_[m - 1] += g[m - 1] * half;

    return VelocityUpdate{factor, {}};
}

double NoseHooverThermostat::conserved_energy(const DynamicsContext& context) const {
    if (friction_.empty()) {
        return 0.0;
    }
    const std::vector<double> q = masses(context);
    const double kt = units::kB * context.temperature;

    double energy = 0.0;
    for (std::size_t j = 0; j < friction_.size(); ++j) {
        energy += 0.5 * q[j] * friction_[j] * friction_[j];
    }
    // The chain's "positions" contribute N_f kB T ln s_1 + kB T sum ln s_j.
    energy += static_cast<double>(context.degrees_of_freedom) * kt * position_[0];
    for (std::size_t j = 1; j < position_.size(); ++j) {
        energy += kt * position_[j];
    }
    return energy;
}

// ---------------------------------------------------------------------------
// Langevin
// ---------------------------------------------------------------------------

LangevinThermostat::LangevinThermostat(Value parameters) : Thermostat(std::move(parameters)) {}

double LangevinThermostat::friction() const {
    if (const double given = parameter("friction").value_or(0.0); given > 0.0) {
        return given;
    }
    // No explicit friction: take it from the coupling time instead.
    return 1.0 / time_constant_fs();
}

void LangevinThermostat::begin(const Atoms&, const DynamicsContext&) {
    const auto seed = parameter("seed").value_or(std::int64_t{0});
    generator_.seed(seed != 0 ? static_cast<std::uint64_t>(seed) : std::random_device{}());
    heat_ = 0.0;
}

VelocityUpdate LangevinThermostat::apply(Atoms& atoms, double dt,
                                         const DynamicsContext& context) {
    if (!(context.temperature > 0.0)) {
        fail("a target temperature is required");
    }
    const double gamma = friction() / units::fs;  // 1/fs -> 1/(internal time)
    if (!(gamma > 0.0)) {
        fail("'friction' must be positive");
    }

    // The Ornstein-Uhlenbeck step, solved exactly over the interval: the old
    // velocity is damped by c1 and the bath supplies exactly enough noise to
    // hold the variance at kB T / m.
    const double c1 = std::exp(-gamma * dt);
    const double c2 = std::sqrt(std::max(0.0, 1.0 - c1 * c1));

    const std::vector<double> before = atoms.velocities();
    const std::vector<double> masses = atoms.masses();
    const double kinetic_before = atoms.kinetic_energy();

    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> after(before.size());
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        if (!(masses[i] > 0.0)) {
            fail("atom " + std::to_string(i) + " has a non-positive mass");
        }
        const double width = c2 * std::sqrt(units::kB * context.temperature / masses[i]);
        for (std::size_t k = 0; k < 3; ++k) {
            after[3 * i + k] = c1 * before[3 * i + k] + width * normal(generator_);
        }
    }
    atoms.set_velocities(after);

    // The noise adds momentum in a random direction. Left alone it would
    // random-walk the whole structure, and that drift is not one of the
    // degrees of freedom the temperature is divided by.
    if (context.fix_center_of_mass) {
        atoms.zero_center_of_mass_momentum();
        after = atoms.velocities();
    }

    // Whatever kinetic energy the thermostat took, the bath gained.
    heat_ -= atoms.kinetic_energy() - kinetic_before;

    // Report the transformation as an affine map so leapfrog can apply the
    // same realisation of the noise to its half-step velocities.
    VelocityUpdate update;
    update.factor = c1;
    update.shift.resize(after.size());
    for (std::size_t k = 0; k < after.size(); ++k) {
        update.shift[k] = after[k] - c1 * before[k];
    }
    return update;
}

double LangevinThermostat::conserved_energy(const DynamicsContext&) const { return heat_; }

std::string LangevinThermostat::to_string() const {
    std::ostringstream out;
    out.precision(4);
    out << name() << "(friction=" << friction() << " /fs)";
    return out.str();
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::shared_ptr<Thermostat> make_thermostat(std::string_view name, const Value& parameters) {
    const std::string key = canonical(name);
    if (key.empty() || key == "none" || key == "off" || key == "nve") {
        return nullptr;
    }
    if (key == "berendsen") {
        return std::make_shared<BerendsenThermostat>(parameters);
    }
    if (key == "langevin") {
        return std::make_shared<LangevinThermostat>(parameters);
    }
    if (key == "nose") {
        return std::make_shared<NoseThermostat>(parameters);
    }
    if (key == "nosehoover" || key == "nh" || key == "nvt" ||
        key == "nosehooverchain" || key == "nhc") {
        return std::make_shared<NoseHooverThermostat>(parameters);
    }
    throw DynamicsError("unknown thermostat '" + std::string(name) +
                        "'; use berendsen, langevin, nose or nose_hoover");
}

std::vector<std::string> thermostat_names() {
    return {"berendsen", "langevin", "nose", "nose_hoover"};
}

}  // namespace fastatoms::md
