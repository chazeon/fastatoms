// SPDX-License-Identifier: MIT
#include "fastatoms/md/barostat.hpp"

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

/// Keep only the isotropic part of a tensor: (1/3) trace(m) I.
Mat3 isotropic_part(const Mat3& m) {
    const double average = (m(0, 0) + m(1, 1) + m(2, 2)) / 3.0;
    return Mat3::diagonal(average, average, average);
}

}  // namespace

// ---------------------------------------------------------------------------
// Base
// ---------------------------------------------------------------------------

Barostat::Barostat(Value parameters) : parameters_(std::move(parameters)) {
    if (parameters_.is_null()) {
        parameters_ = Value(Value::Map{});
    }
    if (!parameters_.is_map()) {
        throw DynamicsError("barostat parameters must be a mapping, got " +
                            std::string(parameters_.type_name()));
    }
}

const Value& Barostat::parameter(std::string_view key) const {
    if (const Value* found = parameters_.find(key)) {
        return *found;
    }
    return Value::null();
}

void Barostat::fail(const std::string& message) const {
    throw DynamicsError(name() + " barostat: " + message);
}

double Barostat::time_constant_fs() const {
    double value = parameter("tau").value_or(0.0);
    if (value <= 0.0) {
        value = parameter("time_constant").value_or(0.0);
    }
    if (value <= 0.0) {
        // The cell should respond an order of magnitude more slowly than the
        // thermostat, or the two fight each other.
        value = 1000.0;
    }
    return value;
}

bool Barostat::is_isotropic() const { return parameter("isotropic").value_or(true); }

void Barostat::require_periodic(const Atoms& atoms) const {
    if (!atoms.cell().is_invertible()) {
        fail("the structure needs a full 3x3 cell before its pressure can be controlled");
    }
    if (!atoms.cell().any_pbc()) {
        fail("the structure needs periodic boundary conditions");
    }
}

void Barostat::deform(Atoms& atoms, const Mat3& deformation) {
    // scale_atoms keeps fractional coordinates fixed, which is exactly an
    // affine transformation of the positions by the same matrix.
    atoms.set_cell(atoms.cell().matrix() * deformation, true);
}

std::string Barostat::to_string() const {
    std::ostringstream out;
    out.precision(4);
    out << name() << "(tau=" << time_constant_fs() << " fs"
        << (is_isotropic() ? ", isotropic" : ", full cell") << ")";
    return out.str();
}

// ---------------------------------------------------------------------------
// Berendsen
// ---------------------------------------------------------------------------

BerendsenBarostat::BerendsenBarostat(Value parameters) : Barostat(std::move(parameters)) {}

double BerendsenBarostat::compressibility() const {
    const double value = parameter("compressibility").value_or(0.1);
    if (!(value > 0.0)) {
        fail("'compressibility' must be positive (units of 1/GPa)");
    }
    return value;
}

Mat3 BerendsenBarostat::deform_cell(Atoms& atoms, double dt,
                                    const DynamicsContext& context) {
    require_periodic(atoms);

    const double tau = time_constant_fs() * units::fs;
    // The compressibility is quoted in 1/GPa and the pressures are in
    // eV/angstrom^3, so convert it into the same system.
    const double kappa = compressibility() / units::GPa;
    const double limit = parameter("max_strain").value_or(0.01);

    Mat3 deformation = Mat3::identity();
    if (is_isotropic()) {
        const double pressure = instantaneous_pressure(atoms);
        double mu = std::cbrt(1.0 - kappa * (dt / tau) * (context.pressure - pressure));
        mu = std::clamp(mu, 1.0 - limit, 1.0 + limit);
        deformation = Mat3::diagonal(mu, mu, mu);
    } else {
        // Each axis follows its own diagonal component of the pressure tensor.
        const Mat3 stress = total_stress(atoms);
        for (std::size_t k = 0; k < 3; ++k) {
            const double pressure = -stress(k, k);
            double mu = std::cbrt(1.0 - kappa * (dt / tau) * (context.pressure - pressure));
            deformation(k, k) = std::clamp(mu, 1.0 - limit, 1.0 + limit);
        }
    }

    deform(atoms, deformation);
    // Berendsen scales the box only; the velocities are left as they are.
    return Mat3::identity();
}

// ---------------------------------------------------------------------------
// Parrinello-Rahman
// ---------------------------------------------------------------------------

ParrinelloRahmanBarostat::ParrinelloRahmanBarostat(Value parameters)
    : Barostat(std::move(parameters)) {}

void ParrinelloRahmanBarostat::begin(const Atoms& atoms, const DynamicsContext&) {
    require_periodic(atoms);
    strain_rate_ = Mat3{};
}

double ParrinelloRahmanBarostat::mass(const DynamicsContext& context) const {
    if (const double given = parameter("mass").value_or(0.0); given > 0.0) {
        return given;
    }
    // W = (N_f + 3) kB T tau_P^2 gives the cell a natural period near tau_P.
    const double tau = time_constant_fs() * units::fs;
    const double temperature = context.temperature > 0.0 ? context.temperature : 300.0;
    return (static_cast<double>(context.degrees_of_freedom) + 3.0) * units::kB * temperature *
           tau * tau;
}

void ParrinelloRahmanBarostat::kick(Atoms& atoms, double dt, const DynamicsContext& context) {
    require_periodic(atoms);

    const double w = mass(context);
    if (!(w > 0.0)) {
        fail("the barostat mass must be positive");
    }

    // The cell accelerates on the imbalance between the internal pressure
    // tensor - which is minus the stress, kinetic term included - and the
    // applied pressure.
    const Mat3 stress = total_stress(atoms);
    Mat3 driving;
    for (std::size_t a = 0; a < 3; ++a) {
        for (std::size_t b = 0; b < 3; ++b) {
            const double external = a == b ? context.pressure : 0.0;
            driving(a, b) = (-stress(a, b) - external) * atoms.volume() / w;
        }
    }
    if (is_isotropic()) {
        driving = isotropic_part(driving);
    }

    for (std::size_t k = 0; k < 9; ++k) {
        strain_rate_.m[k] += driving.m[k] * dt;
    }
    if (is_isotropic()) {
        strain_rate_ = isotropic_part(strain_rate_);
    }
}

Mat3 ParrinelloRahmanBarostat::deform_cell(Atoms& atoms, double dt,
                                           const DynamicsContext& context) {
    require_periodic(atoms);
    (void)context;

    Mat3 deformation = Mat3::identity();
    Mat3 velocity_transform = Mat3::identity();
    for (std::size_t a = 0; a < 3; ++a) {
        for (std::size_t b = 0; b < 3; ++b) {
            deformation(a, b) += strain_rate_(a, b) * dt;
            velocity_transform(a, b) -= strain_rate_(a, b) * dt;
        }
    }

    if (!(deformation.determinant() > 0.0)) {
        fail("the cell collapsed; the barostat mass or timestep is too small");
    }

    deform(atoms, deformation);

    // The atoms' velocities follow the box, which is what makes the volume a
    // real dynamical variable rather than a relaxation target.
    std::vector<double> velocities = atoms.velocities();
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        double* v = velocities.data() + 3 * i;
        (Vec3::load(v) * velocity_transform).store(v);
    }
    atoms.set_velocities(velocities);

    return velocity_transform;
}

double ParrinelloRahmanBarostat::conserved_energy(Atoms& atoms,
                                                  const DynamicsContext& context) const {
    const double w = mass(context);
    double kinetic = 0.0;
    for (const double value : strain_rate_.m) {
        kinetic += value * value;
    }
    // The barostat's own kinetic energy plus the work done against the applied
    // pressure.
    return 0.5 * w * kinetic + context.pressure * atoms.volume();
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::shared_ptr<Barostat> make_barostat(std::string_view name, const Value& parameters) {
    const std::string key = canonical(name);
    if (key.empty() || key == "none" || key == "off" || key == "nvt" || key == "nve") {
        return nullptr;
    }
    if (key == "berendsen") {
        return std::make_shared<BerendsenBarostat>(parameters);
    }
    if (key == "parrinellorahman" || key == "pr" || key == "npt") {
        return std::make_shared<ParrinelloRahmanBarostat>(parameters);
    }
    throw DynamicsError("unknown barostat '" + std::string(name) +
                        "'; use berendsen or parrinello_rahman");
}

std::vector<std::string> barostat_names() { return {"berendsen", "parrinello_rahman"}; }

}  // namespace fastatoms::md
