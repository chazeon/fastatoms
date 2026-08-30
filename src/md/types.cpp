// SPDX-License-Identifier: MIT
#include "fastatoms/md/types.hpp"

#include <cmath>
#include <sstream>

#include "fastatoms/calculators/calculator.hpp"

namespace fastatoms::md {

std::string_view dynamics_stop_reason_name(DynamicsStopReason reason) {
    switch (reason) {
        case DynamicsStopReason::Completed:   return "completed";
        case DynamicsStopReason::Interrupted: return "interrupted";
        case DynamicsStopReason::Unstable:    return "unstable";
    }
    return "unknown";
}

double DynamicsResult::energy_drift_per_atom(std::size_t natoms) const {
    if (history.size() < 2 || natoms == 0) {
        return 0.0;
    }
    const double change = history.back().conserved_energy - history.front().conserved_energy;
    return change / static_cast<double>(natoms);
}

std::string DynamicsResult::to_string() const {
    std::ostringstream out;
    out.precision(6);
    out << "DynamicsResult(" << dynamics_stop_reason_name(reason) << ", steps=" << steps
        << ", evaluations=" << evaluations << ", time=" << time << " fs)";
    return out.str();
}

Mat3 kinetic_stress(const Atoms& atoms) {
    Mat3 stress{};
    if (!atoms.cell().is_invertible()) {
        return stress;
    }
    const std::vector<double> velocities = atoms.velocities();
    const std::vector<double> masses = atoms.masses();

    for (std::size_t i = 0; i < atoms.size(); ++i) {
        const Vec3 v = Vec3::load(velocities.data() + 3 * i);
        const double m = masses[i];
        for (std::size_t a = 0; a < 3; ++a) {
            for (std::size_t b = 0; b < 3; ++b) {
                stress(a, b) += m * v[a] * v[b];
            }
        }
    }

    // Negative, because sigma is an energy derivative: the ideal-gas term has
    // to come out as a positive pressure, and P = -(1/3) trace(sigma).
    const double scale = -1.0 / atoms.volume();
    for (double& value : stress.m) {
        value *= scale;
    }
    return stress;
}

Mat3 total_stress(Atoms& atoms) {
    Mat3 stress = voigt_to_matrix(atoms.get_stress());
    const Mat3 kinetic = kinetic_stress(atoms);
    for (std::size_t k = 0; k < 9; ++k) {
        stress.m[k] += kinetic.m[k];
    }
    return stress;
}

double instantaneous_pressure(Atoms& atoms) {
    const Mat3 stress = total_stress(atoms);
    return -(stress(0, 0) + stress(1, 1) + stress(2, 2)) / 3.0;
}

}  // namespace fastatoms::md
