// SPDX-License-Identifier: MIT
#include "fastatoms/md/constraints.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

#include "fastatoms/md/types.hpp"

namespace fastatoms::md {

std::string ConstraintReport::to_string() const {
    std::ostringstream out;
    out << "ConstraintReport(" << iterations << " iterations, residual " << residual
        << (converged ? ", converged)" : ", NOT converged)");
    return out.str();
}

Constraints::Constraints(std::vector<DistanceConstraint> bonds, double tolerance,
                         int max_iterations)
    : bonds_(std::move(bonds)) {
    set_tolerance(tolerance);
    set_max_iterations(max_iterations);
    for (const DistanceConstraint& bond : bonds_) {
        if (bond.i == bond.j) {
            throw DynamicsError("constraints: a constraint needs two different atoms");
        }
        if (!(bond.length > 0.0)) {
            throw DynamicsError("constraints: a constrained distance must be positive");
        }
    }
}

void Constraints::set_tolerance(double tolerance) {
    if (!(tolerance > 0.0)) {
        throw DynamicsError("constraints: the tolerance must be positive");
    }
    tolerance_ = tolerance;
}

void Constraints::set_max_iterations(int iterations) {
    if (iterations < 1) {
        throw DynamicsError("constraints: at least one iteration is needed");
    }
    max_iterations_ = iterations;
}

void Constraints::cache_cell(const Atoms& atoms) const {
    periodic_ = atoms.cell().any_pbc() && atoms.cell().is_invertible();
    if (periodic_) {
        cell_ = atoms.cell().matrix();
        inverse_cell_ = cell_.inverse();
    }
}

Vec3 Constraints::displacement(const Atoms& atoms, const double* positions, std::size_t i,
                               std::size_t j) const {
    (void)atoms;
    Vec3 d{positions[3 * j] - positions[3 * i], positions[3 * j + 1] - positions[3 * i + 1],
           positions[3 * j + 2] - positions[3 * i + 2]};
    if (!periodic_) {
        return d;
    }
    Vec3 fractional = d * inverse_cell_;
    fractional.x -= std::round(fractional.x);
    fractional.y -= std::round(fractional.y);
    fractional.z -= std::round(fractional.z);
    return fractional * cell_;
}

ConstraintReport Constraints::shake(Atoms& atoms, const std::vector<double>& reference,
                                    std::vector<double>* velocities, double dt) const {
    ConstraintReport report;
    if (bonds_.empty()) {
        return report;
    }
    if (reference.size() != 3 * atoms.size()) {
        throw DynamicsError("shake: the reference positions do not match the structure");
    }
    cache_cell(atoms);

    const std::vector<double> masses = atoms.masses();
    std::vector<double> inverse_mass(masses.size());
    for (std::size_t k = 0; k < masses.size(); ++k) {
        if (!(masses[k] > 0.0)) {
            throw DynamicsError("shake: atom " + std::to_string(k) +
                                " has a non-positive mass");
        }
        inverse_mass[k] = 1.0 / masses[k];
    }

    double* FASTATOMS_RESTRICT positions = atoms.positions_data();
    const bool track_velocities = velocities != nullptr && dt != 0.0;
    const double inverse_dt = track_velocities ? 1.0 / dt : 0.0;

    // The bond directions at the start of the step. The constraint force acts
    // along these, not along the (already violated) new directions - that is
    // what makes the scheme time reversible and keeps it symplectic.
    std::vector<Vec3> reference_bonds(bonds_.size());
    for (std::size_t c = 0; c < bonds_.size(); ++c) {
        reference_bonds[c] =
            displacement(atoms, reference.data(), bonds_[c].i, bonds_[c].j);
    }

    double residual = 0.0;
    int iteration = 0;
    for (; iteration < max_iterations_; ++iteration) {
        residual = 0.0;
        for (std::size_t c = 0; c < bonds_.size(); ++c) {
            const DistanceConstraint& bond = bonds_[c];
            const double target = bond.length * bond.length;
            const Vec3 current = displacement(atoms, positions, bond.i, bond.j);
            const double error = current.norm2() - target;
            residual = std::max(residual, std::abs(error) / target);
            if (std::abs(error) <= tolerance_ * target) {
                continue;
            }

            const Vec3& original = reference_bonds[c];
            const double projection = dot(current, original);
            const double sum_inverse_mass = inverse_mass[bond.i] + inverse_mass[bond.j];
            const double denominator = 2.0 * sum_inverse_mass * projection;
            if (std::abs(denominator) < 1.0e-14) {
                // The bond has turned through ninety degrees in one step: the
                // timestep is far too large for this geometry.
                throw DynamicsError(
                    "shake: the constraint between atoms " + std::to_string(bond.i) + " and " +
                    std::to_string(bond.j) +
                    " cannot be solved; the bond rotated too far in one step, which means "
                    "the timestep is too large or the structure is already broken");
            }

            const double lambda = error / denominator;
            const Vec3 correction = original * lambda;
            positions[3 * bond.i] += correction.x * inverse_mass[bond.i];
            positions[3 * bond.i + 1] += correction.y * inverse_mass[bond.i];
            positions[3 * bond.i + 2] += correction.z * inverse_mass[bond.i];
            positions[3 * bond.j] -= correction.x * inverse_mass[bond.j];
            positions[3 * bond.j + 1] -= correction.y * inverse_mass[bond.j];
            positions[3 * bond.j + 2] -= correction.z * inverse_mass[bond.j];

            if (track_velocities) {
                std::vector<double>& v = *velocities;
                const double scale_i = inverse_mass[bond.i] * inverse_dt;
                const double scale_j = inverse_mass[bond.j] * inverse_dt;
                v[3 * bond.i] += correction.x * scale_i;
                v[3 * bond.i + 1] += correction.y * scale_i;
                v[3 * bond.i + 2] += correction.z * scale_i;
                v[3 * bond.j] -= correction.x * scale_j;
                v[3 * bond.j + 1] -= correction.y * scale_j;
                v[3 * bond.j + 2] -= correction.z * scale_j;
            }
        }
        if (residual <= tolerance_) {
            break;
        }
    }

    report.iterations = iteration + 1;
    report.residual = residual;
    report.converged = residual <= tolerance_;
    if (!report.converged) {
        throw DynamicsError("shake: did not converge in " + std::to_string(max_iterations_) +
                            " iterations (largest relative error " + std::to_string(residual) +
                            "); the timestep is probably too large");
    }
    return report;
}

ConstraintReport Constraints::rattle(const Atoms& atoms, std::vector<double>& velocities) const {
    ConstraintReport report;
    if (bonds_.empty()) {
        return report;
    }
    if (velocities.size() != 3 * atoms.size()) {
        throw DynamicsError("rattle: the velocities do not match the structure");
    }
    cache_cell(atoms);

    const std::vector<double> masses = atoms.masses();
    std::vector<double> inverse_mass(masses.size());
    for (std::size_t k = 0; k < masses.size(); ++k) {
        if (!(masses[k] > 0.0)) {
            throw DynamicsError("rattle: atom " + std::to_string(k) +
                                " has a non-positive mass");
        }
        inverse_mass[k] = 1.0 / masses[k];
    }

    const double* positions = atoms.positions().data();
    std::vector<Vec3> directions(bonds_.size());
    for (std::size_t c = 0; c < bonds_.size(); ++c) {
        directions[c] = displacement(atoms, positions, bonds_[c].i, bonds_[c].j);
    }

    // The velocity condition is linear in the multipliers, so the same
    // Gauss-Seidel sweep converges - and much faster than the position stage,
    // because there is no quadratic term to chase.
    double residual = 0.0;
    int iteration = 0;
    for (; iteration < max_iterations_; ++iteration) {
        residual = 0.0;
        for (std::size_t c = 0; c < bonds_.size(); ++c) {
            const DistanceConstraint& bond = bonds_[c];
            const Vec3& r = directions[c];
            const Vec3 relative{velocities[3 * bond.j] - velocities[3 * bond.i],
                                velocities[3 * bond.j + 1] - velocities[3 * bond.i + 1],
                                velocities[3 * bond.j + 2] - velocities[3 * bond.i + 2]};
            const double projection = dot(r, relative);
            const double target = bond.length * bond.length;
            residual = std::max(residual, std::abs(projection) / target);
            if (std::abs(projection) <= tolerance_ * target) {
                continue;
            }

            const double sum_inverse_mass = inverse_mass[bond.i] + inverse_mass[bond.j];
            const double lambda = projection / (sum_inverse_mass * r.norm2());
            const Vec3 correction = r * lambda;
            velocities[3 * bond.i] += correction.x * inverse_mass[bond.i];
            velocities[3 * bond.i + 1] += correction.y * inverse_mass[bond.i];
            velocities[3 * bond.i + 2] += correction.z * inverse_mass[bond.i];
            velocities[3 * bond.j] -= correction.x * inverse_mass[bond.j];
            velocities[3 * bond.j + 1] -= correction.y * inverse_mass[bond.j];
            velocities[3 * bond.j + 2] -= correction.z * inverse_mass[bond.j];
        }
        if (residual <= tolerance_) {
            break;
        }
    }

    report.iterations = iteration + 1;
    report.residual = residual;
    report.converged = residual <= tolerance_;
    if (!report.converged) {
        throw DynamicsError("rattle: the velocity projection did not converge in " +
                            std::to_string(max_iterations_) + " iterations");
    }
    return report;
}

ConstraintReport Constraints::enforce(Atoms& atoms) const {
    // Using the current positions as their own reference is the standard way
    // to project a nearly-correct structure onto the constraint surface: the
    // bond directions barely change, so the linearisation is accurate.
    const std::vector<double> reference = atoms.positions();
    return shake(atoms, reference, nullptr, 0.0);
}

double Constraints::max_deviation(const Atoms& atoms) const {
    if (bonds_.empty()) {
        return 0.0;
    }
    cache_cell(atoms);
    const double* positions = atoms.positions().data();
    double worst = 0.0;
    for (const DistanceConstraint& bond : bonds_) {
        const double length = displacement(atoms, positions, bond.i, bond.j).norm();
        worst = std::max(worst, std::abs(length - bond.length) / bond.length);
    }
    return worst;
}

double Constraints::max_velocity_deviation(const Atoms& atoms,
                                           const std::vector<double>& velocities) const {
    if (bonds_.empty()) {
        return 0.0;
    }
    if (velocities.size() != 3 * atoms.size()) {
        throw DynamicsError("constraints: the velocities do not match the structure");
    }
    cache_cell(atoms);
    const double* positions = atoms.positions().data();
    double worst = 0.0;
    for (const DistanceConstraint& bond : bonds_) {
        const Vec3 r = displacement(atoms, positions, bond.i, bond.j);
        const Vec3 relative{velocities[3 * bond.j] - velocities[3 * bond.i],
                            velocities[3 * bond.j + 1] - velocities[3 * bond.i + 1],
                            velocities[3 * bond.j + 2] - velocities[3 * bond.i + 2]};
        const double length = r.norm();
        if (length > 0.0) {
            worst = std::max(worst, std::abs(dot(r, relative)) / length);
        }
    }
    return worst;
}

std::string Constraints::to_string() const {
    std::ostringstream out;
    out << "Constraints(" << bonds_.size() << " distances, tolerance " << tolerance_ << ")";
    return out.str();
}

}  // namespace fastatoms::md
