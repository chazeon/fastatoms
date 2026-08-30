// SPDX-License-Identifier: MIT
#include "fastatoms/forcefield/forcefield.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

#include "fastatoms/core/units.hpp"

namespace fastatoms::ff {
namespace {

constexpr double kPi = 3.14159265358979323846;

}  // namespace

std::string EnergyBreakdown::to_string() const {
    std::ostringstream out;
    out << "EnergyBreakdown(lennard_jones=" << lennard_jones;
    if (lennard_jones_tail != 0.0) {
        out << ", tail=" << lennard_jones_tail;
    }
    out << ", coulomb=" << coulomb.total();
    if (self != 0.0) {
        out << ", self=" << self;
    }
    out << ", total=" << total() << " eV)";
    return out.str();
}

ForcefieldCalculator::ForcefieldCalculator(Value parameters)
    : Calculator(std::move(parameters)) {}

std::vector<Property> ForcefieldCalculator::implemented_properties() const {
    return {Property::Energy, Property::FreeEnergy, Property::Forces, Property::Stress,
            Property::Charges};
}

const Topology& ForcefieldCalculator::topology() const {
    if (topology_.empty()) {
        throw ForcefieldError(
            "forcefield: no topology yet; evaluate a structure or call prepare() first");
    }
    return topology_;
}

void ForcefieldCalculator::set_topology(Topology topology) {
    topology_ = std::move(topology);
    model_ = nullptr;
    prepared_ = true;
    prepared_size_ = topology_.size();
    reset();
}

const Ewald& ForcefieldCalculator::ewald() const noexcept { return ewald_; }

const std::vector<DistanceConstraint>& ForcefieldCalculator::constraints() const {
    return topology().constraints();
}

const std::vector<DistanceConstraint>* ForcefieldCalculator::rigid_constraints() const {
    if (topology_.empty() || topology_.constraints().empty()) {
        return nullptr;
    }
    return &topology_.constraints();
}

void ForcefieldCalculator::prepare(const Atoms& atoms) {
    // The topology only has to be rebuilt when the composition changes, which
    // in every realistic case changes the atom count too. A topology set by
    // hand counts as built.
    if (!prepared_ || prepared_size_ != atoms.size()) {
        if (const Value& requested = parameter("model"); requested.is_string()) {
            model_ = &water_model(requested.as_string());
            topology_ = water_topology(atoms, *model_);
        } else if (topology_.empty() || prepared_size_ != atoms.size()) {
            throw ForcefieldError(
                "forcefield: set 'model' to a water model (spc, spce, tip3p) or supply a "
                "topology with set_topology()");
        }
        prepared_ = true;
        prepared_size_ = atoms.size();
    }

    // The splitting is built once and then kept: it is a property of the model,
    // not of the cell, and `Ewald::prepare` only rebuilds the reciprocal
    // vectors when the cell moves. This runs on every call rather than only on
    // the first, because a topology set by hand skips the branch above and
    // would otherwise leave the sum - and the cutoff - unconfigured.
    // Ewald needs a lattice. A cluster - a molecule in vacuum, a solvated ion -
    // has no conditional convergence to worry about, so its electrostatics are
    // just the direct sum, which is the alpha -> 0 limit of the same identity.
    periodic_ = atoms.cell().any_pbc() && atoms.cell().is_invertible();
    if (!periodic_) {
        const double requested = parameter("cutoff").value_or(0.0);
        if (requested > 0.0) {
            cutoff_ = requested;
        } else {
            // Reach across the whole cluster, so nothing is silently dropped.
            double extent = 0.0;
            const Vec3 centre = atoms.center_of_mass();
            for (std::size_t i = 0; i < atoms.size(); ++i) {
                extent = std::max(extent, (atoms.position(i) - centre).norm());
            }
            cutoff_ = std::max(2.0 * extent + 1.0, 1.0);
        }
        return;
    }

    if (!ewald_.prepared()) {
        EwaldParameters ewald_parameters;
        ewald_parameters.accuracy = parameter("accuracy").value_or(1.0e-6);
        ewald_parameters.real_cutoff = parameter("cutoff").value_or(0.0);
        ewald_parameters.alpha = parameter("alpha").value_or(0.0);
        ewald_parameters.kcutoff = parameter("kcutoff").value_or(0.0);
        ewald_ = Ewald(ewald_parameters);
    }
    ewald_.prepare(atoms);

    // One neighbour list serves the dispersion and the screened electrostatics,
    // so they share a cutoff. A water model is parameterised for 9 A or so; a
    // small cell cannot hold that, and the tail correction covers the rest.
    cutoff_ = ewald_.real_cutoff();
}

EwaldEnergy ForcefieldCalculator::direct_coulomb(const Atoms& atoms, const NeighborList& pairs,
                                                 std::vector<double>& forces,
                                                 bool need_forces) const {
    // The bare sum, for a system with no periodic images to worry about.
    const std::vector<double>& charges = topology_.charges();
    const double* FASTATOMS_RESTRICT positions = atoms.positions().data();
    EwaldEnergy energy;

    for (std::size_t p = 0; p < pairs.size(); ++p) {
        const auto i = static_cast<std::size_t>(pairs.i[p]);
        const auto j = static_cast<std::size_t>(pairs.j[p]);
        if (i >= j || topology_.excluded(i, j)) {
            continue;
        }
        const double qq = charges[i] * charges[j];
        if (qq == 0.0) {
            continue;
        }
        const double dx = positions[3 * j] - positions[3 * i];
        const double dy = positions[3 * j + 1] - positions[3 * i + 1];
        const double dz = positions[3 * j + 2] - positions[3 * i + 2];
        const double r_squared = dx * dx + dy * dy + dz * dz;
        if (!(r_squared > 0.0)) {
            continue;
        }
        const double r = std::sqrt(r_squared);
        const double prefactor = units::coulomb * qq;
        energy.real += prefactor / r;
        if (need_forces) {
            const double magnitude = prefactor / (r_squared * r);
            forces[3 * i] -= magnitude * dx;
            forces[3 * i + 1] -= magnitude * dy;
            forces[3 * i + 2] -= magnitude * dz;
            forces[3 * j] += magnitude * dx;
            forces[3 * j + 1] += magnitude * dy;
            forces[3 * j + 2] += magnitude * dz;
        }
    }
    return energy;
}

void ForcefieldCalculator::lennard_jones(const Atoms& atoms, const NeighborList& pairs,
                                         bool need_forces, bool need_stress,
                                         std::vector<double>& forces, double* virial) {
    const std::vector<double>& epsilons = topology_.epsilons();
    const std::vector<double>& sigmas = topology_.sigmas();
    const double* FASTATOMS_RESTRICT positions = atoms.positions().data();
    const Mat3 cell = atoms.cell().matrix();
    const double cutoff_squared = cutoff_ * cutoff_;
    const bool shift_ = parameter("shift").value_or(true);

    double energy = 0.0;
    const std::size_t npairs = pairs.size();
    for (std::size_t p = 0; p < npairs; ++p) {
        const auto i = static_cast<std::size_t>(pairs.i[p]);
        const auto j = static_cast<std::size_t>(pairs.j[p]);
        if (i > j) {
            continue;
        }
        const double epsilon_i = epsilons[i];
        const double epsilon_j = epsilons[j];
        if (epsilon_i <= 0.0 || epsilon_j <= 0.0) {
            continue;  // hydrogens in a three-site water carry no dispersion
        }
        const int s0 = pairs.shifts[3 * p];
        const int s1 = pairs.shifts[3 * p + 1];
        const int s2 = pairs.shifts[3 * p + 2];
        const bool home_cell = (s0 == 0 && s1 == 0 && s2 == 0);
        if (i == j && home_cell) {
            continue;
        }
        if (home_cell && topology_.excluded(i, j)) {
            continue;
        }

        const double sx = s0 * cell.m[0] + s1 * cell.m[3] + s2 * cell.m[6];
        const double sy = s0 * cell.m[1] + s1 * cell.m[4] + s2 * cell.m[7];
        const double sz = s0 * cell.m[2] + s1 * cell.m[5] + s2 * cell.m[8];
        const double dx = positions[3 * j] + sx - positions[3 * i];
        const double dy = positions[3 * j + 1] + sy - positions[3 * i + 1];
        const double dz = positions[3 * j + 2] + sz - positions[3 * i + 2];
        const double r_squared = dx * dx + dy * dy + dz * dz;
        if (r_squared > cutoff_squared || !(r_squared > 0.0)) {
            continue;
        }

        // Lorentz-Berthelot: arithmetic for sigma, geometric for epsilon.
        const double sigma_ij = 0.5 * (sigmas[i] + sigmas[j]);
        const double epsilon_ij = std::sqrt(epsilon_i * epsilon_j);
        const double ratio2 = sigma_ij * sigma_ij / r_squared;
        const double ratio6 = ratio2 * ratio2 * ratio2;
        const double ratio12 = ratio6 * ratio6;
        const double weight = (i == j) ? 0.5 : 1.0;

        // Shifting the potential to zero at the cutoff costs nothing and
        // removes the step a pair takes as it crosses: that step is what turns
        // into a slow energy drift over a long run.
        double offset = 0.0;
        if (shift_) {
            const double cut2 = sigma_ij * sigma_ij / cutoff_squared;
            const double cut6 = cut2 * cut2 * cut2;
            offset = 4.0 * epsilon_ij * (cut6 * cut6 - cut6);
        }
        energy += weight * (4.0 * epsilon_ij * (ratio12 - ratio6) - offset);

        if (need_forces || need_stress) {
            const double magnitude =
                weight * 24.0 * epsilon_ij * (2.0 * ratio12 - ratio6) / r_squared;
            if (need_forces) {
                forces[3 * i] -= magnitude * dx;
                forces[3 * i + 1] -= magnitude * dy;
                forces[3 * i + 2] -= magnitude * dz;
                forces[3 * j] += magnitude * dx;
                forces[3 * j + 1] += magnitude * dy;
                forces[3 * j + 2] += magnitude * dz;
            }
            if (need_stress && virial != nullptr) {
                const double d[3] = {dx, dy, dz};
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        virial[3 * a + b] -= magnitude * d[a] * d[b];
                    }
                }
            }
        }
    }
    breakdown_.lennard_jones = energy;
}

void ForcefieldCalculator::compute(const Atoms& atoms, const std::vector<Property>& properties) {
    if (atoms.empty()) {
        fail("cannot evaluate an empty structure");
    }
    prepare(atoms);
    topology_.validate(atoms);

    // One evaluation gives energy, forces and stress together, so take all
    // three rather than paying for a second pass on the next request.
    (void)properties;
    const bool need_forces = true;
    const bool periodic = atoms.cell().any_pbc() && atoms.cell().is_invertible();
    const bool need_stress = periodic;

    const std::size_t n = atoms.size();
    std::vector<double> forces(3 * n, 0.0);
    double virial[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    // One neighbour list serves the dispersion and the screened electrostatics.
    NeighborListOptions options;
    options.cutoff = cutoff_;
    options.full = true;
    options.self_interaction = false;
    options.store_vectors = false;
    options.store_distances = false;
    const NeighborList pairs = build_neighbor_list(atoms, options);

    breakdown_ = EnergyBreakdown{};
    lennard_jones(atoms, pairs, need_forces, need_stress, forces, virial);

    std::array<double, 6> coulomb_stress{{0, 0, 0, 0, 0, 0}};
    if (periodic) {
        const EwaldResult coulomb =
            ewald_.compute(atoms, topology_, pairs, need_forces, need_stress);
        breakdown_.coulomb = coulomb.energy;
        coulomb_stress = coulomb.stress;
        for (std::size_t k = 0; k < 3 * n; ++k) {
            forces[k] += coulomb.forces[k];
        }
    } else {
        breakdown_.coulomb = direct_coulomb(atoms, pairs, forces, need_forces);
    }

    // The isotropic tail correction: what a uniform fluid beyond the cutoff
    // would contribute, assuming g(r) = 1 there. It shifts the energy and the
    // pressure but not the forces.
    const double volume = periodic ? std::abs(atoms.cell().matrix().determinant()) : 0.0;
    if (parameter("tail").value_or(true) && volume > 0.0) {
        const std::vector<double>& epsilons = topology_.epsilons();
        const std::vector<double>& sigmas = topology_.sigmas();
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            if (!(epsilons[i] > 0.0)) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                if (!(epsilons[j] > 0.0)) {
                    continue;
                }
                const double sigma_ij = 0.5 * (sigmas[i] + sigmas[j]);
                const double epsilon_ij = std::sqrt(epsilons[i] * epsilons[j]);
                const double s3 = sigma_ij * sigma_ij * sigma_ij;
                const double s9 = s3 * s3 * s3;
                const double rc3 = cutoff_ * cutoff_ * cutoff_;
                const double rc9 = rc3 * rc3 * rc3;
                sum += epsilon_ij * (s9 / (9.0 * rc9) - s3 / (3.0 * rc3)) * s3;
            }
        }
        breakdown_.lennard_jones_tail = 8.0 * kPi / volume * sum;
        if (need_stress) {
            // The correction goes as 1/V at fixed cutoff, so its derivative
            // with respect to strain is -E_tail on the diagonal. (The familiar
            // "tail pressure" expression is the virial-route answer to a
            // different question, and using it here would make the stress stop
            // being the derivative of the energy this calculator reports.)
            for (int a = 0; a < 3; ++a) {
                virial[3 * a + a] -= breakdown_.lennard_jones_tail;
            }
        }
    }

    // Per-molecule constants, such as SPC/E's polarisation self-energy.
    if (model_ != nullptr && parameter("self_energy").value_or(true)) {
        breakdown_.self =
            static_cast<double>(topology_.molecules().size()) * model_->self_energy;
    }

    results_.clear();
    results_.energy = breakdown_.total();
    results_.free_energy = results_.energy;
    results_.forces = std::move(forces);
    results_.charges = topology_.charges();

    if (need_stress) {
        const double inverse_volume = 1.0 / volume;
        std::array<double, 6> stress{inverse_volume * virial[0], inverse_volume * virial[4],
                                     inverse_volume * virial[8],
                                     inverse_volume * 0.5 * (virial[5] + virial[7]),
                                     inverse_volume * 0.5 * (virial[2] + virial[6]),
                                     inverse_volume * 0.5 * (virial[1] + virial[3])};
        for (std::size_t k = 0; k < 6; ++k) {
            stress[k] += coulomb_stress[k];
        }
        results_.stress = stress;
    }

    Value extra(Value::Map{});
    extra.insert("lennard_jones", Value(breakdown_.lennard_jones));
    extra.insert("lennard_jones_tail", Value(breakdown_.lennard_jones_tail));
    extra.insert("coulomb", Value(breakdown_.coulomb.total()));
    extra.insert("coulomb_real", Value(breakdown_.coulomb.real));
    extra.insert("coulomb_reciprocal", Value(breakdown_.coulomb.reciprocal));
    extra.insert("coulomb_self", Value(breakdown_.coulomb.self));
    extra.insert("coulomb_excluded", Value(breakdown_.coulomb.excluded));
    extra.insert("self_energy", Value(breakdown_.self));
    extra.insert("alpha", Value(periodic ? ewald_.alpha() : 0.0));
    extra.insert("periodic", Value(periodic));
    extra.insert("cutoff", Value(cutoff_));
    extra.insert("kvectors",
                 Value(static_cast<std::int64_t>(periodic ? ewald_.kvector_count() : 0)));
    extra.insert("molecules", Value(static_cast<std::int64_t>(topology_.molecules().size())));
    results_.extra = std::move(extra);
}

std::string ForcefieldCalculator::to_string() const {
    std::ostringstream out;
    out << "forcefield(";
    if (model_ != nullptr) {
        out << "'" << model_->name << "'";
    } else if (!topology_.empty()) {
        out << topology_.molecules().size() << " molecules";
    } else {
        out << "not prepared";
    }
    if (cutoff_ > 0.0) {
        out << ", cutoff=" << cutoff_ << " A";
    }
    if (ewald_.prepared()) {
        out << ", alpha=" << ewald_.alpha() << ", " << ewald_.kvector_count() << " k-vectors";
    } else if (prepared_) {
        out << ", direct sum (not periodic)";
    }
    out << ")";
    return out.str();
}

}  // namespace fastatoms::ff
