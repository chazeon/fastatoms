// SPDX-License-Identifier: MIT
#include "fastatoms/calculators/lennard_jones.hpp"

#include <cmath>
#include <sstream>

#include "fastatoms/core/neighbor_list.hpp"

namespace fastatoms {

LennardJonesCalculator::LennardJonesCalculator(Value parameters)
    : Calculator(std::move(parameters)) {}

std::vector<Property> LennardJonesCalculator::implemented_properties() const {
    return {Property::Energy, Property::FreeEnergy, Property::Forces, Property::Stress};
}

std::string LennardJonesCalculator::to_string() const {
    std::ostringstream out;
    out << "lennard_jones(epsilon=" << parameter("epsilon").value_or(0.0103)
        << ", sigma=" << parameter("sigma").value_or(3.40) << ")";
    return out.str();
}

void LennardJonesCalculator::compute(const Atoms& atoms,
                                     const std::vector<Property>& properties) {
    (void)properties;  // energy, forces and stress all come from the same loop

    const double epsilon = parameter("epsilon").value_or(0.0103);
    const double sigma = parameter("sigma").value_or(3.40);
    const double cutoff = parameter("cutoff").value_or(3.0 * sigma);
    const bool shift = parameter("shift").value_or(true);

    if (!(epsilon > 0.0) || !(sigma > 0.0) || !(cutoff > 0.0)) {
        fail("epsilon, sigma and cutoff must all be positive");
    }

    double energy_shift = 0.0;
    if (shift) {
        const double ratio6 = std::pow(sigma / cutoff, 6);
        energy_shift = 4.0 * epsilon * (ratio6 * ratio6 - ratio6);
    }

    // A half list visits each pair once; Newton's third law gives the rest.
    NeighborListOptions options;
    options.cutoff = cutoff;
    options.full = false;
    options.store_vectors = true;
    options.store_distances = false;
    const NeighborList pairs = build_neighbor_list(atoms, options);

    const std::size_t natoms = atoms.size();
    double energy = 0.0;
    std::vector<double> forces(3 * natoms, 0.0);
    Mat3 virial{};

    const double sigma2 = sigma * sigma;
    for (std::size_t k = 0; k < pairs.size(); ++k) {
        const Vec3 dr = Vec3::load(pairs.vectors.data() + 3 * k);
        const double r2 = dr.norm2();
        if (r2 <= 0.0) {
            continue;
        }
        const double inv6 = std::pow(sigma2 / r2, 3);
        const double inv12 = inv6 * inv6;
        energy += 4.0 * epsilon * (inv12 - inv6) - energy_shift;

        // F_i = -(24 eps / r^2) (2 (s/r)^12 - (s/r)^6) dr, and F_j = -F_i.
        const double coefficient = 24.0 * epsilon * (2.0 * inv12 - inv6) / r2;
        const Vec3 force_on_i = dr * (-coefficient);

        const auto i = static_cast<std::size_t>(pairs.i[k]);
        const auto j = static_cast<std::size_t>(pairs.j[k]);
        for (std::size_t a = 0; a < 3; ++a) {
            forces[3 * i + a] += force_on_i[a];
            forces[3 * j + a] -= force_on_i[a];
            // sigma_ab = (1/V) sum_pairs F_i,a dr_b
            for (std::size_t b = 0; b < 3; ++b) {
                virial(a, b) += force_on_i[a] * dr[b];
            }
        }
    }

    results_.energy = energy;
    results_.free_energy = energy;
    results_.forces = std::move(forces);

    // Stress only means something for a cell with a volume.
    if (atoms.cell().any_pbc() && atoms.cell().is_invertible()) {
        const double inverse_volume = 1.0 / atoms.volume();
        for (double& value : virial.m) {
            value *= inverse_volume;
        }
        results_.stress = matrix_to_voigt(virial);
    }
}

}  // namespace fastatoms
