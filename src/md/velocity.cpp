// SPDX-License-Identifier: MIT
#include "fastatoms/md/velocity.hpp"

#include <cmath>
#include <random>
#include <vector>

#include "fastatoms/core/units.hpp"
#include "fastatoms/md/types.hpp"

namespace fastatoms::md {

int momentum_degrees_of_freedom(const Atoms& atoms, bool remove_center_of_mass) {
    const int total = 3 * static_cast<int>(atoms.size());
    if (!remove_center_of_mass) {
        return total;
    }
    // Removing the net momentum takes three degrees of freedom away; with only
    // one atom nothing is left to thermalise.
    return total > 3 ? total - 3 : total;
}

void rescale_to_temperature(Atoms& atoms, double temperature, int degrees_of_freedom) {
    if (degrees_of_freedom <= 0) {
        return;
    }
    const double current = atoms.temperature(degrees_of_freedom);
    if (!(current > 0.0) || !(temperature > 0.0)) {
        return;
    }
    const double factor = std::sqrt(temperature / current);
    std::vector<double> momenta = atoms.momenta();
    for (double& value : momenta) {
        value *= factor;
    }
    atoms.set_momenta(momenta);
}

void set_maxwell_boltzmann_velocities(Atoms& atoms, double temperature, std::int64_t seed,
                                      bool remove_center_of_mass, bool exact) {
    if (temperature < 0.0) {
        throw DynamicsError("the temperature must not be negative");
    }
    if (atoms.empty()) {
        return;
    }

    if (temperature == 0.0) {
        atoms.set_momenta(std::vector<double>(3 * atoms.size(), 0.0));
        return;
    }

    // A zero seed means "pick one", so repeated runs differ; any other value
    // makes the run reproducible.
    std::mt19937_64 rng(seed != 0 ? static_cast<std::uint64_t>(seed)
                                  : std::random_device{}());
    std::normal_distribution<double> normal(0.0, 1.0);

    const std::vector<double> masses = atoms.masses();
    std::vector<double> velocities(3 * atoms.size());
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        if (!(masses[i] > 0.0)) {
            throw DynamicsError("atom " + std::to_string(i) + " has a non-positive mass");
        }
        // Each component is normal with variance kB T / m.
        const double sigma = std::sqrt(units::kB * temperature / masses[i]);
        for (std::size_t k = 0; k < 3; ++k) {
            velocities[3 * i + k] = sigma * normal(rng);
        }
    }
    atoms.set_velocities(velocities);

    // Setting the velocities already applied the constraints, so most of what
    // was drawn above may be gone. Removing the net momentum of what is left
    // only makes sense while translation is still a symmetry - with a frozen
    // substrate it is not, and subtracting the mobile atoms' momentum would
    // simply cool them.
    const int translations = free_translations(atoms.constraints(), atoms);
    if (remove_center_of_mass && translations == 3) {
        atoms.zero_center_of_mass_momentum();
    }
    if (exact) {
        // A finite sample lands near the target, not on it. Rescaling makes
        // the starting temperature exactly what was asked for - counted over
        // the degrees of freedom that are actually left, or a slab with one
        // adsorbate starts an order of magnitude too hot.
        int dof = momentum_degrees_of_freedom(atoms, false) -
                  atoms.constrained_degrees_of_freedom();
        if (remove_center_of_mass) {
            dof -= translations;
        }
        rescale_to_temperature(atoms, temperature, dof);
    }
}

}  // namespace fastatoms::md
