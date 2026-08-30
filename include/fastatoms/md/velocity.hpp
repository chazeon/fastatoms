// SPDX-License-Identifier: MIT
#pragma once

/// \file velocity.hpp
/// \brief Setting up initial velocities.

#include <cstdint>

#include "fastatoms/core/atoms.hpp"

namespace fastatoms::md {

/// Draw velocities from a Maxwell-Boltzmann distribution at `temperature`.
///
/// Each Cartesian component is normal with variance `kB T / m`. With
/// `remove_center_of_mass` the net momentum is removed afterwards, which is
/// almost always what you want: a drifting centre of mass contributes to the
/// kinetic energy without being a thermal degree of freedom.
///
/// With `exact` the velocities are then rescaled so the instantaneous
/// temperature is exactly `temperature` rather than a sample from its
/// distribution. That is the usual way to start a run reproducibly; set it to
/// false if you want a genuine draw.
///
/// \throws DynamicsError for a negative temperature.
void set_maxwell_boltzmann_velocities(Atoms& atoms, double temperature,
                                      std::int64_t seed = 0,
                                      bool remove_center_of_mass = true, bool exact = true);

/// Rescale velocities so the instantaneous temperature is exactly
/// `temperature`. Does nothing if the structure is not moving.
void rescale_to_temperature(Atoms& atoms, double temperature, int degrees_of_freedom);

/// Momentum degrees of freedom: `3N`, less three when the centre-of-mass
/// motion is being removed.
[[nodiscard]] int momentum_degrees_of_freedom(const Atoms& atoms,
                                              bool remove_center_of_mass = true);

}  // namespace fastatoms::md
