// SPDX-License-Identifier: MIT
#pragma once

/// \file registry.hpp
/// \brief Building a run from an input file.

#include <memory>

#include "fastatoms/io/parameters.hpp"
#include "fastatoms/md/dynamics.hpp"

namespace fastatoms::md {

/// Build a `Dynamics` from a parameter mapping.
[[nodiscard]] std::shared_ptr<Dynamics> make_dynamics(const Value& parameters = Value{});

/// Build a run from the `simulation.md` section of an input file.
[[nodiscard]] std::shared_ptr<Dynamics> make_dynamics(const MolecularDynamicsParameters& spec);

/// Build a run from a whole `simulation:` section, so that the `output:`
/// settings - trajectory file, log file, interval - are picked up too.
[[nodiscard]] std::shared_ptr<Dynamics> make_dynamics(const SimulationParameters& simulation);

}  // namespace fastatoms::md
