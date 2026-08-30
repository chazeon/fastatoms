// SPDX-License-Identifier: MIT
#pragma once

/// \file registry.hpp
/// \brief Building a phonon calculation from an input file.

#include <memory>

#include "fastatoms/io/parameters.hpp"
#include "fastatoms/phonon/phonons.hpp"

namespace fastatoms::phonon {

/// Build a `Phonons` from a parameter mapping.
[[nodiscard]] std::shared_ptr<Phonons> make_phonons(const Atoms& unit_cell,
                                                    const Value& parameters = Value{});

/// Build one from the `simulation.phonons` section of an input file.
[[nodiscard]] std::shared_ptr<Phonons> make_phonons(const Atoms& unit_cell,
                                                    const PhononParameters& spec);

/// Build one from a whole `simulation:` section.
[[nodiscard]] std::shared_ptr<Phonons> make_phonons(const Atoms& unit_cell,
                                                    const SimulationParameters& simulation);

}  // namespace fastatoms::phonon
