// SPDX-License-Identifier: MIT
#pragma once

/// \file registry.hpp
/// \brief Building a calculator by name, so an input file can choose one.

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/calculators/calculator.hpp"
#include "fastatoms/io/parameters.hpp"

namespace fastatoms {

/// Builds a calculator from a parameter mapping.
using CalculatorFactory = std::function<std::shared_ptr<Calculator>(const Value&)>;

/// Make `name` available to `make_calculator`. Registering an existing name
/// replaces it, which is how a plugin overrides a built-in.
void register_calculator(const std::string& name, CalculatorFactory factory);

/// Build a calculator by name.
///
/// Recognised out of the box: `vasp`, `espresso` (aliases `quantum_espresso`,
/// `qe`, `pw`), `siesta`, `lennard_jones` (alias `lj`), and the machine-learning
/// potentials `mace`, `mattersim`, `chgnet`, `nequip`, `allegro` and the
/// generic `mlip`. Matching ignores case, hyphens and underscores.
///
/// \throws CalculatorError if the name is not registered.
[[nodiscard]] std::shared_ptr<Calculator> make_calculator(std::string_view name,
                                                          const Value& parameters = Value{});

/// Build the calculator described by the `simulation.calculator` section of an
/// input file.
[[nodiscard]] std::shared_ptr<Calculator> make_calculator(const CalculatorParameters& spec);

/// Remove a registered name. Does nothing if it was never registered.
///
/// Language bindings need this: a factory written in Python holds a reference
/// to a Python object, and that reference has to be dropped while the
/// interpreter is still alive rather than when this translation unit's statics
/// are destroyed at process exit.
void unregister_calculator(const std::string& name);

/// Every registered name, in alphabetical order.
[[nodiscard]] std::vector<std::string> registered_calculators();

}  // namespace fastatoms
