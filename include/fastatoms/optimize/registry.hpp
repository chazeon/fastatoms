// SPDX-License-Identifier: MIT
#pragma once

/// \file registry.hpp
/// \brief Building an optimizer by name, so an input file can choose one.

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/io/parameters.hpp"
#include "fastatoms/optimize/optimizer.hpp"

namespace fastatoms::optimize {

using OptimizerFactory = std::function<std::shared_ptr<Optimizer>(const Value&)>;

/// Make `name` available to `make_optimizer`, replacing any existing entry.
void register_optimizer(const std::string& name, OptimizerFactory factory);

/// Build an optimizer by name.
///
/// Recognised out of the box: `sd` (aliases `steepest_descent`, `gd`),
/// `cg` (`conjugate_gradient`), `bfgs`, and `lbfgs` (`l-bfgs`). Matching
/// ignores case, hyphens and underscores.
///
/// \throws OptimizerError if the name is not registered.
[[nodiscard]] std::shared_ptr<Optimizer> make_optimizer(std::string_view name,
                                                        const Value& parameters = Value{});

/// Build the optimizer described by the `simulation.optimizer` section of an
/// input file.
[[nodiscard]] std::shared_ptr<Optimizer> make_optimizer(const OptimizerParameters& spec);

/// Remove a registered name. Language bindings need this so that a factory
/// written in Python can be dropped while the interpreter is still alive.
void unregister_optimizer(const std::string& name);

/// Every registered name, in alphabetical order.
[[nodiscard]] std::vector<std::string> registered_optimizers();

}  // namespace fastatoms::optimize
