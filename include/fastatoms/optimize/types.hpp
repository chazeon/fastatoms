// SPDX-License-Identifier: MIT
#pragma once

/// \file types.hpp
/// \brief The vocabulary of the optimization module.

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fastatoms::optimize {

/// Raised when an optimization cannot be set up or cannot make progress.
class OptimizerError : public std::runtime_error {
public:
    explicit OptimizerError(const std::string& message) : std::runtime_error(message) {}
};

/// The value and gradient of the objective at one point.
struct Evaluation {
    double energy{0.0};
    std::vector<double> gradient{};  ///< dE/dx, so the "force" is its negative
};

/// Anything the optimizers can minimise.
///
/// The optimizers only ever see this: a function from a coordinate vector to
/// an energy and a gradient. Relaxing a structure supplies one that drives the
/// attached calculator, but a plain analytic function works just as well,
/// which is how the algorithms are tested independently of any physics.
using ObjectiveFunction = std::function<Evaluation(const std::vector<double>&)>;

/// One line of the optimization log.
struct OptimizationStep {
    int step{0};                    ///< 0 is the starting point
    int evaluations{0};             ///< objective evaluations so far
    double energy{0.0};             ///< eV
    double fmax{0.0};               ///< largest gradient row norm, eV/angstrom
    double atomic_fmax{0.0};        ///< the same over the atomic rows only
    double energy_change{0.0};      ///< since the previous step
    double max_displacement{0.0};   ///< largest atom movement, angstrom
    double step_length{0.0};        ///< line-search scale factor that was accepted
};

/// Why an optimization stopped.
enum class StopReason {
    Converged,        ///< the force threshold was met
    MaxSteps,         ///< the step budget ran out
    LineSearchFailed, ///< no acceptable step could be found along the direction
    NoProgress,       ///< the proposed displacement shrank below rounding noise
    Interrupted,      ///< a callback asked to stop
};

[[nodiscard]] std::string_view stop_reason_name(StopReason reason);

/// The outcome of a relaxation.
struct OptimizationResult {
    bool converged{false};
    StopReason reason{StopReason::MaxSteps};
    int steps{0};                   ///< accepted steps taken
    int evaluations{0};             ///< objective evaluations, the DFT-cost metric
    double energy{0.0};             ///< at the final point
    double fmax{0.0};               ///< at the final point
    std::string message{};          ///< human-readable summary
    std::vector<OptimizationStep> history{};

    [[nodiscard]] std::string to_string() const;
};

/// Called after every accepted step. Returning false stops the run.
using StepCallback = std::function<bool(const OptimizationStep&)>;

}  // namespace fastatoms::optimize
