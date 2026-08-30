// SPDX-License-Identifier: MIT
#pragma once

/// \file steepest_descent.hpp

#include "fastatoms/optimize/optimizer.hpp"

namespace fastatoms::optimize {

/// Steepest descent: step straight downhill.
///
///     d_k = -g_k / alpha
///
/// The simplest thing that works, and the slowest: on an ill-conditioned
/// surface it zig-zags across the valley instead of running along it, so the
/// step count grows with the condition number. It is here as the baseline the
/// other three improve on, and because it is completely robust - it cannot
/// propose an uphill direction.
///
/// Defaults to a strong Wolfe line search. Backtracking alone is not enough
/// here: it can only shrink the trial step, so if the fixed `alpha` guess
/// overestimates the curvature, steepest descent crawls. The Wolfe search
/// grows the step as well, and the accepted length is carried into the next
/// iteration.
class SteepestDescent : public Optimizer {
public:
    explicit SteepestDescent(Value parameters = {}) : Optimizer(std::move(parameters)) {}

    [[nodiscard]] std::string name() const override { return "sd"; }

protected:
    void begin(std::size_t) override {}
    void restart() override {}
    void accept(const std::vector<double>&, const std::vector<double>&) override {}

    [[nodiscard]] std::vector<double> direction(const std::vector<double>&,
                                                const std::vector<double>& gradient) override {
        return steepest_direction(gradient);
    }

    [[nodiscard]] bool has_natural_unit_step() const override { return false; }
};

}  // namespace fastatoms::optimize
