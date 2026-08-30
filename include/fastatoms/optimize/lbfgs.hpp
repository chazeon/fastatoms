// SPDX-License-Identifier: MIT
#pragma once

/// \file lbfgs.hpp

#include <cstddef>
#include <deque>
#include <vector>

#include "fastatoms/optimize/optimizer.hpp"

namespace fastatoms::optimize {

/// Limited-memory BFGS.
///
/// Identical in spirit to `BFGS`, but instead of an n x n matrix it keeps only
/// the last `memory` pairs of step and gradient change and reconstructs the
/// action of the inverse Hessian on demand with the two-loop recursion:
///
///     q = g
///     for i = k-1 .. k-m:   alpha_i = rho_i s_i.q ;  q -= alpha_i y_i
///     r = gamma q
///     for i = k-m .. k-1:   beta = rho_i y_i.r ;     r += (alpha_i - beta) s_i
///     d = -r
///
/// with `rho_i = 1/(y_i.s_i)` and `gamma = s.y / y.y` from the most recent
/// pair. Memory and time per step are O(m n) rather than O(n^2), which is what
/// makes it the right choice for large structures; with `memory` around ten it
/// converges almost as fast as full BFGS.
///
/// As in `BFGS`, a pair with non-positive curvature is discarded rather than
/// stored, and for the same reason the default line search is a strong Wolfe
/// search.
class LBFGS : public Optimizer {
public:
    explicit LBFGS(Value parameters = {}) : Optimizer(std::move(parameters)) {}

    [[nodiscard]] std::string name() const override { return "lbfgs"; }

    /// Number of correction pairs kept. Default 10.
    [[nodiscard]] std::size_t memory() const;

    /// Pairs currently stored.
    [[nodiscard]] std::size_t history_size() const noexcept { return history_.size(); }

    [[nodiscard]] int skipped_updates() const noexcept { return skipped_; }

protected:
    void begin(std::size_t ndof) override;
    void restart() override;
    void accept(const std::vector<double>& dx, const std::vector<double>& dg) override;

    [[nodiscard]] std::vector<double> direction(const std::vector<double>& x,
                                                const std::vector<double>& gradient) override;

private:
    struct Correction {
        std::vector<double> s;  ///< the step that was taken
        std::vector<double> y;  ///< the gradient change it produced
        double rho{0.0};        ///< 1 / (y.s)
    };

    std::deque<Correction> history_{};
    int skipped_{0};
};

}  // namespace fastatoms::optimize
