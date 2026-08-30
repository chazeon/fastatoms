// SPDX-License-Identifier: MIT
#pragma once

/// \file bfgs.hpp

#include <cstddef>
#include <vector>

#include "fastatoms/optimize/optimizer.hpp"
#include "fastatoms/optimize/vector_math.hpp"

namespace fastatoms::optimize {

/// Broyden-Fletcher-Goldfarb-Shanno, storing the inverse Hessian directly.
///
/// The search direction is `d_k = -H_k g_k`, where `H_k` approximates the
/// inverse Hessian and is updated from the step `s` and the gradient change
/// `y` it produced:
///
///     H_{k+1} = (I - rho s y^T) H_k (I - rho y s^T) + rho s s^T,  rho = 1/(y.s)
///
/// Keeping the *inverse* rather than the Hessian itself is what makes this
/// self-contained: the direction is a matrix-vector product, so no linear
/// solve and no factorisation is ever needed. Expanded for implementation, the
/// update is
///
///     H <- H - rho (s (Hy)^T + (Hy) s^T) + (rho^2 y.Hy + rho) s s^T
///
/// which is one matrix-vector product plus two rank-one updates.
///
/// The update is skipped when `y.s <= 0`, since accepting it would destroy the
/// positive definiteness that guarantees `-H g` points downhill. That happens
/// wherever the surface is genuinely concave - a badly stretched crystal, say -
/// and it is why the default line search is a strong Wolfe search rather than
/// none: with every update discarded the method falls back to a fixed
/// `-g/alpha` step, and only a search that can grow the step recovers from it.
///
/// Cost is O(n^2) memory and time per step, so a few thousand degrees of
/// freedom is the practical ceiling. Above that, use L-BFGS.
class BFGS : public Optimizer {
public:
    explicit BFGS(Value parameters = {}) : Optimizer(std::move(parameters)) {}

    [[nodiscard]] std::string name() const override { return "bfgs"; }

    /// The current inverse Hessian approximation, in angstrom^2/eV.
    [[nodiscard]] const SymmetricMatrix& inverse_hessian() const noexcept { return hessian_; }

    /// Updates skipped so far because the curvature condition failed.
    [[nodiscard]] int skipped_updates() const noexcept { return skipped_; }

protected:
    void begin(std::size_t ndof) override;
    void restart() override;
    void accept(const std::vector<double>& dx, const std::vector<double>& dg) override;

    [[nodiscard]] std::vector<double> direction(const std::vector<double>& x,
                                                const std::vector<double>& gradient) override;

private:
    SymmetricMatrix hessian_{};
    std::size_t ndof_{0};
    bool first_update_{true};
    int skipped_{0};
};

}  // namespace fastatoms::optimize
