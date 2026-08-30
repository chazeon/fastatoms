// SPDX-License-Identifier: MIT
#pragma once

/// \file conjugate_gradient.hpp

#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/optimize/optimizer.hpp"

namespace fastatoms::optimize {

/// Which formula sets the mixing coefficient.
enum class ConjugateGradientVariant {
    PolakRibiere,    ///< with the max(0, beta) safeguard; the practical default
    FletcherReeves,
    HestenesStiefel,
};

[[nodiscard]] ConjugateGradientVariant cg_variant_from_string(std::string_view name);
[[nodiscard]] std::string_view cg_variant_name(ConjugateGradientVariant variant);

/// Nonlinear conjugate gradient.
///
///     d_k = -g_k / alpha + beta_k d_{k-1}
///
/// Each direction is mixed with the previous one so that successive steps stay
/// conjugate, which removes the zig-zag that makes steepest descent slow, at
/// the cost of storing one extra vector - nothing like the n^2 of BFGS.
///
/// Three choices of `beta` are available (`variant` parameter):
///
///   - `polak_ribiere` (default), `beta = g_k.(g_k - g_{k-1}) / |g_{k-1}|^2`,
///     clamped at zero. The clamp is what makes it self-restarting: when the
///     surface stops looking quadratic, beta falls to zero and the method
///     takes a plain downhill step.
///   - `fletcher_reeves`, `beta = |g_k|^2 / |g_{k-1}|^2`.
///   - `hestenes_stiefel`, `beta = g_k.y / d_{k-1}.y` with `y = g_k - g_{k-1}`.
///
/// The direction is reset to steepest descent every `restart` steps (default:
/// the number of degrees of freedom), and whenever the mixed direction would
/// point uphill.
///
/// Conjugacy only holds if the step length satisfies the strong Wolfe
/// conditions, so that is the default line search here.
class ConjugateGradient : public Optimizer {
public:
    explicit ConjugateGradient(Value parameters = {}) : Optimizer(std::move(parameters)) {}

    [[nodiscard]] std::string name() const override { return "cg"; }

    [[nodiscard]] ConjugateGradientVariant variant() const;
    /// Steps between forced restarts; zero means "the number of variables".
    [[nodiscard]] int restart_period() const;

protected:
    void begin(std::size_t ndof) override;
    void restart() override;
    void accept(const std::vector<double>& dx, const std::vector<double>& dg) override;

    [[nodiscard]] std::vector<double> direction(const std::vector<double>& x,
                                                const std::vector<double>& gradient) override;

    /// Conjugacy needs a tighter curvature condition than the quasi-Newton
    /// methods do.
    [[nodiscard]] double default_c2() const override { return 0.1; }
    [[nodiscard]] bool has_natural_unit_step() const override { return false; }

private:
    std::vector<double> previous_gradient_{};
    std::vector<double> previous_direction_{};
    bool has_history_{false};
    int since_restart_{0};
    std::size_t ndof_{0};
};

}  // namespace fastatoms::optimize
