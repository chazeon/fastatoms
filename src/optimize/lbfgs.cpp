// SPDX-License-Identifier: MIT
#include "fastatoms/optimize/lbfgs.hpp"

#include <cmath>

#include "fastatoms/optimize/vector_math.hpp"

namespace fastatoms::optimize {

std::size_t LBFGS::memory() const {
    const auto value = parameter("memory").value_or(std::int64_t{10});
    if (value < 1) {
        fail("'memory' must be at least 1");
    }
    return static_cast<std::size_t>(value);
}

void LBFGS::begin(std::size_t) { restart(); }

void LBFGS::restart() { history_.clear(); }

void LBFGS::accept(const std::vector<double>& dx, const std::vector<double>& dg) {
    const double sy = dot(dx, dg);
    const double ss = dot(dx, dx);
    const double yy = dot(dg, dg);

    if (!(sy > 1e-12 * std::sqrt(ss * yy)) || !std::isfinite(sy)) {
        ++skipped_;
        return;
    }

    history_.push_back(Correction{dx, dg, 1.0 / sy});
    while (history_.size() > memory()) {
        history_.pop_front();
    }
}

std::vector<double> LBFGS::direction(const std::vector<double>&,
                                     const std::vector<double>& gradient) {
    if (history_.empty()) {
        return steepest_direction(gradient);
    }

    // First loop: walk backwards through the corrections, peeling each one off
    // the gradient and remembering how much was removed.
    std::vector<double> q = gradient;
    std::vector<double> alphas(history_.size(), 0.0);

    for (std::size_t k = history_.size(); k > 0; --k) {
        const Correction& correction = history_[k - 1];
        const double a = correction.rho * dot(correction.s, q);
        alphas[k - 1] = a;
        axpy(-a, correction.y, q);
    }

    // Scale by the curvature of the most recent pair: gamma = s.y / y.y is the
    // best available estimate of the inverse Hessian's magnitude. Falling back
    // to 1/alpha keeps the physical guess when scaling is switched off.
    const Correction& newest = history_.back();
    double gamma = 1.0 / alpha();
    if (parameter("initial_scaling").value_or(true)) {
        const double yy = dot(newest.y, newest.y);
        if (yy > 0.0) {
            gamma = dot(newest.s, newest.y) / yy;
        }
    }
    scale(gamma, q);

    // Second loop: put the corrections back, forwards this time.
    for (std::size_t k = 0; k < history_.size(); ++k) {
        const Correction& correction = history_[k];
        const double beta = correction.rho * dot(correction.y, q);
        axpy(alphas[k] - beta, correction.s, q);
    }

    return scaled(-1.0, q);
}

}  // namespace fastatoms::optimize
