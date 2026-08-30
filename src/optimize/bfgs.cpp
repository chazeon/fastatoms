// SPDX-License-Identifier: MIT
#include "fastatoms/optimize/bfgs.hpp"

#include <cmath>

namespace fastatoms::optimize {

void BFGS::begin(std::size_t ndof) {
    ndof_ = ndof;
    restart();
}

void BFGS::restart() {
    // Start from a diagonal inverse Hessian: 1/alpha angstrom^2/eV, so the
    // very first step is the same one steepest descent would take.
    hessian_.reset(ndof_, 1.0 / alpha());
    first_update_ = true;
}

std::vector<double> BFGS::direction(const std::vector<double>&,
                                    const std::vector<double>& gradient) {
    if (hessian_.size() != gradient.size()) {
        hessian_.reset(gradient.size(), 1.0 / alpha());
    }
    return scaled(-1.0, hessian_.apply(gradient));
}

void BFGS::accept(const std::vector<double>& dx, const std::vector<double>& dg) {
    const double sy = dot(dx, dg);
    const double ss = dot(dx, dx);
    const double yy = dot(dg, dg);

    // Positive curvature along the step is what keeps H positive definite. A
    // step that fails it carries no usable information, so skip it rather than
    // corrupt the model.
    if (!(sy > 1e-12 * std::sqrt(ss * yy)) || !std::isfinite(sy)) {
        ++skipped_;
        return;
    }

    if (first_update_ && parameter("initial_scaling").value_or(true)) {
        // Rescale the initial guess to the curvature actually observed. Without
        // this the first few steps of a badly scaled problem are wasted
        // undoing the arbitrary 1/alpha.
        hessian_.reset(dx.size(), sy / yy);
    }
    first_update_ = false;

    const std::vector<double> hy = hessian_.apply(dg);
    const double yhy = dot(dg, hy);
    const double rho = 1.0 / sy;

    // H <- H - rho (s (Hy)^T + (Hy) s^T) + (rho^2 y.Hy + rho) s s^T
    hessian_.add_symmetric_updates(-rho, dx, hy, rho * rho * yhy + rho);
}

}  // namespace fastatoms::optimize
