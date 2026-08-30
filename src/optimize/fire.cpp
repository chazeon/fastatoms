// SPDX-License-Identifier: MIT
#include "fastatoms/optimize/fire.hpp"

#include <cmath>

#include "fastatoms/optimize/vector_math.hpp"

namespace fastatoms::optimize {

double Fire::initial_timestep() const {
    const double value = parameter("dt").value_or(parameter("timestep").value_or(0.1));
    if (!(value > 0.0)) {
        fail("'dt' must be positive");
    }
    return value;
}

double Fire::maximum_timestep() const {
    const double value = parameter("dt_max").value_or(1.0);
    if (!(value > 0.0)) {
        fail("'dt_max' must be positive");
    }
    return value;
}

double Fire::initial_mixing() const {
    // Deliberately not the base class's `alpha`: there it is a curvature guess
    // in eV/A^2, here it is a dimensionless mixing fraction, and silently
    // sharing the name would make one of them wrong.
    const double value = parameter("mixing").value_or(parameter("f_alpha_start")
                                                          .value_or(0.1));
    if (!(value > 0.0) || value > 1.0) {
        fail("'mixing' must be in (0, 1]");
    }
    return value;
}

void Fire::begin(std::size_t ndof) {
    velocity_.assign(ndof, 0.0);
    dt_ = initial_timestep();
    mixing_ = initial_mixing();
    downhill_ = 0;
}

void Fire::restart() {
    std::fill(velocity_.begin(), velocity_.end(), 0.0);
    dt_ = initial_timestep();
    mixing_ = initial_mixing();
    downhill_ = 0;
}

void Fire::accept(const std::vector<double>&, const std::vector<double>&) {
    // The velocity is advanced in direction(); there is no curvature history.
}

std::vector<double> Fire::direction(const std::vector<double>&,
                                    const std::vector<double>& gradient) {
    const std::size_t n = gradient.size();
    if (velocity_.size() != n) {
        velocity_.assign(n, 0.0);
    }

    const double dt_max = maximum_timestep();
    const int n_min = static_cast<int>(parameter("n_min").value_or(std::int64_t{5}));
    const double f_inc = parameter("f_inc").value_or(1.1);
    const double f_dec = parameter("f_dec").value_or(0.5);
    const double f_alpha = parameter("f_alpha").value_or(0.99);

    // The force is the negative gradient.
    std::vector<double> force(n);
    for (std::size_t k = 0; k < n; ++k) {
        force[k] = -gradient[k];
    }

    const double power = dot(force, velocity_);
    if (power > 0.0) {
        // Rolling downhill: bend the velocity towards the force and, once that
        // has held for a while, take longer steps.
        const double speed = norm(velocity_);
        const double force_norm = norm(force);
        if (force_norm > 0.0) {
            const double scale = mixing_ * speed / force_norm;
            for (std::size_t k = 0; k < n; ++k) {
                velocity_[k] = (1.0 - mixing_) * velocity_[k] + scale * force[k];
            }
        }
        ++downhill_;
        if (downhill_ > n_min) {
            dt_ = std::min(dt_ * f_inc, dt_max);
            mixing_ *= f_alpha;
        }
    } else {
        // Turned uphill: stop dead, shorten the step, and start mixing hard
        // again. Freezing rather than reversing is what keeps FIRE from
        // oscillating across a narrow valley.
        std::fill(velocity_.begin(), velocity_.end(), 0.0);
        downhill_ = 0;
        dt_ *= f_dec;
        mixing_ = initial_mixing();
    }

    for (std::size_t k = 0; k < n; ++k) {
        velocity_[k] += dt_ * force[k];
    }

    std::vector<double> step(n);
    for (std::size_t k = 0; k < n; ++k) {
        step[k] = dt_ * velocity_[k];
    }
    return step;
}

}  // namespace fastatoms::optimize
