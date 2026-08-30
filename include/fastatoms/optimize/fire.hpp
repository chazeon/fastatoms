// SPDX-License-Identifier: MIT
#pragma once

/// \file fire.hpp

#include <vector>

#include "fastatoms/optimize/optimizer.hpp"

namespace fastatoms::optimize {

/// FIRE: fast inertial relaxation engine (Bitzek et al., PRL 97, 170201).
///
/// A ball rolling downhill with friction that adapts. The state is a velocity;
/// each step mixes it towards the current force, and the timestep grows while
/// progress is being made and collapses the moment the velocity turns uphill:
///
///     P = F . v
///     P > 0:  v <- (1-a) v + a |v| F^   and, after N_min such steps,
///             dt <- min(dt f_inc, dt_max),  a <- a f_alpha
///     P <= 0: v <- 0,  dt <- dt f_dec,  a <- a_start
///     v <- v + dt F,   dx = dt v
///
/// It is here because of the nudged elastic band. The NEB force is not the
/// gradient of anything - the spring term is added by hand and the climbing
/// image has its parallel component reversed - so there is no energy for a
/// line search to decrease, and a quasi-Newton method building a curvature
/// model out of that force is modelling a function that does not exist. FIRE
/// asks only for a direction to roll in, which is exactly what a band
/// provides, and is what nearly every NEB implementation uses for the reason.
///
/// It is a perfectly good structural relaxer too, and on a rough starting
/// geometry it is often more robust than L-BFGS, though it needs more steps
/// near a minimum.
///
/// Parameters, with the values from the paper:
///   - `dt`         - initial timestep (0.1)
///   - `dt_max`     - largest it may grow to (1.0)
///   - `n_min`      - downhill steps before the timestep may grow (5)
///   - `f_inc`      - growth factor (1.1)
///   - `f_dec`      - collapse factor (0.5)
///   - `alpha`      - initial velocity mixing (0.1); note this is *not* the
///                    `alpha` of the quasi-Newton methods, which is a
///                    curvature guess
///   - `f_alpha`    - mixing decay (0.99)
class Fire : public Optimizer {
public:
    explicit Fire(Value parameters = {}) : Optimizer(std::move(parameters)) {}

    [[nodiscard]] std::string name() const override { return "fire"; }

    /// The current timestep and mixing, for a caller that wants to watch the
    /// adaptation rather than infer it.
    [[nodiscard]] double timestep() const noexcept { return dt_; }
    [[nodiscard]] double mixing() const noexcept { return mixing_; }
    [[nodiscard]] int downhill_steps() const noexcept { return downhill_; }

protected:
    void begin(std::size_t ndof) override;
    void restart() override;
    void accept(const std::vector<double>& dx, const std::vector<double>& dg) override;

    [[nodiscard]] std::vector<double> direction(const std::vector<double>& x,
                                                const std::vector<double>& gradient) override;

    /// The step FIRE proposes is already the step to take; scaling it with a
    /// line search would discard the velocity that is the whole method.
    [[nodiscard]] bool has_natural_unit_step() const override { return true; }

    /// No line search. There is nothing for it to decrease when the force is
    /// not a gradient, and even for an ordinary relaxation FIRE's adaptive
    /// timestep already plays that role.
    [[nodiscard]] LineSearchKind default_line_search() const override {
        return LineSearchKind::None;
    }

private:
    [[nodiscard]] double initial_timestep() const;
    [[nodiscard]] double maximum_timestep() const;
    [[nodiscard]] double initial_mixing() const;

    std::vector<double> velocity_{};
    double dt_{0.1};
    double mixing_{0.1};
    int downhill_{0};
};

}  // namespace fastatoms::optimize
