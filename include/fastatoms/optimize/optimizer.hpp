// SPDX-License-Identifier: MIT
#pragma once

/// \file optimizer.hpp
/// \brief The geometry optimizer base class.

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/io/parameters.hpp"
#include "fastatoms/optimize/line_search.hpp"
#include "fastatoms/optimize/types.hpp"
#include "fastatoms/optimize/variables.hpp"

namespace fastatoms::optimize {

/// Base class for the geometry optimizers.
///
/// A subclass supplies one thing: a search direction for the current gradient,
/// plus whatever bookkeeping its history needs. Everything shared - the
/// convergence test, the trust radius, the line search, the step log, and the
/// mapping between a structure and a coordinate vector - lives here.
///
/// The numerical core is `minimize()`, which works on any `ObjectiveFunction`.
/// `run()` is the structure-facing wrapper that builds such a function from an
/// `Atoms` object and its calculator. Because the two are separate, the
/// algorithms can be exercised on analytic test functions with no physics
/// involved, which is exactly how they are tested.
///
/// Parameters (all optional):
///   - `fmax`        - convergence threshold in eV/angstrom. Default 0.05.
///                     Convergence means the largest force on any single atom,
///                     not the largest Cartesian component.
///   - `max_steps`   - step budget. Default 200.
///   - `max_step`    - trust radius in angstrom, the furthest any atom may
///                     move in one step. Default 0.2. It bounds the accepted
///                     displacement, so a line search is free to grow a short
///                     direction up to it.
///   - `alpha`       - initial inverse-curvature guess in eV/angstrom^2, used
///                     to turn a gradient into a first displacement.
///                     Default 70, the usual value for atomic systems.
///   - `line_search` - "none", "backtracking" or "wolfe" (the default).
///                     "none" takes the proposed direction as given, capped by
///                     the trust radius: one force evaluation per step, but it
///                     cannot recover when the curvature model has nothing to
///                     say.
///   - `relax_cell`  - also relax the cell. Default false.
///   - `cell_factor`, `pressure`, `mask` - see `CellVariablesOptions`.
///   - `verbose`     - print a step table while running. Default false.
class Optimizer {
public:
    virtual ~Optimizer() = default;

    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;

    [[nodiscard]] virtual std::string name() const = 0;

    /// Relax `atoms` in place using its attached calculator.
    ///
    /// On return the structure holds the final geometry and its calculator
    /// holds the matching results, so a following `get_potential_energy()`
    /// costs nothing.
    ///
    /// \throws OptimizerError if no calculator is attached, or if the cell is
    ///         to be relaxed but the structure is not periodic.
    OptimizationResult run(Atoms& atoms);

    /// Minimise an arbitrary differentiable function.
    ///
    /// `x` is both the starting point and, on return, the best point found.
    /// The gradient is expected in groups of three so that the convergence
    /// test means "largest force on one atom"; for a general function of `n`
    /// variables the length just has to be a multiple of three.
    OptimizationResult minimize(const ObjectiveFunction& objective, std::vector<double>& x);

    /// \name Configuration
    /// @{
    [[nodiscard]] const Value& parameters() const noexcept { return parameters_; }
    void set_parameters(Value parameters);
    void set_parameter(const std::string& key, Value value);

    [[nodiscard]] double fmax() const;
    [[nodiscard]] int max_steps() const;
    [[nodiscard]] double max_step() const;
    [[nodiscard]] double alpha() const;
    [[nodiscard]] LineSearchKind line_search_kind() const;
    /// @}

    /// Add an observer called after every accepted step. Returning false from
    /// it stops the run with `StopReason::Interrupted`.
    void attach(StepCallback callback);
    void clear_callbacks();

    [[nodiscard]] virtual std::string to_string() const;

protected:
    explicit Optimizer(Value parameters = {});

    /// Prepare for a fresh minimisation of `ndof` variables.
    virtual void begin(std::size_t ndof) = 0;

    /// Displacement to try next, in length units. Must point downhill.
    [[nodiscard]] virtual std::vector<double> direction(
        const std::vector<double>& x, const std::vector<double>& gradient) = 0;

    /// Record an accepted step and the gradient change it produced.
    virtual void accept(const std::vector<double>& dx, const std::vector<double>& dg) = 0;

    /// Throw away accumulated history, after a failed or uphill step.
    virtual void restart() = 0;

    /// Line search used when the `line_search` parameter is absent.
    ///
    /// A strong Wolfe search for every algorithm. Skipping it saves force
    /// evaluations only while the curvature model is good: near a minimum it
    /// costs about 10-60% more evaluations, but on a stretched or concave
    /// starting geometry - where every curvature update is legitimately
    /// discarded and a quasi-Newton method degenerates to a fixed `-g/alpha`
    /// step - it is several times cheaper overall, because it converges at
    /// all. Set `line_search: none` to take the direction as proposed.
    [[nodiscard]] virtual LineSearchKind default_line_search() const {
        return LineSearchKind::StrongWolfe;
    }

    /// Curvature constant for the strong Wolfe search. Conjugate gradient
    /// wants a tighter value than the quasi-Newton methods.
    [[nodiscard]] virtual double default_c2() const { return 0.9; }

    /// Whether a scale factor of one along the proposed direction is already
    /// the natural step.
    ///
    /// It is for the quasi-Newton methods, whose direction carries a real
    /// curvature estimate. It is not for steepest descent or conjugate
    /// gradient, whose direction is only `-g / alpha` with a fixed guess at
    /// alpha: if that guess is far off, every line search would have to climb
    /// back from a scale of one all over again. Those methods instead start
    /// each search where the previous one finished, so the step size adapts to
    /// the surface within a handful of iterations.
    [[nodiscard]] virtual bool has_natural_unit_step() const { return true; }

    /// `-gradient / alpha`: the fallback direction, in length units.
    [[nodiscard]] std::vector<double> steepest_direction(
        const std::vector<double>& gradient) const;

    [[nodiscard]] const Value& parameter(std::string_view key) const;

    [[noreturn]] void fail(const std::string& message) const;

    Value parameters_{};

private:
    [[nodiscard]] LineSearchOptions line_search_options() const;
    [[nodiscard]] std::unique_ptr<Variables> make_variables(const Atoms& atoms) const;
    bool notify(const OptimizationStep& step) const;

    std::vector<StepCallback> callbacks_{};
    /// Rows that belong to atoms, for reporting; 0 means "all of them".
    std::size_t atom_rows_{0};
};

}  // namespace fastatoms::optimize
