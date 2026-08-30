// SPDX-License-Identifier: MIT
#include "fastatoms/optimize/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

#include "fastatoms/optimize/vector_math.hpp"

namespace fastatoms::optimize {

std::string_view stop_reason_name(StopReason reason) {
    switch (reason) {
        case StopReason::Converged:        return "converged";
        case StopReason::MaxSteps:         return "max_steps";
        case StopReason::LineSearchFailed: return "line_search_failed";
        case StopReason::NoProgress:       return "no_progress";
        case StopReason::Interrupted:      return "interrupted";
    }
    return "unknown";
}

std::string OptimizationResult::to_string() const {
    std::ostringstream out;
    out.precision(6);
    out << "OptimizationResult(" << (converged ? "converged" : "not converged")
        << ", reason=" << stop_reason_name(reason) << ", steps=" << steps
        << ", evaluations=" << evaluations << ", energy=" << energy << ", fmax=" << fmax << ")";
    return out.str();
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

Optimizer::Optimizer(Value parameters) : parameters_(std::move(parameters)) {
    if (parameters_.is_null()) {
        parameters_ = Value(Value::Map{});
    }
    if (!parameters_.is_map()) {
        throw OptimizerError("optimizer parameters must be a mapping, got " +
                             std::string(parameters_.type_name()));
    }
}

const Value& Optimizer::parameter(std::string_view key) const {
    if (const Value* found = parameters_.find(key)) {
        return *found;
    }
    return Value::null();
}

void Optimizer::set_parameters(Value parameters) {
    if (parameters.is_null()) {
        parameters = Value(Value::Map{});
    }
    if (!parameters.is_map()) {
        throw OptimizerError("optimizer parameters must be a mapping, got " +
                             std::string(parameters.type_name()));
    }
    parameters_ = std::move(parameters);
}

void Optimizer::set_parameter(const std::string& key, Value value) {
    parameters_.insert(key, std::move(value));
}

void Optimizer::fail(const std::string& message) const {
    throw OptimizerError(name() + ": " + message);
}

double Optimizer::fmax() const {
    const double value = parameter("fmax").value_or(0.05);
    if (!(value > 0.0)) {
        fail("'fmax' must be positive");
    }
    return value;
}

int Optimizer::max_steps() const {
    const auto value = static_cast<int>(parameter("max_steps").value_or(std::int64_t{200}));
    if (value < 0) {
        fail("'max_steps' must not be negative");
    }
    return value;
}

double Optimizer::max_step() const {
    const double value = parameter("max_step").value_or(parameter("maxstep").value_or(0.2));
    if (!(value > 0.0)) {
        fail("'max_step' must be positive");
    }
    return value;
}

double Optimizer::alpha() const {
    const double value = parameter("alpha").value_or(70.0);
    if (!(value > 0.0)) {
        fail("'alpha' must be positive");
    }
    return value;
}

LineSearchKind Optimizer::line_search_kind() const {
    const Value& value = parameter("line_search");
    if (value.is_null()) {
        return default_line_search();
    }
    if (value.is_bool()) {
        return value.as_bool() ? LineSearchKind::StrongWolfe : LineSearchKind::None;
    }
    return line_search_from_string(value.as_string());
}

LineSearchOptions Optimizer::line_search_options() const {
    LineSearchOptions options;
    options.kind = line_search_kind();
    options.c1 = parameter("c1").value_or(1e-4);
    options.c2 = parameter("c2").value_or(default_c2());
    options.max_evaluations =
        static_cast<int>(parameter("max_line_search_evaluations").value_or(std::int64_t{20}));
    // The direction has already been clipped to the trust radius, so a scale
    // of one is the full trust-radius step. Backtracking only ever shrinks it;
    // the Wolfe search may grow it, but not without evidence.
    // The scale ceiling is set per step from the trust radius; see minimize().
    options.max_step = 1.0;
    options.initial_step = 1.0;
    return options;
}

void Optimizer::attach(StepCallback callback) {
    if (callback) {
        callbacks_.push_back(std::move(callback));
    }
}

void Optimizer::clear_callbacks() { callbacks_.clear(); }

std::string Optimizer::to_string() const {
    std::ostringstream out;
    out.precision(4);
    out << name() << "(fmax=" << fmax() << ", max_steps=" << max_steps()
        << ", max_step=" << max_step()
        << ", line_search='" << line_search_name(line_search_kind()) << "'";
    if (parameter("relax_cell").value_or(false)) {
        out << ", relax_cell=True";
    }
    out << ")";
    return out.str();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::vector<double> Optimizer::steepest_direction(const std::vector<double>& gradient) const {
    return scaled(-1.0 / alpha(), gradient);
}

bool Optimizer::notify(const OptimizationStep& step) const {
    if (parameter("verbose").value_or(false)) {
        if (step.step == 0) {
            std::printf("%-10s %6s %16s %14s %14s\n", name().c_str(), "step", "energy [eV]",
                        "fmax [eV/A]", "dE [eV]");
        }
        std::printf("%-10s %6d %16.6f %14.6f %14.3e\n", "", step.step, step.energy, step.fmax,
                    step.energy_change);
        std::fflush(stdout);
    }
    for (const StepCallback& callback : callbacks_) {
        if (!callback(step)) {
            return false;
        }
    }
    return true;
}

std::unique_ptr<Variables> Optimizer::make_variables(const Atoms& atoms) const {
    if (!parameter("relax_cell").value_or(false)) {
        return std::make_unique<AtomicVariables>(atoms.size());
    }

    CellVariablesOptions options;
    options.cell_factor = parameter("cell_factor").value_or(0.0);
    options.pressure = parameter("pressure").value_or(0.0);
    if (const Value& mask = parameter("mask"); !mask.is_null()) {
        const std::vector<double> values = mask.as_doubles();
        if (values.size() != 6) {
            fail("'mask' must have six entries in Voigt order (xx, yy, zz, yz, xz, xy)");
        }
        std::copy(values.begin(), values.end(), options.mask.begin());
    }
    return std::make_unique<CellVariables>(atoms, options);
}

// ---------------------------------------------------------------------------
// The optimization loop
// ---------------------------------------------------------------------------

OptimizationResult Optimizer::minimize(const ObjectiveFunction& objective,
                                       std::vector<double>& x) {
    if (x.empty()) {
        fail("there is nothing to optimise: the coordinate vector is empty");
    }
    if (x.size() % 3 != 0) {
        fail("the coordinate vector length must be a multiple of three");
    }

    const double threshold = fmax();
    const int budget = max_steps();
    const LineSearchOptions search = line_search_options();
    const std::size_t atom_rows = atom_rows_ != 0 ? atom_rows_ : x.size() / 3;

    OptimizationResult result;
    Evaluation current = objective(x);
    ++result.evaluations;

    const auto record = [&](int step_index, double energy_change, double displacement,
                            double step_length) {
        OptimizationStep step;
        step.step = step_index;
        step.evaluations = result.evaluations;
        step.energy = current.energy;
        step.fmax = max_row_norm(current.gradient);
        step.atomic_fmax = max_row_norm(current.gradient, 0, atom_rows);
        step.energy_change = energy_change;
        step.max_displacement = displacement;
        step.step_length = step_length;
        result.history.push_back(step);
        return step;
    };

    const auto finish = [&](StopReason reason, std::string message) {
        result.reason = reason;
        result.converged = reason == StopReason::Converged;
        result.energy = current.energy;
        result.fmax = max_row_norm(current.gradient);
        result.message = std::move(message);
        return result;
    };

    if (!notify(record(0, 0.0, 0.0, 0.0))) {
        return finish(StopReason::Interrupted, "a callback stopped the run before it started");
    }
    if (max_row_norm(current.gradient) <= threshold) {
        return finish(StopReason::Converged, "the starting geometry already meets fmax");
    }

    begin(x.size());

    // Methods without a natural unit step need a starting guess for the line
    // search. Reusing the previous accepted scale outright would lock a
    // method into whatever step it once took; scaling by the ratio of the
    // directional derivatives keeps the first trial the same *distance* as
    // last time even when the direction's magnitude changes.
    double carried_scale = 1.0;
    double previous_slope = 0.0;

    for (int step_index = 1; step_index <= budget; ++step_index) {
        std::vector<double> proposed = direction(x, current.gradient);
        if (proposed.size() != x.size()) {
            fail("the search direction has the wrong length");
        }

        // A history-based method can propose an uphill direction when its
        // model has gone stale. Drop the history and fall back to the gradient.
        if (dot(proposed, current.gradient) >= 0.0) {
            restart();
            proposed = steepest_direction(current.gradient);
        }

        double slope = dot(current.gradient, proposed);

        // Run one line search along `d`, letting it use any scale that keeps
        // the displacement inside the trust radius.
        const auto search_along = [&](std::vector<double> d, double initial_guess) {
            LineSearchOptions attempt = search;
            const double longest = max_row_norm(d);
            if (longest <= 0.0) {
                return LineSearchResult{};
            }

            // The trust radius limits how far any atom may move, so the scale
            // at which the displacement reaches it is the natural ceiling. A
            // direction shorter than the trust radius may therefore be grown
            // by the line search, which is what lets these methods recover
            // from a badly chosen `alpha` instead of creeping.
            const double ceiling = std::min(max_step() / longest, 1e6);

            if (attempt.kind == LineSearchKind::None) {
                // Nothing will rescale the direction, so clip it here.
                if (longest > max_step()) {
                    scale(max_step() / longest, d);
                    slope = dot(current.gradient, d);
                }
            } else if (attempt.kind == LineSearchKind::Backtracking) {
                // Backtracking only ever shrinks, so it starts at the ceiling.
                attempt.max_step = std::min(1.0, ceiling);
                attempt.initial_step = attempt.max_step;
            } else {
                attempt.max_step = ceiling;
                attempt.initial_step = std::clamp(initial_guess, attempt.min_step, ceiling);
            }
            return line_search(objective, x, current, d, attempt);
        };

        // Where to start the search. Quasi-Newton directions already carry a
        // curvature estimate, so one is the right first guess; for the others
        // the previous step is rescaled by the ratio of directional
        // derivatives, which keeps the trial the same distance as last time.
        double initial_guess = 1.0;
        if (!has_natural_unit_step()) {
            initial_guess = carried_scale;
            if (previous_slope < 0.0 && slope < 0.0) {
                initial_guess = carried_scale * previous_slope / slope;
            }
        }

        LineSearchResult found = search_along(proposed, initial_guess);
        result.evaluations += found.evaluations;

        if (!found.ok) {
            // One retry straight downhill before giving up: the direction may
            // simply have been a bad extrapolation.
            restart();
            std::vector<double> fallback = steepest_direction(current.gradient);
            slope = dot(current.gradient, fallback);
            found = search_along(fallback, 1.0);
            result.evaluations += found.evaluations;
            carried_scale = 1.0;
            previous_slope = 0.0;
        }
        if (!found.ok) {
            return finish(StopReason::LineSearchFailed,
                          "no downhill step could be found along the search direction");
        }

        const std::vector<double> dx = subtract(found.x, x);
        const std::vector<double> dg = subtract(found.evaluation.gradient, current.gradient);
        const double displacement = max_row_norm(dx);
        const double energy_change = found.evaluation.energy - current.energy;

        if (!has_natural_unit_step() && found.step > 0.0) {
            carried_scale = found.step;
            previous_slope = slope;
        }
        accept(dx, dg);
        x = std::move(found.x);
        current = std::move(found.evaluation);
        ++result.steps;

        if (!notify(record(step_index, energy_change, displacement, found.step))) {
            return finish(StopReason::Interrupted, "a callback stopped the run");
        }
        if (max_row_norm(current.gradient) <= threshold) {
            return finish(StopReason::Converged,
                          "converged in " + std::to_string(result.steps) + " steps");
        }
        if (displacement < 1e-12) {
            return finish(StopReason::NoProgress,
                          "the step shrank below rounding noise before reaching fmax");
        }
    }

    return finish(StopReason::MaxSteps,
                  "stopped after " + std::to_string(budget) + " steps without reaching fmax");
}

OptimizationResult Optimizer::run(Atoms& atoms) {
    if (atoms.empty()) {
        fail("cannot relax an empty structure");
    }

    const std::unique_ptr<Variables> variables = make_variables(atoms);
    atom_rows_ = variables->atom_rows();

    std::vector<double> x = variables->pack(atoms);
    const ObjectiveFunction objective = [&variables, &atoms](const std::vector<double>& trial) {
        variables->unpack(trial, atoms);
        return variables->evaluate(atoms);
    };

    OptimizationResult result;
    try {
        result = minimize(objective, x);
    } catch (...) {
        atom_rows_ = 0;
        throw;
    }
    atom_rows_ = 0;

    // Leave the structure at the point the result describes. The last
    // objective evaluation was already made there, so the calculator's cache
    // still applies and nothing is recomputed.
    variables->unpack(x, atoms);
    return result;
}

}  // namespace fastatoms::optimize
