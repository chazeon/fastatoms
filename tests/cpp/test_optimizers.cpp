// SPDX-License-Identifier: MIT
//
// The optimization algorithms as pure numerics: analytic objective functions,
// exact algebraic identities, and the line searches. Nothing here touches an
// Atoms object, which is the point - if these pass, any failure in a real
// relaxation is in the physics layer, not the math.

#include <cmath>
#include <string>
#include <vector>

#include "fastatoms/optimize/bfgs.hpp"
#include "fastatoms/optimize/conjugate_gradient.hpp"
#include "fastatoms/optimize/lbfgs.hpp"
#include "fastatoms/optimize/line_search.hpp"
#include "fastatoms/optimize/registry.hpp"
#include "fastatoms/optimize/steepest_descent.hpp"
#include "fastatoms/optimize/vector_math.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::optimize;

namespace {

/// Expose the protected hooks so the update formulas can be checked directly.
template <typename Base>
struct Testable : Base {
    using Base::Base;
    using Base::accept;
    using Base::begin;
    using Base::direction;
};

Value settings(std::initializer_list<std::pair<const char*, Value>> entries) {
    Value map{Value::Map{}};
    for (const auto& [key, value] : entries) {
        map.insert(key, value);
    }
    return map;
}

/// f(x) = 0.5 sum_i k_i (x_i - c_i)^2. The condition number is max(k)/min(k),
/// which is exactly what separates steepest descent from the rest.
ObjectiveFunction quadratic(const std::vector<double>& curvature,
                            const std::vector<double>& centre, int* counter = nullptr) {
    return [curvature, centre, counter](const std::vector<double>& x) {
        Evaluation out;
        out.gradient.assign(x.size(), 0.0);
        for (std::size_t k = 0; k < x.size(); ++k) {
            const double d = x[k] - centre[k];
            out.energy += 0.5 * curvature[k] * d * d;
            out.gradient[k] = curvature[k] * d;
        }
        if (counter != nullptr) {
            ++(*counter);
        }
        return out;
    };
}

/// The extended Rosenbrock function: a curved narrow valley, the standard
/// stress test for a nonlinear minimiser. Minimum 0 at x = (1, 1, ...).
Evaluation rosenbrock(const std::vector<double>& x) {
    Evaluation out;
    out.gradient.assign(x.size(), 0.0);
    for (std::size_t k = 0; k + 1 < x.size(); k += 2) {
        const double a = x[k];
        const double b = x[k + 1];
        const double t = b - a * a;
        out.energy += 100.0 * t * t + (1.0 - a) * (1.0 - a);
        out.gradient[k] += -400.0 * a * t - 2.0 * (1.0 - a);
        out.gradient[k + 1] += 200.0 * t;
    }
    return out;
}

std::vector<double> filled(std::size_t n, double value) {
    return std::vector<double>(n, value);
}

}  // namespace

// ---------------------------------------------------------------------------
// The hand-written linear algebra
// ---------------------------------------------------------------------------

FA_TEST(vector_helpers) {
    const std::vector<double> a{1.0, 2.0, 3.0};
    const std::vector<double> b{4.0, 5.0, 6.0};
    FA_CHECK_CLOSE(dot(a, b), 32.0, 1e-15);
    FA_CHECK_CLOSE(norm(std::vector<double>{3.0, 4.0}), 5.0, 1e-15);
    FA_CHECK(subtract(b, a) == (std::vector<double>{3.0, 3.0, 3.0}));

    std::vector<double> y = a;
    axpy(2.0, b, y);
    FA_CHECK(y == (std::vector<double>{9.0, 12.0, 15.0}));

    // fmax means "largest force on one atom", not "largest component".
    FA_CHECK_CLOSE(max_row_norm(std::vector<double>{3.0, 4.0, 0.0, 1.0, 0.0, 0.0}), 5.0, 1e-15);
    FA_CHECK_CLOSE(max_row_norm(std::vector<double>{3.0, 4.0, 0.0, 9.0, 0.0, 0.0}, 0, 1), 5.0,
                   1e-15);
    FA_CHECK_THROWS(max_row_norm(std::vector<double>{1.0, 2.0}), std::invalid_argument);
    FA_CHECK_THROWS(dot(a, std::vector<double>{1.0}), std::invalid_argument);
}

FA_TEST(symmetric_matrix_apply_and_update) {
    SymmetricMatrix m(3, 2.0);
    FA_CHECK(m.size() == 3);
    const std::vector<double> v{1.0, 2.0, 3.0};
    FA_CHECK(m.apply(v) == (std::vector<double>{2.0, 4.0, 6.0}));

    // M += 1 * (a b^T + b a^T) + 2 * a a^T, checked entry by entry.
    const std::vector<double> a{1.0, 0.0, 0.0};
    const std::vector<double> b{0.0, 1.0, 0.0};
    m.add_symmetric_updates(1.0, a, b, 2.0);
    FA_CHECK_CLOSE(m(0, 0), 2.0 + 2.0, 1e-15);
    FA_CHECK_CLOSE(m(0, 1), 1.0, 1e-15);
    FA_CHECK_CLOSE(m(1, 0), 1.0, 1e-15);
    FA_CHECK_CLOSE(m(2, 2), 2.0, 1e-15);
}

// ---------------------------------------------------------------------------
// Line searches
// ---------------------------------------------------------------------------

FA_TEST(backtracking_satisfies_the_armijo_condition) {
    const ObjectiveFunction f = quadratic({1.0, 10.0, 100.0}, {0.0, 0.0, 0.0});
    const std::vector<double> x{1.0, 1.0, 1.0};
    const Evaluation start = f(x);

    LineSearchOptions options;
    options.kind = LineSearchKind::Backtracking;
    // A deliberately over-long direction, so the search has to pull it back.
    const std::vector<double> d = scaled(-1.0, start.gradient);

    const LineSearchResult result = line_search(f, x, start, d, options);
    FA_CHECK(result.ok);
    FA_CHECK(result.step > 0.0 && result.step <= 1.0);
    FA_CHECK(result.evaluation.energy < start.energy);

    const double slope = dot(start.gradient, d);
    FA_CHECK(result.evaluation.energy <= start.energy + options.c1 * result.step * slope);
}

FA_TEST(strong_wolfe_satisfies_both_conditions) {
    const ObjectiveFunction f = quadratic({1.0, 4.0, 25.0}, {0.5, -0.5, 0.25});
    const std::vector<double> x{3.0, 2.0, 1.0};
    const Evaluation start = f(x);
    const std::vector<double> d = scaled(-0.05, start.gradient);

    LineSearchOptions options;
    options.kind = LineSearchKind::StrongWolfe;
    options.c2 = 0.1;

    const LineSearchResult result = line_search(f, x, start, d, options);
    FA_CHECK(result.ok);

    const double slope0 = dot(start.gradient, d);
    const double slope_new = dot(result.evaluation.gradient, d);
    // Sufficient decrease...
    FA_CHECK(result.evaluation.energy <= start.energy + options.c1 * result.step * slope0);
    // ...and the curvature condition, which is what conjugacy relies on.
    FA_CHECK(std::abs(slope_new) <= -options.c2 * slope0 + 1e-12);
}

FA_TEST(line_search_refuses_an_uphill_direction) {
    const ObjectiveFunction f = quadratic({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    const std::vector<double> x{1.0, 1.0, 1.0};
    const Evaluation start = f(x);

    LineSearchOptions options;
    options.kind = LineSearchKind::StrongWolfe;
    // Straight uphill.
    const LineSearchResult result = line_search(f, x, start, start.gradient, options);
    FA_CHECK(!result.ok);
    FA_CHECK(result.evaluations == 0);
}

FA_TEST(no_line_search_takes_the_direction_as_given) {
    const ObjectiveFunction f = quadratic({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    const std::vector<double> x{1.0, 1.0, 1.0};
    const Evaluation start = f(x);
    const std::vector<double> d{-0.5, -0.5, -0.5};

    LineSearchOptions options;
    options.kind = LineSearchKind::None;
    const LineSearchResult result = line_search(f, x, start, d, options);
    FA_CHECK(result.ok);
    FA_CHECK(result.evaluations == 1);
    FA_CHECK_CLOSE(result.step, 1.0, 1e-15);
    FA_CHECK_CLOSE(result.x[0], 0.5, 1e-15);
}

FA_TEST(line_search_names_round_trip) {
    FA_CHECK(line_search_from_string("none") == LineSearchKind::None);
    FA_CHECK(line_search_from_string("Armijo") == LineSearchKind::Backtracking);
    FA_CHECK(line_search_from_string("strong-wolfe") == LineSearchKind::StrongWolfe);
    FA_CHECK(line_search_name(LineSearchKind::Backtracking) == "backtracking");
    FA_CHECK_THROWS(line_search_from_string("golden"), OptimizerError);
}

// ---------------------------------------------------------------------------
// The update formulas, checked exactly
// ---------------------------------------------------------------------------

FA_TEST(bfgs_update_satisfies_the_secant_equation) {
    // The defining property of the BFGS inverse update: after absorbing a step
    // s and the gradient change y it caused, H y must be exactly s.
    Testable<BFGS> optimizer(settings({{"alpha", Value(2.0)}}));
    optimizer.begin(6);

    const std::vector<double> s{0.1, -0.2, 0.3, 0.05, -0.15, 0.25};
    const std::vector<double> y{1.0, 0.5, -0.25, 2.0, -1.5, 0.75};
    FA_CHECK(dot(s, y) > 0.0);

    optimizer.accept(s, y);
    const std::vector<double> reconstructed = optimizer.inverse_hessian().apply(y);
    for (std::size_t k = 0; k < s.size(); ++k) {
        FA_CHECK_CLOSE(reconstructed[k], s[k], 1e-12);
    }

    // And -H g is the direction, so feeding y back gives -s.
    const std::vector<double> d = optimizer.direction({}, y);
    for (std::size_t k = 0; k < s.size(); ++k) {
        FA_CHECK_CLOSE(d[k], -s[k], 1e-12);
    }
}

FA_TEST(bfgs_skips_updates_with_negative_curvature) {
    Testable<BFGS> optimizer;
    optimizer.begin(3);
    const std::vector<double> s{1.0, 0.0, 0.0};
    const std::vector<double> y{-1.0, 0.0, 0.0};  // y.s < 0
    optimizer.accept(s, y);
    FA_CHECK(optimizer.skipped_updates() == 1);

    // The untouched matrix still gives a downhill direction.
    const std::vector<double> g{1.0, 2.0, 3.0};
    FA_CHECK(dot(optimizer.direction({}, g), g) < 0.0);
}

FA_TEST(lbfgs_two_loop_reproduces_the_last_secant_pair) {
    // With a single stored correction the two-loop recursion is an exact BFGS
    // update from gamma*I, so it must satisfy the same secant equation.
    Testable<LBFGS> optimizer;
    optimizer.begin(6);

    const std::vector<double> s{0.1, -0.2, 0.3, 0.05, -0.15, 0.25};
    const std::vector<double> y{1.0, 0.5, -0.25, 2.0, -1.5, 0.75};
    optimizer.accept(s, y);
    FA_CHECK(optimizer.history_size() == 1);

    const std::vector<double> d = optimizer.direction({}, y);
    for (std::size_t k = 0; k < s.size(); ++k) {
        FA_CHECK_CLOSE(d[k], -s[k], 1e-12);
    }
}

FA_TEST(lbfgs_keeps_only_the_requested_number_of_pairs) {
    Testable<LBFGS> optimizer(settings({{"memory", Value(3)}}));
    optimizer.begin(3);
    for (int k = 1; k <= 8; ++k) {
        optimizer.accept({0.1 * k, 0.0, 0.0}, {0.2 * k, 0.0, 0.0});
    }
    FA_CHECK(optimizer.history_size() == 3);
    FA_CHECK(optimizer.memory() == 3);
}

FA_TEST(all_directions_point_downhill) {
    const std::vector<double> g{1.0, -2.0, 0.5, 3.0, -0.25, 1.5};
    Testable<SteepestDescent> sd;
    Testable<ConjugateGradient> cg;
    Testable<BFGS> bfgs;
    Testable<LBFGS> lbfgs;
    sd.begin(6);
    cg.begin(6);
    bfgs.begin(6);
    lbfgs.begin(6);

    FA_CHECK(dot(sd.direction({}, g), g) < 0.0);
    FA_CHECK(dot(cg.direction({}, g), g) < 0.0);
    FA_CHECK(dot(bfgs.direction({}, g), g) < 0.0);
    FA_CHECK(dot(lbfgs.direction({}, g), g) < 0.0);
}

// ---------------------------------------------------------------------------
// Whole minimisations
// ---------------------------------------------------------------------------

FA_TEST(every_algorithm_finds_a_quadratic_minimum) {
    const std::vector<double> curvature{1.0, 3.0, 7.0, 12.0, 20.0, 35.0};
    const std::vector<double> centre{0.5, -1.0, 2.0, 0.0, -0.5, 1.5};

    for (const std::string& algorithm : {"sd", "cg", "bfgs", "lbfgs"}) {
        auto optimizer = make_optimizer(
            algorithm, settings({{"fmax", Value(1e-6)}, {"max_steps", Value(500)},
                                 {"alpha", Value(20.0)}, {"max_step", Value(10.0)}}));
        std::vector<double> x = filled(6, 5.0);
        const OptimizationResult result = optimizer->minimize(quadratic(curvature, centre), x);

        FA_CHECK(result.converged);
        FA_CHECK(result.reason == StopReason::Converged);
        for (std::size_t k = 0; k < x.size(); ++k) {
            FA_CHECK_CLOSE(x[k], centre[k], 1e-4);
        }
        FA_CHECK_CLOSE(result.energy, 0.0, 1e-8);
        FA_CHECK(result.fmax <= 1e-6);
        FA_CHECK(result.history.size() == static_cast<std::size_t>(result.steps) + 1);
    }
}

FA_TEST(quasi_newton_beats_steepest_descent_on_a_stiff_problem) {
    // Condition number 1000: steepest descent zig-zags, the others do not.
    const std::vector<double> curvature{1.0, 1.0, 10.0, 100.0, 500.0, 1000.0};
    const std::vector<double> centre = filled(6, 0.0);

    const auto steps_for = [&](const std::string& algorithm) {
        auto optimizer = make_optimizer(
            algorithm, settings({{"fmax", Value(1e-5)}, {"max_steps", Value(5000)},
                                 {"alpha", Value(1000.0)}, {"max_step", Value(10.0)}}));
        std::vector<double> x = filled(6, 1.0);
        const OptimizationResult result = optimizer->minimize(quadratic(curvature, centre), x);
        FA_CHECK(result.converged);
        return result.steps;
    };

    const int sd_steps = steps_for("sd");
    const int cg_steps = steps_for("cg");
    const int bfgs_steps = steps_for("bfgs");
    const int lbfgs_steps = steps_for("lbfgs");

    // Steepest descent needs a number of steps that grows with the condition
    // number; the others do not, so the gap is an order of magnitude, not a
    // few percent.
    FA_CHECK(bfgs_steps * 10 < sd_steps);
    FA_CHECK(lbfgs_steps * 10 < sd_steps);
    FA_CHECK(cg_steps * 10 < sd_steps);
}

FA_TEST(the_quadratic_termination_property_holds) {
    // Given exact line searches, both conjugate gradient and BFGS minimise an
    // n-variable quadratic in at most n steps. That finite termination is the
    // whole theoretical content of these methods, so it is worth testing
    // directly. The strong Wolfe search is close to exact but not exact, and
    // the initial inverse Hessian is only a guess, so one extra step is
    // allowed - the point is that it is n, not hundreds.
    const std::vector<double> curvature{1.0, 4.0, 9.0, 30.0, 80.0, 150.0};
    const std::vector<double> centre{1.0, -1.0, 0.5, -0.5, 2.0, -2.0};
    const std::size_t n = curvature.size();

    for (const std::string& algorithm : {"cg", "bfgs", "lbfgs"}) {
        auto optimizer = make_optimizer(algorithm,
                                        settings({{"fmax", Value(1e-8)},
                                                  {"max_steps", Value(100)},
                                                  {"alpha", Value(50.0)},
                                                  {"max_step", Value(100.0)},
                                                  {"line_search", Value("wolfe")},
                                                  {"c2", Value(0.01)},
                                                  {"memory", Value(20)}}));
        std::vector<double> x = filled(n, 4.0);
        const OptimizationResult result = optimizer->minimize(quadratic(curvature, centre), x);

        FA_CHECK(result.converged);
        FA_CHECK(result.steps <= static_cast<int>(n) + 1);
        for (std::size_t k = 0; k < n; ++k) {
            FA_CHECK_CLOSE(x[k], centre[k], 1e-6);
        }
    }
}

FA_TEST(every_algorithm_climbs_the_rosenbrock_valley) {
    for (const std::string& algorithm : {"cg", "bfgs", "lbfgs"}) {
        auto optimizer =
            make_optimizer(algorithm, settings({{"fmax", Value(1e-5)},
                                                {"max_steps", Value(3000)},
                                                {"alpha", Value(1.0)},
                                                {"max_step", Value(1.0)},
                                                {"line_search", Value("wolfe")}}));
        std::vector<double> x{-1.2, 1.0, -1.2, 1.0, -1.2, 1.0};
        const OptimizationResult result = optimizer->minimize(rosenbrock, x);

        FA_CHECK(result.converged);
        for (const double value : x) {
            FA_CHECK_CLOSE(value, 1.0, 1e-3);
        }
        FA_CHECK_CLOSE(result.energy, 0.0, 1e-6);
    }
}

FA_TEST(conjugate_gradient_variants_all_converge) {
    const std::vector<double> curvature{1.0, 5.0, 25.0, 50.0, 100.0, 200.0};
    for (const char* variant : {"polak_ribiere", "fletcher_reeves", "hestenes_stiefel"}) {
        ConjugateGradient optimizer(settings({{"variant", Value(variant)},
                                              {"fmax", Value(1e-6)},
                                              {"max_steps", Value(2000)},
                                              {"alpha", Value(100.0)},
                                              {"max_step", Value(10.0)}}));
        FA_CHECK(cg_variant_name(optimizer.variant()) == variant);
        std::vector<double> x = filled(6, 2.0);
        const OptimizationResult result =
            optimizer.minimize(quadratic(curvature, filled(6, 0.0)), x);
        FA_CHECK(result.converged);
        FA_CHECK_CLOSE(result.energy, 0.0, 1e-9);
    }
    FA_CHECK_THROWS(cg_variant_from_string("nonsense"), OptimizerError);
}

// ---------------------------------------------------------------------------
// Convergence handling
// ---------------------------------------------------------------------------

FA_TEST(a_converged_start_costs_a_single_evaluation) {
    BFGS optimizer(settings({{"fmax", Value(1.0)}}));
    std::vector<double> x = filled(3, 0.0);
    const OptimizationResult result = optimizer.minimize(quadratic({1, 1, 1}, {0, 0, 0}), x);

    FA_CHECK(result.converged);
    FA_CHECK(result.steps == 0);
    FA_CHECK(result.evaluations == 1);
    FA_CHECK(result.history.size() == 1);
}

FA_TEST(the_step_budget_is_respected) {
    BFGS optimizer(settings({{"fmax", Value(1e-12)}, {"max_steps", Value(3)}}));
    std::vector<double> x = filled(6, 10.0);
    const OptimizationResult result = optimizer.minimize(quadratic(filled(6, 1.0), filled(6, 0.0)), x);

    FA_CHECK(!result.converged);
    FA_CHECK(result.reason == StopReason::MaxSteps);
    FA_CHECK(result.steps == 3);
    FA_CHECK(result.message.find("without reaching fmax") != std::string::npos);
}

FA_TEST(a_callback_sees_every_step_and_can_stop_the_run) {
    BFGS optimizer(settings({{"fmax", Value(1e-8)}, {"max_steps", Value(100)}}));

    std::vector<double> energies;
    optimizer.attach([&energies](const OptimizationStep& step) {
        energies.push_back(step.energy);
        return step.step < 2;  // stop after the second accepted step
    });

    std::vector<double> x = filled(6, 4.0);
    const OptimizationResult result = optimizer.minimize(quadratic(filled(6, 3.0), filled(6, 0.0)), x);

    FA_CHECK(result.reason == StopReason::Interrupted);
    FA_CHECK(!result.converged);
    FA_CHECK(energies.size() == 3);  // the start plus two steps
    FA_CHECK(energies[1] < energies[0]);
    FA_CHECK(energies[2] < energies[1]);
}

FA_TEST(the_trust_radius_bounds_the_first_step) {
    // With no line search the trust radius is the only thing limiting how far
    // a step can go.
    BFGS optimizer(settings({{"max_step", Value(0.05)},
                             {"fmax", Value(1e-8)},
                             {"max_steps", Value(1)},
                             {"line_search", Value("none")}}));
    std::vector<double> x = filled(3, 100.0);
    const std::vector<double> start = x;
    (void)optimizer.minimize(quadratic({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}), x);

    FA_CHECK_CLOSE(max_row_norm(subtract(x, start)), 0.05, 1e-12);
}

FA_TEST(bad_configuration_is_rejected) {
    std::vector<double> x = filled(3, 1.0);
    const ObjectiveFunction f = quadratic({1, 1, 1}, {0, 0, 0});

    FA_CHECK_THROWS(BFGS(settings({{"fmax", Value(-1.0)}})).minimize(f, x), OptimizerError);
    FA_CHECK_THROWS(BFGS(settings({{"max_step", Value(0.0)}})).minimize(f, x), OptimizerError);
    FA_CHECK_THROWS(BFGS(settings({{"alpha", Value(0.0)}})).minimize(f, x), OptimizerError);
    FA_CHECK_THROWS(LBFGS(settings({{"memory", Value(0)}})).minimize(f, x), OptimizerError);

    std::vector<double> empty;
    FA_CHECK_THROWS(BFGS().minimize(f, empty), OptimizerError);
    std::vector<double> ragged = filled(4, 1.0);
    FA_CHECK_THROWS(BFGS().minimize(f, ragged), OptimizerError);
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

FA_TEST(registry_builds_the_built_in_optimizers) {
    FA_CHECK(make_optimizer("sd")->name() == "sd");
    FA_CHECK(make_optimizer("cg")->name() == "cg");
    FA_CHECK(make_optimizer("bfgs")->name() == "bfgs");
    FA_CHECK(make_optimizer("lbfgs")->name() == "lbfgs");

    FA_CHECK(make_optimizer("Steepest-Descent")->name() == "sd");
    FA_CHECK(make_optimizer("conjugate_gradient")->name() == "cg");
    FA_CHECK(make_optimizer("L-BFGS")->name() == "lbfgs");
    FA_CHECK(make_optimizer("BFGS")->name() == "bfgs");

    FA_CHECK_THROWS(make_optimizer("newton"), OptimizerError);
    FA_CHECK(registered_optimizers().size() == 4);
}

FA_TEST(registry_merges_the_typed_input_file_fields) {
    OptimizerParameters spec;
    spec.algorithm = "lbfgs";
    spec.fmax = 0.01;
    spec.max_steps = 42;
    spec.relax_cell = true;

    auto optimizer = make_optimizer(spec);
    FA_CHECK(optimizer->name() == "lbfgs");
    FA_CHECK_CLOSE(optimizer->fmax(), 0.01, 1e-15);
    FA_CHECK(optimizer->max_steps() == 42);
    FA_CHECK(optimizer->parameters().at("relax_cell").as_bool());

    // Anything the file set that the typed struct does not know about still
    // reaches the optimizer.
    OptimizerParameters rich;
    rich.algorithm = "cg";
    rich.parameters = settings({{"variant", Value("fletcher_reeves")}, {"fmax", Value(0.2)}});
    auto configured = make_optimizer(rich);
    FA_CHECK_CLOSE(configured->fmax(), 0.2, 1e-15);
    FA_CHECK(configured->parameters().at("variant").as_string() == "fletcher_reeves");
}

FA_TEST_MAIN()
