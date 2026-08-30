// SPDX-License-Identifier: MIT
#include "fastatoms/optimize/line_search.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

#include "fastatoms/optimize/vector_math.hpp"

namespace fastatoms::optimize {
namespace {

std::vector<double> shifted(const std::vector<double>& x, double alpha,
                            const std::vector<double>& direction) {
    std::vector<double> out = x;
    axpy(alpha, direction, out);
    return out;
}

/// Minimiser of the quadratic through (a, phi_a) with slope dphi_a and
/// (b, phi_b). Returns a non-finite value when the fit is degenerate, which
/// the caller treats as "fall back to bisection".
double quadratic_minimum(double a, double phi_a, double dphi_a, double b, double phi_b) {
    const double width = b - a;
    const double denominator = 2.0 * (phi_b - phi_a - dphi_a * width);
    if (std::abs(denominator) < 1e-300) {
        return std::nan("");
    }
    return a - dphi_a * width * width / denominator;
}

/// Lowest-energy point seen so far, so a search that runs out of budget can
/// still hand back progress rather than nothing.
struct BestPoint {
    bool valid{false};
    double step{0.0};
    Evaluation evaluation{};
    std::vector<double> x{};

    void offer(double alpha, const Evaluation& evaluation_at, std::vector<double> point) {
        if (!valid || evaluation_at.energy < evaluation.energy) {
            valid = true;
            step = alpha;
            evaluation = evaluation_at;
            x = std::move(point);
        }
    }
};

LineSearchResult backtracking(const ObjectiveFunction& objective, const std::vector<double>& x,
                              const Evaluation& start, const std::vector<double>& direction,
                              const LineSearchOptions& options, double slope) {
    LineSearchResult result;
    BestPoint best;

    double alpha = options.initial_step;
    for (int attempt = 0; attempt < options.max_evaluations; ++attempt) {
        std::vector<double> trial = shifted(x, alpha, direction);
        const Evaluation evaluation = objective(trial);
        ++result.evaluations;
        best.offer(alpha, evaluation, trial);

        // Armijo: the energy must fall by a fraction of what the slope promised.
        if (evaluation.energy <= start.energy + options.c1 * alpha * slope) {
            result.ok = true;
            result.step = alpha;
            result.evaluation = evaluation;
            result.x = std::move(trial);
            return result;
        }

        // Fit a quadratic through what we know and jump to its minimum, but
        // never trust it to move less than 10% or more than 90% of the way.
        double next = quadratic_minimum(0.0, start.energy, slope, alpha, evaluation.energy);
        if (!std::isfinite(next) || next < 0.1 * alpha || next > 0.9 * alpha) {
            next = options.contraction * alpha;
        }
        alpha = next;
        if (alpha < options.min_step) {
            break;
        }
    }

    // The strict condition was not met in budget. Any genuine decrease still
    // beats standing still, so hand back the best point we saw.
    if (best.valid && best.evaluation.energy < start.energy) {
        result.ok = true;
        result.step = best.step;
        result.evaluation = best.evaluation;
        result.x = std::move(best.x);
    }
    return result;
}

LineSearchResult strong_wolfe(const ObjectiveFunction& objective, const std::vector<double>& x,
                              const Evaluation& start, const std::vector<double>& direction,
                              const LineSearchOptions& options, double slope) {
    LineSearchResult result;
    BestPoint best;

    const double phi0 = start.energy;
    const double dphi0 = slope;

    // The refinement stage of Nocedal and Wright Algorithm 3.6. The interval
    // (lo, hi) always brackets a point satisfying the strong Wolfe conditions.
    const auto zoom = [&](double lo, double phi_lo, double dphi_lo, double hi,
                          double phi_hi) -> bool {
        while (result.evaluations < options.max_evaluations) {
            double trial_step = quadratic_minimum(lo, phi_lo, dphi_lo, hi, phi_hi);
            const double low = std::min(lo, hi);
            const double high = std::max(lo, hi);
            const double margin = 0.1 * (high - low);
            if (!std::isfinite(trial_step) || trial_step < low + margin ||
                trial_step > high - margin) {
                trial_step = 0.5 * (lo + hi);
            }

            std::vector<double> trial = shifted(x, trial_step, direction);
            const Evaluation evaluation = objective(trial);
            ++result.evaluations;
            best.offer(trial_step, evaluation, trial);
            const double phi_j = evaluation.energy;
            const double dphi_j = dot(evaluation.gradient, direction);

            if (phi_j > phi0 + options.c1 * trial_step * dphi0 || phi_j >= phi_lo) {
                hi = trial_step;
                phi_hi = phi_j;
            } else {
                if (std::abs(dphi_j) <= -options.c2 * dphi0) {
                    result.ok = true;
                    result.step = trial_step;
                    result.evaluation = evaluation;
                    result.x = std::move(trial);
                    return true;
                }
                if (dphi_j * (hi - lo) >= 0.0) {
                    hi = lo;
                    phi_hi = phi_lo;
                }
                lo = trial_step;
                phi_lo = phi_j;
                dphi_lo = dphi_j;
            }

            if (std::abs(hi - lo) < options.min_step) {
                break;
            }
        }
        return false;
    };

    double previous_step = 0.0;
    double previous_phi = phi0;
    double previous_dphi = dphi0;
    double alpha = std::min(options.initial_step, options.max_step);

    // The bracketing stage, Algorithm 3.5.
    while (result.evaluations < options.max_evaluations) {
        std::vector<double> trial = shifted(x, alpha, direction);
        const Evaluation evaluation = objective(trial);
        ++result.evaluations;
        best.offer(alpha, evaluation, trial);
        const double phi_a = evaluation.energy;

        const bool violates_armijo = phi_a > phi0 + options.c1 * alpha * dphi0;
        const bool rose_again = result.evaluations > 1 && phi_a >= previous_phi;
        if (violates_armijo || rose_again) {
            if (zoom(previous_step, previous_phi, previous_dphi, alpha, phi_a)) {
                return result;
            }
            break;
        }

        const double dphi_a = dot(evaluation.gradient, direction);
        if (std::abs(dphi_a) <= -options.c2 * dphi0) {
            result.ok = true;
            result.step = alpha;
            result.evaluation = evaluation;
            result.x = std::move(trial);
            return result;
        }
        if (dphi_a >= 0.0) {
            // The slope turned uphill, so the minimum is behind us.
            if (zoom(alpha, phi_a, dphi_a, previous_step, previous_phi)) {
                return result;
            }
            break;
        }

        if (alpha >= options.max_step) {
            break;
        }
        previous_step = alpha;
        previous_phi = phi_a;
        previous_dphi = dphi_a;
        alpha = std::min(alpha * options.expansion, options.max_step);
    }

    if (best.valid && best.evaluation.energy < phi0) {
        result.ok = true;
        result.step = best.step;
        result.evaluation = best.evaluation;
        result.x = std::move(best.x);
    }
    return result;
}

}  // namespace

LineSearchKind line_search_from_string(std::string_view name) {
    std::string key;
    for (const char c : name) {
        if (c != '_' && c != '-' && c != ' ') {
            key += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    if (key.empty() || key == "none" || key == "off" || key == "false" || key == "fixed") {
        return LineSearchKind::None;
    }
    if (key == "backtracking" || key == "armijo") {
        return LineSearchKind::Backtracking;
    }
    if (key == "wolfe" || key == "strongwolfe" || key == "true") {
        return LineSearchKind::StrongWolfe;
    }
    throw OptimizerError("unknown line search '" + std::string(name) +
                         "'; use none, backtracking or wolfe");
}

std::string_view line_search_name(LineSearchKind kind) {
    switch (kind) {
        case LineSearchKind::None:         return "none";
        case LineSearchKind::Backtracking: return "backtracking";
        case LineSearchKind::StrongWolfe:  return "wolfe";
    }
    return "none";
}

LineSearchResult line_search(const ObjectiveFunction& objective, const std::vector<double>& x,
                             const Evaluation& start, const std::vector<double>& direction,
                             const LineSearchOptions& options) {
    LineSearchResult result;
    const double slope = dot(start.gradient, direction);
    if (!(slope < 0.0)) {
        // Not a descent direction: there is nothing for a line search to find.
        return result;
    }

    if (options.kind == LineSearchKind::None) {
        std::vector<double> trial = shifted(x, 1.0, direction);
        result.evaluation = objective(trial);
        result.evaluations = 1;
        result.ok = true;
        result.step = 1.0;
        result.x = std::move(trial);
        return result;
    }

    return options.kind == LineSearchKind::Backtracking
               ? backtracking(objective, x, start, direction, options, slope)
               : strong_wolfe(objective, x, start, direction, options, slope);
}

}  // namespace fastatoms::optimize
