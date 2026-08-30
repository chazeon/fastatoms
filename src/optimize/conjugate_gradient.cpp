// SPDX-License-Identifier: MIT
#include "fastatoms/optimize/conjugate_gradient.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

#include "fastatoms/optimize/vector_math.hpp"

namespace fastatoms::optimize {
namespace {

std::string canonical(std::string_view name) {
    std::string key;
    for (const char c : name) {
        if (c != '_' && c != '-' && c != ' ' && c != '+') {
            key += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    return key;
}

}  // namespace

ConjugateGradientVariant cg_variant_from_string(std::string_view name) {
    const std::string key = canonical(name);
    if (key.empty() || key == "polakribiere" || key == "pr" || key == "prplus") {
        return ConjugateGradientVariant::PolakRibiere;
    }
    if (key == "fletcherreeves" || key == "fr") {
        return ConjugateGradientVariant::FletcherReeves;
    }
    if (key == "hestenesstiefel" || key == "hs") {
        return ConjugateGradientVariant::HestenesStiefel;
    }
    throw OptimizerError("unknown conjugate gradient variant '" + std::string(name) +
                         "'; use polak_ribiere, fletcher_reeves or hestenes_stiefel");
}

std::string_view cg_variant_name(ConjugateGradientVariant variant) {
    switch (variant) {
        case ConjugateGradientVariant::PolakRibiere:    return "polak_ribiere";
        case ConjugateGradientVariant::FletcherReeves:  return "fletcher_reeves";
        case ConjugateGradientVariant::HestenesStiefel: return "hestenes_stiefel";
    }
    return "polak_ribiere";
}

ConjugateGradientVariant ConjugateGradient::variant() const {
    const Value& value = parameter("variant");
    if (value.is_null()) {
        return ConjugateGradientVariant::PolakRibiere;
    }
    return cg_variant_from_string(value.as_string());
}

int ConjugateGradient::restart_period() const {
    const auto value = static_cast<int>(parameter("restart").value_or(std::int64_t{0}));
    if (value < 0) {
        fail("'restart' must not be negative");
    }
    return value;
}

void ConjugateGradient::begin(std::size_t ndof) {
    ndof_ = ndof;
    restart();
}

void ConjugateGradient::restart() {
    previous_gradient_.clear();
    previous_direction_.clear();
    has_history_ = false;
    since_restart_ = 0;
}

void ConjugateGradient::accept(const std::vector<double>&, const std::vector<double>&) {
    // The state conjugate gradient needs is captured in `direction`, which
    // sees both the previous and the current gradient.
}

std::vector<double> ConjugateGradient::direction(const std::vector<double>&,
                                                 const std::vector<double>& gradient) {
    const int period = restart_period() > 0 ? restart_period() : static_cast<int>(ndof_);

    // Periodic restarts stop stale curvature information from accumulating on
    // a surface that is not really quadratic.
    bool force_restart = has_history_ && period > 0 && since_restart_ >= period;

    // Powell's test: conjugacy only means something while successive gradients
    // stay close to orthogonal. Once they are not, the mixed direction rests on
    // a model that no longer holds.
    if (has_history_ && !force_restart) {
        const double overlap = std::abs(dot(gradient, previous_gradient_));
        if (overlap >= 0.2 * dot(gradient, gradient)) {
            force_restart = true;
        }
    }
    if (force_restart) {
        restart();
    }

    // The recursion is carried in the textbook form, d = -g + beta d_prev, and
    // only converted to a displacement on the way out. Mixing the 1/alpha
    // scaling into the recursion would be wrong for Hestenes-Stiefel, whose
    // beta depends on d_prev and so does not carry the scaling through.
    std::vector<double> d = scaled(-1.0, gradient);

    if (has_history_) {
        const double previous_norm2 = dot(previous_gradient_, previous_gradient_);
        if (previous_norm2 > 0.0) {
            const std::vector<double> y = subtract(gradient, previous_gradient_);
            double beta = 0.0;

            switch (variant()) {
                case ConjugateGradientVariant::FletcherReeves:
                    beta = dot(gradient, gradient) / previous_norm2;
                    break;
                case ConjugateGradientVariant::PolakRibiere:
                    // The max(0, .) clamp turns a negative beta into an
                    // automatic restart, which is what makes PR+ reliable.
                    beta = std::max(0.0, dot(gradient, y) / previous_norm2);
                    break;
                case ConjugateGradientVariant::HestenesStiefel: {
                    const double denominator = dot(previous_direction_, y);
                    beta = std::abs(denominator) > 0.0 ? dot(gradient, y) / denominator : 0.0;
                    beta = std::max(0.0, beta);
                    break;
                }
            }

            if (std::isfinite(beta) && beta != 0.0) {
                axpy(beta, previous_direction_, d);
                // Mixing can tip the direction uphill; fall back if it does.
                if (dot(d, gradient) >= 0.0) {
                    d = scaled(-1.0, gradient);
                    since_restart_ = 0;
                }
            }
        }
    }

    previous_gradient_ = gradient;
    previous_direction_ = d;
    has_history_ = true;
    ++since_restart_;

    // Turn the direction into a displacement; the line search sets the length
    // from here.
    return scaled(1.0 / alpha(), d);
}

}  // namespace fastatoms::optimize
