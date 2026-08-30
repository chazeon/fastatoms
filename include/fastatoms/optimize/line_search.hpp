// SPDX-License-Identifier: MIT
#pragma once

/// \file line_search.hpp
/// \brief Step-length selection along a search direction.

#include <string_view>
#include <vector>

#include "fastatoms/optimize/types.hpp"

namespace fastatoms::optimize {

/// How far to move along a search direction.
enum class LineSearchKind {
    /// Take the direction as proposed, capped by the trust radius. No extra
    /// objective evaluations - the right default when each one is a DFT run.
    None,
    /// Backtrack until the Armijo sufficient-decrease condition holds.
    Backtracking,
    /// Satisfy the strong Wolfe conditions. More evaluations, but it is what
    /// keeps conjugate gradient conjugate and guarantees a positive curvature
    /// condition for the quasi-Newton updates.
    StrongWolfe,
};

[[nodiscard]] LineSearchKind line_search_from_string(std::string_view name);
[[nodiscard]] std::string_view line_search_name(LineSearchKind kind);

struct LineSearchOptions {
    LineSearchKind kind{LineSearchKind::None};
    double c1{1e-4};              ///< Armijo (sufficient decrease) constant
    double c2{0.9};               ///< curvature constant; 0.1 is usual for CG
    double initial_step{1.0};     ///< first trial scale along the direction
    double max_step{10.0};        ///< largest trial scale
    double min_step{1e-10};       ///< give up below this
    double contraction{0.5};      ///< backtracking factor
    double expansion{2.0};        ///< bracketing growth factor
    int max_evaluations{20};      ///< objective evaluations allowed per search
};

/// What a line search found.
struct LineSearchResult {
    bool ok{false};
    double step{0.0};             ///< accepted scale factor along the direction
    Evaluation evaluation{};      ///< objective at the accepted point
    std::vector<double> x{};      ///< the accepted point
    int evaluations{0};
};

/// Find a step length along `direction` from `x`.
///
/// `start` must already hold the objective at `x`. `direction` must be a
/// descent direction; the caller is responsible for that.
///
/// Both algorithms follow Nocedal and Wright, "Numerical Optimization":
/// backtracking is the standard Armijo loop, and the strong Wolfe search is
/// the bracketing procedure of Algorithm 3.5 with the `zoom` refinement of
/// Algorithm 3.6, using cubic interpolation where it is well behaved and
/// bisection otherwise.
[[nodiscard]] LineSearchResult line_search(const ObjectiveFunction& objective,
                                           const std::vector<double>& x,
                                           const Evaluation& start,
                                           const std::vector<double>& direction,
                                           const LineSearchOptions& options);

}  // namespace fastatoms::optimize
