// SPDX-License-Identifier: MIT
#pragma once

/// \file basin_hopping.hpp
/// \brief Basin hopping: searching for the global minimum, not a nearby one.
///
/// A local optimizer answers "what is the nearest minimum". Basin hopping
/// answers "what is the lowest one" by replacing the real energy landscape
/// with the landscape of its local minima: every structure is relaxed before
/// its energy is looked at, which turns the basin around each minimum into a
/// flat plateau and leaves a Monte Carlo walk with nothing to do but step
/// between basins.
///
/// The walk is Metropolis on those relaxed energies. The lowest structure ever
/// seen is kept separately from the walker's current position, because the two
/// are different things: the walk is allowed to go uphill and often should,
/// and the answer is the best it ever found rather than where it stopped.

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/optimize/optimizer.hpp"

namespace fastatoms::globalopt {

/// Raised for a search that cannot be set up or carried out.
class GlobalOptimizationError : public std::runtime_error {
public:
    explicit GlobalOptimizationError(const std::string& message)
        : std::runtime_error(message) {}
};

/// How to perturb a structure between relaxations.
enum class Move {
    Cartesian,  ///< displace every atom by an independent random vector
    Single,     ///< displace one atom, chosen at random
    Twist,      ///< rotate a random hemisphere of the cluster about a random axis
    Exchange,   ///< swap the positions of two atoms of different species
    Mixed,      ///< choose among the others at random, each step
};

[[nodiscard]] std::string_view move_name(Move move);
[[nodiscard]] Move move_from_string(std::string_view name);

/// What to search, and how hard.
struct BasinHoppingOptions {
    /// Monte Carlo steps. Each one costs a local relaxation, which is where
    /// essentially all the time goes.
    int steps{100};

    /// The Metropolis temperature in kelvin. Not a physical temperature: it
    /// sets how willing the walk is to accept a worse minimum, and the useful
    /// range is whatever makes the acceptance ratio land near a half.
    double temperature{100.0};

    /// Perturbation size in angstrom.
    double step_size{0.4};

    Move move{Move::Cartesian};

    /// Zero draws a seed from the system, which makes runs differ. Any other
    /// value makes the search exactly reproducible.
    std::uint64_t seed{0};

    /// \name Adaptive step size
    ///
    /// A step that is too small never leaves the basin and a step that is too
    /// large is always rejected; both waste the relaxation. The size is scaled
    /// towards whatever gives `target_acceptance`, which is the standard
    /// remedy and costs nothing.
    /// @{
    bool adapt_step{true};
    double target_acceptance{0.5};
    double adapt_factor{0.9};
    int adapt_interval{25};
    /// @}

    /// \name The local relaxation
    /// @{
    std::string optimizer{"lbfgs"};
    double fmax{0.05};
    int max_steps{200};
    /// @}

    /// Give up when the best energy has not improved for this many steps.
    /// Zero runs the full budget.
    int stall_limit{0};

    /// How much lower a structure must be, in eV, to count as an improvement.
    ///
    /// Reaching the same minimum twice by different relaxation paths gives
    /// energies that differ in the last bits. Counting those as improvements
    /// reports a handful of discoveries that are all the same structure, and -
    /// worse - resets `stall_limit` forever, so a search that has plainly
    /// converged never stops. The lowest structure is still kept whenever it
    /// is strictly lower; this only governs what is *called* an improvement.
    double improvement_tolerance{1e-9};

    /// Reject a perturbation that brings two atoms closer than this, in
    /// angstrom. Zero accepts anything, which for a Lennard-Jones cluster
    /// means occasionally handing the optimizer a structure with an enormous
    /// repulsion in it.
    double minimum_distance{0.0};
};

/// What one step did.
struct BasinHoppingStep {
    int step{0};
    /// The relaxed energy of the structure proposed this step.
    double energy{0.0};
    /// The energy of the state the walk is sitting on after this step.
    double current_energy{0.0};
    /// The lowest energy seen so far, over the whole run.
    double best_energy{0.0};
    bool accepted{false};
    /// True when this step set a new global best.
    bool improved{false};
    double step_size{0.0};
    int evaluations{0};

    [[nodiscard]] std::string to_string() const;
};

/// The outcome of a search.
struct BasinHoppingResult {
    /// The lowest structure found, which is what the search is for.
    Atoms best{};
    double best_energy{0.0};
    /// The energy of the structure that was handed in, after one relaxation.
    double initial_energy{0.0};

    int steps{0};
    int accepted{0};
    /// How many steps set a new global best.
    int improved{0};
    /// Force evaluations over every relaxation, which is the cost that matters.
    int evaluations{0};
    double acceptance_ratio{0.0};
    /// The step at which the best structure was found.
    int best_step{0};

    std::vector<BasinHoppingStep> history{};
    std::string message{};

    [[nodiscard]] std::string to_string() const;
};

/// Called after each step. Returning false stops the search.
using BasinHoppingCallback = std::function<bool(const BasinHoppingStep&)>;

/// Basin hopping over the local minima of an energy landscape.
///
/// ```
/// BasinHoppingOptions options;
/// options.steps = 200;
/// options.temperature = 100.0;
///
/// BasinHopping search(options);
/// const BasinHoppingResult result = search.run(cluster);
/// // `cluster` now holds the best structure found.
/// ```
class BasinHopping {
public:
    explicit BasinHopping(BasinHoppingOptions options = {});

    [[nodiscard]] const BasinHoppingOptions& options() const noexcept { return options_; }

    /// Watch the search. Returning false from the callback stops it.
    void attach(BasinHoppingCallback callback);

    /// Search, leaving `atoms` holding the best structure found.
    ///
    /// The structure is relaxed once before the walk starts, so the reported
    /// initial energy is a minimum and not whatever was handed in.
    ///
    /// \throws GlobalOptimizationError if the structure is empty or has no
    ///         calculator, or if the named local optimizer does not exist
    BasinHoppingResult run(Atoms& atoms);

    /// Apply one perturbation, without relaxing. Exposed because it is worth
    /// being able to look at what the move actually does.
    void perturb(Atoms& atoms, std::uint64_t seed) const;

private:
    BasinHoppingOptions options_;
    BasinHoppingCallback callback_{};
};

}  // namespace fastatoms::globalopt
