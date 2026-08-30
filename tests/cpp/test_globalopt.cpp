// SPDX-License-Identifier: MIT
///
/// Basin hopping. Checked against the Lennard-Jones clusters, whose global
/// minima have been known and tabulated for decades: a search that cannot find
/// those is not a search.

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "fastatoms/fastatoms.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::globalopt;

namespace {

/// Lennard-Jones in reduced units, unshifted, so energies can be compared with
/// the published cluster minima directly.
std::shared_ptr<Calculator> reduced_lennard_jones() {
    return std::make_shared<LennardJonesCalculator>(
        Value(Value::Map{{"epsilon", Value(1.0)},
                         {"sigma", Value(1.0)},
                         {"cutoff", Value(100.0)},
                         {"shift", Value(false)}}));
}

/// `n` atoms at random positions in a box that roughly fits them.
Atoms random_cluster(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const double half = 0.6 * std::cbrt(static_cast<double>(n));
    std::uniform_real_distribution<double> uniform(-half, half);
    std::vector<double> positions;
    for (std::size_t k = 0; k < 3 * n; ++k) {
        positions.push_back(uniform(rng));
    }
    Atoms atoms(std::vector<std::string>(n, "Ar"), positions);
    atoms.set_calculator(reduced_lennard_jones());
    return atoms;
}

BasinHoppingOptions searching(int steps, std::uint64_t seed) {
    BasinHoppingOptions options;
    options.steps = steps;
    options.temperature = 8000.0;
    options.step_size = 0.45;
    options.seed = seed;
    options.fmax = 1e-4;
    options.max_steps = 500;
    options.minimum_distance = 0.6;
    return options;
}

}  // namespace

// ---------------------------------------------------------------------------
// It finds the answer
// ---------------------------------------------------------------------------

FA_TEST(the_known_lennard_jones_minima_are_found) {
    // The published global minima, in units of epsilon. These have been in the
    // literature since the seventies and nothing here knows them.
    struct Case { std::size_t atoms; double energy; int steps; };
    const std::vector<Case> cases{
        {7, -16.505384, 120},
        {13, -44.326801, 400},
        {19, -72.659782, 400},
    };
    for (const Case& one : cases) {
        Atoms cluster = random_cluster(one.atoms, 7);
        BasinHopping search(searching(one.steps, 12345));
        const BasinHoppingResult result = search.run(cluster);

        FA_CHECK_CLOSE(result.best_energy, one.energy, 1e-4);
        // And the structure handed back is the one with that energy, not the
        // walker's last position.
        FA_CHECK_CLOSE(cluster.get_potential_energy(), one.energy, 1e-4);
        FA_CHECK(result.best_step > 0);
    }
}

FA_TEST(it_escapes_the_minimum_it_started_in) {
    // The point of the whole algorithm. A local optimizer relaxes this start
    // into a minimum that is not the global one; basin hopping has to get out
    // of it.
    Atoms cluster = random_cluster(13, 7);
    BasinHopping search(searching(400, 12345));
    const BasinHoppingResult result = search.run(cluster);

    FA_CHECK(result.initial_energy > -44.0);          // the local minimum it fell into
    FA_CHECK(result.best_energy < result.initial_energy - 2.0);
    FA_CHECK_CLOSE(result.best_energy, -44.326801, 1e-4);

    // Exactly one improvement, not several. Reaching the same minimum again by
    // a different relaxation path gives an energy that differs in the last
    // bits, and counting those as discoveries both miscounts them and stops a
    // stall limit from ever firing.
    FA_CHECK(result.improved == 1);
}

FA_TEST(the_same_seed_gives_the_same_search) {
    // Reproducibility, which a stochastic method needs more than a
    // deterministic one does.
    Atoms first = random_cluster(7, 3);
    Atoms second = random_cluster(7, 3);
    const BasinHoppingResult a = BasinHopping(searching(60, 999)).run(first);
    const BasinHoppingResult b = BasinHopping(searching(60, 999)).run(second);

    FA_CHECK_CLOSE(a.best_energy, b.best_energy, 1e-15);
    FA_CHECK(a.accepted == b.accepted);
    FA_CHECK(a.best_step == b.best_step);
    FA_CHECK(a.history.size() == b.history.size());
    for (std::size_t k = 0; k < a.history.size(); ++k) {
        FA_CHECK_CLOSE(a.history[k].energy, b.history[k].energy, 1e-15);
    }
}

// ---------------------------------------------------------------------------
// The Metropolis rule
// ---------------------------------------------------------------------------

FA_TEST(at_zero_temperature_nothing_uphill_is_accepted) {
    Atoms cluster = random_cluster(13, 5);
    BasinHoppingOptions options = searching(60, 4242);
    options.temperature = 0.0;
    options.adapt_step = false;

    BasinHopping search(options);
    const BasinHoppingResult result = search.run(cluster);

    // Every accepted step went downhill, so the walk's energy never rises.
    double previous = result.initial_energy;
    for (const BasinHoppingStep& step : result.history) {
        if (step.accepted) {
            FA_CHECK(step.energy <= previous + 1e-12);
        }
        FA_CHECK(step.current_energy <= previous + 1e-12);
        previous = step.current_energy;
    }
    FA_CHECK(result.accepted < result.steps);   // some were rejected
}

FA_TEST(a_hot_search_accepts_more_than_a_cold_one) {
    // What the temperature is for: it buys uphill moves, and more of them.
    const auto acceptance = [](double temperature) {
        Atoms cluster = random_cluster(13, 11);
        BasinHoppingOptions options = searching(80, 777);
        options.temperature = temperature;
        options.adapt_step = false;   // so only the temperature differs
        return BasinHopping(options).run(cluster).acceptance_ratio;
    };
    FA_CHECK(acceptance(50000.0) > acceptance(10.0));
}

FA_TEST(the_best_is_kept_apart_from_the_walker) {
    // The walk is allowed to wander uphill; the answer is the lowest it ever
    // saw, not where it stopped. So the best can only fall, and always sits at
    // or below the current state.
    Atoms cluster = random_cluster(13, 2);
    const BasinHoppingResult result = BasinHopping(searching(100, 31337)).run(cluster);

    double previous_best = result.initial_energy;
    for (const BasinHoppingStep& step : result.history) {
        FA_CHECK(step.best_energy <= previous_best + 1e-12);
        FA_CHECK(step.best_energy <= step.current_energy + 1e-12);
        previous_best = step.best_energy;
    }
    FA_CHECK_CLOSE(result.history.back().best_energy, result.best_energy, 1e-15);
}

// ---------------------------------------------------------------------------
// The moves
// ---------------------------------------------------------------------------

FA_TEST(every_move_moves_something) {
    const std::vector<Move> moves{Move::Cartesian, Move::Single, Move::Twist, Move::Mixed};
    for (const Move move : moves) {
        BasinHoppingOptions options;
        options.move = move;
        options.step_size = 0.5;
        const BasinHopping search(options);

        Atoms cluster = random_cluster(13, 1);
        const std::vector<double> before = cluster.positions();
        search.perturb(cluster, 4242);

        double moved = 0.0;
        for (std::size_t k = 0; k < before.size(); ++k) {
            moved = std::max(moved, std::abs(cluster.positions()[k] - before[k]));
        }
        FA_CHECK(moved > 1e-6);
    }
}

FA_TEST(a_single_atom_move_moves_a_single_atom) {
    BasinHoppingOptions options;
    options.move = Move::Single;
    options.step_size = 0.5;
    const BasinHopping search(options);

    Atoms cluster = random_cluster(13, 1);
    const std::vector<double> before = cluster.positions();
    search.perturb(cluster, 99);

    int moved = 0;
    for (std::size_t k = 0; k < cluster.size(); ++k) {
        const Vec3 shift{cluster.positions()[3 * k] - before[3 * k],
                         cluster.positions()[3 * k + 1] - before[3 * k + 1],
                         cluster.positions()[3 * k + 2] - before[3 * k + 2]};
        if (shift.norm() > 1e-9) {
            ++moved;
        }
    }
    FA_CHECK(moved == 1);
}

FA_TEST(an_exchange_swaps_two_species_and_leaves_the_positions_alone) {
    // The move an alloy needs: the shape may already be right and the ordering
    // wrong, and no displacement explores that.
    Atoms alloy = random_cluster(12, 8);
    for (std::size_t k = 0; k < 6; ++k) {
        alloy.set_number(k, 28);   // half nickel, half argon
    }
    BasinHoppingOptions options;
    options.move = Move::Exchange;
    const BasinHopping search(options);

    const std::vector<double> before = alloy.positions();
    search.perturb(alloy, 5);

    // The set of positions is unchanged; only which species sits where.
    std::vector<double> sorted_before = before;
    std::vector<double> sorted_after = alloy.positions();
    std::sort(sorted_before.begin(), sorted_before.end());
    std::sort(sorted_after.begin(), sorted_after.end());
    for (std::size_t k = 0; k < sorted_before.size(); ++k) {
        FA_CHECK_CLOSE(sorted_before[k], sorted_after[k], 1e-12);
    }
    int swapped = 0;
    for (std::size_t k = 0; k < alloy.size(); ++k) {
        const Vec3 shift{alloy.positions()[3 * k] - before[3 * k],
                         alloy.positions()[3 * k + 1] - before[3 * k + 1],
                         alloy.positions()[3 * k + 2] - before[3 * k + 2]};
        if (shift.norm() > 1e-9) {
            ++swapped;
        }
    }
    FA_CHECK(swapped == 2);
}

FA_TEST(a_perturbation_respects_the_constraints) {
    // A frozen atom stays frozen under a Monte Carlo move as much as under a
    // relaxation, because the move goes through `move_to`.
    Atoms cluster = random_cluster(13, 4);
    cluster.set_constraints({std::make_shared<FixAtoms>(std::vector<std::size_t>{0, 1})});
    const std::vector<double> before = cluster.positions();

    BasinHoppingOptions options;
    options.step_size = 0.6;
    BasinHopping(options).perturb(cluster, 7);

    for (std::size_t c = 0; c < 6; ++c) {
        FA_CHECK_CLOSE(cluster.positions()[c], before[c], 1e-15);
    }
    FA_CHECK(std::abs(cluster.positions()[6] - before[6]) > 1e-9);
}

FA_TEST(move_names_round_trip) {
    for (const Move move : {Move::Cartesian, Move::Single, Move::Twist, Move::Exchange,
                            Move::Mixed}) {
        FA_CHECK(move_from_string(move_name(move)) == move);
    }
    FA_CHECK(move_from_string("swap") == Move::Exchange);
    FA_CHECK_THROWS(move_from_string("teleport"), GlobalOptimizationError);
}

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

FA_TEST(a_callback_can_watch_and_stop) {
    Atoms cluster = random_cluster(13, 6);
    BasinHopping search(searching(200, 1));

    int seen = 0;
    search.attach([&](const BasinHoppingStep& step) {
        ++seen;
        FA_CHECK(step.step == seen);
        return seen < 20;
    });
    const BasinHoppingResult result = search.run(cluster);

    FA_CHECK(seen == 20);
    FA_CHECK(result.steps == 20);
    FA_CHECK(result.message.find("callback") != std::string::npos);
}

FA_TEST(a_stalled_search_gives_up) {
    Atoms cluster = random_cluster(7, 9);
    BasinHoppingOptions options = searching(500, 3);
    options.stall_limit = 15;
    const BasinHoppingResult result = BasinHopping(options).run(cluster);

    FA_CHECK(result.steps < 500);
    FA_CHECK(result.message.find("without improvement") != std::string::npos);
    // It still found the answer before giving up.
    FA_CHECK_CLOSE(result.best_energy, -16.505384, 1e-4);
}

FA_TEST(the_step_size_adapts_towards_the_acceptance_asked_for) {
    Atoms cluster = random_cluster(13, 12);
    BasinHoppingOptions options = searching(150, 8);
    options.step_size = 0.02;          // far too small: almost everything accepted
    options.adapt_interval = 10;
    options.target_acceptance = 0.5;

    const BasinHoppingResult result = BasinHopping(options).run(cluster);
    // Starting from a step that is accepted too often, adaptation grows it.
    FA_CHECK(result.history.back().step_size > result.history.front().step_size);
}

FA_TEST(the_search_refuses_what_it_cannot_do) {
    Atoms empty;
    FA_CHECK_THROWS(BasinHopping().run(empty), GlobalOptimizationError);

    Atoms without_calculator(std::vector<std::string>{"Ar", "Ar"},
                             {0.0, 0.0, 0.0, 3.8, 0.0, 0.0});
    FA_CHECK_THROWS(BasinHopping().run(without_calculator), GlobalOptimizationError);

    BasinHoppingOptions bad_step;
    bad_step.step_size = -1.0;
    FA_CHECK_THROWS(BasinHopping(bad_step), GlobalOptimizationError);

    BasinHoppingOptions bad_temperature;
    bad_temperature.temperature = -5.0;
    FA_CHECK_THROWS(BasinHopping(bad_temperature), GlobalOptimizationError);

    BasinHoppingOptions bad_optimizer;
    bad_optimizer.optimizer = "newton";
    Atoms cluster = random_cluster(7, 1);
    FA_CHECK_THROWS(BasinHopping(bad_optimizer).run(cluster), optimize::OptimizerError);
}

FA_TEST_MAIN()
