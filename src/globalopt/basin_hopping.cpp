// SPDX-License-Identifier: MIT

#include "fastatoms/globalopt/basin_hopping.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <sstream>

#include "fastatoms/core/units.hpp"
#include "fastatoms/optimize/registry.hpp"

namespace fastatoms::globalopt {
namespace {

/// The shortest distance between any two atoms, ignoring periodicity: these
/// are clusters.
double closest_pair(const Atoms& atoms) {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        for (std::size_t j = i + 1; j < atoms.size(); ++j) {
            best = std::min(best, (atoms.position(i) - atoms.position(j)).norm());
        }
    }
    return best;
}

}  // namespace

std::string_view move_name(Move move) {
    switch (move) {
        case Move::Cartesian: return "cartesian";
        case Move::Single:    return "single";
        case Move::Twist:     return "twist";
        case Move::Exchange:  return "exchange";
        case Move::Mixed:     return "mixed";
    }
    return "cartesian";
}

Move move_from_string(std::string_view name) {
    std::string key(name);
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    key.erase(std::remove_if(key.begin(), key.end(),
                             [](unsigned char c) { return c == '-' || c == '_' || c == ' '; }),
              key.end());
    if (key == "cartesian" || key == "displace" || key == "all") return Move::Cartesian;
    if (key == "single" || key == "oneatom") return Move::Single;
    if (key == "twist" || key == "rotate") return Move::Twist;
    if (key == "exchange" || key == "swap") return Move::Exchange;
    if (key == "mixed" || key == "random") return Move::Mixed;
    throw GlobalOptimizationError("'" + std::string(name) +
                                  "' is not a move; use cartesian, single, twist, exchange "
                                  "or mixed");
}

std::string BasinHoppingStep::to_string() const {
    std::ostringstream out;
    out << "step " << step << ": E = " << energy << (accepted ? " accepted" : " rejected")
        << (improved ? " (new best)" : "") << ", best " << best_energy;
    return out.str();
}

std::string BasinHoppingResult::to_string() const {
    std::ostringstream out;
    out << "BasinHoppingResult(best " << best_energy << " eV at step " << best_step << ", "
        << steps << " steps, " << accepted << " accepted, " << evaluations << " evaluations)";
    return out.str();
}

BasinHopping::BasinHopping(BasinHoppingOptions options) : options_(std::move(options)) {
    if (options_.steps < 0) {
        throw GlobalOptimizationError("the number of steps must not be negative");
    }
    if (options_.temperature < 0.0) {
        throw GlobalOptimizationError("the temperature must not be negative");
    }
    if (!(options_.step_size > 0.0)) {
        throw GlobalOptimizationError("the step size must be positive");
    }
    if (options_.adapt_step &&
        (!(options_.adapt_factor > 0.0) || options_.adapt_factor >= 1.0)) {
        throw GlobalOptimizationError("the adaptation factor must lie between zero and one");
    }
}

void BasinHopping::attach(BasinHoppingCallback callback) { callback_ = std::move(callback); }

void BasinHopping::perturb(Atoms& atoms, std::uint64_t seed) const {
    if (atoms.empty()) {
        throw GlobalOptimizationError("there is nothing to perturb");
    }
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    Move move = options_.move;
    if (move == Move::Mixed) {
        // Exchange only makes sense when there is more than one species to
        // exchange, so it is offered only then.
        bool mixed_species = false;
        for (std::size_t k = 1; k < atoms.size(); ++k) {
            mixed_species = mixed_species || atoms.number(k) != atoms.number(0);
        }
        const int choices = mixed_species ? 4 : 3;
        const int pick = static_cast<int>(unit(rng) * choices);
        move = pick == 0   ? Move::Cartesian
               : pick == 1 ? Move::Single
               : pick == 2 ? Move::Twist
                           : Move::Exchange;
    }

    std::vector<double> positions = atoms.positions();
    const std::size_t n = atoms.size();

    switch (move) {
        case Move::Cartesian: {
            for (double& value : positions) {
                value += options_.step_size * uniform(rng);
            }
            break;
        }
        case Move::Single: {
            const auto which = static_cast<std::size_t>(unit(rng) * n) % n;
            for (std::size_t c = 0; c < 3; ++c) {
                positions[3 * which + c] += options_.step_size * uniform(rng);
            }
            break;
        }
        case Move::Twist: {
            // Turn everything on one side of a random plane through the centre
            // about a random axis. A cluster stuck in a funnel often cannot be
            // shaken out of it by independent displacements, but a coordinated
            // twist of half of it rearranges the packing outright.
            const Vec3 centre = atoms.center_of_mass();
            Vec3 axis{uniform(rng), uniform(rng), uniform(rng)};
            const double length = axis.norm();
            if (!(length > 1e-9)) {
                break;
            }
            axis = Vec3{axis.x / length, axis.y / length, axis.z / length};
            Vec3 plane{uniform(rng), uniform(rng), uniform(rng)};
            const double plane_length = plane.norm();
            if (!(plane_length > 1e-9)) {
                break;
            }
            plane = Vec3{plane.x / plane_length, plane.y / plane_length,
                         plane.z / plane_length};

            // The angle is scaled by the step size so that adapting it adapts
            // this move too.
            const double angle = options_.step_size * uniform(rng) * 3.14159265358979323846;
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);

            for (std::size_t k = 0; k < n; ++k) {
                const Vec3 relative{positions[3 * k] - centre.x, positions[3 * k + 1] - centre.y,
                                    positions[3 * k + 2] - centre.z};
                if (dot(relative, plane) < 0.0) {
                    continue;   // the other half stays put
                }
                // Rodrigues, for a row vector.
                const Vec3 crossed = cross(axis, relative);
                const double along = dot(axis, relative);
                const Vec3 turned{
                    relative.x * cosine + crossed.x * sine + axis.x * along * (1.0 - cosine),
                    relative.y * cosine + crossed.y * sine + axis.y * along * (1.0 - cosine),
                    relative.z * cosine + crossed.z * sine + axis.z * along * (1.0 - cosine)};
                positions[3 * k] = centre.x + turned.x;
                positions[3 * k + 1] = centre.y + turned.y;
                positions[3 * k + 2] = centre.z + turned.z;
            }
            break;
        }
        case Move::Exchange: {
            // Swap two atoms of different species. In an alloy cluster the
            // ordering matters as much as the shape, and no amount of
            // displacement explores it.
            std::vector<std::pair<std::size_t, std::size_t>> pairs;
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = i + 1; j < n; ++j) {
                    if (atoms.number(i) != atoms.number(j)) {
                        pairs.emplace_back(i, j);
                    }
                }
            }
            if (pairs.empty()) {
                break;   // one species: there is nothing to exchange
            }
            const auto& [i, j] = pairs[static_cast<std::size_t>(unit(rng) * pairs.size()) %
                                       pairs.size()];
            for (std::size_t c = 0; c < 3; ++c) {
                std::swap(positions[3 * i + c], positions[3 * j + c]);
            }
            break;
        }
        case Move::Mixed:
            break;   // resolved above
    }

    // Through `move_to`, so any constraints on the structure get their say:
    // a frozen substrate stays frozen under a perturbation as much as under a
    // relaxation.
    atoms.move_to(positions);
}

BasinHoppingResult BasinHopping::run(Atoms& atoms) {
    if (atoms.empty()) {
        throw GlobalOptimizationError("there is nothing to optimise");
    }
    if (!atoms.has_calculator()) {
        throw GlobalOptimizationError(
            "basin hopping needs a calculator: every step relaxes the structure, and a "
            "relaxation needs forces");
    }

    const auto make_local = [&] {
        return optimize::make_optimizer(
            options_.optimizer,
            Value(Value::Map{
                {"fmax", Value(options_.fmax)},
                {"max_steps", Value(static_cast<std::int64_t>(options_.max_steps))}}));
    };
    // Built once up front so a bad optimizer name fails before any work.
    auto local = make_local();

    BasinHoppingResult result;
    std::mt19937_64 rng(options_.seed != 0 ? options_.seed : std::random_device{}());
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    // The walk starts from a minimum, not from whatever was handed in, so the
    // first Metropolis comparison is between like and like.
    const optimize::OptimizationResult first = local->run(atoms);
    result.evaluations += first.evaluations;

    Atoms current = atoms;
    double current_energy = current.get_potential_energy();
    result.initial_energy = current_energy;

    result.best = current;
    result.best_energy = current_energy;
    result.best_step = 0;

    const double kT = units::kB * options_.temperature;
    double step_size = options_.step_size;
    const double configured = options_.step_size;

    int since_improvement = 0;
    int accepted_in_window = 0;
    int window = 0;

    for (int step = 1; step <= options_.steps; ++step) {
        Atoms trial = current;

        // Perturb, refusing anything that has put two atoms on top of each
        // other; the relaxation from such a structure is wasted work.
        BasinHopping mover(*this);
        mover.options_.step_size = step_size;
        int attempts = 0;
        do {
            trial = current;
            mover.perturb(trial, rng());
            ++attempts;
        } while (options_.minimum_distance > 0.0 && trial.size() > 1 &&
                 closest_pair(trial) < options_.minimum_distance && attempts < 50);

        auto relaxer = make_local();
        const optimize::OptimizationResult relaxed = relaxer->run(trial);
        const double energy = trial.get_potential_energy();
        result.evaluations += relaxed.evaluations;

        // Metropolis on the relaxed energies. Downhill is always taken; uphill
        // with probability exp(-dE/kT), which at zero temperature is never.
        const double difference = energy - current_energy;
        bool accept = difference <= 0.0;
        if (!accept && kT > 0.0) {
            accept = unit(rng) < std::exp(-difference / kT);
        }

        BasinHoppingStep record;
        record.step = step;
        record.energy = energy;
        record.accepted = accept;
        record.step_size = step_size;
        record.evaluations = relaxed.evaluations;

        if (accept) {
            current = trial;
            current_energy = energy;
            ++result.accepted;
            ++accepted_in_window;
        }
        // The best is tracked separately from the walker, because the walk is
        // allowed to go uphill and the answer is the lowest it ever saw rather
        // than where it happens to be standing.
        //
        // Whether it counts as an *improvement* is a separate question: two
        // relaxations into the same minimum differ in the last bits, and
        // treating that as a discovery both miscounts them and stops
        // `stall_limit` from ever firing.
        const bool meaningful = energy < result.best_energy - options_.improvement_tolerance;
        if (energy < result.best_energy) {
            result.best = trial;
            result.best_energy = energy;
            if (meaningful) {
                result.best_step = step;
            }
        }
        if (meaningful) {
            record.improved = true;
            ++result.improved;
            since_improvement = 0;
        } else {
            ++since_improvement;
        }
        record.current_energy = current_energy;
        record.best_energy = result.best_energy;
        result.history.push_back(record);
        result.steps = step;

        if (callback_ && !callback_(record)) {
            result.message = "stopped by the callback after " + std::to_string(step) + " steps";
            break;
        }

        // Scale the step towards the acceptance ratio that was asked for.
        ++window;
        if (options_.adapt_step && window >= options_.adapt_interval) {
            const double ratio = static_cast<double>(accepted_in_window) / window;
            step_size *= ratio > options_.target_acceptance ? 1.0 / options_.adapt_factor
                                                            : options_.adapt_factor;
            // Kept within two orders of magnitude of what was asked for, so a
            // pathological run cannot drive it to zero or to nonsense.
            step_size = std::clamp(step_size, 0.01 * configured, 100.0 * configured);
            accepted_in_window = 0;
            window = 0;
        }

        if (options_.stall_limit > 0 && since_improvement >= options_.stall_limit) {
            result.message = "stopped after " + std::to_string(since_improvement) +
                             " steps without improvement";
            break;
        }
    }

    result.acceptance_ratio =
        result.steps > 0 ? static_cast<double>(result.accepted) / result.steps : 0.0;
    if (result.message.empty()) {
        result.message = "completed " + std::to_string(result.steps) + " steps";
    }

    // Hand back the best structure, which is the whole point.
    atoms = result.best;
    return result;
}

}  // namespace fastatoms::globalopt
