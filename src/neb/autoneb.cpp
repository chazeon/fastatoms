// SPDX-License-Identifier: MIT
#include "fastatoms/neb/autoneb.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

#include "fastatoms/optimize/registry.hpp"
#include "fastatoms/util/text.hpp"

namespace fastatoms::neb {
namespace {

double distance_between(const Atoms& a, const Atoms& b) {
    const std::vector<double>& from = a.positions();
    const std::vector<double>& to = b.positions();
    double total = 0.0;
    for (std::size_t k = 0; k < from.size(); ++k) {
        const double d = to[k] - from[k];
        total += d * d;
    }
    return std::sqrt(total);
}

/// A structure on the straight line between two others.
Atoms blend(const Atoms& a, const Atoms& b, double fraction) {
    Atoms out = a;
    const std::vector<double>& from = a.positions();
    const std::vector<double>& to = b.positions();
    std::vector<double> positions(from.size());
    for (std::size_t k = 0; k < from.size(); ++k) {
        positions[k] = from[k] + fraction * (to[k] - from[k]);
    }
    out.set_positions(positions);
    return out;
}

}  // namespace

std::string_view insertion_name(Insertion where) {
    return where == Insertion::Distance ? "distance" : "energy";
}

Insertion insertion_from_string(std::string_view name) {
    const std::string key = util::lowercase(name);
    if (key == "energy") {
        return Insertion::Energy;
    }
    if (key == "distance" || key == "spacing") {
        return Insertion::Distance;
    }
    throw NEBError("autoneb: unknown insertion rule '" + std::string(name) +
                   "'; expected energy or distance");
}

std::string AutoNEBResult::to_string() const {
    std::ostringstream out;
    out.precision(6);
    out << "AutoNEBResult(" << (converged ? "converged" : "not converged") << ", "
        << profile.energies.size() << " images, " << stages << " stages, " << steps
        << " steps, " << evaluations << " evaluations, barrier " << profile.forward_barrier
        << " eV)";
    return out.str();
}

OptimizerFactory optimizer_factory(const std::string& algorithm, Value parameters) {
    return [algorithm, parameters]() {
        Value settings = parameters.is_map() ? parameters : Value(Value::Map{});
        // Forced, not defaulted: a line search on a force that is not a
        // gradient searches for a decrease in something the force knows
        // nothing about.
        settings.insert("line_search", Value(std::string("none")));
        return optimize::make_optimizer(algorithm, settings);
    };
}

// ---------------------------------------------------------------------------
AutoNEB::AutoNEB(Atoms initial, Atoms final, AutoNEBParameters parameters,
                 NEBParameters band)
    : parameters_(parameters), band_parameters_(band) {
    if (parameters_.initial_images < 3) {
        throw NEBError("autoneb: a band needs at least three images to start from");
    }
    if (parameters_.max_images < parameters_.initial_images) {
        throw NEBError("autoneb: max_images (" + std::to_string(parameters_.max_images) +
                       ") is below initial_images (" +
                       std::to_string(parameters_.initial_images) + ")");
    }
    // The band grows, so it starts without climbing whatever was asked for:
    // the final stage turns it on.
    NEBParameters growing = band_parameters_;
    growing.climbing = false;
    band_ = std::make_unique<NEB>(
        NEB::from_endpoints(initial, final, parameters_.initial_images - 2, growing));
}

NEB& AutoNEB::band() { return *band_; }
const NEB& AutoNEB::band() const { return *band_; }

std::size_t AutoNEB::widest_gap() {
    // Which neighbouring pair the band resolves worst. Energy by default: a
    // large jump between two images means the profile between them is a guess,
    // and near a barrier that guess is the part that matters.
    const std::vector<Atoms>& images = band_->images();
    const std::vector<double> energies =
        parameters_.insert == Insertion::Energy ? band_->profile().energies
                                                : std::vector<double>{};

    std::size_t best = 0;
    double worst = -1.0;
    for (std::size_t k = 0; k + 1 < images.size(); ++k) {
        const double measure = parameters_.insert == Insertion::Distance
                                   ? distance_between(images[k], images[k + 1])
                                   : std::abs(energies[k + 1] - energies[k]);
        if (measure > worst) {
            worst = measure;
            best = k;
        }
    }
    return best;
}

std::size_t AutoNEB::insert_image() {
    if (band_->size() >= parameters_.max_images) {
        return band_->size();
    }
    const std::size_t gap = widest_gap();
    std::vector<Atoms> images = band_->images();
    Atoms inserted = blend(images[gap], images[gap + 1], 0.5);
    images.insert(images.begin() + static_cast<std::ptrdiff_t>(gap + 1), std::move(inserted));

    NEBParameters growing = band_parameters_;
    growing.climbing = false;
    band_ = std::make_unique<NEB>(std::move(images), growing);
    return gap + 1;
}

std::size_t AutoNEB::prune_image() {
    if (band_->size() <= 3) {
        return band_->size();
    }
    const BandProfile profile = band_->profile();
    const double barrier = std::max(std::abs(profile.forward_barrier), 1e-30);

    // The image whose energy is closest to what interpolating its neighbours
    // would have given: it is telling the profile the least.
    std::size_t candidate = band_->size();
    double smallest = 0.0;
    for (std::size_t k = 1; k + 1 < band_->size(); ++k) {
        if (k == profile.highest) {
            continue;   // never the top of the barrier
        }
        const double left = profile.distances[k] - profile.distances[k - 1];
        const double right = profile.distances[k + 1] - profile.distances[k];
        const double span = left + right;
        if (!(span > 0.0)) {
            continue;
        }
        const double interpolated =
            profile.energies[k - 1] + (profile.energies[k + 1] - profile.energies[k - 1]) *
                                          (left / span);
        const double miss = std::abs(profile.energies[k] - interpolated) / barrier;
        if (candidate == band_->size() || miss < smallest) {
            smallest = miss;
            candidate = k;
        }
    }
    if (candidate == band_->size() || smallest > parameters_.prune_tolerance) {
        return band_->size();
    }

    std::vector<Atoms> images = band_->images();
    images.erase(images.begin() + static_cast<std::ptrdiff_t>(candidate));
    NEBParameters growing = band_parameters_;
    growing.climbing = false;
    band_ = std::make_unique<NEB>(std::move(images), growing);
    return candidate;
}

void AutoNEB::redistribute() {
    const std::vector<Atoms>& images = band_->images();
    const std::size_t count = images.size();

    // Arc length along the current path.
    std::vector<double> arc(count, 0.0);
    for (std::size_t k = 1; k < count; ++k) {
        arc[k] = arc[k - 1] + distance_between(images[k - 1], images[k]);
    }
    if (!(arc.back() > 0.0)) {
        return;
    }

    std::vector<Atoms> spaced;
    spaced.reserve(count);
    spaced.push_back(images.front());
    for (std::size_t k = 1; k + 1 < count; ++k) {
        const double target = arc.back() * static_cast<double>(k) / static_cast<double>(count - 1);
        // The segment this target falls in.
        std::size_t segment = 0;
        while (segment + 2 < count && arc[segment + 1] < target) {
            ++segment;
        }
        const double span = arc[segment + 1] - arc[segment];
        const double fraction = span > 0.0 ? (target - arc[segment]) / span : 0.0;
        spaced.push_back(blend(images[segment], images[segment + 1], fraction));
    }
    spaced.push_back(images.back());

    NEBParameters growing = band_parameters_;
    growing.climbing = false;
    band_ = std::make_unique<NEB>(std::move(spaced), growing);
}

AutoNEBResult AutoNEB::run(const OptimizerFactory& factory) {
    AutoNEBResult result;

    const auto relax = [&](double tolerance, bool climbing) {
        NEBParameters settings = band_parameters_;
        settings.climbing = climbing;
        band_->set_parameters(settings);

        std::shared_ptr<optimize::Optimizer> optimizer = factory();
        optimizer->set_parameter("fmax", Value(tolerance));
        optimizer->set_parameter("max_steps",
                                 Value(static_cast<std::int64_t>(parameters_.max_steps)));
        const optimize::OptimizationResult stage = band_->run(*optimizer);
        ++result.stages;
        result.steps += stage.steps;
        result.evaluations += stage.evaluations;
        result.history.push_back(band_->size());
        return stage;
    };

    // A first relaxation of the coarse band, so the insertions that follow are
    // guided by a path rather than by the straight line it started as.
    relax(parameters_.coarse_fmax, false);

    while (band_->size() < parameters_.max_images) {
        if (parameters_.prune) {
            prune_image();
        }
        insert_image();
        relax(parameters_.coarse_fmax, false);
    }

    if (parameters_.redistribute) {
        redistribute();
        relax(parameters_.coarse_fmax, false);
    }

    // Tighten the whole band *before* anything climbs. Turning the climbing
    // image on while the band is still only coarsely relaxed is what lets it
    // run past a neighbour: the images fold over each other, the tangent
    // between the crossed pair reverses, and the band converges to a path that
    // doubles back on itself with the "highest" image sitting in a well. The
    // springs cannot undo that, because they act along the reversed tangent.
    optimize::OptimizationResult last = relax(parameters_.fmax, false);

    // Only now the climb, from a band that already knows where the path is.
    if (parameters_.climb) {
        last = relax(parameters_.fmax, true);
    }
    result.converged = last.converged;
    result.profile = band_->profile();
    return result;
}

AutoNEBResult AutoNEB::run(const std::string& algorithm, Value parameters) {
    return run(optimizer_factory(algorithm, std::move(parameters)));
}

}  // namespace fastatoms::neb
