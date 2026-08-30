// SPDX-License-Identifier: MIT
#include "fastatoms/neb/neb.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

#include "fastatoms/util/text.hpp"

namespace fastatoms::neb {
namespace {

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double total = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        total += a[k] * b[k];
    }
    return total;
}

double norm(const std::vector<double>& a) { return std::sqrt(dot(a, a)); }

void normalise(std::vector<double>& a) {
    const double length = norm(a);
    if (length > 0.0) {
        for (double& value : a) {
            value /= length;
        }
    }
}

/// Largest per-atom force magnitude in a 3N vector.
double max_row_norm(const std::vector<double>& forces) {
    double worst = 0.0;
    for (std::size_t atom = 0; atom * 3 + 2 < forces.size(); ++atom) {
        const double x = forces[3 * atom];
        const double y = forces[3 * atom + 1];
        const double z = forces[3 * atom + 2];
        worst = std::max(worst, std::sqrt(x * x + y * y + z * z));
    }
    return worst;
}

}  // namespace

std::string_view tangent_name(Tangent tangent) {
    switch (tangent) {
        case Tangent::Bisect: return "bisect";
        case Tangent::Upwind: return "upwind";
        case Tangent::Improved: break;
    }
    return "improved";
}

Tangent tangent_from_string(std::string_view name) {
    const std::string key = util::lowercase(name);
    if (key == "improved" || key == "henkelman") {
        return Tangent::Improved;
    }
    if (key == "bisect" || key == "central" || key == "simple") {
        return Tangent::Bisect;
    }
    if (key == "upwind") {
        return Tangent::Upwind;
    }
    throw NEBError("neb: unknown tangent '" + std::string(name) +
                   "'; expected improved, bisect or upwind");
}

std::string BandProfile::to_string() const {
    std::ostringstream out;
    out.precision(6);
    out << "BandProfile(" << energies.size() << " images, forward barrier " << forward_barrier
        << " eV, reverse " << reverse_barrier << " eV, reaction " << reaction_energy
        << " eV, highest image " << highest << (climbing ? ", climbing)" : ")");
    return out.str();
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
NEB::NEB(std::vector<Atoms> images, NEBParameters parameters)
    : images_(std::move(images)), parameters_(parameters) {
    validate();
    natoms_ = images_.front().size();
    energies_.assign(images_.size(), 0.0);
    forces_.assign(3 * natoms_ * images_.size(), 0.0);
}

void NEB::validate() {
    if (images_.size() < 3) {
        throw NEBError("neb: a band needs at least three images - two endpoints and something "
                       "between them - but was given " + std::to_string(images_.size()));
    }
    const std::size_t count = images_.front().size();
    if (count == 0) {
        throw NEBError("neb: the images have no atoms");
    }
    for (const std::size_t atom : parameters_.fixed_atoms) {
        if (atom >= images_.front().size()) {
            throw NEBError("neb: fixed atom " + std::to_string(atom) +
                           " does not exist in a system of " +
                           std::to_string(images_.front().size()) + " atoms");
        }
    }
    for (std::size_t index = 0; index < images_.size(); ++index) {
        const Atoms& atoms = images_[index];
        if (atoms.size() != count) {
            throw NEBError("neb: image " + std::to_string(index) + " has " +
                           std::to_string(atoms.size()) + " atoms but image 0 has " +
                           std::to_string(count) + "; every image must be the same system");
        }
        if (atoms.numbers() != images_.front().numbers()) {
            throw NEBError("neb: image " + std::to_string(index) +
                           " has different species from image 0; a band interpolates one "
                           "system between two geometries, not one system into another");
        }
        if (!atoms.has_calculator()) {
            throw NEBError("neb: image " + std::to_string(index) +
                           " has no calculator; every image is evaluated independently, so "
                           "each one needs one");
        }
    }
}

std::vector<Atoms> NEB::interpolate(const Atoms& initial, const Atoms& final,
                                    std::size_t intermediate, bool minimum_image) {
    if (initial.size() != final.size()) {
        throw NEBError("neb: the endpoints have different numbers of atoms");
    }
    if (intermediate == 0) {
        throw NEBError("neb: interpolation needs at least one intermediate image");
    }

    const std::vector<double>& start = initial.positions();
    const std::vector<double>& stop = final.positions();

    // The straight line each atom travels. Under periodic boundaries the raw
    // coordinate difference can run most of the way around the cell when an
    // atom simply crossed a face, so the shortest image is taken instead.
    std::vector<double> delta(start.size());
    const bool periodic =
        minimum_image && initial.cell().any_pbc() && initial.cell().is_invertible();
    for (std::size_t atom = 0; atom < initial.size(); ++atom) {
        Vec3 step{stop[3 * atom] - start[3 * atom], stop[3 * atom + 1] - start[3 * atom + 1],
                  stop[3 * atom + 2] - start[3 * atom + 2]};
        if (periodic) {
            step = initial.cell().minimum_image(step);
        }
        delta[3 * atom] = step.x;
        delta[3 * atom + 1] = step.y;
        delta[3 * atom + 2] = step.z;
    }

    std::vector<Atoms> band;
    band.reserve(intermediate + 2);
    band.push_back(initial);
    for (std::size_t k = 1; k <= intermediate; ++k) {
        const double fraction = static_cast<double>(k) / static_cast<double>(intermediate + 1);
        Atoms image = initial;
        std::vector<double> positions = start;
        for (std::size_t c = 0; c < positions.size(); ++c) {
            positions[c] += fraction * delta[c];
        }
        image.set_positions(positions);
        band.push_back(std::move(image));
    }
    band.push_back(final);
    return band;
}

NEB NEB::from_endpoints(const Atoms& initial, const Atoms& final, std::size_t intermediate,
                        NEBParameters parameters) {
    std::vector<Atoms> band = interpolate(initial, final, intermediate,
                                          parameters.minimum_image);
    // Every image is evaluated on its own, so each needs a calculator. Copying
    // the endpoint's shares one object between them, which is correct: the
    // band evaluates one image at a time and never holds two sets of results.
    if (initial.has_calculator()) {
        for (Atoms& image : band) {
            image.set_calculator(initial.calculator());
        }
    }
    return NEB(std::move(band), parameters);
}

const Atoms& NEB::image(std::size_t index) const {
    if (index >= images_.size()) {
        throw NEBError("neb: image " + std::to_string(index) + " is out of range for a band of " +
                       std::to_string(images_.size()));
    }
    return images_[index];
}

Atoms& NEB::image(std::size_t index) {
    return const_cast<Atoms&>(static_cast<const NEB&>(*this).image(index));
}

// ---------------------------------------------------------------------------
// Coordinates
// ---------------------------------------------------------------------------
std::vector<double> NEB::positions() const {
    std::vector<double> x(ndof());
    const std::size_t stride = 3 * natoms_;
    for (std::size_t k = 1; k + 1 < images_.size(); ++k) {
        const std::vector<double>& p = images_[k].positions();
        std::copy(p.begin(), p.end(), x.begin() + static_cast<std::ptrdiff_t>((k - 1) * stride));
    }
    return x;
}

void NEB::set_positions(const std::vector<double>& x) {
    if (x.size() != ndof()) {
        throw NEBError("neb: expected " + std::to_string(ndof()) + " coordinates for " +
                       std::to_string(movable()) + " movable images, got " +
                       std::to_string(x.size()));
    }
    const std::size_t stride = 3 * natoms_;
    for (std::size_t k = 1; k + 1 < images_.size(); ++k) {
        std::vector<double> p(x.begin() + static_cast<std::ptrdiff_t>((k - 1) * stride),
                              x.begin() + static_cast<std::ptrdiff_t>(k * stride));
        // `move_to`, so an image's own constraints get to veto the step the
        // optimizer proposed, exactly as they would in a plain relaxation.
        images_[k].move_to(p);
    }
    evaluated_ = false;
}

std::vector<double> NEB::displacement(std::size_t a, std::size_t b) const {
    const std::vector<double>& from = images_[a].positions();
    const std::vector<double>& to = images_[b].positions();
    const bool periodic = parameters_.minimum_image && images_[a].cell().any_pbc() &&
                          images_[a].cell().is_invertible();

    std::vector<double> delta(from.size());
    for (std::size_t atom = 0; atom < natoms_; ++atom) {
        Vec3 step{to[3 * atom] - from[3 * atom], to[3 * atom + 1] - from[3 * atom + 1],
                  to[3 * atom + 2] - from[3 * atom + 2]};
        if (periodic) {
            step = images_[a].cell().minimum_image(step);
        }
        delta[3 * atom] = step.x;
        delta[3 * atom + 1] = step.y;
        delta[3 * atom + 2] = step.z;
    }
    return delta;
}

// ---------------------------------------------------------------------------
// Physics
// ---------------------------------------------------------------------------
void NEB::evaluate_images() {
    const std::size_t stride = 3 * natoms_;
    for (std::size_t k = 0; k < images_.size(); ++k) {
        Atoms& atoms = images_[k];
        // Energy then forces on the same image, back to back, so a calculator
        // shared between images answers both from one evaluation.
        energies_[k] = atoms.get_potential_energy();
        const std::vector<double> f = atoms.get_forces();
        std::copy(f.begin(), f.end(),
                  forces_.begin() + static_cast<std::ptrdiff_t>(k * stride));
    }
    evaluated_ = true;
}

std::vector<double> NEB::improved_tangent(std::size_t index) const {
    const std::vector<double> forward = displacement(index, index + 1);
    const std::vector<double> backward = displacement(index - 1, index);

    std::vector<double> tau(forward.size());
    const double previous = energies_[index - 1];
    const double current = energies_[index];
    const double next = energies_[index + 1];

    if (parameters_.tangent == Tangent::Bisect) {
        // The plain central difference. Kept for comparison: it is what the
        // original formulation used, and it kinks wherever the force along the
        // band is large - which is exactly at the saddle.
        for (std::size_t k = 0; k < tau.size(); ++k) {
            tau[k] = forward[k] + backward[k];
        }
    } else if (parameters_.tangent == Tangent::Upwind) {
        tau = next > previous ? forward : backward;
    } else if (next > current && current > previous) {
        // Rising: the tangent is the direction of the higher neighbour.
        tau = forward;
    } else if (next < current && current < previous) {
        // Falling: likewise, downhill behind us.
        tau = backward;
    } else {
        // An extremum, where neither neighbour is unambiguously "up the path".
        // Mix the two, weighted by the energy change on each side, leaning
        // towards the side with the larger change.
        const double up = std::abs(next - current);
        const double down = std::abs(previous - current);
        const double larger = std::max(up, down);
        const double smaller = std::min(up, down);
        const double a = next > previous ? larger : smaller;
        const double b = next > previous ? smaller : larger;
        for (std::size_t k = 0; k < tau.size(); ++k) {
            tau[k] = a * forward[k] + b * backward[k];
        }
    }

    normalise(tau);
    return tau;
}

std::vector<double> NEB::tangent(std::size_t index) {
    if (index == 0 || index + 1 >= images_.size()) {
        throw NEBError("neb: the tangent is only defined for a movable image, not the endpoints");
    }
    if (!evaluated_) {
        evaluate_images();
    }
    return improved_tangent(index);
}

std::vector<double> NEB::true_forces() {
    if (!evaluated_) {
        evaluate_images();
    }
    return forces_;
}

std::size_t NEB::highest_image() {
    if (!evaluated_) {
        evaluate_images();
    }
    std::size_t best = 1;
    for (std::size_t k = 2; k + 1 < images_.size(); ++k) {
        if (energies_[k] > energies_[best]) {
            best = k;
        }
    }
    return best;
}

std::vector<double> NEB::project(bool climbing, std::size_t climber) const {
    const std::size_t stride = 3 * natoms_;
    std::vector<double> out(ndof(), 0.0);

    // Variable springs: stiffer near the top of the barrier, so images bunch
    // where the resolution matters.
    const double reference = std::max(energies_.front(), energies_.back());
    double highest_energy = energies_[1];
    for (std::size_t k = 2; k + 1 < images_.size(); ++k) {
        highest_energy = std::max(highest_energy, energies_[k]);
    }

    const auto spring_between = [&](std::size_t lower) {
        if (!parameters_.variable_springs) {
            return parameters_.spring_constant;
        }
        const double k_max = parameters_.spring_constant;
        const double k_min = k_max * std::clamp(parameters_.spring_softening, 0.0, 1.0);
        const double energy = std::max(energies_[lower], energies_[lower + 1]);
        if (energy <= reference || highest_energy <= reference) {
            return k_min;
        }
        const double weight = (highest_energy - energy) / (highest_energy - reference);
        return k_max - (k_max - k_min) * weight;
    };

    for (std::size_t k = 1; k + 1 < images_.size(); ++k) {
        const std::vector<double> tau = improved_tangent(k);
        std::vector<double> f(stride);
        std::copy(forces_.begin() + static_cast<std::ptrdiff_t>(k * stride),
                  forces_.begin() + static_cast<std::ptrdiff_t>((k + 1) * stride), f.begin());
        const double parallel = dot(f, tau);

        std::vector<double> projected(stride);
        if (climbing && k == climber) {
            // The climbing image: no springs, and the component along the path
            // reversed, so it walks uphill along the band while relaxing in
            // every perpendicular direction. Its converged position is the
            // saddle point itself.
            for (std::size_t c = 0; c < stride; ++c) {
                projected[c] = f[c] - 2.0 * parallel * tau[c];
            }
        } else {
            const double forward = norm(displacement(k, k + 1));
            const double backward = norm(displacement(k - 1, k));
            const double spring = spring_between(k) * forward - spring_between(k - 1) * backward;
            for (std::size_t c = 0; c < stride; ++c) {
                projected[c] = f[c] - parallel * tau[c] + spring * tau[c];
            }
        }
        // Held atoms feel nothing, in every image.
        for (const std::size_t atom : parameters_.fixed_atoms) {
            if (3 * atom + 2 < projected.size()) {
                projected[3 * atom] = 0.0;
                projected[3 * atom + 1] = 0.0;
                projected[3 * atom + 2] = 0.0;
            }
        }
        // And whatever the image itself carries. The true forces arrived here
        // already projected, by `get_forces`, but the band adds a spring term
        // along the tangent and the climbing image reverses a component, and
        // neither of those is obliged to respect a constraint. Re-applying
        // afterwards is what makes the band honour the same restrictions as
        // every other part of the code.
        images_[k].adjust_forces(projected);
        std::copy(projected.begin(), projected.end(),
                  out.begin() + static_cast<std::ptrdiff_t>((k - 1) * stride));
    }
    return out;
}

std::vector<double> NEB::forces() {
    if (!evaluated_) {
        evaluate_images();
    }
    if (!parameters_.climbing) {
        climbing_active_ = false;
        return project(false, 0);
    }

    const std::size_t climber = highest_image();
    if (!climbing_active_ && parameters_.climb_threshold > 0.0) {
        // Climbing from a badly interpolated start can drive an image up the
        // wrong ridge, so it waits until the band has settled. The test is on
        // the plain band force - the very quantity the optimizer converges on
        // and `fmax()` reports - so "climb below 1e-2" means what a caller who
        // has watched an optimizer log would expect it to mean, and the gate
        // is guaranteed to open before a run that converges below it stops.
        std::vector<double> plain = project(false, climber);
        if (max_row_norm(plain) > parameters_.climb_threshold) {
            return plain;
        }
    }
    climbing_active_ = true;
    return project(true, climber);
}

double NEB::fmax() { return max_row_norm(forces()); }

optimize::Evaluation NEB::evaluate(const std::vector<double>& x) {
    set_positions(x);
    const std::vector<double> f = forces();

    optimize::Evaluation result;
    result.gradient.resize(f.size());
    for (std::size_t k = 0; k < f.size(); ++k) {
        result.gradient[k] = -f[k];
    }
    // Something for an optimizer to log. It is the sum over the movable
    // images, and it is emphatically not a potential whose gradient is above:
    // the spring term is added by hand and the climbing image's parallel
    // component is reversed, so no scalar function has this force as its
    // gradient.
    double total = 0.0;
    for (std::size_t k = 1; k + 1 < images_.size(); ++k) {
        total += energies_[k];
    }
    result.energy = total;
    return result;
}

optimize::ObjectiveFunction NEB::objective() {
    return [this](const std::vector<double>& x) { return this->evaluate(x); };
}

optimize::OptimizationResult NEB::run(optimize::Optimizer& optimizer) {
    if (optimizer.line_search_kind() != optimize::LineSearchKind::None) {
        throw NEBError(
            "neb: '" + optimizer.name() +
            "' is configured with a line search, which the band force cannot support: it is "
            "not the gradient of any energy, so there is nothing for the search to decrease. "
            "Use the 'fire' optimizer, or pass line_search='none'.");
    }
    std::vector<double> x = positions();
    optimize::OptimizationResult result = optimizer.minimize(objective(), x);
    set_positions(x);
    // Leave the band evaluated at its final geometry, so a following profile()
    // costs nothing.
    (void)forces();
    return result;
}

BandProfile NEB::profile() {
    if (!evaluated_) {
        evaluate_images();
    }
    BandProfile profile;
    profile.energies = energies_;
    profile.climbing = climbing_active_;

    profile.distances.assign(images_.size(), 0.0);
    for (std::size_t k = 1; k < images_.size(); ++k) {
        profile.distances[k] = profile.distances[k - 1] + norm(displacement(k - 1, k));
    }

    const std::vector<double> band = forces();
    const std::size_t stride = 3 * natoms_;
    profile.image_fmax.assign(movable(), 0.0);
    for (std::size_t k = 0; k < movable(); ++k) {
        std::vector<double> f(band.begin() + static_cast<std::ptrdiff_t>(k * stride),
                              band.begin() + static_cast<std::ptrdiff_t>((k + 1) * stride));
        profile.image_fmax[k] = max_row_norm(f);
    }

    profile.highest = highest_image();
    profile.forward_barrier = energies_[profile.highest] - energies_.front();
    profile.reverse_barrier = energies_[profile.highest] - energies_.back();
    profile.reaction_energy = energies_.back() - energies_.front();
    return profile;
}

std::string NEB::to_string() const {
    std::ostringstream out;
    out << "NEB(" << images_.size() << " images, " << natoms_ << " atoms, k="
        << parameters_.spring_constant << ", tangent='" << tangent_name(parameters_.tangent)
        << "'";
    if (parameters_.climbing) {
        // `climbing()` reports whether the climbing image is running *now*,
        // which is not the same as having asked for one when a threshold
        // delays it. Say which of the two this is, so the repr cannot be read
        // as the other.
        if (climbing_active_) {
            out << ", climbing";
        } else if (parameters_.climb_threshold > 0.0) {
            out << ", climbing once below " << parameters_.climb_threshold << " eV/A";
        } else {
            out << ", climbing from the first step";
        }
    }
    out << ")";
    return out.str();
}

}  // namespace fastatoms::neb
