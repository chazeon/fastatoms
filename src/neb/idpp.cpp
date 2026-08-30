// SPDX-License-Identifier: MIT

#include "fastatoms/neb/idpp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

#include "fastatoms/neb/neb.hpp"
#include "fastatoms/optimize/fire.hpp"

namespace fastatoms::neb {
namespace {

/// The full distance matrix, N*N row-major. O(N^2), which is what IDPP costs.
std::vector<double> distance_matrix(const Atoms& atoms, bool mic) {
    const std::size_t n = atoms.size();
    std::vector<double> out(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double d = atoms.distance(i, j, mic);
            out[i * n + j] = d;
            out[j * n + i] = d;
        }
    }
    return out;
}

}  // namespace

IDPPCalculator::IDPPCalculator(std::vector<double> targets, std::size_t natoms, bool mic)
    : targets_(std::move(targets)), natoms_(natoms), mic_(mic) {
    // The weights go as 1/t^4, so their size is set entirely by how far apart
    // the atoms happen to be - a factor of 10^4 between a molecule and a
    // loosely packed solid. Normalising by their sum makes the objective, and
    // therefore any force tolerance quoted against it, independent of that.
    double total = 0.0;
    for (std::size_t i = 0; i < natoms_; ++i) {
        for (std::size_t j = i + 1; j < natoms_; ++j) {
            const double target = targets_[i * natoms_ + j];
            if (target > 0.0) {
                total += 1.0 / (target * target * target * target);
            }
        }
    }
    scale_ = total > 0.0 ? 1.0 / total : 1.0;
}

std::vector<Property> IDPPCalculator::implemented_properties() const {
    return {Property::Energy, Property::Forces};
}

void IDPPCalculator::compute(const Atoms& atoms, const std::vector<Property>&) {
    const std::size_t n = atoms.size();
    if (n != natoms_) {
        fail("the structure changed size between images");
    }

    double energy = 0.0;
    std::vector<double> forces(3 * n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double target = targets_[i * n + j];
            if (target <= 0.0) {
                continue;
            }
            Vec3 dr = atoms.distance_vector(i, j, mic_);   // r_j - r_i
            double d = dr.norm();
            if (d < 1e-9) {
                // Two atoms exactly on top of one another: the direction to
                // separate them along is undefined, and skipping the pair
                // would make this configuration a fixed point of the very
                // objective meant to remove it. Pick a deterministic axis
                // instead, so the run stays reproducible.
                const std::size_t axis = (i + j) % 3;
                dr = Vec3{axis == 0 ? 1e-6 : 0.0, axis == 1 ? 1e-6 : 0.0,
                          axis == 2 ? 1e-6 : 0.0};
                d = 1e-6;
            }
            // w = 1/t^4 weights close pairs, which are the ones that decide
            // whether the image is physically sane at all. `scale_` divides
            // out the overall size of the weights, which otherwise depends on
            // the fourth power of the system's typical distance and would make
            // any fixed force tolerance meaningless.
            const double weight = scale_ / (target * target * target * target);
            const double gap = d - target;
            energy += weight * gap * gap;

            // dS/dr_j = 2 w (d - t) * (r_j - r_i)/d, and the opposite on i.
            const double scale = 2.0 * weight * gap / d;
            forces[3 * i + 0] += scale * dr.x;
            forces[3 * i + 1] += scale * dr.y;
            forces[3 * i + 2] += scale * dr.z;
            forces[3 * j + 0] -= scale * dr.x;
            forces[3 * j + 1] -= scale * dr.y;
            forces[3 * j + 2] -= scale * dr.z;
        }
    }

    results_.energy = energy;
    results_.free_energy = energy;
    results_.forces = std::move(forces);
}

std::vector<Atoms> idpp_interpolate(const Atoms& initial, const Atoms& final, std::size_t images,
                                    const IDPPOptions& options) {
    if (images < 3) {
        throw NEBError("idpp: a band needs at least 3 images, including the endpoints");
    }
    // The straight line is the starting point; IDPP moves it somewhere usable.
    std::vector<Atoms> band = NEB::interpolate(initial, final, images - 2, options.minimum_image);

    // A symmetric exchange puts the straight line exactly on a saddle of the
    // objective, where the gradient vanishes for the wrong reason and nothing
    // moves. A small deterministic nudge of the interior images costs nothing
    // and takes the band off it. Deterministic, so two runs agree.
    for (std::size_t k = 1; k + 1 < band.size(); ++k) {
        for (std::size_t a = 0; a < band[k].size(); ++a) {
            const std::size_t seed = 2654435761u * (k * band[k].size() + a + 1);
            const Vec3 nudge{1e-3 * (static_cast<double>((seed >> 3) % 1000) / 1000.0 - 0.5),
                             1e-3 * (static_cast<double>((seed >> 13) % 1000) / 1000.0 - 0.5),
                             1e-3 * (static_cast<double>((seed >> 23) % 1000) / 1000.0 - 0.5)};
            band[k].set_position(a, band[k].position(a) + nudge);
        }
    }

    const std::size_t n = initial.size();
    const std::vector<double> first = distance_matrix(initial, options.minimum_image);
    const std::vector<double> last = distance_matrix(final, options.minimum_image);

    for (std::size_t k = 0; k < band.size(); ++k) {
        // Each image aims at its own linear interpolation of the endpoint
        // distances - "image-dependent" is the whole of the name.
        const double t = static_cast<double>(k) / static_cast<double>(band.size() - 1);
        std::vector<double> targets(n * n, 0.0);
        for (std::size_t c = 0; c < n * n; ++c) {
            targets[c] = (1.0 - t) * first[c] + t * last[c];
        }
        band[k].set_calculator(
            std::make_shared<IDPPCalculator>(std::move(targets), n, options.minimum_image));
    }

    NEBParameters parameters;
    parameters.spring_constant = options.spring_constant;
    parameters.minimum_image = options.minimum_image;
    NEB relaxing(band, parameters);

    optimize::Fire optimizer(Value(Value::Map{
        {"fmax", Value(options.fmax)},
        {"max_steps", Value(static_cast<std::int64_t>(options.max_steps))},
    }));
    relaxing.run(optimizer);

    // Hand back plain structures: the objective was scaffolding.
    std::vector<Atoms> out = relaxing.images();
    for (Atoms& image : out) {
        image.set_calculator(nullptr);
    }
    return out;
}

double closest_approach(const Atoms& atoms, bool mic) {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        for (std::size_t j = i + 1; j < atoms.size(); ++j) {
            best = std::min(best, atoms.distance(i, j, mic));
        }
    }
    return best;
}

}  // namespace fastatoms::neb
