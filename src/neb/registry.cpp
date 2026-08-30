// SPDX-License-Identifier: MIT

#include "fastatoms/neb/registry.hpp"

#include <utility>

#include "fastatoms/io/formats.hpp"
#include "fastatoms/io/yaml_input.hpp"

namespace fastatoms::neb {
namespace {

/// Indices come out of the deck as signed, because that is how the YAML reader
/// sees them; the parser has already rejected negatives.
std::vector<std::size_t> to_indices(const std::vector<std::int64_t>& values) {
    std::vector<std::size_t> out;
    out.reserve(values.size());
    for (const std::int64_t value : values) {
        out.push_back(static_cast<std::size_t>(value));
    }
    return out;
}

}  // namespace

NEBParameters band_parameters(const BandParameters& spec) {
    NEBParameters parameters;
    parameters.spring_constant = spec.spring_constant;
    parameters.climbing = spec.climbing;
    parameters.climb_threshold = spec.climb_threshold;
    parameters.tangent = tangent_from_string(spec.tangent);
    parameters.variable_springs = spec.variable_springs;
    parameters.spring_softening = spec.spring_softening;
    parameters.minimum_image = spec.minimum_image;
    parameters.fixed_atoms = to_indices(spec.fixed_atoms);
    return parameters;
}

AutoNEBParameters autoneb_parameters(const BandParameters& spec,
                                     const OptimizerParameters& optimizer) {
    AutoNEBParameters parameters;
    parameters.initial_images = static_cast<std::size_t>(spec.images);
    parameters.max_images = static_cast<std::size_t>(spec.max_images);
    parameters.fmax = optimizer.fmax;
    parameters.coarse_fmax = spec.coarse_fmax;
    parameters.max_steps = static_cast<int>(optimizer.max_steps);
    parameters.insert = insertion_from_string(spec.insert);
    parameters.climb = spec.climbing;
    parameters.prune = spec.prune;
    parameters.prune_tolerance = spec.prune_tolerance;
    parameters.redistribute = spec.redistribute;
    return parameters;
}

std::vector<Atoms> band_images(const BandParameters& spec, const Atoms& structure) {
    std::vector<Atoms> images;

    if (!spec.band.empty()) {
        images = io::read_frames(spec.band);
        if (images.size() < 3) {
            throw NEBError("neb.band: '" + spec.band + "' holds " + std::to_string(images.size()) +
                           " frames; a band needs at least 3");
        }
    } else {
        const Atoms initial = !spec.initial_structure.is_null()
                                  ? io::atoms_from_value(spec.initial_structure)
                                  : (spec.initial.empty() ? structure : io::read(spec.initial));
        const Atoms final = !spec.final_structure.is_null()
                                ? io::atoms_from_value(spec.final_structure)
                                : io::read(spec.final);
        // `images` counts the endpoints, the way the deck and the Python
        // interface both spell it; `interpolate` counts the ones in between.
        images = NEB::interpolate(initial, final, static_cast<std::size_t>(spec.images - 2),
                                  spec.minimum_image);
    }

    // A band read from a file carries no calculator; the deck's structure has
    // the one the file configured, and every image needs it.
    if (structure.has_calculator()) {
        for (Atoms& image : images) {
            image.set_calculator(structure.calculator());
        }
    }
    return images;
}

NEB make_band(const Atoms& structure, const SimulationParameters& simulation) {
    return NEB(band_images(simulation.neb, structure), band_parameters(simulation.neb));
}

bool wants_autoneb(const SimulationParameters& simulation) { return simulation.neb.automatic; }

AutoNEB make_autoneb(const Atoms& structure, const SimulationParameters& simulation) {
    std::vector<Atoms> images = band_images(simulation.neb, structure);
    return AutoNEB(images.front(), images.back(),
                   autoneb_parameters(simulation.neb, simulation.optimizer),
                   band_parameters(simulation.neb));
}

}  // namespace fastatoms::neb
