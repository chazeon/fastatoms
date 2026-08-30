// SPDX-License-Identifier: MIT
#pragma once

/// \file registry.hpp
/// \brief Building a band from a parsed input deck.

#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/io/parameters.hpp"
#include "fastatoms/neb/autoneb.hpp"
#include "fastatoms/neb/neb.hpp"

namespace fastatoms::neb {

/// Translate the deck's `neb:` section into band parameters.
///
/// \throws NEBError if `tangent` or `insert` names something that does not
///         exist, which is worth catching here rather than silently running
///         with a default the file did not ask for.
[[nodiscard]] NEBParameters band_parameters(const BandParameters& spec);

/// As above, for the AutoNEB-specific settings. `fmax` and `max_steps` come
/// from the deck's `optimizer:` section, since that is what they control.
[[nodiscard]] AutoNEBParameters autoneb_parameters(const BandParameters& spec,
                                                   const OptimizerParameters& optimizer);

/// Read the images the deck names.
///
/// Either a prepared multi-frame `band:` file, or two endpoints interpolated
/// into `images` of them. `structure` is the deck's own structure, used as the
/// first endpoint when `initial:` is not given. Every image gets a copy of the
/// calculator attached to `structure`, if it has one.
///
/// \throws InputError if a file named cannot be read.
/// \throws NEBError if what comes back cannot form a band.
[[nodiscard]] std::vector<Atoms> band_images(const BandParameters& spec, const Atoms& structure);

/// The band a deck describes, ready to relax.
[[nodiscard]] NEB make_band(const Atoms& structure, const SimulationParameters& simulation);

/// Whether the deck asked for AutoNEB rather than a fixed band.
[[nodiscard]] bool wants_autoneb(const SimulationParameters& simulation);

/// The AutoNEB run a deck describes. Uses the deck's endpoints; a prepared
/// `band:` file contributes only its two ends, since AutoNEB builds its own.
[[nodiscard]] AutoNEB make_autoneb(const Atoms& structure, const SimulationParameters& simulation);

}  // namespace fastatoms::neb
