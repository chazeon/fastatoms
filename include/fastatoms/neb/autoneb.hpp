// SPDX-License-Identifier: MIT
#pragma once

/// \file autoneb.hpp
/// \brief Growing a band where the resolution is actually needed.
///
/// A fixed band spends its images evenly along the path, which is the wrong
/// place for most of them: the flat approach to a barrier needs two or three,
/// and the region around the saddle needs the rest. Guessing the right number
/// up front means either a coarse path or paying for images that sit in the
/// valley contributing nothing.
///
/// AutoNEB starts with the fewest images that can describe a path at all,
/// relaxes them loosely, and then repeatedly puts a new image wherever the
/// band is least resolved - by default across the largest energy jump between
/// neighbours, which is where the saddle is. Each addition is relaxed before
/// the next is chosen, so the band decides where it needs detail rather than
/// being told. Only the last stage runs to the tight tolerance, and only that
/// stage climbs.
///
/// It can also go the other way. `prune` removes an image the profile does not
/// need - one whose energy is nearly what interpolating its neighbours would
/// give - and `redistribute` respaces the whole band evenly along its own arc
/// length, which is worth doing after several insertions have bunched it.

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "fastatoms/neb/neb.hpp"
#include "fastatoms/optimize/optimizer.hpp"

namespace fastatoms::neb {

/// Where a new image goes.
enum class Insertion {
    Energy,    ///< across the largest energy difference between neighbours
    Distance,  ///< across the largest geometric gap
};

[[nodiscard]] std::string_view insertion_name(Insertion where);
[[nodiscard]] Insertion insertion_from_string(std::string_view name);

struct AutoNEBParameters {
    /// Images to start with, endpoints included. Three is the fewest that can
    /// describe a path.
    std::size_t initial_images{5};

    /// Images to grow to, endpoints included.
    std::size_t max_images{11};

    /// Final force tolerance, eV/angstrom.
    double fmax{0.05};

    /// Tolerance while growing. Relaxing intermediate bands to the final
    /// tolerance wastes most of the work: they are about to change shape when
    /// the next image lands in the middle of them.
    double coarse_fmax{0.2};

    /// Step ceiling for one stage.
    int max_steps{200};

    Insertion insert{Insertion::Energy};

    /// Climb in the final stage, once the path is resolved.
    bool climb{true};

    /// Drop an image the profile does not need before each insertion, so the
    /// band can move detail from where it is not wanted to where it is.
    bool prune{false};

    /// How close an image's energy must be to the interpolation of its
    /// neighbours, relative to the barrier, before it counts as redundant.
    double prune_tolerance{0.01};

    /// Respace the band evenly along its own arc length before the final
    /// stage.
    bool redistribute{false};
};

/// What a run did.
struct AutoNEBResult {
    BandProfile profile{};
    int stages{0};                        ///< relaxations performed
    int steps{0};                         ///< optimizer steps over all stages
    int evaluations{0};                   ///< band evaluations over all stages
    bool converged{false};
    /// Image count after each stage, so the growth is visible.
    std::vector<std::size_t> history{};

    [[nodiscard]] std::string to_string() const;
};

/// Builds an optimizer for one stage. A fresh one each time: the previous
/// stage's velocity or curvature history describes a band that no longer
/// exists, because an image was just inserted into the middle of it.
using OptimizerFactory = std::function<std::shared_ptr<optimize::Optimizer>()>;

/// Build a factory for a named algorithm.
///
/// The line search is forced off whatever the parameters say: the band force
/// is not the gradient of an energy, so there is nothing for a search to
/// decrease.
[[nodiscard]] OptimizerFactory optimizer_factory(const std::string& algorithm,
                                                 Value parameters = {});

class AutoNEB {
public:
    AutoNEB(Atoms initial, Atoms final, AutoNEBParameters parameters = {},
            NEBParameters band = {});

    /// Grow and relax the band.
    AutoNEBResult run(const OptimizerFactory& factory);

    /// The same, with an optimizer named rather than built.
    AutoNEBResult run(const std::string& algorithm = "fire", Value parameters = {});

    [[nodiscard]] NEB& band();
    [[nodiscard]] const NEB& band() const;

    /// \name The three moves, exposed so they can be used on their own
    /// @{
    /// Insert one image where the band is least resolved. Returns where it
    /// went, or `size()` when the band is already at `max_images`.
    std::size_t insert_image();

    /// Remove the least informative interior image. Returns which one went, or
    /// `size()` when none was redundant enough or the band is at its minimum.
    std::size_t prune_image();

    /// Respace every movable image evenly along the current path.
    void redistribute();
    /// @}

private:
    [[nodiscard]] std::size_t widest_gap();

    std::unique_ptr<NEB> band_{};
    AutoNEBParameters parameters_{};
    NEBParameters band_parameters_{};
};

}  // namespace fastatoms::neb
