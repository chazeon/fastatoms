// SPDX-License-Identifier: MIT
#pragma once

/// \file neb.hpp
/// \brief The nudged elastic band: reaction paths and saddle points.
///
/// A band is a chain of images between two fixed endpoints. Each image feels
/// the true force with its component along the path removed, plus a spring
/// force with everything *but* the along-path component removed:
///
///     F_i = F_true,i - (F_true,i . t^) t^   +   k (|R_i+1 - R_i| - |R_i - R_i-1|) t^
///
/// That split is the "nudging". Keeping the perpendicular true force lets the
/// band relax down onto the reaction path; keeping only the parallel spring
/// force keeps the images spread along it without the springs pulling the band
/// off the valley floor - which is what a plain elastic band does, cutting
/// corners around the saddle.
///
/// The tangent is Henkelman and Jonsson's improved estimate (JCP 113, 9978):
/// on a rising or falling stretch it is simply the direction of the
/// higher-energy neighbour, and only at an extremum are the two neighbours
/// mixed, weighted by their energy differences. The obvious central difference
/// `R_i+1 - R_i-1` develops kinks wherever the force along the band is large,
/// and those kinks are exactly where the saddle is.
///
/// Climbing image
/// --------------
/// The band alone converges to a path, not to a saddle: the highest image sits
/// wherever the springs put it. With `climbing` on, that image loses its
/// springs entirely and has its parallel force *inverted*, so it climbs uphill
/// along the path while relaxing downhill in every other direction. It ends on
/// the saddle point itself, and no interpolation between images is needed to
/// estimate the barrier.
///
/// Driving it
/// ----------
/// The band exposes an `ObjectiveFunction`, so any optimizer in
/// `fastatoms::optimize` can drive it:
///
///     NEB band = NEB::from_endpoints(initial, final, 7);
///     Fire optimizer;
///     band.run(optimizer);
///
/// **The NEB force is not the gradient of anything.** The spring term is added
/// by hand and the climbing image's parallel component is reversed, so no
/// scalar function has it as a gradient. A line search - which exists to find a
/// point where an energy is lower - therefore has nothing to search on, and a
/// quasi-Newton method building curvature from that force is modelling a
/// surface that does not exist. Use FIRE, or set `line_search: none`. `run()`
/// says so out loud rather than quietly converging badly.

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/optimize/optimizer.hpp"
#include "fastatoms/optimize/types.hpp"

namespace fastatoms::neb {

/// Raised when a band cannot be built or evaluated.
class NEBError : public std::runtime_error {
public:
    explicit NEBError(const std::string& message) : std::runtime_error(message) {}
};

/// How the local direction of the path is estimated.
enum class Tangent {
    Improved,  ///< Henkelman and Jonsson's energy-weighted estimate; the default
    Bisect,    ///< the normalised central difference, R_i+1 - R_i-1
    Upwind,    ///< always towards the higher-energy neighbour
};

[[nodiscard]] std::string_view tangent_name(Tangent tangent);
/// \throws NEBError if the name is not one of improved, bisect or upwind.
[[nodiscard]] Tangent tangent_from_string(std::string_view name);

/// How a band behaves.
struct NEBParameters {
    /// Spring constant along the band, eV/angstrom^2.
    double spring_constant{0.1};

    /// Drive the highest image to the saddle instead of letting the springs
    /// place it.
    bool climbing{false};

    /// Start climbing only once the band's largest force has fallen below
    /// this, in eV/angstrom. Zero climbs from the first step.
    ///
    /// Climbing from a badly interpolated start pushes one image uphill before
    /// the band knows where the path is, and it can climb the wrong ridge.
    /// Relaxing first and then climbing is both faster and more reliable.
    double climb_threshold{0.0};

    Tangent tangent{Tangent::Improved};

    /// Stiffen the springs near the top of the barrier, so images bunch where
    /// the resolution is wanted (Henkelman and Jonsson's variable springs).
    bool variable_springs{false};
    /// The softest the springs get, as a fraction of `spring_constant`.
    double spring_softening{0.5};

    /// Measure image-to-image displacements with the minimum-image convention.
    /// On by default for a fully periodic cell, where an atom that crosses a
    /// boundary would otherwise appear to have moved most of a cell.
    bool minimum_image{true};

    /// Atoms held still in every image, by index.
    ///
    /// Usually the substrate. This is not a convenience: the projected band
    /// force is not required to sum to zero, because the component removed
    /// from one atom is not the component removed from another. A spectator
    /// atom - a surface layer, a wall - therefore keeps a force that the
    /// tangent does not describe, the band drifts bodily along the reaction
    /// coordinate, and nothing restores it, because the springs only equalise
    /// the spacing between images and are blind to the band as a whole moving.
    /// A calculation on an adsorbate over a slab needs this.
    std::vector<std::size_t> fixed_atoms{};
};

/// What the band looks like at a moment: the energy profile and where the top
/// of it is.
struct BandProfile {
    std::vector<double> energies{};        ///< eV, one per image including the ends
    std::vector<double> distances{};       ///< cumulative path length, angstrom
    std::vector<double> image_fmax{};      ///< largest NEB force on each movable image
    std::size_t highest{0};                ///< index of the highest-energy image
    double forward_barrier{0.0};           ///< highest energy minus the first
    double reverse_barrier{0.0};           ///< highest energy minus the last
    double reaction_energy{0.0};           ///< last minus first
    bool climbing{false};                  ///< whether the highest image was climbing

    [[nodiscard]] std::string to_string() const;
};

/// A chain of images between two fixed endpoints.
class NEB {
public:
    /// \param images at least three, the first and last held fixed
    /// \throws NEBError on too few images, a size mismatch, or a missing
    ///         calculator
    explicit NEB(std::vector<Atoms> images, NEBParameters parameters = {});

    /// Build a band by interpolating between two endpoints.
    ///
    /// The endpoints' calculator is shared with every new image, so the band
    /// is ready to evaluate.
    [[nodiscard]] static NEB from_endpoints(const Atoms& initial, const Atoms& final,
                                            std::size_t intermediate,
                                            NEBParameters parameters = {});

    /// Straight-line interpolation between two structures, endpoints included,
    /// giving `intermediate + 2` images.
    ///
    /// \param minimum_image move each atom by the shortest periodic
    ///        displacement rather than the raw coordinate difference
    [[nodiscard]] static std::vector<Atoms> interpolate(const Atoms& initial, const Atoms& final,
                                                        std::size_t intermediate,
                                                        bool minimum_image = true);

    /// \name The band
    /// @{
    [[nodiscard]] std::size_t size() const noexcept { return images_.size(); }
    /// Images that are free to move: everything but the two endpoints.
    [[nodiscard]] std::size_t movable() const noexcept { return images_.size() - 2; }
    [[nodiscard]] std::size_t natoms() const noexcept { return natoms_; }

    [[nodiscard]] const Atoms& image(std::size_t index) const;
    [[nodiscard]] Atoms& image(std::size_t index);
    [[nodiscard]] const std::vector<Atoms>& images() const noexcept { return images_; }

    [[nodiscard]] const NEBParameters& parameters() const noexcept { return parameters_; }
    void set_parameters(NEBParameters parameters) { parameters_ = parameters; }
    /// @}

    /// \name Driving an optimizer
    ///
    /// The coordinate vector holds the movable images only, concatenated:
    /// `3 * natoms * (size() - 2)` numbers.
    /// @{
    [[nodiscard]] std::size_t ndof() const noexcept { return 3 * natoms_ * movable(); }
    [[nodiscard]] std::vector<double> positions() const;
    void set_positions(const std::vector<double>& x);

    /// Energy and gradient at `x`. The gradient is the negative of the NEB
    /// force; the energy is the sum over the movable images, which is reported
    /// so an optimizer has something to log and is *not* a potential the
    /// gradient derives from.
    [[nodiscard]] optimize::Evaluation evaluate(const std::vector<double>& x);

    /// The same, as a function an optimizer can be handed.
    [[nodiscard]] optimize::ObjectiveFunction objective();

    /// Relax the band with `optimizer`, leaving the images at their final
    /// geometry.
    ///
    /// \throws NEBError if the optimizer is configured with a line search,
    ///         which the NEB force cannot support.
    optimize::OptimizationResult run(optimize::Optimizer& optimizer);
    /// @}

    /// \name What the band is doing
    /// @{
    /// Evaluate every image and return the profile. Cheap after a `run`, which
    /// leaves the results cached.
    [[nodiscard]] BandProfile profile();

    /// True forces on every image, `3 * natoms * size()`, endpoints included.
    [[nodiscard]] std::vector<double> true_forces();

    /// The projected NEB forces on the movable images, `ndof()` long.
    [[nodiscard]] std::vector<double> forces();

    /// The unit tangent at one image, `3 * natoms` long. Only defined for a
    /// movable image.
    [[nodiscard]] std::vector<double> tangent(std::size_t index);

    /// Largest NEB force on any movable image, the quantity convergence tests.
    [[nodiscard]] double fmax();

    /// Which movable image currently has the highest energy.
    [[nodiscard]] std::size_t highest_image();

    /// Whether the climbing image is active right now.
    [[nodiscard]] bool is_climbing() const noexcept { return climbing_active_; }
    /// @}

    [[nodiscard]] std::string to_string() const;

private:
    void validate();
    /// Displacement from image `a` to image `b`, minimum image when asked.
    [[nodiscard]] std::vector<double> displacement(std::size_t a, std::size_t b) const;
    void evaluate_images();
    [[nodiscard]] std::vector<double> improved_tangent(std::size_t index) const;

    /// The projected band force, with or without the climbing image.
    ///
    /// Split out of `forces()` so the climb threshold can be tested against a
    /// plain projection, without the climbing image perturbing the very number
    /// that decides whether climbing starts.
    [[nodiscard]] std::vector<double> project(bool climbing, std::size_t climber) const;

    std::vector<Atoms> images_{};
    NEBParameters parameters_{};
    std::size_t natoms_{0};

    /// Per-image energies and forces from the most recent evaluation.
    std::vector<double> energies_{};
    std::vector<double> forces_{};
    bool evaluated_{false};
    bool climbing_active_{false};
};

}  // namespace fastatoms::neb
