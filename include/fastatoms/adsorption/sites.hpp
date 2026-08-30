// SPDX-License-Identifier: MIT
#pragma once

/// \file sites.hpp
/// \brief Finding where an adsorbate can sit, and putting one there.
///
/// The geometry of adsorption is the same on a nanoparticle and on a slab:
/// find the atoms that are exposed, enumerate the atop, bridge and hollow
/// positions their topology offers, throw away the ones that are buried, and
/// work out which way is out. What differs is only how "out" is defined, and
/// that is measured here from the local arrangement rather than assumed to be
/// a fixed axis - so a facet of an icosahedron and the top of a slab are
/// handled by the same code.

#include <stdexcept>
#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"

namespace fastatoms::adsorption {

/// Raised for a surface or a request that cannot be worked with.
class AdsorptionError : public std::runtime_error {
public:
    explicit AdsorptionError(const std::string& message) : std::runtime_error(message) {}
};

/// How many surface atoms an adsorbate touches.
enum class SiteType {
    Atop,      ///< directly above one atom
    Bridge,    ///< between two
    Hollow,    ///< in the dip between three, the usual site on a close-packed facet
    FourFold,  ///< in the dip between four, which square facets offer and close-packed ones do not
};

[[nodiscard]] std::string_view site_type_name(SiteType type);
[[nodiscard]] SiteType site_type_from_string(std::string_view name);

/// One place an adsorbate can bind.
struct AdsorptionSite {
    SiteType type{SiteType::Atop};

    /// Where the adsorbate's binding atom goes: the site's own centre pushed
    /// out along the normal by the requested height.
    Vec3 position{};

    /// The outward unit normal, measured from the local arrangement of atoms.
    Vec3 normal{};

    /// The centre of the site itself, on the surface rather than above it.
    Vec3 centre{};

    /// The surface atoms that form it, ascending.
    std::vector<std::size_t> atoms{};

    /// Index into `AdsorptionFinder::unique_sites()` of the representative
    /// this site is equivalent to.
    std::size_t unique{0};

    [[nodiscard]] std::string to_string() const;
};

/// What to look for.
struct SiteOptions {
    /// How far above the surface the binding atom is placed, in angstrom.
    double height{2.0};

    /// Two atoms are bonded when they are closer than this multiple of the sum
    /// of their covalent radii.
    double bond_scale{1.25};

    /// An atom counts as surface when its coordination is at most this. Zero
    /// asks for the default: one less than the highest coordination present,
    /// which separates the skin of a nanoparticle from its interior without
    /// needing to know what the bulk number is.
    int surface_coordination{0};

    /// Which kinds of site to enumerate.
    bool atop{true};
    bool bridge{true};
    bool hollow{true};
    bool four_fold{true};

    /// A site is discarded when any atom is closer to the placed position than
    /// this fraction of the height - which is what removes the sites on the
    /// far side of a bridge, and the hollows that are really inside the
    /// particle.
    double clearance{0.75};

    /// Distances agreeing to this, in angstrom, count as equal when deciding
    /// which sites are equivalent.
    double tolerance{0.05};
};

/// Adsorption sites on a surface or a nanoparticle.
///
/// ```
/// const AdsorptionFinder finder(particle);
/// for (const std::size_t index : finder.unique_sites()) {
///     const Atoms decorated = finder.place_atom("H", index);
/// }
/// ```
class AdsorptionFinder {
public:
    /// \throws AdsorptionError if the structure is empty or has no bonds at
    ///         the given scale
    explicit AdsorptionFinder(Atoms surface, SiteOptions options = {});

    [[nodiscard]] const Atoms& surface() const noexcept { return surface_; }
    [[nodiscard]] const SiteOptions& options() const noexcept { return options_; }

    /// Every site found.
    [[nodiscard]] const std::vector<AdsorptionSite>& sites() const noexcept { return sites_; }

    /// The atoms judged to be exposed.
    [[nodiscard]] const std::vector<std::size_t>& surface_atoms() const noexcept {
        return surface_atoms_;
    }

    /// One representative of each distinct site, as indices into `sites()`.
    [[nodiscard]] const std::vector<std::size_t>& unique_sites() const noexcept {
        return unique_;
    }

    /// Every site equivalent to unique site `unique`, including itself.
    ///
    /// \param unique an index into `unique_sites()`, not into `sites()`
    [[nodiscard]] const std::vector<std::size_t>& equivalent_sites(std::size_t unique) const;

    /// The sites of one kind, as indices into `sites()`.
    [[nodiscard]] std::vector<std::size_t> sites_of_type(SiteType type,
                                                         bool unique_only = false) const;

    /// How many sites of each kind were found.
    [[nodiscard]] std::string summary() const;

    /// \name Placing an adsorbate
    /// @{

    /// A single atom on one site.
    [[nodiscard]] Atoms place_atom(const std::string& symbol, std::size_t site) const;

    /// A molecule on one site.
    ///
    /// The molecule is moved so that `binding_atom` sits at the site, and
    /// turned so that the direction from the binding atom to the rest of the
    /// molecule points away from the surface. That is what makes an OH bind
    /// through its oxygen with the hydrogen pointing out, rather than lying
    /// down or burying itself.
    ///
    /// \param anchor the atom whose direction from the binding atom is aligned
    ///        with the outward normal. Negative means the centroid of
    ///        everything except the binding atom, which is the sensible
    ///        default and is exactly the O-to-H direction for a diatomic.
    /// \throws AdsorptionError if the indices do not exist
    [[nodiscard]] Atoms place(const Atoms& adsorbate, std::size_t site,
                              std::size_t binding_atom = 0, std::ptrdiff_t anchor = -1) const;

    /// Several adsorbates at once, on the given sites.
    [[nodiscard]] Atoms place_many(const Atoms& adsorbate, const std::vector<std::size_t>& sites,
                                   std::size_t binding_atom = 0,
                                   std::ptrdiff_t anchor = -1) const;
    /// @}

    /// \name Series of structures
    /// @{

    /// One structure per site, for writing out as a trajectory.
    ///
    /// \param unique_only visit one representative of each distinct site
    ///        rather than every site; the usual thing to want, and on a
    ///        symmetric particle it is a great deal fewer calculations
    [[nodiscard]] std::vector<Atoms> trajectory(const Atoms& adsorbate,
                                                bool unique_only = true,
                                                std::size_t binding_atom = 0,
                                                std::ptrdiff_t anchor = -1) const;

    /// The same, for a single-atom adsorbate.
    [[nodiscard]] std::vector<Atoms> trajectory_atom(const std::string& symbol,
                                                     bool unique_only = true) const;

    /// A series of increasing coverage.
    ///
    /// Structure `k` carries `k + 1` adsorbates. Each new one is placed on the
    /// candidate site furthest from those already occupied, so the series
    /// spreads out rather than clustering in one corner - which is what a
    /// coverage study wants and what simply taking the sites in order does
    /// not give.
    ///
    /// \param candidates which sites may be used; empty means every site
    /// \param steps how many structures to produce; zero means one per
    ///        candidate site, up to all of them
    [[nodiscard]] std::vector<Atoms> coverage_series(
        const Atoms& adsorbate, const std::vector<std::size_t>& candidates = {}, int steps = 0,
        std::size_t binding_atom = 0, std::ptrdiff_t anchor = -1) const;

    /// The coverage of each structure a series returns, as adsorbates per
    /// surface atom.
    [[nodiscard]] std::vector<double> coverage_fractions(
        const std::vector<std::size_t>& candidates = {}, int steps = 0) const;
    /// @}

private:
    Atoms surface_;
    SiteOptions options_;
    std::vector<AdsorptionSite> sites_{};
    std::vector<std::size_t> surface_atoms_{};
    std::vector<std::size_t> unique_{};
    std::vector<std::vector<std::size_t>> equivalent_{};

    /// The order sites are taken in for a coverage series: greedily furthest
    /// from everything already chosen.
    [[nodiscard]] std::vector<std::size_t> spread_order(
        const std::vector<std::size_t>& candidates) const;
};

/// A few small molecules, ready to adsorb, with the binding atom first.
///
/// `molecule("OH")` gives an oxygen at the origin and a hydrogen above it, so
/// placing it with the default binding atom puts the oxygen on the surface and
/// the hydrogen pointing away - which is how hydroxyl actually binds.
[[nodiscard]] Atoms molecule(const std::string& name);

/// Which molecules `molecule` knows.
[[nodiscard]] std::vector<std::string> known_molecules();

}  // namespace fastatoms::adsorption
