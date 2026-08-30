// SPDX-License-Identifier: MIT
///
/// Adsorption sites. The sites found on a closed nanoparticle are checked
/// against Euler's formula: the atop sites are its vertices, the bridges its
/// edges and the hollows its faces, so V - E + F must be two. Nothing in the
/// site search knows that, and almost nothing that is wrong with a site search
/// leaves it true.

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "fastatoms/fastatoms.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::adsorption;

namespace {

std::map<std::string, int> counts_of(const AdsorptionFinder& finder) {
    std::map<std::string, int> counts;
    for (const AdsorptionSite& site : finder.sites()) {
        ++counts[std::string(site_type_name(site.type))];
    }
    return counts;
}

Atoms platinum_icosahedron(int shells) { return build::icosahedron("Pt", shells, 2.77); }

}  // namespace

// ---------------------------------------------------------------------------
// Site detection
// ---------------------------------------------------------------------------

FA_TEST(the_sites_of_a_closed_particle_satisfy_eulers_formula) {
    // Vertices, edges and faces of the surface polyhedron. If the search
    // invented a face, missed an edge or found a site inside the particle,
    // this stops being two.
    struct Case { Atoms particle; };
    const std::vector<Atoms> particles{
        platinum_icosahedron(2), platinum_icosahedron(3),
        build::cuboctahedron("Pt", 2, 3.92), build::cuboctahedron("Pt", 3, 3.92),
    };
    for (const Atoms& particle : particles) {
        const AdsorptionFinder finder(particle, SiteOptions{1.9});
        const std::map<std::string, int> counts = counts_of(finder);

        const int vertices = static_cast<int>(finder.surface_atoms().size());
        const int edges = counts.count("bridge") ? counts.at("bridge") : 0;
        const int faces = (counts.count("hollow") ? counts.at("hollow") : 0) +
                          (counts.count("4-fold") ? counts.at("4-fold") : 0);
        FA_CHECK(vertices - edges + faces == 2);

        // And every surface atom carries exactly one atop site.
        FA_CHECK(counts.at("atop") == vertices);
    }
}

FA_TEST(only_a_surface_with_square_facets_has_four_fold_sites) {
    // An icosahedron is made of triangles and has no square facet anywhere on
    // it. Building a four-fold site out of a triangle plus a fourth atom finds
    // every rhombus of two adjacent triangles instead, which is wrong: the
    // centre of such a rhombus is over the shared edge, which is the bridge
    // site already found.
    const AdsorptionFinder icosahedron(platinum_icosahedron(2), SiteOptions{1.9});
    FA_CHECK(icosahedron.sites_of_type(SiteType::FourFold).empty());

    // A cuboctahedron has six square facets, and they do carry four-fold sites.
    const AdsorptionFinder cuboctahedron(build::cuboctahedron("Pt", 2, 3.92), SiteOptions{1.9});
    FA_CHECK(cuboctahedron.sites_of_type(SiteType::FourFold).size() == 24);
}

FA_TEST(the_interior_of_a_particle_is_not_surface) {
    // A Mackay icosahedron of two shells is a thirteen-atom core inside a
    // forty-two atom skin, and only the skin can adsorb anything.
    const AdsorptionFinder finder(platinum_icosahedron(2), SiteOptions{1.9});
    FA_CHECK(finder.surface().size() == 55);
    FA_CHECK(finder.surface_atoms().size() == 42);

    // Every site sits outside the particle, further from the centre than the
    // atoms that form it.
    const Vec3 centre = finder.surface().center_of_mass();
    for (const AdsorptionSite& site : finder.sites()) {
        const double site_radius = (site.position - centre).norm();
        for (const std::size_t atom : site.atoms) {
            FA_CHECK(site_radius > (finder.surface().position(atom) - centre).norm());
        }
        // And the normal points outwards, not in.
        FA_CHECK(dot(site.normal, site.centre - centre) > 0.0);
    }
}

FA_TEST(the_site_types_can_be_asked_for_separately) {
    SiteOptions only_hollow;
    only_hollow.height = 1.9;
    only_hollow.atop = false;
    only_hollow.bridge = false;
    only_hollow.four_fold = false;
    const AdsorptionFinder finder(platinum_icosahedron(2), only_hollow);
    FA_CHECK(finder.sites().size() == 80);
    for (const AdsorptionSite& site : finder.sites()) {
        FA_CHECK(site.type == SiteType::Hollow);
    }

    FA_CHECK(site_type_from_string("3-fold") == SiteType::Hollow);
    FA_CHECK(site_type_from_string("ontop") == SiteType::Atop);
    FA_CHECK_THROWS(site_type_from_string("quintuple"), AdsorptionError);
}

// ---------------------------------------------------------------------------
// Symmetry reduction
// ---------------------------------------------------------------------------

FA_TEST(the_unique_sites_of_an_icosahedron_are_what_its_symmetry_says) {
    const AdsorptionFinder finder(platinum_icosahedron(2), SiteOptions{1.9});

    // Two kinds of atop site, because a two-shell icosahedron has twelve
    // vertex atoms and thirty edge atoms and nothing else.
    const std::vector<std::size_t> atop = finder.sites_of_type(SiteType::Atop, true);
    FA_CHECK(atop.size() == 2);
    std::vector<std::size_t> multiplicities;
    for (const std::size_t site : atop) {
        multiplicities.push_back(finder.equivalent_sites(finder.sites()[site].unique).size());
    }
    std::sort(multiplicities.begin(), multiplicities.end());
    FA_CHECK(multiplicities[0] == 12);   // the vertices
    FA_CHECK(multiplicities[1] == 30);   // the edge atoms

    // Two kinds of hollow: the twenty at the centres of the icosahedron's own
    // faces, and the sixty others.
    std::vector<std::size_t> hollows;
    for (const std::size_t site : finder.sites_of_type(SiteType::Hollow, true)) {
        hollows.push_back(finder.equivalent_sites(finder.sites()[site].unique).size());
    }
    std::sort(hollows.begin(), hollows.end());
    FA_CHECK(hollows.size() == 2);
    FA_CHECK(hollows[0] == 20);
    FA_CHECK(hollows[1] == 60);
}

FA_TEST(every_site_belongs_to_exactly_one_class) {
    const AdsorptionFinder finder(platinum_icosahedron(2), SiteOptions{1.9});
    std::size_t covered = 0;
    std::vector<char> seen(finder.sites().size(), 0);
    for (std::size_t k = 0; k < finder.unique_sites().size(); ++k) {
        for (const std::size_t site : finder.equivalent_sites(k)) {
            FA_CHECK(!seen[site]);
            seen[site] = 1;
            FA_CHECK(finder.sites()[site].unique == k);
            ++covered;
        }
    }
    FA_CHECK(covered == finder.sites().size());
    FA_CHECK_THROWS(finder.equivalent_sites(finder.unique_sites().size()), AdsorptionError);
}

// ---------------------------------------------------------------------------
// Placement and orientation
// ---------------------------------------------------------------------------

FA_TEST(an_atom_lands_at_the_site_and_nowhere_else) {
    const AdsorptionFinder finder(platinum_icosahedron(2), SiteOptions{1.9});
    const std::size_t site = finder.unique_sites().front();

    const Atoms decorated = finder.place_atom("H", site);
    FA_CHECK(decorated.size() == finder.surface().size() + 1);
    FA_CHECK(decorated.symbol(decorated.size() - 1) == "H");
    FA_CHECK_CLOSE((decorated.position(decorated.size() - 1) - finder.sites()[site].position)
                       .norm(),
                   0.0, 1e-12);
    // The surface itself is untouched.
    for (std::size_t k = 0; k < finder.surface().size(); ++k) {
        FA_CHECK_CLOSE((decorated.position(k) - finder.surface().position(k)).norm(), 0.0,
                       1e-15);
    }
}

FA_TEST(a_hydroxyl_binds_through_its_oxygen_with_the_hydrogen_pointing_out) {
    // The thing the orientation code exists for. Left unturned, an OH built
    // along z would lie flat or bury its hydrogen on most of the facets of a
    // particle; it has to be rotated so the O-H direction is the outward
    // normal of whichever site it lands on.
    const AdsorptionFinder finder(platinum_icosahedron(2), SiteOptions{1.9});
    const Vec3 centre = finder.surface().center_of_mass();

    for (const std::size_t site : finder.unique_sites()) {
        const Atoms decorated = finder.place(molecule("OH"), site);
        const std::size_t n = finder.surface().size();
        FA_CHECK(decorated.symbol(n) == "O");
        FA_CHECK(decorated.symbol(n + 1) == "H");

        // The oxygen is at the site.
        FA_CHECK_CLOSE((decorated.position(n) - finder.sites()[site].position).norm(), 0.0,
                       1e-12);
        // The hydrogen is further out than the oxygen, always.
        FA_CHECK((decorated.position(n + 1) - centre).norm() >
                 (decorated.position(n) - centre).norm());
        // And the O-H direction is the site normal, to within rounding.
        const Vec3 bond = decorated.position(n + 1) - decorated.position(n);
        const Vec3 unit{bond.x / bond.norm(), bond.y / bond.norm(), bond.z / bond.norm()};
        FA_CHECK_CLOSE(dot(unit, finder.sites()[site].normal), 1.0, 1e-9);
        // The bond length survived the rotation.
        FA_CHECK_CLOSE(bond.norm(), 0.97, 1e-9);
    }
}

FA_TEST(the_binding_atom_and_the_anchor_can_both_be_chosen) {
    const AdsorptionFinder finder(platinum_icosahedron(2), SiteOptions{1.9});
    const std::size_t site = finder.unique_sites().front();
    const std::size_t n = finder.surface().size();

    // Bind a carbon monoxide through its oxygen instead of its carbon.
    const Atoms through_oxygen = finder.place(molecule("CO"), site, 1);
    FA_CHECK(through_oxygen.symbol(n + 1) == "O");
    FA_CHECK_CLOSE((through_oxygen.position(n + 1) - finder.sites()[site].position).norm(), 0.0,
                   1e-12);

    // Water, oriented by one of its hydrogens rather than by the centroid.
    const Atoms water = finder.place(molecule("H2O"), site, 0, 1);
    const Vec3 bond = water.position(n + 1) - water.position(n);
    const Vec3 unit{bond.x / bond.norm(), bond.y / bond.norm(), bond.z / bond.norm()};
    FA_CHECK_CLOSE(dot(unit, finder.sites()[site].normal), 1.0, 1e-9);

    FA_CHECK_THROWS(finder.place(molecule("OH"), site, 5), AdsorptionError);
    FA_CHECK_THROWS(finder.place(molecule("OH"), site, 0, 0), AdsorptionError);
    FA_CHECK_THROWS(finder.place_atom("H", 10000), AdsorptionError);
    FA_CHECK_THROWS(molecule("unobtainium"), AdsorptionError);
}

// ---------------------------------------------------------------------------
// Trajectories and coverage
// ---------------------------------------------------------------------------

FA_TEST(a_trajectory_has_one_frame_per_site) {
    const AdsorptionFinder finder(platinum_icosahedron(2), SiteOptions{1.9});

    const std::vector<Atoms> unique = finder.trajectory_atom("H", true);
    FA_CHECK(unique.size() == finder.unique_sites().size());

    const std::vector<Atoms> all = finder.trajectory_atom("H", false);
    FA_CHECK(all.size() == finder.sites().size());

    for (const Atoms& frame : all) {
        FA_CHECK(frame.size() == finder.surface().size() + 1);
    }
    // Visiting only the unique sites is the point: six calculations instead of
    // two hundred and forty-two.
    FA_CHECK(unique.size() * 10 < all.size());

    const std::vector<Atoms> hydroxyls = finder.trajectory(molecule("OH"), true);
    FA_CHECK(hydroxyls.front().size() == finder.surface().size() + 2);
}

FA_TEST(a_coverage_series_grows_and_spreads) {
    const AdsorptionFinder finder(platinum_icosahedron(2), SiteOptions{1.9});
    const std::vector<std::size_t> hollows = finder.sites_of_type(SiteType::Hollow);

    const std::vector<Atoms> series = finder.coverage_series(molecule("H"), hollows, 8);
    FA_CHECK(series.size() == 8);
    for (std::size_t k = 0; k < series.size(); ++k) {
        // One more adsorbate each time.
        FA_CHECK(series[k].size() == finder.surface().size() + k + 1);
    }

    const std::vector<double> fractions = finder.coverage_fractions(hollows, 8);
    FA_CHECK(fractions.size() == 8);
    FA_CHECK_CLOSE(fractions[0], 1.0 / 42, 1e-12);
    FA_CHECK_CLOSE(fractions[7], 8.0 / 42, 1e-12);

    // The adsorbates spread out rather than piling into one corner: the
    // closest pair among the first six is further apart than the site spacing.
    const Atoms sixth = series[5];
    double closest = 1e9;
    for (std::size_t a = finder.surface().size(); a < sixth.size(); ++a) {
        for (std::size_t b = a + 1; b < sixth.size(); ++b) {
            closest = std::min(closest, (sixth.position(a) - sixth.position(b)).norm());
        }
    }
    FA_CHECK(closest > 4.0);
}

FA_TEST(the_finder_refuses_what_it_cannot_work_with) {
    FA_CHECK_THROWS(AdsorptionFinder(Atoms{}), AdsorptionError);

    SiteOptions bad_height;
    bad_height.height = -1.0;
    FA_CHECK_THROWS(AdsorptionFinder(platinum_icosahedron(2), bad_height), AdsorptionError);

    // Two atoms far enough apart to have no bonds at all.
    Atoms scattered(std::vector<std::string>{"Pt", "Pt"}, {0.0, 0.0, 0.0, 40.0, 0.0, 0.0});
    FA_CHECK_THROWS(AdsorptionFinder(scattered), AdsorptionError);

    // A bulk crystal, where every atom has the same coordination.
    FA_CHECK_THROWS(AdsorptionFinder(build::fcc("Pt", 3.92, true).repeat(3, 3, 3)),
                    AdsorptionError);
}

FA_TEST_MAIN()
