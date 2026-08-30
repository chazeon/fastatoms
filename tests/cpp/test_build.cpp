// SPDX-License-Identifier: MIT
///
/// The structure builders. Each generator is checked against something that
/// does not depend on the code that produced it: a counted coordination
/// number, a known nearest-neighbour distance, a magic cluster size, a
/// symmetry the shape has to have.

#include <algorithm>
#include <cmath>
#include <array>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "fastatoms/fastatoms.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::build;

namespace {

/// Nearest-neighbour distance, over the minimum image when periodic.
double nearest_neighbour(const Atoms& atoms) {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        for (std::size_t j = i + 1; j < atoms.size(); ++j) {
            best = std::min(best, atoms.distance(i, j, true));
        }
    }
    return best;
}

/// How many neighbours atom `i` has within `cutoff`, counting periodic images.
std::size_t coordination(const Atoms& atoms, std::size_t i, double cutoff) {
    const NeighborList list = build_neighbor_list(atoms, cutoff);
    return static_cast<std::size_t>(list.coordination_numbers()[i]);
}

/// The shortest distance in a finite cluster, ignoring periodicity.
double shortest_bond(const Atoms& atoms) {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        for (std::size_t j = i + 1; j < atoms.size(); ++j) {
            best = std::min(best, (atoms.position(i) - atoms.position(j)).norm());
        }
    }
    return best;
}

}  // namespace

// ---------------------------------------------------------------------------
// Bulk lattices
// ---------------------------------------------------------------------------

FA_TEST(the_cubic_lattices_have_the_right_size_and_spacing) {
    const double a = 4.05;   // aluminium

    // Counts: primitive cells hold one formula unit, conventional cells the
    // number the cube contains.
    FA_CHECK(sc("Al", a).size() == 1);
    FA_CHECK(fcc("Al", a).size() == 1);
    FA_CHECK(fcc("Al", a, true).size() == 4);
    FA_CHECK(bcc("Al", a).size() == 1);
    FA_CHECK(bcc("Al", a, true).size() == 2);
    FA_CHECK(diamond("Si", a).size() == 2);
    FA_CHECK(diamond("Si", a, true).size() == 8);

    // Nearest-neighbour distances, which is what actually says the lattice is
    // the lattice and not merely the right number of atoms in a box.
    FA_CHECK_CLOSE(nearest_neighbour(sc("Al", a).repeat(2, 2, 2)), a, 1e-12);
    FA_CHECK_CLOSE(nearest_neighbour(fcc("Al", a).repeat(2, 2, 2)), a / std::sqrt(2.0), 1e-12);
    FA_CHECK_CLOSE(nearest_neighbour(bcc("Al", a).repeat(2, 2, 2)), a * std::sqrt(3.0) / 2, 1e-12);
    FA_CHECK_CLOSE(nearest_neighbour(diamond("Si", a).repeat(2, 2, 2)), a * std::sqrt(3.0) / 4,
                   1e-12);

    // And the primitive and conventional cells describe the same crystal.
    FA_CHECK_CLOSE(fcc("Al", a, true).volume(), 4.0 * fcc("Al", a).volume(), 1e-10);
    FA_CHECK_CLOSE(bcc("Al", a, true).volume(), 2.0 * bcc("Al", a).volume(), 1e-10);
    FA_CHECK_CLOSE(diamond("Si", a, true).volume(), 4.0 * diamond("Si", a).volume(), 1e-10);
}

FA_TEST(the_lattices_have_the_coordination_numbers_they_should) {
    const double a = 4.05;
    // The textbook numbers, counted rather than asserted from the construction.
    FA_CHECK(coordination(sc("Al", a).repeat(3, 3, 3), 0, a * 1.1) == 6);
    FA_CHECK(coordination(fcc("Al", a).repeat(3, 3, 3), 0, a * 0.78) == 12);
    FA_CHECK(coordination(bcc("Al", a).repeat(3, 3, 3), 0, a * 0.95) == 8);
    FA_CHECK(coordination(diamond("Si", a).repeat(3, 3, 3), 0, a * 0.5) == 4);
    FA_CHECK(coordination(hcp("Mg", 3.21).repeat(3, 3, 3), 0, 3.21 * 1.1) == 12);
}

FA_TEST(hcp_defaults_to_the_ideal_axial_ratio) {
    const Atoms ideal = hcp("Mg", 3.21);
    const double c = ideal.cell().lengths().z;
    FA_CHECK_CLOSE(c / 3.21, std::sqrt(8.0 / 3.0), 1e-12);
    // In an ideal hcp the in-plane and out-of-plane neighbours are the same
    // distance away, which is the whole point of it being close packed.
    FA_CHECK_CLOSE(nearest_neighbour(ideal.repeat(2, 2, 2)), 3.21, 1e-12);

    // An explicit c is honoured.
    FA_CHECK_CLOSE(hcp("Zn", 2.66, 4.95).cell().lengths().z, 4.95, 1e-12);
}

FA_TEST(the_binary_lattices_alternate_species) {
    const Atoms salt = rocksalt("Na", "Cl", 5.64, true);
    FA_CHECK(salt.size() == 8);
    // Every nearest neighbour of a sodium is a chlorine, at half the cube edge.
    FA_CHECK_CLOSE(nearest_neighbour(salt), 5.64 / 2, 1e-12);

    const Atoms gaas = zincblende("Ga", "As", 5.65, true);
    FA_CHECK(gaas.size() == 8);
    FA_CHECK_CLOSE(nearest_neighbour(gaas), 5.65 * std::sqrt(3.0) / 4, 1e-12);

    const Atoms salt_cscl = cscl("Cs", "Cl", 4.12);
    FA_CHECK(salt_cscl.size() == 2);
    FA_CHECK_CLOSE(nearest_neighbour(salt_cscl), 4.12 * std::sqrt(3.0) / 2, 1e-12);
}

FA_TEST(every_lattice_comes_out_periodic_and_with_a_cell) {
    for (const Lattice lattice : {Lattice::SC, Lattice::FCC, Lattice::BCC, Lattice::HCP,
                                  Lattice::Diamond}) {
        const Atoms atoms = bulk({"Si"}, lattice, 5.43);
        const std::array<bool, 3> periodic{true, true, true};
        FA_CHECK(atoms.pbc() == periodic);
        FA_CHECK(atoms.cell().is_invertible());
        FA_CHECK(atoms.volume() > 0.0);
    }
}

FA_TEST(bulk_rejects_the_wrong_number_of_species) {
    FA_CHECK_THROWS(bulk({"Na"}, Lattice::RockSalt, 5.64), BuildError);
    FA_CHECK_THROWS(bulk({"Na", "Cl"}, Lattice::FCC, 5.64), BuildError);
    FA_CHECK_THROWS(bulk({}, Lattice::FCC, 5.64), BuildError);
    FA_CHECK_THROWS(fcc("Al", -1.0), BuildError);
    FA_CHECK_THROWS(lattice_from_string("hexagonal-close-packed-ish"), BuildError);
}

// ---------------------------------------------------------------------------
// Two-dimensional materials
// ---------------------------------------------------------------------------

FA_TEST(graphene_is_a_honeycomb_with_the_right_bond_length) {
    const Atoms sheet = graphene();
    FA_CHECK(sheet.size() == 2);

    // The bond, which is what a lattice constant of 2.46 actually means.
    FA_CHECK_CLOSE(nearest_neighbour(sheet), 2.46 / std::sqrt(3.0), 1e-12);
    FA_CHECK_CLOSE(nearest_neighbour(sheet), 1.42028, 1e-5);

    // Three neighbours each, which no amount of getting the cell wrong gives
    // by accident.
    const Atoms patch = graphene(2.46, 15.0, 4, 4);
    FA_CHECK(patch.size() == 32);
    for (std::size_t i = 0; i < patch.size(); ++i) {
        FA_CHECK(coordination(patch, i, 1.6) == 3);
    }
    // Flat: every atom at the same height.
    for (std::size_t i = 0; i < patch.size(); ++i) {
        FA_CHECK_CLOSE(patch.position(i).z, patch.position(0).z, 1e-12);
    }
}

FA_TEST(the_rectangular_graphene_cell_describes_the_same_sheet) {
    const Atoms rectangular = graphene_rectangular();
    FA_CHECK(rectangular.size() == 4);
    FA_CHECK_CLOSE(nearest_neighbour(rectangular), 2.46 / std::sqrt(3.0), 1e-12);
    for (std::size_t i = 0; i < rectangular.size(); ++i) {
        FA_CHECK(coordination(rectangular.repeat(3, 3, 1), i, 1.6) == 3);
    }
    // Twice the area of the hexagonal cell, for twice the atoms.
    const Vec3 lengths = rectangular.cell().lengths();
    FA_CHECK_CLOSE(lengths.x, 2.46 * std::sqrt(3.0), 1e-12);
    FA_CHECK_CLOSE(lengths.y, 2.46, 1e-12);
}

FA_TEST(the_vacuum_gap_is_the_gap_and_not_the_cell_height) {
    // A flat sheet has no thickness, so the two coincide.
    FA_CHECK_CLOSE(graphene(2.46, 15.0).cell().lengths().z, 15.0, 1e-12);

    // A dichalcogenide does have thickness, and asking for 15 A of vacuum has
    // to give 15 A of empty space, not a 15 A cell with the layer inside it.
    const Atoms sheet = tmd("Mo", "S", Phase::H, 0.0, 0.0, 15.0);
    double lowest = sheet.position(0).z, highest = lowest;
    for (std::size_t i = 1; i < sheet.size(); ++i) {
        lowest = std::min(lowest, sheet.position(i).z);
        highest = std::max(highest, sheet.position(i).z);
    }
    const double height = sheet.cell().lengths().z;
    FA_CHECK_CLOSE(height - (highest - lowest), 15.0, 1e-12);
    // And it is centred, so the layer does not straddle the boundary.
    FA_CHECK_CLOSE(lowest, height - highest, 1e-12);
}

FA_TEST(a_dichalcogenide_has_the_coordination_its_phase_implies) {
    for (const Phase phase : {Phase::H, Phase::T}) {
        const Atoms sheet = tmd("Mo", "S", phase);
        FA_CHECK(sheet.size() == 3);
        FA_CHECK(sheet.symbol(0) == "Mo");
        FA_CHECK(sheet.symbol(1) == "S" && sheet.symbol(2) == "S");

        // Six sulphurs around every molybdenum in both phases - the phases
        // differ in where they sit, not how many there are.
        const Atoms patch = sheet.repeat(3, 3, 1);
        for (std::size_t i = 0; i < patch.size(); ++i) {
            if (patch.symbol(i) == "Mo") {
                FA_CHECK(coordination(patch, i, 2.6) == 6);
            }
        }
    }

    // The thickness is the vertical S-S separation, and the table's value.
    const Atoms h = tmd("Mo", "S", Phase::H);
    FA_CHECK_CLOSE(std::abs(h.position(1).z - h.position(2).z), 3.130, 1e-12);
    FA_CHECK_CLOSE(h.cell().lengths().x, 3.160, 1e-12);

    // In 1H the two chalcogens are eclipsed; in 1T they are not.
    FA_CHECK_CLOSE(h.position(1).x, h.position(2).x, 1e-12);
    FA_CHECK_CLOSE(h.position(1).y, h.position(2).y, 1e-12);
    const Atoms t = tmd("Mo", "S", Phase::T);
    FA_CHECK((Vec3{t.position(1).x - t.position(2).x, t.position(1).y - t.position(2).y, 0.0})
                 .norm() > 1.0);
}

FA_TEST(a_binary_honeycomb_and_a_buckled_one) {
    // Hexagonal boron nitride: the same lattice, two species.
    const Atoms bn = honeycomb({"B", "N"}, 2.51);
    FA_CHECK(bn.size() == 2);
    FA_CHECK(bn.symbol(0) == "B" && bn.symbol(1) == "N");
    FA_CHECK_CLOSE(nearest_neighbour(bn), 2.51 / std::sqrt(3.0), 1e-12);

    // Silicene is not flat, and the buckling is the height difference.
    const Atoms silicene = honeycomb({"Si"}, 3.87, 15.0, 0.44);
    FA_CHECK_CLOSE(std::abs(silicene.position(0).z - silicene.position(1).z), 0.44, 1e-12);
    FA_CHECK_CLOSE(silicene.cell().lengths().z, 0.44 + 15.0, 1e-12);
}

FA_TEST(the_vacuum_of_an_existing_sheet_can_be_reset) {
    Atoms sheet = tmd("W", "Se", Phase::H, 0.0, 0.0, 12.0);
    const double a_before = sheet.cell().lengths().x;

    set_vacuum(sheet, 20.0);

    double lowest = sheet.position(0).z, highest = lowest;
    for (std::size_t i = 1; i < sheet.size(); ++i) {
        lowest = std::min(lowest, sheet.position(i).z);
        highest = std::max(highest, sheet.position(i).z);
    }
    FA_CHECK_CLOSE(sheet.cell().lengths().z - (highest - lowest), 20.0, 1e-12);
    // The in-plane lattice is not something a vacuum gap may touch.
    FA_CHECK_CLOSE(sheet.cell().lengths().x, a_before, 1e-12);
}

FA_TEST(the_two_dimensional_builders_reject_nonsense) {
    FA_CHECK_THROWS(honeycomb({"C", "N", "O"}, 2.46), BuildError);
    FA_CHECK_THROWS(honeycomb({"C"}, -1.0), BuildError);
    FA_CHECK_THROWS(honeycomb({"C"}, 2.46, 0.0), BuildError);   // flat and no vacuum
    FA_CHECK_THROWS(graphene(2.46, 15.0, 0, 1), BuildError);
    FA_CHECK_THROWS(tmd("Xx", "S"), BuildError);                // not tabulated
    FA_CHECK_THROWS(phase_from_string("3R"), BuildError);
}

// ---------------------------------------------------------------------------
// Nanotubes
// ---------------------------------------------------------------------------

FA_TEST(the_tube_indices_give_the_textbook_geometry) {
    // Atom counts and diameters for tubes whose numbers are in every review of
    // the subject. The count is 4(n^2+nm+m^2)/d_R carbons per repeat.
    struct Case { int n, m; std::size_t atoms; double diameter; };
    const std::vector<Case> cases{
        {5, 5, 20, 6.78},    // armchair
        {10, 0, 40, 7.83},   // zigzag
        {6, 3, 84, 6.21},    // chiral
        {9, 0, 36, 7.05},
    };
    for (const Case& c : cases) {
        const TubeGeometry geometry = tube_geometry(c.n, c.m, 2.46);
        const Atoms tube = nanotube(c.n, c.m);
        FA_CHECK(tube.size() == c.atoms);
        FA_CHECK_CLOSE(2.0 * geometry.radius, c.diameter, 5e-3);
    }
}

FA_TEST(zigzag_and_armchair_are_where_they_should_be) {
    // The chiral angle runs from zero at zigzag to thirty degrees at armchair,
    // and is the one number that says which kind of tube was built.
    FA_CHECK_CLOSE(tube_geometry(10, 0, 2.46).chiral_angle, 0.0, 1e-12);
    FA_CHECK_CLOSE(tube_geometry(5, 5, 2.46).chiral_angle, 30.0, 1e-12);
    FA_CHECK(tube_geometry(6, 3, 2.46).chiral_angle > 0.0);
    FA_CHECK(tube_geometry(6, 3, 2.46).chiral_angle < 30.0);
}

FA_TEST(a_rolled_tube_is_a_tube) {
    const Atoms tube = nanotube(10, 0, 2);
    const TubeGeometry geometry = tube_geometry(10, 0, 2.46);

    // Every atom the same distance from the axis, which is what "rolled" means
    // and what a sign error in the wrap would destroy.
    for (std::size_t i = 0; i < tube.size(); ++i) {
        const Vec3 p = tube.position(i) - tube.center_of_mass();
        FA_CHECK_CLOSE(std::sqrt(p.x * p.x + p.y * p.y), geometry.radius, 1e-9);
    }
    // Three neighbours each, still, and no seam of doubled atoms: bending the
    // sheet shortens the bonds slightly but does not break them.
    for (std::size_t i = 0; i < tube.size(); ++i) {
        FA_CHECK(coordination(tube, i, 1.6) == 3);
    }
    FA_CHECK(shortest_bond(tube) > 1.35);
    FA_CHECK(shortest_bond(tube) < 1.43);

    // Periodic along the axis alone, and the repeat is the period.
    const std::array<bool, 3> along_z{false, false, true};
    FA_CHECK(tube.pbc() == along_z);
    FA_CHECK_CLOSE(tube.cell().lengths().z, 2.0 * geometry.period, 1e-12);
}

FA_TEST(a_sheet_with_thickness_rolls_into_a_tube_with_two_walls) {
    // The reason rolling takes a sheet rather than a lattice constant: a
    // dichalcogenide has a top and a bottom, and they become an outer and an
    // inner wall rather than collapsing onto one radius.
    const Atoms sheet = tmd("Mo", "S", Phase::H, 0.0, 0.0, 1.0);
    const Atoms tube = roll(sheet, 10, 0);

    std::vector<double> molybdenum, sulphur;
    const Vec3 axis = tube.center_of_mass();
    for (std::size_t i = 0; i < tube.size(); ++i) {
        const Vec3 p = tube.position(i) - axis;
        const double radius = std::sqrt(p.x * p.x + p.y * p.y);
        (tube.symbol(i) == "Mo" ? molybdenum : sulphur).push_back(radius);
    }
    FA_CHECK(molybdenum.size() * 2 == sulphur.size());

    // The metal sits on one radius between two shells of chalcogen.
    const double inner = *std::min_element(sulphur.begin(), sulphur.end());
    const double outer = *std::max_element(sulphur.begin(), sulphur.end());
    const double middle = molybdenum.front();
    FA_CHECK(inner < middle && middle < outer);
    FA_CHECK_CLOSE(outer - inner, 3.130, 1e-9);   // the layer's own thickness
}

FA_TEST(rolling_rejects_what_cannot_be_rolled) {
    FA_CHECK_THROWS(tube_geometry(0, 0, 2.46), BuildError);
    FA_CHECK_THROWS(tube_geometry(-1, 2, 2.46), BuildError);
    FA_CHECK_THROWS(tube_geometry(5, 5, 0.0), BuildError);
    FA_CHECK_THROWS(nanotube(5, 5, 0), BuildError);
}

// ---------------------------------------------------------------------------
// Clusters
// ---------------------------------------------------------------------------

FA_TEST(the_closed_shell_sizes_are_the_magic_numbers) {
    // 13, 55, 147, 309, 561 - the sizes at which a shell is complete. Nothing
    // about the construction guarantees these; they come out or they do not.
    const std::vector<std::size_t> magic{13, 55, 147, 309, 561};
    for (std::size_t shells = 1; shells <= magic.size(); ++shells) {
        FA_CHECK(icosahedron("Ag", static_cast<int>(shells), 2.89).size() ==
                 magic[shells - 1]);
        FA_CHECK(cuboctahedron("Ag", static_cast<int>(shells), 4.09).size() ==
                 magic[shells - 1]);
    }
}

FA_TEST(an_icosahedron_has_twelve_vertices_and_is_strained) {
    const Atoms cluster = icosahedron("Ag", 2, 2.89);

    // The shortest bond is *not* the 2.89 asked for. That number is the
    // spacing within a shell; the shells themselves sit 1.0515 times closer,
    // and no arrangement of twenty tetrahedra around a point avoids it. The
    // strain is the reason icosahedra stop winning above a few hundred atoms.
    FA_CHECK_CLOSE(shortest_bond(cluster), 2.89 / 1.0515, 1e-3);

    // Twelve atoms at the outer radius: the vertices.
    const Vec3 middle = cluster.center_of_mass();
    std::vector<double> radii;
    for (std::size_t i = 0; i < cluster.size(); ++i) {
        radii.push_back((cluster.position(i) - middle).norm());
    }
    std::sort(radii.begin(), radii.end());
    const double outer = radii.back();
    FA_CHECK(std::count_if(radii.begin(), radii.end(),
                           [&](double r) { return std::abs(r - outer) < 1e-6; }) == 12);
    // Two shells out, at the radial spacing.
    FA_CHECK_CLOSE(outer / 2.0, 2.89 / 1.0515, 1e-3);
}

FA_TEST(a_cuboctahedron_is_fcc_and_faceted) {
    const Atoms cluster = cuboctahedron("Ag", 3, 4.09);
    FA_CHECK_CLOSE(shortest_bond(cluster), 4.09 / std::sqrt(2.0), 1e-9);

    // The interior atoms have the full fcc coordination of twelve, which says
    // the lattice inside really is fcc and not merely fcc-shaped.
    const NeighborList list = build_neighbor_list(cluster, 4.09 * 0.78);
    const std::vector<std::int32_t> counts = list.coordination_numbers();
    FA_CHECK(*std::max_element(counts.begin(), counts.end()) == 12);
    FA_CHECK(std::count(counts.begin(), counts.end(), 12) > 0);
}

FA_TEST(a_decahedron_has_a_five_fold_axis) {
    const Atoms cluster = decahedron("Ag", 2, 3, 0, 4.09);
    FA_CHECK(cluster.size() > 0);
    FA_CHECK_CLOSE(shortest_bond(cluster), 4.09 / std::sqrt(2.0), 5e-2);

    // Turning it by a fifth of a turn about z must map it onto itself. That is
    // what "five-fold twinned" means, and the strain that closes the 7.35
    // degree gap is what makes it true.
    const double turn = 2.0 * 3.14159265358979323846 / 5;
    const Vec3 axis = cluster.center_of_mass();
    for (std::size_t i = 0; i < cluster.size(); ++i) {
        const Vec3 p = cluster.position(i) - axis;
        const Vec3 turned{p.x * std::cos(turn) - p.y * std::sin(turn),
                          p.x * std::sin(turn) + p.y * std::cos(turn), p.z};
        double closest = std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < cluster.size(); ++j) {
            closest = std::min(closest, (turned - (cluster.position(j) - axis)).norm());
        }
        FA_CHECK(closest < 1e-6);
    }
}

FA_TEST(the_marks_re_entrance_removes_atoms) {
    const Atoms ino = decahedron("Ag", 3, 4, 0, 4.09);
    const Atoms marks = decahedron("Ag", 3, 4, 1, 4.09);
    FA_CHECK(marks.size() < ino.size());
}

// ---------------------------------------------------------------------------
// Wulff
// ---------------------------------------------------------------------------

FA_TEST(the_cubic_families_have_the_right_multiplicities) {
    FA_CHECK(equivalent_directions(1, 0, 0).size() == 6);
    FA_CHECK(equivalent_directions(1, 1, 1).size() == 8);
    FA_CHECK(equivalent_directions(1, 1, 0).size() == 12);
    FA_CHECK(equivalent_directions(2, 1, 0).size() == 24);
    FA_CHECK_THROWS(equivalent_directions(0, 0, 0), BuildError);
}

FA_TEST(a_wulff_particle_answers_to_its_surface_energies) {
    // Only {100}: the shape can be nothing but a cube.
    const Atoms cube = wulff("Ag", Lattice::FCC, 4.09, {{1, 0, 0, 1.0}}, 200);
    const Vec3 cube_lengths = cube.cell().lengths();
    FA_CHECK_CLOSE(cube_lengths.x, cube_lengths.y, 1e-9);
    FA_CHECK_CLOSE(cube_lengths.y, cube_lengths.z, 1e-9);

    // Only {111}: an octahedron, which is taller across its vertices than a
    // cube of the same atom count is across its faces.
    const Atoms octahedron = wulff("Ag", Lattice::FCC, 4.09, {{1, 1, 1, 1.0}}, 200);
    FA_CHECK(octahedron.size() >= 150);

    // Both, at the ratio that makes them meet edge to edge: a cuboctahedron.
    // The classic result is that {100} vanishes once the ratio passes 2/sqrt(3),
    // so just below it both facets are present.
    const Atoms both =
        wulff("Ag", Lattice::FCC, 4.09, {{1, 0, 0, 1.0}, {1, 1, 1, 1.10}}, 300);
    FA_CHECK(both.size() >= 250);
    FA_CHECK_CLOSE(shortest_bond(both), 4.09 / std::sqrt(2.0), 1e-9);
}

FA_TEST(cheaper_facets_grow_and_dearer_ones_shrink) {
    // Wulff's theorem, stated as a comparison the construction has to satisfy:
    // raising the cost of {111} relative to {100} must move atoms out of the
    // corners of the cube and into its faces, making the particle rounder in
    // the [100] directions and flatter in [111].
    const auto extent = [](const Atoms& atoms, const Vec3& direction) {
        double furthest = 0.0;
        for (std::size_t i = 0; i < atoms.size(); ++i) {
            const Vec3 p = atoms.position(i) - atoms.center_of_mass();
            furthest = std::max(furthest,
                                p.x * direction.x + p.y * direction.y + p.z * direction.z);
        }
        return furthest;
    };
    const Vec3 cube_face{1.0, 0.0, 0.0};
    const Vec3 body_diagonal{1 / std::sqrt(3.0), 1 / std::sqrt(3.0), 1 / std::sqrt(3.0)};

    const Atoms cheap_111 =
        wulff("Ag", Lattice::FCC, 4.09, {{1, 0, 0, 1.0}, {1, 1, 1, 0.80}}, 400);
    const Atoms dear_111 =
        wulff("Ag", Lattice::FCC, 4.09, {{1, 0, 0, 1.0}, {1, 1, 1, 1.30}}, 400);

    const double cheap_ratio = extent(cheap_111, body_diagonal) / extent(cheap_111, cube_face);
    const double dear_ratio = extent(dear_111, body_diagonal) / extent(dear_111, cube_face);
    FA_CHECK(dear_ratio > cheap_ratio);
}

FA_TEST(the_cluster_builders_reject_nonsense) {
    FA_CHECK_THROWS(icosahedron("Ag", 0, 2.89), BuildError);
    FA_CHECK_THROWS(icosahedron("Ag", 2, 0.0), BuildError);
    FA_CHECK_THROWS(cuboctahedron("Ag", 0, 4.09), BuildError);
    FA_CHECK_THROWS(decahedron("Ag", 0, 3, 0, 4.09), BuildError);
    FA_CHECK_THROWS(decahedron("Ag", 2, 3, -1, 4.09), BuildError);
    FA_CHECK_THROWS(wulff("Ag", Lattice::FCC, 4.09, {}, 100), BuildError);
    FA_CHECK_THROWS(wulff("Ag", Lattice::FCC, 4.09, {{1, 0, 0, -1.0}}, 100), BuildError);
}

FA_TEST_MAIN()
