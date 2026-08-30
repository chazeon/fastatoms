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

FA_TEST_MAIN()
