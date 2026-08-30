// SPDX-License-Identifier: MIT
//
// The neighbour list is the one piece of FastAtoms with enough index
// arithmetic to hide a subtle bug, so most of these tests check it against a
// brute-force O(N^2 * images) reference rather than against hand-written
// expectations.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "fastatoms/core/elements.hpp"
#include "fastatoms/core/neighbor_list.hpp"
#include "testing.hpp"

using namespace fastatoms;

namespace {

using Pair = std::tuple<int, int, int, int, int>;  // i, j, shift x/y/z

/// Every pair within `cutoff`, found by checking every periodic image within a
/// generous range. Slow but obviously correct.
std::set<Pair> brute_force(const Atoms& atoms, double cutoff, bool self_interaction = false) {
    std::set<Pair> pairs;
    const Cell& cell = atoms.cell();
    const Mat3& m = cell.matrix();

    std::array<int, 3> reach{0, 0, 0};
    if (cell.any_pbc()) {
        const Vec3 heights = cell.heights();
        for (std::size_t k = 0; k < 3; ++k) {
            if (cell.pbc()[k]) {
                reach[k] = static_cast<int>(std::ceil(cutoff / heights[k])) + 1;
            }
        }
    }

    const double cutoff2 = cutoff * cutoff;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        for (std::size_t j = 0; j < atoms.size(); ++j) {
            for (int sx = -reach[0]; sx <= reach[0]; ++sx) {
                for (int sy = -reach[1]; sy <= reach[1]; ++sy) {
                    for (int sz = -reach[2]; sz <= reach[2]; ++sz) {
                        if (i == j && sx == 0 && sy == 0 && sz == 0 && !self_interaction) {
                            continue;
                        }
                        const Vec3 offset =
                            m.row(0) * sx + m.row(1) * sy + m.row(2) * sz;
                        const Vec3 dr = atoms.position(j) + offset - atoms.position(i);
                        if (dr.norm2() < cutoff2) {
                            pairs.emplace(static_cast<int>(i), static_cast<int>(j), sx, sy, sz);
                        }
                    }
                }
            }
        }
    }
    return pairs;
}

std::set<Pair> to_set(const NeighborList& list) {
    std::set<Pair> pairs;
    for (std::size_t k = 0; k < list.size(); ++k) {
        pairs.emplace(list.i[k], list.j[k], list.shifts[3 * k], list.shifts[3 * k + 1],
                      list.shifts[3 * k + 2]);
    }
    return pairs;
}

/// A jittered lattice, which gives a realistic density without needing a file.
Atoms random_structure(std::size_t nx, const Cell& cell, unsigned seed, double jitter = 0.3) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> noise(-jitter, jitter);

    std::vector<int> numbers;
    std::vector<double> positions;
    for (std::size_t a = 0; a < nx; ++a) {
        for (std::size_t b = 0; b < nx; ++b) {
            for (std::size_t c = 0; c < nx; ++c) {
                const Vec3 scaled{(static_cast<double>(a) + 0.5) / static_cast<double>(nx),
                                  (static_cast<double>(b) + 0.5) / static_cast<double>(nx),
                                  (static_cast<double>(c) + 0.5) / static_cast<double>(nx)};
                const Vec3 r = cell.cartesian(scaled) +
                               Vec3{noise(rng), noise(rng), noise(rng)};
                numbers.push_back(14);
                positions.push_back(r.x);
                positions.push_back(r.y);
                positions.push_back(r.z);
            }
        }
    }
    return Atoms(std::move(numbers), std::move(positions), cell);
}

}  // namespace

FA_TEST(simple_cubic_has_six_nearest_neighbors) {
    Atoms atoms(std::vector<std::string>{"Cu"}, {0.0, 0.0, 0.0}, Cell::cubic(3.0));
    const Atoms crystal = atoms.repeat(4, 4, 4);

    const NeighborList list = build_neighbor_list(crystal, 3.1);
    FA_CHECK(list.size() == 6 * crystal.size());
    for (std::size_t a = 0; a < crystal.size(); ++a) {
        FA_CHECK(list.count(a) == 6);
    }

    // Second shell: 6 face + 12 edge neighbours.
    const NeighborList wider = build_neighbor_list(crystal, 3.0 * std::sqrt(2.0) + 0.1);
    FA_CHECK(wider.size() == 18 * crystal.size());
}

FA_TEST(csr_layout_is_sorted_and_consistent) {
    const Atoms crystal = random_structure(4, Cell::cubic(12.0), 7);
    const NeighborList list = build_neighbor_list(crystal, 3.5);

    FA_CHECK(list.natoms() == crystal.size());
    FA_CHECK(list.first.front() == 0);
    FA_CHECK(list.first.back() == list.size());
    for (std::size_t a = 0; a < list.natoms(); ++a) {
        for (std::size_t k = list.first[a]; k < list.first[a + 1]; ++k) {
            FA_CHECK(static_cast<std::size_t>(list.i[k]) == a);
        }
    }
    FA_CHECK(list.distances.size() == list.size());
    FA_CHECK(list.vectors.size() == 3 * list.size());
}

FA_TEST(matches_brute_force_in_a_cubic_cell) {
    const Atoms crystal = random_structure(3, Cell::cubic(9.0), 11);
    const double cutoff = 3.7;
    FA_CHECK(to_set(build_neighbor_list(crystal, cutoff)) == brute_force(crystal, cutoff));
}

FA_TEST(matches_brute_force_in_a_triclinic_cell) {
    const Cell cell = Cell::from_parameters(9.0, 10.0, 11.0, 71.0, 84.0, 105.0);
    const Atoms crystal = random_structure(3, cell, 23);
    const double cutoff = 4.0;
    FA_CHECK(to_set(build_neighbor_list(crystal, cutoff)) == brute_force(crystal, cutoff));
}

FA_TEST(matches_brute_force_with_a_cutoff_larger_than_the_cell) {
    // Multiple images of the same pair must all be reported, each with its own
    // shift. This is where a naive minimum-image implementation breaks.
    const Atoms crystal = random_structure(2, Cell::cubic(4.0), 31, 0.2);
    const double cutoff = 6.5;
    FA_CHECK(to_set(build_neighbor_list(crystal, cutoff)) == brute_force(crystal, cutoff));
}

FA_TEST(matches_brute_force_for_a_slab) {
    Cell cell = Cell::orthorhombic(8.0, 8.0, 30.0);
    cell.set_pbc({true, true, false});
    const Atoms slab = random_structure(3, cell, 47);
    const double cutoff = 4.5;
    FA_CHECK(to_set(build_neighbor_list(slab, cutoff)) == brute_force(slab, cutoff));
}

FA_TEST(matches_brute_force_for_an_isolated_molecule) {
    Atoms cluster = random_structure(3, Cell::cubic(10.0), 53);
    cluster.set_pbc(false);
    const double cutoff = 4.0;
    FA_CHECK(to_set(build_neighbor_list(cluster, cutoff)) == brute_force(cluster, cutoff));

    // A structure with no cell at all must work too.
    Atoms free_cluster(cluster.numbers(), cluster.positions());
    FA_CHECK(to_set(build_neighbor_list(free_cluster, cutoff)) ==
             brute_force(free_cluster, cutoff));
}

FA_TEST(unwrapped_input_gives_shifts_relative_to_the_given_positions) {
    // Atoms deliberately placed outside the cell: the reported shift must
    // satisfy r_ij = r_j + shift * cell - r_i for the positions as given.
    Atoms atoms(std::vector<std::string>{"H", "H"}, {-11.0, 0.5, 0.5, 1.0, 0.5, 0.5},
                Cell::cubic(10.0));
    const NeighborList list = build_neighbor_list(atoms, 3.0);
    FA_CHECK(list.size() == 2);

    const Mat3& m = atoms.cell().matrix();
    for (std::size_t k = 0; k < list.size(); ++k) {
        const Vec3 offset = m.row(0) * list.shifts[3 * k] + m.row(1) * list.shifts[3 * k + 1] +
                            m.row(2) * list.shifts[3 * k + 2];
        const Vec3 expected = atoms.position(static_cast<std::size_t>(list.j[k])) + offset -
                              atoms.position(static_cast<std::size_t>(list.i[k]));
        FA_CHECK_CLOSE(expected.x, list.vectors[3 * k], 1e-12);
        FA_CHECK_CLOSE(expected.norm(), list.distances[k], 1e-12);
    }
}

FA_TEST(half_list_holds_each_pair_once) {
    const Atoms crystal = random_structure(3, Cell::cubic(9.0), 61);
    const double cutoff = 3.6;
    const NeighborList full = build_neighbor_list(crystal, cutoff, true);
    const NeighborList half = build_neighbor_list(crystal, cutoff, false);

    FA_CHECK(half.size() * 2 == full.size());

    // Mirroring the half list must reproduce the full one exactly.
    std::set<Pair> mirrored;
    for (std::size_t k = 0; k < half.size(); ++k) {
        mirrored.emplace(half.i[k], half.j[k], half.shifts[3 * k], half.shifts[3 * k + 1],
                         half.shifts[3 * k + 2]);
        mirrored.emplace(half.j[k], half.i[k], -half.shifts[3 * k], -half.shifts[3 * k + 1],
                         -half.shifts[3 * k + 2]);
    }
    FA_CHECK(mirrored == to_set(full));

    const std::vector<std::int32_t> from_half = half.coordination_numbers();
    const std::vector<std::int32_t> from_full = full.coordination_numbers();
    FA_CHECK(from_half == from_full);
}

FA_TEST(self_interaction_adds_the_zero_shift_pair) {
    const Atoms crystal = random_structure(2, Cell::cubic(8.0), 67);
    NeighborListOptions options;
    options.cutoff = 3.0;
    options.self_interaction = true;
    const NeighborList list = build_neighbor_list(crystal, options);
    FA_CHECK(to_set(list) == brute_force(crystal, 3.0, true));
}

FA_TEST(per_atom_cutoffs_detect_bonds) {
    // O-H at 0.96 A is a bond; the H...H distance of 1.54 A is not.
    Atoms molecule(std::vector<std::string>{"O", "H", "H"},
                   {0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0});
    NeighborListOptions options;
    options.per_atom_cutoffs = natural_cutoffs(molecule.numbers(), 1.2);
    options.full = false;
    const NeighborList bonds = build_neighbor_list(molecule, options);
    FA_CHECK(bonds.size() == 2);

    const NeighborList full_bonds =
        build_neighbor_list(molecule, [&] {
            NeighborListOptions o = options;
            o.full = true;
            return o;
        }());
    const std::vector<std::int32_t> coordination = full_bonds.coordination_numbers();
    FA_CHECK(coordination[0] == 2);
    FA_CHECK(coordination[1] == 1);
}

FA_TEST(degenerate_clusters_do_not_blow_up_the_search) {
    // A collinear or planar non-periodic cluster has essentially zero extent
    // in one or two directions. The bin search range has to stay finite there,
    // or the pair loop never terminates.
    Atoms dimer(std::vector<std::string>{"Ar", "Ar"}, {0.0, 0.0, 0.0, 4.2, 0.0, 0.0});
    const NeighborList pairs = build_neighbor_list(dimer, 20.0);
    FA_CHECK(pairs.size() == 2);
    FA_CHECK_CLOSE(pairs.distances[0], 4.2, 1e-12);

    Atoms square(std::vector<std::string>{"H", "H", "H", "H"},
                 {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0});
    FA_CHECK(build_neighbor_list(square, 5.0).size() == 12);

    Atoms single(std::vector<std::string>{"He"}, {0.0, 0.0, 0.0});
    FA_CHECK(build_neighbor_list(single, 10.0).empty());
}

FA_TEST(invalid_arguments_are_rejected) {
    const Atoms crystal = random_structure(2, Cell::cubic(8.0), 71);
    FA_CHECK_THROWS(build_neighbor_list(crystal, 0.0), std::invalid_argument);
    FA_CHECK_THROWS(build_neighbor_list(crystal, -1.0), std::invalid_argument);

    NeighborListOptions options;
    options.per_atom_cutoffs = {1.0, 2.0};
    FA_CHECK_THROWS(build_neighbor_list(crystal, options), std::invalid_argument);

    Atoms broken(std::vector<std::string>{"H"}, {0.0, 0.0, 0.0});
    broken.set_pbc(true);
    FA_CHECK_THROWS(build_neighbor_list(broken, 2.0), std::invalid_argument);

    const Atoms empty;
    FA_CHECK(build_neighbor_list(empty, 2.0).empty());
}

FA_TEST_MAIN()
