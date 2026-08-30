// SPDX-License-Identifier: MIT
#include <cmath>
#include <stdexcept>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/core/units.hpp"
#include "testing.hpp"

using namespace fastatoms;

namespace {

constexpr double kHalfPi = 1.57079632679489661923;

Atoms water() {
    return Atoms(std::vector<std::string>{"O", "H", "H"},
                 {0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0});
}

}  // namespace

FA_TEST(construction_and_species) {
    const Atoms atoms = water();
    FA_CHECK(atoms.size() == 3);
    FA_CHECK(atoms.number(0) == 8);
    FA_CHECK(atoms.symbol(1) == "H");
    FA_CHECK(atoms.formula() == "H2O");
    FA_CHECK_CLOSE(atoms.mass(0), 15.999, 1e-9);

    FA_CHECK_THROWS(Atoms(std::vector<int>{1, 1}, {0.0, 0.0, 0.0}), std::invalid_argument);
    FA_CHECK_THROWS(Atoms(std::vector<std::string>{"Xx"}, {0.0, 0.0, 0.0}),
                    std::invalid_argument);
}

FA_TEST(hill_formula_ordering) {
    const Atoms ethanol(std::vector<std::string>{"O", "H", "C", "H", "C", "H"},
                        std::vector<double>(18, 0.0));
    FA_CHECK(ethanol.formula() == "C2H3O");

    const Atoms salt(std::vector<std::string>{"Na", "Cl"}, std::vector<double>(6, 0.0));
    FA_CHECK(salt.formula() == "ClNa");
}

FA_TEST(scaled_positions_round_trip) {
    Atoms atoms(std::vector<std::string>{"Si", "Si"},
                {0.0, 0.0, 0.0, 1.3575, 1.3575, 1.3575}, Cell::cubic(5.43));
    const std::vector<double> scaled = atoms.scaled_positions();
    FA_CHECK_CLOSE(scaled[3], 0.25, 1e-12);
    atoms.set_scaled_positions(scaled);
    FA_CHECK_CLOSE(atoms.position(1).x, 1.3575, 1e-12);

    Atoms free_atoms = water();
    FA_CHECK_THROWS(free_atoms.scaled_positions(), std::invalid_argument);
}

FA_TEST(wrap_folds_atoms_back) {
    Atoms atoms(std::vector<std::string>{"H"}, {11.0, -1.0, 5.0}, Cell::cubic(10.0));
    atoms.wrap();
    FA_CHECK_CLOSE(atoms.position(0).x, 1.0, 1e-9);
    FA_CHECK_CLOSE(atoms.position(0).y, 9.0, 1e-9);
}

FA_TEST(distances_use_the_minimum_image_when_asked) {
    Atoms atoms(std::vector<std::string>{"H", "H"}, {0.0, 0.0, 0.0, 9.0, 0.0, 0.0},
                Cell::cubic(10.0));
    FA_CHECK_CLOSE(atoms.distance(0, 1, false), 9.0, 1e-12);
    FA_CHECK_CLOSE(atoms.distance(0, 1, true), 1.0, 1e-12);

    const std::vector<double> matrix = atoms.all_distances(true);
    FA_CHECK_CLOSE(matrix[1], 1.0, 1e-12);
    FA_CHECK_CLOSE(matrix[0], 0.0, 1e-12);
    FA_CHECK_THROWS(atoms.distance(0, 5), std::out_of_range);
}

FA_TEST(center_of_mass_and_geometry) {
    Atoms atoms(std::vector<std::string>{"H", "H"}, {0.0, 0.0, 0.0, 2.0, 0.0, 0.0});
    FA_CHECK_CLOSE(atoms.center_of_mass().x, 1.0, 1e-12);

    // A heavy atom pulls the centre of mass towards itself.
    atoms.set_number(1, 8);
    FA_CHECK(atoms.center_of_mass().x > 1.5);
    FA_CHECK_CLOSE(atoms.center_of_geometry().x, 1.0, 1e-12);
}

FA_TEST(translate_and_rotate) {
    Atoms atoms(std::vector<std::string>{"H"}, {1.0, 0.0, 0.0});
    atoms.translate(Vec3{0.0, 2.0, 0.0});
    FA_CHECK_CLOSE(atoms.position(0).y, 2.0, 1e-12);

    Atoms pair(std::vector<std::string>{"H", "H"}, {1.0, 0.0, 0.0, -1.0, 0.0, 0.0});
    pair.rotate(kHalfPi, Vec3{0.0, 0.0, 1.0}, Vec3{0.0, 0.0, 0.0});
    FA_CHECK_CLOSE(pair.position(0).x, 0.0, 1e-12);
    FA_CHECK_CLOSE(pair.position(0).y, 1.0, 1e-12);
    FA_CHECK_THROWS(pair.rotate(1.0, Vec3{0.0, 0.0, 0.0}, Vec3{}), std::invalid_argument);
}

FA_TEST(center_adds_vacuum) {
    Atoms atoms(std::vector<std::string>{"H", "H"}, {0.0, 0.0, 0.0, 2.0, 0.0, 0.0});
    atoms.center(5.0);
    // 2 angstrom of atoms plus 5 on each side.
    FA_CHECK_CLOSE(atoms.cell().lengths().x, 12.0, 1e-9);
    FA_CHECK_CLOSE(atoms.position(0).x, 5.0, 1e-9);
    FA_CHECK_CLOSE(atoms.position(1).x, 7.0, 1e-9);
}

FA_TEST(repeat_builds_a_supercell) {
    Atoms atoms(std::vector<std::string>{"Si"}, {0.0, 0.0, 0.0}, Cell::cubic(5.0));
    atoms.set_tags({7});
    const Atoms super = atoms.repeat(2, 3, 4);
    FA_CHECK(super.size() == 24);
    FA_CHECK_CLOSE(super.cell().lengths().x, 10.0, 1e-12);
    FA_CHECK_CLOSE(super.cell().lengths().z, 20.0, 1e-12);
    FA_CHECK_CLOSE(super.volume(), 125.0 * 24.0, 1e-9);
    FA_CHECK(super.has_tags());
    FA_CHECK(super.tags()[23] == 7);
    FA_CHECK_THROWS(atoms.repeat(0, 1, 1), std::invalid_argument);
}

FA_TEST(subset_extend_and_remove_keep_arrays_consistent) {
    Atoms atoms = water();
    atoms.set_charges({-0.8, 0.4, 0.4});
    atoms.set_array("forces", std::vector<double>(9, 1.5), 3);

    const Atoms hydrogens = atoms.subset({1, 2});
    FA_CHECK(hydrogens.size() == 2);
    FA_CHECK(hydrogens.formula() == "H2");
    FA_CHECK_CLOSE(hydrogens.charges()[0], 0.4, 1e-12);
    FA_CHECK(hydrogens.array("forces").data.size() == 6);

    Atoms combined = atoms;
    combined.extend(hydrogens);
    FA_CHECK(combined.size() == 5);
    FA_CHECK(combined.charges().size() == 5);
    FA_CHECK(combined.array("forces").data.size() == 15);

    combined.remove({0, 0, 4});
    FA_CHECK(combined.size() == 3);
    FA_CHECK(combined.formula() == "H3");
}

FA_TEST(extend_materialises_missing_arrays) {
    Atoms charged = water();
    charged.set_charges({-0.8, 0.4, 0.4});
    Atoms plain = water();

    charged.extend(plain);
    FA_CHECK(charged.charges().size() == 6);
    FA_CHECK_CLOSE(charged.charges()[5], 0.0, 1e-12);

    plain.extend(charged);
    FA_CHECK(plain.has_charges());
    FA_CHECK_CLOSE(plain.charges()[0], 0.0, 1e-12);
    FA_CHECK_CLOSE(plain.charges()[3], -0.8, 1e-12);
}

FA_TEST(append_grows_optional_arrays) {
    Atoms atoms = water();
    atoms.set_charges({-0.8, 0.4, 0.4});
    atoms.append("Na", Vec3{3.0, 0.0, 0.0});
    FA_CHECK(atoms.size() == 4);
    FA_CHECK(atoms.charges().size() == 4);
    FA_CHECK_CLOSE(atoms.charges()[3], 0.0, 1e-12);
}

FA_TEST(velocities_momenta_and_temperature) {
    Atoms atoms(std::vector<std::string>{"H"}, {0.0, 0.0, 0.0});
    FA_CHECK_CLOSE(atoms.kinetic_energy(), 0.0, 1e-12);

    atoms.set_velocities({1.0, 0.0, 0.0});
    const double mass = atoms.mass(0);
    FA_CHECK_CLOSE(atoms.momenta()[0], mass, 1e-12);
    FA_CHECK_CLOSE(atoms.kinetic_energy(), 0.5 * mass, 1e-12);
    FA_CHECK_CLOSE(atoms.temperature(), 2.0 * 0.5 * mass / (3.0 * units::kB), 1e-9);
    FA_CHECK_CLOSE(atoms.velocities()[0], 1.0, 1e-12);

    atoms.zero_center_of_mass_momentum();
    FA_CHECK_CLOSE(atoms.total_momentum().norm(), 0.0, 1e-12);
}

FA_TEST(set_cell_can_scale_the_atoms) {
    Atoms atoms(std::vector<std::string>{"Si"}, {1.0, 1.0, 1.0}, Cell::cubic(4.0));
    atoms.set_cell(Mat3::diagonal(8.0, 8.0, 8.0), true);
    FA_CHECK_CLOSE(atoms.position(0).x, 2.0, 1e-12);

    atoms.set_cell(Mat3::diagonal(4.0, 4.0, 4.0), false);
    FA_CHECK_CLOSE(atoms.position(0).x, 2.0, 1e-12);
}

FA_TEST(reserved_array_names_are_rejected) {
    Atoms atoms = water();
    FA_CHECK_THROWS(atoms.set_array("positions", std::vector<double>(9, 0.0), 3),
                    std::invalid_argument);
    FA_CHECK_THROWS(atoms.set_array("forces", std::vector<double>(8, 0.0), 3),
                    std::invalid_argument);
    atoms.set_array("forces", std::vector<double>(9, 0.0), 3);
    FA_CHECK(atoms.has_array("forces"));
    atoms.remove_array("forces");
    FA_CHECK(!atoms.has_array("forces"));
}

FA_TEST_MAIN()
