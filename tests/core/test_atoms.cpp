// SPDX-License-Identifier: MIT
#include "fastatoms/core/atoms.hpp"
#include "testing.hpp"

using namespace fastatoms;

namespace {
Atoms two_atoms() {
    Atoms atoms(std::vector<std::string>{"Fe", "O"}, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                Cell(Mat3::from_rows({2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 2.0}),
                     {true, true, true}));
    atoms.set_string_array("labels", {"Fe1", "O"});
    atoms.set_int_array("type", {1, 2});
    atoms.info()["comment"] = Value("hello");
    return atoms;
}
}  // namespace

FA_TEST(labels_fall_back_to_symbols) {
    Atoms atoms(std::vector<std::string>{"Si", "C"}, {0, 0, 0, 1, 1, 1});
    FA_CHECK(atoms.labels() == (std::vector<std::string>{"Si", "C"}));
    atoms.set_string_array("labels", {"Si1", "C_h"});
    FA_CHECK(atoms.labels()[1] == "C_h");
}

FA_TEST(repeat_uses_ase_order_and_keeps_arrays) {
    const Atoms atoms = two_atoms();
    const Atoms big = atoms.repeat(2, 1, 3);
    FA_CHECK(big.size() == 12);
    // Atom index runs fastest: image k holds atoms 2k, 2k+1.
    for (std::size_t i = 0; i < big.size(); ++i) {
        FA_CHECK(big.labels()[i] == (i % 2 == 0 ? "Fe1" : "O"));
        FA_CHECK(big.int_array("type").data[i] == static_cast<std::int64_t>(i % 2 + 1));
    }
    // Image order: x slowest, z fastest (ASE): image 1 is shifted along z.
    FA_CHECK_CLOSE(big.position(2).z, 2.0, 1e-12);
    FA_CHECK_CLOSE(big.position(6).x, 2.0, 1e-12);
    FA_CHECK_CLOSE(big.cell().matrix().row(2).z, 6.0, 1e-12);
    FA_CHECK(big.info()["comment"].as_string() == "hello");
}

FA_TEST(subset_and_extend_keep_string_and_int_arrays) {
    const Atoms atoms = two_atoms();
    Atoms one = atoms.subset({1});
    FA_CHECK(one.labels() == std::vector<std::string>{"O"});
    FA_CHECK(one.int_array("type").data == std::vector<std::int64_t>{2});

    Atoms plain(std::vector<std::string>{"H"}, {0.5, 0.5, 0.5});
    one.extend(plain);
    FA_CHECK(one.size() == 2);
    FA_CHECK(one.labels()[1] == "H");             // filled from the symbol
    FA_CHECK(one.int_array("type").data[1] == 0);  // filled with zero

    plain.extend(atoms);
    FA_CHECK(plain.labels() == (std::vector<std::string>{"H", "Fe1", "O"}));
}

FA_TEST(array_length_is_checked) {
    Atoms atoms = two_atoms();
    FA_CHECK_THROWS(atoms.set_string_array("labels", {"x"}), std::invalid_argument);
    FA_CHECK_THROWS(atoms.set_int_array("image", {0, 0, 0}, 3), std::invalid_argument);
    FA_CHECK_THROWS((void)atoms.string_array("nope"), std::out_of_range);
}

FA_TEST(append_labels_new_atom_with_its_symbol) {
    Atoms atoms = two_atoms();
    atoms.append("C", Vec3{0.5, 0.5, 0.5});
    FA_CHECK(atoms.labels()[2] == "C");
}

FA_TEST(extend_with_itself) {
    Atoms atoms = two_atoms();
    atoms.extend(atoms);
    FA_CHECK(atoms.size() == 4);
    FA_CHECK(atoms.labels()[2] == "Fe1");
}

FA_TEST(right_angles_are_exact) {
    const Cell cubic = Cell::from_parameters(5, 5, 5, 90, 90, 90);
    const Mat3& m = cubic.matrix();
    FA_CHECK(m.row(1).x == 0.0 && m.row(2).x == 0.0 && m.row(2).y == 0.0);
    const Cell hex = Cell::from_parameters(3, 3, 15, 90, 90, 120);
    FA_CHECK(hex.matrix().row(2).x == 0.0 && hex.matrix().row(2).y == 0.0);
}

FA_TEST_MAIN()
