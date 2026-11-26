// SPDX-License-Identifier: MIT
#include "fastatoms/core/atoms.hpp"
#include "testing.hpp"

#include <limits>
#include <stdexcept>

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

FA_TEST(array_shape_overflow_is_rejected) {
    Atoms atoms(std::vector<int>{1, 1}, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0});
    // 2 * (SIZE_MAX / 2 + 1) wraps to 0, which would match empty data.
    const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2 + 1;
    FA_CHECK_THROWS(atoms.set_array("big", {}, huge), std::length_error);
    FA_CHECK_THROWS(atoms.set_int_array("big", {}, huge), std::length_error);
    FA_CHECK(!atoms.has_array("big"));
    FA_CHECK(!atoms.has_int_array("big"));
}

FA_TEST(to_string_handles_huge_cell) {
    const Atoms atoms(std::vector<int>{1}, {0.0, 0.0, 0.0}, Cell::cubic(1e40));
    const std::string text = atoms.to_string();
    // The %.4f rendering of 1e40 is far longer than the old 96-byte buffer.
    FA_CHECK(text.size() > 96);
    FA_CHECK(text.find("cell=[1000000000") != std::string::npos);
    FA_CHECK(text.find("pbc=[True, True, True])") != std::string::npos);
    const std::string cell_text = Cell::cubic(1e40).to_string();
    FA_CHECK(cell_text.find("1e+40") != std::string::npos);
}

FA_TEST(minimum_image_is_defined_for_large_coordinates) {
    const Cell unit = Cell::cubic(1.0);
    // 2^31 + 0.25 used to be cast to int before the image search.
    std::array<int, 3> offset{};
    const Vec3 image = unit.minimum_image(Vec3{2147483648.25, 0.0, 0.0}, offset);
    FA_CHECK_CLOSE(image.x, 0.25, 1e-6);
    FA_CHECK_CLOSE(image.y, 0.0, 1e-12);
    FA_CHECK(offset[0] == std::numeric_limits<int>::min());  // representable: -2^31
    FA_CHECK(offset[1] == 0 && offset[2] == 0);
    FA_CHECK_CLOSE(unit.minimum_image(Vec3{2147483648.25, 0.0, 0.0}).x, 0.25, 1e-6);

    // -2^32 is not representable as an int offset.
    FA_CHECK_THROWS(unit.minimum_image(Vec3{4294967296.0, 0.0, 0.0}, offset), std::overflow_error);
}

FA_TEST(value_as_int_rejects_out_of_range_floats) {
    FA_CHECK_THROWS(Value(1e30).as_int(), InputError);
    FA_CHECK_THROWS(Value(9223372036854775808.0).as_int(), InputError);  // 2^63
    FA_CHECK(Value(-9223372036854775808.0).as_int() == std::numeric_limits<std::int64_t>::min());
    FA_CHECK(Value(4096.0).as_int() == 4096);
    FA_CHECK_THROWS(Value(1.5).as_int(), InputError);
}

FA_TEST(rank_deficient_cell_uses_completed_basis_like_ase) {
    // ASE: complete_cell adds (0,0,1); scaled_positions((1,1.5,4)) == (0.5,0.5,4).
    const Mat3 matrix = Mat3::from_rows({2.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 0.0});
    const Cell cell(matrix, {true, true, false});
    FA_CHECK(!cell.is_invertible());  // the raw matrix is singular
    FA_CHECK(cell.has_scaled_coordinates());

    Atoms atoms(std::vector<int>{1}, {1.0, 1.5, 4.0}, cell);
    const std::vector<double> s = atoms.scaled_positions();
    FA_CHECK_CLOSE(s[0], 0.5, 1e-12);
    FA_CHECK_CLOSE(s[1], 0.5, 1e-12);
    FA_CHECK_CLOSE(s[2], 4.0, 1e-12);

    atoms.set_scaled_positions({0.25, 0.5, 1.0});
    FA_CHECK_CLOSE(atoms.position(0).x, 0.5, 1e-12);
    FA_CHECK_CLOSE(atoms.position(0).y, 1.5, 1e-12);
    FA_CHECK_CLOSE(atoms.position(0).z, 1.0, 1e-12);

    // Wrapping folds only the periodic fractional coordinates: x 5 -> 1.
    atoms.set_position(0, Vec3{5.0, 1.5, 4.0});
    atoms.wrap();
    FA_CHECK_CLOSE(atoms.position(0).x, 1.0, 1e-12);
    FA_CHECK_CLOSE(atoms.position(0).z, 4.0, 1e-12);

    // Minimum image uses only the periodic sublattice: 1.8 -> -0.2 along x,
    // while the non-periodic z displacement is kept.
    const Vec3 mic = cell.minimum_image(Vec3{1.8, 0.0, 7.0});
    FA_CHECK_CLOSE(mic.x, -0.2, 1e-12);
    FA_CHECK_CLOSE(mic.z, 7.0, 1e-12);
}

FA_TEST(cells_without_a_fractional_basis_are_reported) {
    const Mat3 zero{};
    FA_CHECK(!Cell(zero, {true, true, true}).has_scaled_coordinates());
    const Mat3 dependent = Mat3::from_rows({1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
    const Cell cell(dependent, {true, true, true});
    FA_CHECK(!cell.has_scaled_coordinates());
    const Atoms atoms(std::vector<int>{1}, {0.0, 0.0, 0.0}, cell);
    FA_CHECK_THROWS(atoms.scaled_positions(), std::invalid_argument);
}

FA_TEST(extend_rejects_width_mismatch_before_mutating) {
    Atoms left(std::vector<int>{1, 1}, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0});
    left.set_array("foo", {1.0, 2.0}, 1);
    left.set_int_array("bar", {3, 4}, 1);
    Atoms right(std::vector<int>{6}, {0.0, 0.0, 0.0});
    right.set_array("foo", {5.0, 6.0}, 2);
    FA_CHECK_THROWS(left.extend(right), std::invalid_argument);
    FA_CHECK(left.size() == 2);
    FA_CHECK(left.array("foo").ncols == 1 && left.array("foo").data.size() == 2);
    FA_CHECK(left.array("foo").data[1] == 2.0);

    right.remove_array("foo");
    right.set_int_array("bar", {7, 8}, 2);
    FA_CHECK_THROWS(left.extend(right), std::invalid_argument);
    FA_CHECK(left.size() == 2 && left.int_array("bar").data.size() == 2);
}

FA_TEST_MAIN()
