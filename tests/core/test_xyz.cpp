// SPDX-License-Identifier: MIT
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/core/cell.hpp"
#include "fastatoms/io/format.hpp"
#include "fastatoms/io/xyz.hpp"
#include "fastatoms/util/text.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::io;

namespace {

std::string data_path(const char* name) {
    return std::string(FASTATOMS_TEST_DATA) + "/xyz/" + name;
}

/// The FormatError message for input that should be rejected, or "" when it reads.
std::string error_of(std::string_view text) {
    try {
        (void)read_xyz(text);
    } catch (const FormatError& error) {
        return error.what();
    }
    return {};
}

bool same_doubles(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {  // exact: the round trip must be bit-identical
            return false;
        }
    }
    return true;
}

}  // namespace

FA_TEST(plain_xyz_file) {
    const std::vector<Atoms> frames = read_xyz(util::read_text_file(data_path("water.xyz")));
    FA_CHECK(frames.size() == 1);
    const Atoms& water = frames.front();
    FA_CHECK(water.size() == 3);
    FA_CHECK(water.numbers() == (std::vector<int>{8, 1, 1}));
    FA_CHECK(water.labels() == (std::vector<std::string>{"O", "H", "H"}));
    FA_CHECK(water.info()["comment"].as_string() == "water molecule");
    FA_CHECK(water.cell().is_zero());
    FA_CHECK(!water.pbc()[0] && !water.pbc()[1] && !water.pbc()[2]);
    FA_CHECK_CLOSE(water.position(1).y, 0.757, 1e-12);
    FA_CHECK_CLOSE(water.position(2).z, -0.476, 1e-12);
}

FA_TEST(plain_xyz_labels_are_kept) {
    const std::string text = "2\n\nFe1 0 0 0\nFe_pv 0 0 2.5\n";
    const std::vector<Atoms> frames = read_xyz(text);
    FA_CHECK(frames.size() == 1);
    FA_CHECK(frames[0].numbers() == (std::vector<int>{26, 26}));
    FA_CHECK(frames[0].labels() == (std::vector<std::string>{"Fe1", "Fe_pv"}));
}

FA_TEST(multi_frame_with_blank_lines) {
    const std::vector<Atoms> frames = read_xyz(util::read_text_file(data_path("multi.xyz")));
    FA_CHECK(frames.size() == 3);
    for (const Atoms& frame : frames) {
        FA_CHECK(frame.size() == 2);
    }
    FA_CHECK(frames[0].info()["comment"].as_string() == "step 1");
    FA_CHECK(frames[2].info()["comment"].as_string() == "step 3");
    FA_CHECK_CLOSE(frames[1].position(1).z, 0.80, 1e-12);
    FA_CHECK(read_xyz("").empty());
    FA_CHECK(read_xyz("\n\n  \n").empty());
}

FA_TEST(extxyz_lattice_defaults_to_periodic) {
    const std::string text =
        "2\n"
        "Lattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3 energy=-1.5\n"
        "O 0 0 0\n"
        "H 1 0 0\n";
    const std::vector<Atoms> frames = read_xyz(text);
    FA_CHECK(frames.size() == 1);
    const Atoms& atoms = frames[0];
    FA_CHECK(atoms.pbc()[0] && atoms.pbc()[1] && atoms.pbc()[2]);
    FA_CHECK_CLOSE(atoms.cell().matrix().m[0], 10.0, 1e-12);
    FA_CHECK_CLOSE(atoms.cell().matrix().m[4], 10.0, 1e-12);
    FA_CHECK_CLOSE(atoms.cell().matrix().m[8], 10.0, 1e-12);
    FA_CHECK_CLOSE(atoms.cell().volume(), 1000.0, 1e-9);
    FA_CHECK(atoms.labels() == (std::vector<std::string>{"O", "H"}));
    FA_CHECK(!atoms.info().contains("Lattice"));
    FA_CHECK(!atoms.info().contains("Properties"));
    FA_CHECK(atoms.info()["energy"].is_float());
    FA_CHECK_CLOSE(atoms.info()["energy"].as_double(), -1.5, 1e-12);
}

FA_TEST(extxyz_pbc_overrides_lattice) {
    const std::string text =
        "1\n"
        "Lattice=\"4 0 0 0 4 0 0 0 4\" Properties=species:S:1:pos:R:3 pbc=\"T T F\"\n"
        "Na 1 1 1\n";
    const Atoms atoms = read_xyz(text).front();
    FA_CHECK(atoms.pbc()[0] && atoms.pbc()[1] && !atoms.pbc()[2]);
    FA_CHECK_CLOSE(atoms.cell().volume(), 64.0, 1e-9);
}

FA_TEST(extra_r_i_l_s_columns) {
    const std::string text =
        "2\n"
        "Lattice=\"5 0 0 0 5 0 0 0 5\" Properties=species:S:1:pos:R:3:forces:R:3:index:I:1:"
        "flag:L:1:label:S:1 pbc=\"F F F\"\n"
        "C 0 0 0 0.1 0.2 0.3 7 T C_a\n"
        "C 1 1 1 -0.1 -0.2 -0.3 8 F C_b\n";
    const Atoms atoms = read_xyz(text).front();
    FA_CHECK(!atoms.pbc()[0] && !atoms.pbc()[1] && !atoms.pbc()[2]);
    FA_CHECK(atoms.labels() == (std::vector<std::string>{"C", "C"}));

    FA_CHECK(atoms.has_array("forces"));
    FA_CHECK(atoms.array("forces").ncols == 3);
    FA_CHECK(same_doubles(atoms.array("forces").data, {0.1, 0.2, 0.3, -0.1, -0.2, -0.3}));

    FA_CHECK(atoms.has_int_array("index"));
    FA_CHECK(atoms.int_array("index").data == (std::vector<std::int64_t>{7, 8}));
    FA_CHECK(atoms.has_int_array("flag"));
    FA_CHECK(atoms.int_array("flag").data == (std::vector<std::int64_t>{1, 0}));

    FA_CHECK(atoms.has_string_array("label"));
    FA_CHECK(atoms.string_array("label") == (std::vector<std::string>{"C_a", "C_b"}));
}

FA_TEST(velocities_become_momenta) {
    const std::string text =
        "1\n"
        "Properties=species:S:1:pos:R:3:masses:R:1:velo:R:3 pbc=\"F F F\"\n"
        "H 0 0 0 2.0 0.5 0 0\n";
    const Atoms atoms = read_xyz(text).front();
    FA_CHECK(atoms.has_masses());
    FA_CHECK(atoms.has_momenta());
    // Masses are applied before the velocity conversion, so p = m * v.
    FA_CHECK_CLOSE(atoms.momenta()[0], 1.0, 1e-12);
}

FA_TEST(info_parsing) {
    const std::string text =
        "1\n"
        "energy=-1.5 n=3 flag bool=T off=False name=\"my water\" "
        "vec=\"1 2 3\" Properties=species:S:1:pos:R:3\n"
        "H 0 0 0\n";
    const Atoms atoms = read_xyz(text).front();
    const Value& info = atoms.info();
    FA_CHECK(info["energy"].is_float());
    FA_CHECK_CLOSE(info["energy"].as_double(), -1.5, 1e-12);
    FA_CHECK(info["n"].is_int());
    FA_CHECK(info["n"].as_int() == 3);
    FA_CHECK(info["flag"].is_bool() && info["flag"].as_bool());
    FA_CHECK(info["bool"].is_bool() && info["bool"].as_bool());
    FA_CHECK(info["off"].is_bool() && !info["off"].as_bool());
    FA_CHECK(info["name"].as_string() == "my water");
    FA_CHECK(info["vec"].is_list());
    FA_CHECK(info["vec"].size() == 3);
    FA_CHECK(info["vec"].at(1).as_int() == 2);
}

FA_TEST(gemmy_protein_file) {
    const std::vector<Atoms> frames = read_xyz(util::read_text_file(data_path("1bna.extxyz")));
    FA_CHECK(frames.size() == 1);
    const Atoms& dna = frames.front();
    FA_CHECK(dna.size() == 486);
    FA_CHECK(!dna.pbc()[0] && !dna.pbc()[1] && !dna.pbc()[2]);
    FA_CHECK_CLOSE(dna.cell().matrix().m[0], 38.481, 1e-12);
    FA_CHECK_CLOSE(dna.cell().matrix().m[8], 58.485, 1e-12);
    FA_CHECK(dna.labels().size() == 486);
    FA_CHECK(dna.labels()[0] == "O");
    FA_CHECK_CLOSE(dna.position(0).x, 22.91, 1e-12);
}

FA_TEST(round_trip_is_exact) {
    const Cell cell(Mat3(10.0, 0.0, 0.0, 0.5, 9.5, 0.0, 0.25, 0.125, 8.0), {true, true, false});
    const std::vector<int> numbers{8, 1, 1};
    const std::vector<double> positions{0.1,       1.0 / 3.0,  -2.5e-7,
                                        1e10 + 0.5, 0.7071067811865476, 3.0,
                                        -0.0,      5.5,        1e-300};
    Atoms atoms(numbers, positions, cell);
    atoms.set_string_array("labels", {"O1", "H", "H_pv"});
    atoms.set_array("forces", {0.5, -0.25, 1e-9, 2.0, 3.0, -4.0, 0.0, 0.0, 1.0 / 7.0}, 3);
    atoms.set_int_array("type", {1, 2, 2});
    atoms.set_momenta({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9});
    atoms.set_info(Value(Value::Map{
        {"energy", Value(-1.25)},
        {"converged", Value(true)},
        {"label", Value("a b")},
        {"stress", Value(Value::List{Value(1.0), Value(2.5)})},
        {"count", Value(7)},
        {"whole", Value(2.0)},
    }));

    const std::string text = write_xyz(atoms);
    const std::vector<Atoms> frames = read_xyz(text);
    FA_CHECK(frames.size() == 1);
    const Atoms& back = frames.front();

    FA_CHECK(back.size() == atoms.size());
    FA_CHECK(back.numbers() == atoms.numbers());
    FA_CHECK(same_doubles(back.positions(), atoms.positions()));
    FA_CHECK(back.labels() == atoms.labels());
    FA_CHECK(back.pbc()[0] && back.pbc()[1] && !back.pbc()[2]);
    FA_CHECK(back.cell().matrix().m == atoms.cell().matrix().m);
    FA_CHECK(same_doubles(back.array("forces").data, atoms.array("forces").data));
    FA_CHECK(back.int_array("type").data == (std::vector<std::int64_t>{1, 2, 2}));
    FA_CHECK(same_doubles(back.momenta(), atoms.momenta()));

    FA_CHECK(back.info()["energy"].as_double() == -1.25);
    FA_CHECK(back.info()["converged"].as_bool());
    FA_CHECK(back.info()["label"].as_string() == "a b");
    FA_CHECK(back.info()["stress"].is_list());
    FA_CHECK(back.info()["stress"].at(1).as_double() == 2.5);
    FA_CHECK(back.info()["count"].as_int() == 7);
    FA_CHECK(back.info()["whole"].is_float());
    FA_CHECK(back.info()["whole"].as_double() == 2.0);

    // Writing the frame again gives the same text.
    FA_CHECK(write_xyz(back) == text);
}

FA_TEST(round_trip_multi_frame) {
    Atoms a(std::vector<int>{1, 1}, std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.74});
    Atoms b(std::vector<int>{1, 1}, std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.80});
    const std::vector<Atoms> frames{a, b};
    const std::vector<Atoms> back = read_xyz(write_xyz(frames));
    FA_CHECK(back.size() == 2);
    FA_CHECK_CLOSE(back[1].position(1).z, 0.80, 0.0);
}

FA_TEST(plain_write_and_read) {
    Atoms atoms(std::vector<int>{26, 26},
                std::vector<double>{0.0, 0.0, 0.0, 0.1, 1.0 / 3.0, 2.5});
    atoms.set_string_array("labels", {"Fe1", "Fe_pv"});
    XyzWriteOptions plain;
    plain.extended = false;
    const std::string text = write_xyz(atoms, plain);
    FA_CHECK(text.rfind("2\n\nFe1 0 0 0\n", 0) == 0);
    const Atoms back = read_xyz(text).front();
    FA_CHECK(back.numbers() == atoms.numbers());
    FA_CHECK(back.labels() == atoms.labels());
    FA_CHECK(same_doubles(back.positions(), atoms.positions()));
    FA_CHECK(back.cell().is_zero());
}

FA_TEST(extended_write_layout) {
    Atoms molecule(std::vector<int>{8, 1}, std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.96});
    const std::string text = write_xyz(molecule);
    FA_CHECK(text.find("Lattice=") == std::string::npos);
    FA_CHECK(text.find("Properties=species:S:1:pos:R:3 pbc=\"F F F\"") != std::string::npos);
    FA_CHECK(text.find("\nO 0 0 0\n") != std::string::npos);
    FA_CHECK(text.find("\nH 0 0 0.96\n") != std::string::npos);

    Atoms periodic = molecule;
    periodic.set_cell(Mat3::diagonal(5.0, 5.0, 5.0), false);
    periodic.set_pbc(true);
    const std::string periodic_text = write_xyz(periodic);
    FA_CHECK(periodic_text.find("Lattice=\"5 0 0 0 5 0 0 0 5\" ") != std::string::npos);
    FA_CHECK(periodic_text.find("pbc=\"T T T\"") != std::string::npos);
}

FA_TEST(precision_option_limits_digits) {
    Atoms atoms(std::vector<int>{6}, std::vector<double>{1.0 / 3.0, 0.0, 0.0});
    XyzWriteOptions rounded;
    rounded.precision = 6;
    const std::string text = write_xyz(atoms, rounded);
    FA_CHECK(text.find("0.333333 ") != std::string::npos);
    FA_CHECK(text.find("0.3333333") == std::string::npos);
    FA_CHECK(text.find("0.333333333") == std::string::npos);
}

FA_TEST(writer_rejects_unrepresentable_info) {
    Atoms atoms({1}, {0.0, 0.0, 0.0});
    atoms.set_info(Value(Value::Map{{"bad key", Value(1)}}));
    FA_CHECK_THROWS(write_xyz(atoms), FormatError);

    atoms.set_info(Value(Value::Map{{"text", Value("two\nlines")}}));
    FA_CHECK_THROWS(write_xyz(atoms), FormatError);
}

FA_TEST(nested_info_round_trips_as_json) {
    Atoms atoms({1}, {0.0, 0.0, 0.0});
    atoms.set_info(Value(Value::Map{
        {"nested", Value(Value::Map{{"a", Value(1)},
                                     {"b", Value(Value::List{Value(true), Value()})}})},
        {"empty", Value(Value::List{})},
    }));
    const std::string text = write_xyz(atoms);
    FA_CHECK(text.find("nested=\"_JSON {\\\"a\\\": 1, \\\"b\\\": [true, null]}\"") !=
             std::string::npos);
    FA_CHECK(text.find("empty=\"_JSON []\"") != std::string::npos);

    const Atoms back = read_xyz(text).front();
    FA_CHECK(back.info()["nested"].at("a").as_int() == 1);
    FA_CHECK(back.info()["nested"].at("b").at(0).as_bool());
    FA_CHECK(back.info()["nested"].at("b").at(1).is_null());
    FA_CHECK(back.info()["empty"].is_list() && back.info()["empty"].size() == 0);
    FA_CHECK(write_xyz(back) == text);
}

FA_TEST(virial_is_read_and_written_in_fortran_order) {
    // Nine numbers in Fortran order: row r, column c is number r + 3c (ASE's rule).
    // Other keys keep the nine numbers as a flat list.
    const std::string text =
        "1\nvirial=\"1 4 7 2 5 8 3 6 9\" not_a_3x3=\"1 4 7 2 5 8 3 6 9\"\nH 0 0 0\n";
    const Atoms atoms = read_xyz(text).front();
    const Value& virial = atoms.info()["virial"];
    FA_CHECK(virial.is_list() && virial.size() == 3);
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            FA_CHECK(virial.at(r).at(c).as_int() == static_cast<std::int64_t>(3 * r + c + 1));
        }
    }
    FA_CHECK(atoms.info()["not_a_3x3"].size() == 9);

    const std::string out = write_xyz(atoms);
    FA_CHECK(out.find("virial=\"1 4 7 2 5 8 3 6 9\"") != std::string::npos);
    FA_CHECK(out.find("not_a_3x3=\"1 4 7 2 5 8 3 6 9\"") != std::string::npos);
    FA_CHECK(write_xyz(read_xyz(out).front()) == out);
}

FA_TEST(json_values_are_read_as_nested_lists_and_maps) {
    const std::string text =
        "1\nf_float_array=\"_JSON [[1.5, 2, 3], [4, 5, 6]]\" "
        "f_int_array=\"_JSON [[1, 2], [3, 4]]\" f_dict={_JSON {\"a\" : 1}}\nH 0 0 0\n";
    const Atoms atoms = read_xyz(text).front();

    // A float anywhere makes the whole numeric array floats, as numpy does.
    const Value& floats = atoms.info()["f_float_array"];
    FA_CHECK(floats.is_list() && floats.size() == 2);
    FA_CHECK(floats.at(0).at(0).is_float() && floats.at(0).at(0).as_double() == 1.5);
    FA_CHECK(floats.at(0).at(1).is_float() && floats.at(0).at(1).as_double() == 2.0);
    FA_CHECK(floats.at(1).at(2).as_double() == 6.0);

    const Value& ints = atoms.info()["f_int_array"];
    FA_CHECK(ints.at(1).at(0).is_int() && ints.at(1).at(0).as_int() == 3);

    const Value& dict = atoms.info()["f_dict"];
    FA_CHECK(dict.is_map() && dict.at("a").as_int() == 1);

    const std::string out = write_xyz(atoms);
    FA_CHECK(out.find("f_float_array=\"_JSON [[1.5, 2.0, 3.0], [4.0, 5.0, 6.0]]\"") !=
             std::string::npos);
    FA_CHECK(out.find("f_int_array=\"_JSON [[1, 2], [3, 4]]\"") != std::string::npos);
    FA_CHECK(out.find("f_dict=\"_JSON {\\\"a\\\": 1}\"") != std::string::npos);
    FA_CHECK(write_xyz(read_xyz(out).front()) == out);
}

FA_TEST(vec_lines_give_the_cell) {
    // ASE's vec_cell form: the comment is not read as key=value pairs, and the
    // VEC lines after the atoms give the cell. A missing vector is an aperiodic axis.
    const std::string text =
        "2\nComment with a = sign\nH 0 0 0\nH 0 0 0.74\nVEC1 4 0 0\nVEC2 0 5 0\n"
        "1\nc\nHe 1 1 1\n";
    const std::vector<Atoms> frames = read_xyz(text);
    FA_CHECK(frames.size() == 2);
    const Atoms& atoms = frames[0];
    FA_CHECK(atoms.size() == 2);
    FA_CHECK(atoms.info()["comment"].as_string() == "Comment with a = sign");
    FA_CHECK(atoms.cell().matrix().m[0] == 4.0);
    FA_CHECK(atoms.cell().matrix().m[4] == 5.0);
    FA_CHECK(atoms.cell().matrix().m[8] == 0.0);
    FA_CHECK(atoms.pbc()[0] && atoms.pbc()[1] && !atoms.pbc()[2]);
    FA_CHECK(frames[1].numbers() == std::vector<int>({2}));
    FA_CHECK(frames[1].cell().is_zero());

    FA_CHECK(error_of("1\nc\nH 0 0 0\nVEC2 0 1 0\n").find("line 4") != std::string::npos);
    FA_CHECK(error_of("1\nc\nH 0 0 0\nVEC1 1 0\n").find("line 4") != std::string::npos);
    FA_CHECK_THROWS(read_xyz("1\nc\nH 0 0 0\nVEC1 1 0 0\nVEC2 0 1 0\nVEC3 0 0 1\nVEC3 0 0 1\n"),
                    FormatError);
}

FA_TEST(malformed_input_is_rejected_with_line_numbers) {
    FA_CHECK(error_of("abc\ncomment\nH 0 0 0\n").find("line 1") != std::string::npos);
    FA_CHECK(error_of("3\nc\nH 0 0 0\nH 0 0 0\n").find("line 1") != std::string::npos);
    FA_CHECK(error_of("1\n").find("line 1") != std::string::npos);
    FA_CHECK(error_of("1\nc\nQq 0 0 0\n").find("line 3") != std::string::npos);
    FA_CHECK(error_of("1\nc\nH 0 x 0\n").find("line 3") != std::string::npos);
    FA_CHECK(error_of("1\nc\nH 0\n").find("line 3") != std::string::npos);
    FA_CHECK(error_of("2\nc\nH 0 0 0\n\nH 1 1 1\n").find("line 4") != std::string::npos);
    FA_CHECK(error_of("1\nProperties=species:S:1:pos:Q:3\nH 0 0 0\n").find("line 2") !=
             std::string::npos);
    FA_CHECK(error_of("1\nProperties=pos:R:3\nH 0 0 0\n").find("line 2") != std::string::npos);
    FA_CHECK(error_of("1\nProperties=species:S:1:pos:R:3 Lattice=\"1 2\"\nH 0 0 0\n").find(
                 "line 2") != std::string::npos);
    FA_CHECK(error_of("1\nLattice=\"1 0 0 0 1 0 0 0 1 Properties=species:S:1:pos:R:3\nH 0 0 0\n")
                 .find("line 2") != std::string::npos);
    FA_CHECK(error_of("1\nProperties=species:S:1:pos:R:3 pbc=\"T X F\"\nH 0 0 0\n").find(
                 "line 2") != std::string::npos);
    FA_CHECK(error_of("1\nProperties=species:S:1:pos:R:3 index=\"\"\nH 0 0 0\n").empty());
    FA_CHECK(!error_of("1\nProperties=species:S:1:pos:R:3:n:I:1\nH 0 0 0 2.5\n").empty());
}

FA_TEST(malformed_input_throws_format_error) {
    FA_CHECK_THROWS(read_xyz("3\nc\nH 0 0 0\n"), FormatError);
    FA_CHECK_THROWS(read_xyz("1\nc\nQq 0 0 0\n"), FormatError);
    FA_CHECK_THROWS(read_xyz("1\nProperties=species:S:1:pos:R\nH 0 0 0\n"), FormatError);
    FA_CHECK_THROWS(read_xyz("1\nProperties=species:S:2:pos:R:3\nH 0 0 0\n"), FormatError);
    FA_CHECK_THROWS(read_xyz("1\nProperties=species:S:1:pos:R:3:x:S:2\nH 0 0 0 a b\n"),
                    FormatError);
    FA_CHECK_THROWS(read_xyz("1\nc\nH 0 0 nan_text\n"), FormatError);
}

FA_TEST(nan_and_inf_are_read) {
    const Atoms atoms =
        read_xyz("1\nProperties=species:S:1:pos:R:3 energy=-inf\nH nan 0 inf\n").front();
    FA_CHECK(std::isnan(atoms.position(0).x));
    FA_CHECK(std::isinf(atoms.position(0).z));
    FA_CHECK(std::isinf(atoms.info()["energy"].as_double()));
}

FA_TEST(key_value_whitespace_around_equals) {
    const std::string text = "1\nm = 7 n= 8 Properties=species:S:1:pos:R:3\nH 0 0 0\n";
    const Atoms atoms = read_xyz(text).front();
    FA_CHECK(atoms.info()["m"].is_int() && atoms.info()["m"].as_int() == 7);
    FA_CHECK(atoms.info()["n"].is_int() && atoms.info()["n"].as_int() == 8);
    FA_CHECK(!atoms.info().contains("7"));
    FA_CHECK(!atoms.info().contains("8"));
}

FA_TEST(key_value_brackets_and_braces_are_single_values) {
    const std::string text =
        "1\na=[1 2 3] b={x y} Properties=species:S:1:pos:R:3\nH 0 0 0\n";
    const Atoms atoms = read_xyz(text).front();
    FA_CHECK(atoms.info()["a"].is_list());
    FA_CHECK(atoms.info()["a"].size() == 3);
    FA_CHECK(atoms.info()["a"].at(2).as_int() == 3);
    FA_CHECK(atoms.info()["b"].is_string());
    FA_CHECK(atoms.info()["b"].as_string() == "x y");
}

FA_TEST(key_value_quoted_flags_become_bool_list) {
    const std::string text =
        "1\nd=\"T F T\" e=\"true\" Properties=species:S:1:pos:R:3\nH 0 0 0\n";
    const Atoms atoms = read_xyz(text).front();
    const Value& d = atoms.info()["d"];
    FA_CHECK(d.is_list() && d.size() == 3);
    FA_CHECK(d.at(0).is_bool() && d.at(0).as_bool());
    FA_CHECK(d.at(1).is_bool() && !d.at(1).as_bool());
    FA_CHECK(d.at(2).is_bool() && d.at(2).as_bool());
    FA_CHECK(atoms.info()["e"].is_bool() && atoms.info()["e"].as_bool());
}

FA_TEST(key_value_empty_quotes_end_their_pair) {
    const std::string text = "1\ne=\"\" n=2 Properties=species:S:1:pos:R:3\nH 0 0 0\n";
    const Atoms atoms = read_xyz(text).front();
    FA_CHECK(atoms.info()["n"].is_int() && atoms.info()["n"].as_int() == 2);
    FA_CHECK(atoms.info()["e"].is_string() && atoms.info()["e"].as_string().empty());
}

FA_TEST(velocities_round_trip_bit_exact) {
    const std::string text =
        "2\nProperties=species:S:1:pos:R:3:velo:R:3 pbc=\"F F F\"\n"
        "H 0 0 0 1 2 3\n"
        "X 0 0 1 0.1 -0.2 0.30000000000000004\n";
    const Atoms atoms = read_xyz(text).front();
    FA_CHECK(atoms.has_velocities());
    FA_CHECK(same_doubles(atoms.velocities(), {1.0, 2.0, 3.0, 0.1, -0.2, 0.30000000000000004}));
    // The mass-0 atom keeps its velocity: velo is stored verbatim, not as momenta.
    FA_CHECK(atoms.mass(1) == 0.0);

    const std::string written = write_xyz(atoms);
    FA_CHECK(written.find("velo:R:3") != std::string::npos);
    FA_CHECK(written.find("momenta") == std::string::npos);
    const Atoms back = read_xyz(written).front();
    FA_CHECK(same_doubles(back.velocities(), atoms.velocities()));
    FA_CHECK(write_xyz(back) == written);
}

FA_TEST(momenta_column_sets_momenta) {
    const std::string text =
        "1\nProperties=species:S:1:pos:R:3:masses:R:1:momenta:R:3 pbc=\"F F F\"\n"
        "H 0 0 0 2 4 0 0\n";
    const Atoms atoms = read_xyz(text).front();
    // Momentum 4 on a mass-2 atom is velocity 2.
    FA_CHECK_CLOSE(atoms.momenta()[0], 4.0, 1e-12);
    FA_CHECK_CLOSE(atoms.velocities()[0], 2.0, 1e-12);
}

FA_TEST(charges_written_as_initial_charges) {
    Atoms atoms(std::vector<int>{1, 8}, std::vector<double>{0, 0, 0, 0, 0, 1});
    atoms.set_charges({0.5, -0.5});
    const std::string text = write_xyz(atoms);
    FA_CHECK(text.find("initial_charges:R:1") != std::string::npos);
    FA_CHECK(text.find(":charges:") == std::string::npos);
    const Atoms back = read_xyz(text).front();
    FA_CHECK(same_doubles(back.charges(), {0.5, -0.5}));

    const std::string legacy =
        "1\nProperties=species:S:1:pos:R:3:charges:R:1 pbc=\"F F F\"\nH 0 0 0 0.25\n";
    FA_CHECK_CLOSE(read_xyz(legacy).front().charges()[0], 0.25, 0.0);
}

FA_TEST(logical_columns_stay_logical) {
    const std::string text =
        "2\nProperties=species:S:1:pos:R:3:flag:L:1 pbc=\"F F F\"\n"
        "C 0 0 0 T\n"
        "C 1 1 1 F\n";
    const Atoms atoms = read_xyz(text).front();
    FA_CHECK(atoms.int_array("flag").data == (std::vector<std::int64_t>{1, 0}));

    const std::string written = write_xyz(atoms);
    FA_CHECK(written.find("Properties=species:S:1:pos:R:3:flag:L:1") != std::string::npos);
    FA_CHECK(written.find("\nC 0 0 0 T\n") != std::string::npos);
    FA_CHECK(written.find("\nC 1 1 1 F\n") != std::string::npos);
    FA_CHECK(written.find("_logical_columns") == std::string::npos);
    FA_CHECK(write_xyz(read_xyz(written).front()) == written);
}

FA_TEST(lattice_from_parameters_has_no_noise) {
    Atoms atoms(std::vector<int>{6}, std::vector<double>{0.0, 0.0, 0.0});
    atoms.set_cell(Cell::from_parameters(3, 3, 3, 90, 90, 90));
    atoms.set_pbc(true);
    FA_CHECK(write_xyz(atoms).find("Lattice=\"3 0 0 0 3 0 0 0 3\" ") != std::string::npos);
}

FA_TEST(writer_rejects_line_breaks_in_info) {
    Atoms atoms({1}, {0.0, 0.0, 0.0});
    for (const char* bad : {"first\nsecond", "first\rsecond"}) {
        atoms.set_info(Value(Value::Map{{"note", Value(std::string(bad))}}));
        std::string message;
        try {
            (void)write_xyz(atoms);
        } catch (const FormatError& error) {
            message = error.what();
        }
        FA_CHECK(message.find("'note'") != std::string::npos);
    }
}

FA_TEST_MAIN()
