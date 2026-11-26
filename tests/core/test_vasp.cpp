// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "fastatoms/io/format.hpp"
#include "fastatoms/io/vasp.hpp"
#include "testing.hpp"

using fastatoms::Atoms;
using fastatoms::Cell;
using fastatoms::io::FormatError;
using fastatoms::io::PoscarReadOptions;
using fastatoms::io::PoscarWriteOptions;

namespace {

std::string fixture(const std::string& name) {
    std::ifstream in(std::string(FASTATOMS_TEST_DATA) + "/vasp/" + name, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

std::string with_crlf(const std::string& text) {
    std::string out;
    for (const char c : text) {
        if (c == '\n') {
            out += '\r';
        }
        out += c;
    }
    return out;
}

/// The message of the FormatError thrown by `f`, or empty when it does not throw.
template <typename F>
std::string format_error(F&& f) {
    try {
        f();
    } catch (const FormatError& error) {
        return error.what();
    }
    return {};
}

/// |a - b| <= tol * max(|a|, |b|): a relative bound that also holds at zero.
bool within_relative(double a, double b, double tol) {
    return std::abs(a - b) <= tol * std::max(std::abs(a), std::abs(b));
}

std::vector<int> flags_of(const Atoms& a) {
    std::vector<int> out;
    for (const std::int64_t v : a.int_array("selective_dynamics").data) {
        out.push_back(static_cast<int>(v));
    }
    return out;
}

const char* kHex = R"(hex cell
1.0
  1.0 0.0 0.0
  0.5 0.866 0.0
  0.0 0.0 1.0
H
1
Cartesian
0.1 0.2 0.3
)";

}  // namespace

// ---------------------------------------------------------------- labels

// phonopy SPOSCAR: `Fe1 Fe2` on the species line. ASE rejects these labels
// (it does not strip the trailing digit), so there is no ASE cross-check.
FA_TEST(phonopy_sposcar_labels_are_kept_verbatim) {
    const Atoms a = fastatoms::io::read_poscar(fixture("SPOSCAR_phonopy"));
    FA_CHECK(a.size() == 4);
    FA_CHECK((a.string_array("labels") == std::vector<std::string>{"Fe1", "Fe1", "Fe2", "Fe2"}));
    FA_CHECK((a.numbers() == std::vector<int>{26, 26, 26, 26}));
    FA_CHECK(a.pbc()[0] && a.pbc()[1] && a.pbc()[2]);
    // Direct (0.5, 0.5, 0.5) on a 2.8 A cube.
    FA_CHECK_CLOSE(a.position(1).x, 1.4, 1e-12);
    FA_CHECK_CLOSE(a.position(2).x, 0.7, 1e-12);
    FA_CHECK_CLOSE(a.position(2).y, 0.7, 1e-12);
    FA_CHECK(!a.has_int_array("selective_dynamics"));
    FA_CHECK(a.info()["comment"].as_string() == "Fe1 Fe2 phonopy supercell");
}

// PAW `Fe_pv1`, partial selective dynamics, a trailing `! Fe1` site label.
// ASE gives the same positions and element (it strips `_pv`).
FA_TEST(pv_label_partial_selective_dynamics) {
    const Atoms a = fastatoms::io::read_poscar(fixture("POSCAR_pv"));
    FA_CHECK((a.string_array("labels") == std::vector<std::string>{"Fe_pv1", "Fe_pv1"}));
    FA_CHECK((a.numbers() == std::vector<int>{26, 26}));
    FA_CHECK((flags_of(a) == std::vector<int>{1, 1, 1, 0, 1, 0}));
    // Scale 2.87 times Direct (0.5, 0.5, 0.5) of (a0 + a1 + a2) = (1.435, 1.435, 1.435).
    FA_CHECK_CLOSE(a.position(1).x, 2.87 * 0.7175, 1e-12);
    FA_CHECK_CLOSE(a.position(1).z, 2.87 * 0.7175, 1e-12);
    FA_CHECK(a.info()["comment"].as_string() == "bcc Fe with PAW pv potential");
}

// VASP 6 writes a POTCAR hash after the element: `Fe/hash`.
FA_TEST(vasp6_hash_species_line) {
    const Atoms a = fastatoms::io::read_poscar(fixture("POSCAR_hash"));
    FA_CHECK((a.string_array("labels") ==
              std::vector<std::string>{"Fe/5c3b9e0d1a2f", "Fe/5c3b9e0d1a2f", "O/9f8e7d6c5b4a"}));
    FA_CHECK((a.numbers() == std::vector<int>{26, 26, 8}));
    // Cartesian positions, scale 1.
    FA_CHECK_CLOSE(a.position(1).x, 2.0, 1e-12);
    FA_CHECK_CLOSE(a.position(2).z, 2.0, 1e-12);
}

// Labels that do not name an element are an error that names the label.
FA_TEST(unknown_species_label_is_an_error) {
    const std::string text = R"(bad
1.0
  2.0 0.0 0.0
  0.0 2.0 0.0
  0.0 0.0 2.0
Qq1
1
Direct
0.0 0.0 0.0
)";
    const std::string message = format_error([&] { (void)fastatoms::io::read_poscar(text); });
    FA_CHECK(message.find("Qq1") != std::string::npos);
}

// ---------------------------------------------------------------- VASP 4

FA_TEST(vasp4_species_from_comment_line) {
    const Atoms a = fastatoms::io::read_poscar(fixture("POSCAR_vasp4"));
    FA_CHECK((a.string_array("labels") == std::vector<std::string>{"Fe", "Fe", "O"}));
    FA_CHECK((a.numbers() == std::vector<int>{26, 26, 8}));
    FA_CHECK_CLOSE(a.position(1).x, 1.5, 1e-12);  // Direct 0.5 * 3 A
}

FA_TEST(vasp4_species_from_option) {
    const std::string text = R"(Mystery cell
1.0
  2.0 0.0 0.0
  0.0 2.0 0.0
  0.0 0.0 2.0
1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
)";
    PoscarReadOptions options;
    options.species = {"Fe", "O"};
    const Atoms a = fastatoms::io::read_poscar(text, options);
    FA_CHECK((a.string_array("labels") == std::vector<std::string>{"Fe", "O"}));
    FA_CHECK((a.numbers() == std::vector<int>{26, 8}));
}

FA_TEST(vasp4_without_species_is_an_error) {
    const std::string text = R"(Mystery 12345
1.0
  2.0 0.0 0.0
  0.0 2.0 0.0
  0.0 0.0 2.0
1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
)";
    FA_CHECK_THROWS(fastatoms::io::read_poscar(text), FormatError);
}

// ---------------------------------------------------------------- scale

FA_TEST(negative_scale_is_a_target_volume) {
    const std::string text = R"(volume
-54.0
  3.0 0.0 0.0
  0.0 3.0 0.0
  0.0 0.0 3.0
H
1
Cartesian
1.0 0.0 0.0
)";
    const Atoms a = fastatoms::io::read_poscar(text);
    FA_CHECK_CLOSE(a.volume(), 54.0, 1e-10);
    const double f = std::cbrt(2.0);  // (54 / 27)^(1/3)
    FA_CHECK_CLOSE(a.cell().vector(0).x, 3.0 * f, 1e-12);
    FA_CHECK_CLOSE(a.position(0).x, f, 1e-12);  // Cartesian coordinates scale too
}

// Three scales follow ASE: each Cartesian component k of every lattice vector
// and of every Cartesian position is multiplied by scale[k]. For a diagonal
// cell that is the same as scaling each lattice vector; for a non-diagonal
// cell it is not. ASE gives cell (2,0,0), (1,2.598,0), (0,0,4) and position
// (0.2,0.6,1.2) for kHex with scales 2 3 4.
FA_TEST(three_scales_match_ase) {
    const std::string text = std::string(kHex).replace(std::string(kHex).find("1.0\n"), 4,
                                                       "2.0 3.0 4.0\n");
    const Atoms a = fastatoms::io::read_poscar(text);
    FA_CHECK_CLOSE(a.cell().vector(0).x, 2.0, 1e-12);
    FA_CHECK_CLOSE(a.cell().vector(1).x, 1.0, 1e-12);
    FA_CHECK_CLOSE(a.cell().vector(1).y, 2.598, 1e-12);
    FA_CHECK_CLOSE(a.cell().vector(2).z, 4.0, 1e-12);
    FA_CHECK_CLOSE(a.position(0).x, 0.2, 1e-12);
    FA_CHECK_CLOSE(a.position(0).y, 0.6, 1e-12);
    FA_CHECK_CLOSE(a.position(0).z, 1.2, 1e-12);
}

FA_TEST(bad_scale_lines_are_errors) {
    const std::string three_negative = std::string(kHex).replace(
        std::string(kHex).find("1.0\n"), 4, "-1.0 -2.0 -3.0\n");
    FA_CHECK_THROWS(fastatoms::io::read_poscar(three_negative), FormatError);
    const std::string two_values = std::string(kHex).replace(std::string(kHex).find("1.0\n"),
                                                             4, "1.0 2.0\n");
    FA_CHECK_THROWS(fastatoms::io::read_poscar(two_values), FormatError);
}

// ---------------------------------------------------------------- coordinates

// Every spelling of Cartesian gives the same positions, with the scale
// applied; every other first letter is Direct.
FA_TEST(coordinate_mode_keyed_on_first_letter) {
    const char* cartesian_lines[] = {"Cartesian", "  cart", "K", "k", "C", "\tCARTESIAN"};
    for (const char* mode : cartesian_lines) {
        const std::string text = std::string(R"(mode
1.0
  4.0 0.0 0.0
  0.0 4.0 0.0
  0.0 0.0 4.0
H
1
)") + mode + "\n0.5 0.5 0.5\n";
        const Atoms a = fastatoms::io::read_poscar(text);
        FA_CHECK_CLOSE(a.position(0).x, 0.5, 1e-12);
    }
    const char* direct_lines[] = {"Direct", "  d", "frac", "D"};
    for (const char* mode : direct_lines) {
        const std::string text = std::string(R"(mode
1.0
  4.0 0.0 0.0
  0.0 4.0 0.0
  0.0 0.0 4.0
H
1
)") + mode + "\n0.5 0.5 0.5\n";
        const Atoms a = fastatoms::io::read_poscar(text);
        FA_CHECK_CLOSE(a.position(0).x, 2.0, 1e-12);
    }
}

FA_TEST(selective_keyword_is_case_insensitive_and_flags_are_per_axis) {
    const std::string text = R"(sel
1.0
  4.0 0.0 0.0
  0.0 4.0 0.0
  0.0 0.0 4.0
H
2
selective
Cartesian
0.5 0.5 0.5 T F T
1.0 1.0 1.0 f t F
)";
    const Atoms a = fastatoms::io::read_poscar(text);
    FA_CHECK((flags_of(a) == std::vector<int>{1, 0, 1, 0, 1, 0}));
    FA_CHECK(a.int_array("selective_dynamics").ncols == 3);
}

FA_TEST(missing_selective_flag_is_an_error) {
    const std::string text = R"(sel
1.0
  4.0 0.0 0.0
  0.0 4.0 0.0
  0.0 0.0 4.0
H
1
Selective dynamics
Direct
0.5 0.5 0.5
)";
    FA_CHECK_THROWS(fastatoms::io::read_poscar(text), FormatError);
}

FA_TEST(fortran_double_exponent_and_trailing_tokens) {
    const std::string text = R"(fortran
1.0
  4.0 0.0 0.0
  0.0 4.0 0.0
  0.0 0.0 4.0
H
1
Cartesian
1.0D-01 2.5D+00 0.0D0 ! the site
)";
    const Atoms a = fastatoms::io::read_poscar(text);
    FA_CHECK_CLOSE(a.position(0).x, 0.1, 1e-15);
    FA_CHECK_CLOSE(a.position(0).y, 2.5, 1e-15);
}

FA_TEST(too_few_positions_name_the_line) {
    const std::string text = R"(short
1.0
  4.0 0.0 0.0
  0.0 4.0 0.0
  0.0 0.0 4.0
H
3
Direct
0.0 0.0 0.0
0.5 0.5 0.5
)";
    const std::string message = format_error([&] { (void)fastatoms::io::read_poscar(text); });
    FA_CHECK(message.find("expected 3 position lines") != std::string::npos);
    FA_CHECK(message.find("line 10") != std::string::npos);
}

FA_TEST(bad_atom_count_names_the_line) {
    const std::string text = R"(count
1.0
  4.0 0.0 0.0
  0.0 4.0 0.0
  0.0 0.0 4.0
H
two
Direct
0.0 0.0 0.0
)";
    const std::string message = format_error([&] { (void)fastatoms::io::read_poscar(text); });
    FA_CHECK(message.find("line 7") != std::string::npos);
    FA_CHECK(message.find("two") != std::string::npos);
}

// ---------------------------------------------------------------- text handling

FA_TEST(crlf_and_tabs) {
    const std::string text = with_crlf(
        "crlf file\n1.0\n\t4.0\t0.0\t0.0\n0.0\t4.0\t0.0\n0.0\t0.0\t4.0\nH\n2\nDirect\n"
        "0.25\t0.25\t0.25\n0.75 0.75 0.75 ! x\n\n\n");
    const Atoms a = fastatoms::io::read_poscar(text);
    FA_CHECK(a.size() == 2);
    FA_CHECK(a.info()["comment"].as_string() == "crlf file");
    FA_CHECK_CLOSE(a.position(0).x, 1.0, 1e-12);
    FA_CHECK_CLOSE(a.position(1).z, 3.0, 1e-12);
    FA_CHECK_CLOSE(a.cell().vector(0).x, 4.0, 1e-12);
}

// ---------------------------------------------------------------- velocities

// A CONTCAR from an MD step: a Direct positions block, a blank line, Cartesian
// velocities (no mode line), then a predictor-corrector block that is skipped.
// ASE agrees on the values (its velocities are these times Ang/fs).
FA_TEST(contcar_velocity_block_is_read_and_predictor_corrector_skipped) {
    const Atoms a = fastatoms::io::read_poscar(fixture("CONTCAR_md"));
    FA_CHECK(a.has_momenta());
    // Stored verbatim, so the file's decimal reads back to the same double.
    const std::vector<double> v = a.velocities();
    FA_CHECK((v.size() == 9));
    FA_CHECK(v[0] == 0.01);
    FA_CHECK(v[1] == -0.02);
    FA_CHECK(v[2] == 0.03);
    FA_CHECK(v[3] == -0.04);
    FA_CHECK(v[8] == -0.01);
}

FA_TEST(velocity_mode_direct_converts_by_cell) {
    const char* direct = R"(vel
1.0
  3.0 0.0 0.0
  0.0 3.0 0.0
  0.0 0.0 3.0
H
1
Direct
0.0 0.0 0.0

Direct
1.0 0.0 0.0
)";
    const Atoms a = fastatoms::io::read_poscar(direct);
    FA_CHECK(a.velocities()[0] == 3.0);
    FA_CHECK(a.velocities()[1] == 0.0);

    // A blank mode line and a Cartesian mode line both use the values as-is.
    const char* cartesian = R"(vel
1.0
  3.0 0.0 0.0
  0.0 3.0 0.0
  0.0 0.0 3.0
H
1
Direct
0.0 0.0 0.0

Cartesian
1.0 0.0 0.0
)";
    const Atoms b = fastatoms::io::read_poscar(cartesian);
    FA_CHECK(b.velocities()[0] == 1.0);
}

// ---------------------------------------------------------------- writer

FA_TEST(writer_keeps_labels_and_groups_consecutive_runs) {
    const Atoms a = fastatoms::io::read_poscar(fixture("SPOSCAR_phonopy"));
    const std::string text = fastatoms::io::write_poscar(a);
    FA_CHECK(text.find("\nFe1 Fe2\n2 2\nDirect\n") != std::string::npos);

    // Interleaved labels: runs by default, grouped by label with sort.
    Atoms b(std::vector<int>{26, 26, 26, 26},
            std::vector<double>{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3},
            Cell::cubic(10.0));
    b.set_string_array("labels", {"Fe1", "Fe2", "Fe1", "Fe2"});
    const std::string runs = fastatoms::io::write_poscar(b);
    FA_CHECK(runs.find("\nFe1 Fe2 Fe1 Fe2\n1 1 1 1\n") != std::string::npos);
    PoscarWriteOptions sorted;
    sorted.sort = true;
    const std::string grouped = fastatoms::io::write_poscar(b, sorted);
    FA_CHECK(grouped.find("\nFe1 Fe2\n2 2\n") != std::string::npos);
    const Atoms reread = fastatoms::io::read_poscar(grouped);
    FA_CHECK((reread.string_array("labels") ==
              std::vector<std::string>{"Fe1", "Fe1", "Fe2", "Fe2"}));
    FA_CHECK_CLOSE(reread.position(1).x, 2.0, 1e-12);  // atom 2 was the second Fe1

    PoscarWriteOptions no_labels;
    no_labels.write_labels = false;
    FA_CHECK(fastatoms::io::write_poscar(b, no_labels).find("\nFe\n4\n") != std::string::npos);
}

FA_TEST(writer_selective_cartesian_and_vasp4) {
    const Atoms a = fastatoms::io::read_poscar(fixture("POSCAR_pv"));
    const std::string text = fastatoms::io::write_poscar(a);
    FA_CHECK(text.find("\nSelective dynamics\nDirect\n") != std::string::npos);
    FA_CHECK(text.find(" F T F\n") != std::string::npos);

    PoscarWriteOptions cart;
    cart.direct = false;
    const std::string c = fastatoms::io::write_poscar(a, cart);
    FA_CHECK(c.find("\nSelective dynamics\nCartesian\n") != std::string::npos);

    PoscarWriteOptions v4;
    v4.vasp5 = false;
    const std::string four = fastatoms::io::write_poscar(a, v4);
    FA_CHECK(four.find("Fe_pv1") == std::string::npos);
    FA_CHECK(four.find("\n2\nSelective dynamics") != std::string::npos);
}

// Read -> write -> read. Cartesian output is bit-identical: positions are
// stored Cartesian and the shortest round-trip decimal reads back unchanged.
// Direct output is not bit-identical. Positions are stored Cartesian, so the
// fractions are re-derived through the cell inverse and can move by an ulp
// or two on a skewed cell. The bound is 1e-15 relative. On the diagonal
// fixture below the text also happens to be stable after that first trip.
FA_TEST(round_trip) {
    const Atoms first = fastatoms::io::read_poscar(fixture("CONTCAR_md"));

    PoscarWriteOptions cart;
    cart.direct = false;
    const std::string text = fastatoms::io::write_poscar(first, cart);
    const Atoms second = fastatoms::io::read_poscar(text);
    FA_CHECK(second.positions() == first.positions());
    FA_CHECK(second.cell().matrix() == first.cell().matrix());
    FA_CHECK((second.string_array("labels") == first.string_array("labels")));
    FA_CHECK(second.velocities() == first.velocities());
    FA_CHECK(fastatoms::io::write_poscar(second, cart) == text);

    const std::string direct = fastatoms::io::write_poscar(first);
    const Atoms third = fastatoms::io::read_poscar(direct);
    FA_CHECK(fastatoms::io::write_poscar(third) == direct);
    for (std::size_t k = 0; k < first.positions().size(); ++k) {
        FA_CHECK(within_relative(third.positions()[k], first.positions()[k], 1e-15));
    }
}

// The same round trip on a skewed cell with fractions that are not exact in
// binary, where Direct is the one that can drift.
FA_TEST(round_trip_skewed_cell) {
    const std::string text = R"(skewed
1.0
  1.0 0.0 0.0
  0.5 0.866 0.0
  0.0 0.0 1.0
H O
1 1
Direct
0.3 0.7 0.11
0.61 0.13 0.97
)";
    const Atoms first = fastatoms::io::read_poscar(text);

    PoscarWriteOptions cart;
    cart.direct = false;
    const std::string cart_text = fastatoms::io::write_poscar(first, cart);
    const Atoms cart_back = fastatoms::io::read_poscar(cart_text);
    FA_CHECK(cart_back.positions() == first.positions());
    FA_CHECK(fastatoms::io::write_poscar(cart_back, cart) == cart_text);

    const std::string direct = fastatoms::io::write_poscar(first);
    const Atoms back = fastatoms::io::read_poscar(direct);
    for (std::size_t k = 0; k < first.positions().size(); ++k) {
        FA_CHECK(within_relative(back.positions()[k], first.positions()[k], 1e-15));
    }
}

// ---------------------------------------------------------------- XDATCAR

FA_TEST(xdatcar_fixed_cell_layout) {
    const auto frames = fastatoms::io::read_xdatcar(fixture("XDATCAR_fixed"));
    FA_CHECK(frames.size() == 2);
    FA_CHECK_CLOSE(frames[0].cell().vector(0).x, 3.0, 1e-12);
    FA_CHECK_CLOSE(frames[1].cell().vector(0).x, 3.0, 1e-12);
    FA_CHECK_CLOSE(frames[1].position(0).x, 0.03, 1e-12);  // Direct 0.01 * 3 A
    FA_CHECK((frames[1].string_array("labels") == std::vector<std::string>{"Fe", "Fe", "O"}));
}

FA_TEST(xdatcar_variable_cell_layout) {
    const auto frames = fastatoms::io::read_xdatcar(fixture("XDATCAR_variable"));
    FA_CHECK(frames.size() == 2);
    FA_CHECK_CLOSE(frames[0].cell().vector(0).x, 3.0, 1e-12);
    FA_CHECK_CLOSE(frames[1].cell().vector(0).x, 3.1, 1e-12);
    FA_CHECK_CLOSE(frames[1].position(1).x, 1.55, 1e-12);  // Direct 0.5 * 3.1 A
    FA_CHECK((frames[1].numbers() == std::vector<int>{26, 26, 8}));
}

// ---------------------------------------------------------------- edge cases

// "Cubic" is a word, not a label, so a VASP 4 comment line of it gives no species.
FA_TEST(vasp4_cubic_comment_asks_for_species_option) {
    const std::string text = R"(Cubic
1.0
  3.0 0.0 0.0
  0.0 3.0 0.0
  0.0 0.0 3.0
2
Direct
0.0 0.0 0.0
0.5 0.5 0.5
)";
    const std::string message = format_error([&] { (void)fastatoms::io::read_poscar(text); });
    FA_CHECK(message.find("species option") != std::string::npos);
    PoscarReadOptions options;
    options.species = {"Fe"};
    FA_CHECK((fastatoms::io::read_poscar(text, options).numbers() == std::vector<int>{26, 26}));
}

FA_TEST(vasp4_selective_dynamics) {
    const std::string text = R"(Fe O
1.0
  3.0 0.0 0.0
  0.0 3.0 0.0
  0.0 0.0 3.0
2 1
Selective dynamics
Direct
0.0 0.0 0.0 T T F
0.5 0.5 0.5 F F T
0.25 0.25 0.25 T F F
)";
    const Atoms a = fastatoms::io::read_poscar(text);
    FA_CHECK((a.string_array("labels") == std::vector<std::string>{"Fe", "Fe", "O"}));
    FA_CHECK((flags_of(a) == std::vector<int>{1, 1, 0, 0, 0, 1, 1, 0, 0}));
}

FA_TEST(species_option_is_ignored_when_species_line_present) {
    PoscarReadOptions options;
    options.species = {"Xx", "Yy"};
    const Atoms a = fastatoms::io::read_poscar(fixture("POSCAR_pv"), options);
    FA_CHECK((a.string_array("labels") == std::vector<std::string>{"Fe_pv1", "Fe_pv1"}));
    FA_CHECK((a.numbers() == std::vector<int>{26, 26}));
}

FA_TEST(duplicate_species_blocks_are_allowed) {
    const std::string text = R"(dup
1.0
  3.0 0.0 0.0
  0.0 3.0 0.0
  0.0 0.0 3.0
Fe O Fe
1 1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
0.25 0.25 0.25
)";
    const Atoms a = fastatoms::io::read_poscar(text);
    FA_CHECK((a.string_array("labels") == std::vector<std::string>{"Fe", "O", "Fe"}));
    FA_CHECK((a.numbers() == std::vector<int>{26, 8, 26}));
    const std::string out = fastatoms::io::write_poscar(a);
    FA_CHECK(out.find("\nFe O Fe\n1 1 1\n") != std::string::npos);
}

// Trailing `#` and `!` comments on the species and counts lines, and on the scale line.
FA_TEST(species_counts_and_scale_lines_take_trailing_comments) {
    const char* variants[] = {"Fe O   # species\n1 1  # counts\n", "Fe O ! species\n1 1 ! counts\n"};
    for (const char* lines : variants) {
        const std::string text = std::string(R"(comments
1.0   # scale
  3.0 0.0 0.0
  0.0 3.0 0.0
  0.0 0.0 3.0
)") + lines + "Direct\n0.0 0.0 0.0\n0.5 0.5 0.5\n";
        const Atoms a = fastatoms::io::read_poscar(text);
        FA_CHECK((a.string_array("labels") == std::vector<std::string>{"Fe", "O"}));
        FA_CHECK((a.numbers() == std::vector<int>{26, 8}));
        FA_CHECK_CLOSE(a.position(1).x, 1.5, 1e-12);
    }
}

// A malformed or short velocity block after the positions is skipped: the
// read succeeds, with no velocities and the positions intact.
FA_TEST(malformed_velocity_block_is_skipped) {
    const std::string head = R"(vel
1.0
  3.0 0.0 0.0
  0.0 3.0 0.0
  0.0 0.0 3.0
H
2
Direct
0.0 0.0 0.0
0.5 0.5 0.5
)";
    const char* tails[] = {
        "\n0.1 0.2 0.3\n",                       // short: one row of two
        "\n0.1 0.2 0.3\n0.4 0.5\n",             // a row with two numbers
        "\n0.1 0.2 0.3\n0.4 abc 0.6\n",         // a row with a bad number
        "\n0.5000E+00 0.1E+01\n0.0 0.0 0.0\n",  // first row short
        "\n  Predictor-corrector\n0.0 0.0\n",   // predictor-corrector block
        "\nweird words here\n0.1 0.2 0.3\n",    // not a mode line
    };
    for (const char* tail : tails) {
        // Any exception fails the case, so a plain read is the check that it does not throw.
        const Atoms a = fastatoms::io::read_poscar(head + tail);
        FA_CHECK(!a.has_momenta());
        FA_CHECK_CLOSE(a.position(1).x, 1.5, 1e-12);
    }
}

// A velocity block whose rows are all zero is still a velocity block.
FA_TEST(velocity_block_all_zero_and_no_blank_line) {
    const std::string zero = R"(zero
1.0
  3.0 0.0 0.0
  0.0 3.0 0.0
  0.0 0.0 3.0
H
1
Direct
0.0 0.0 0.0

0.0 0.0 0.0
)";
    const Atoms a = fastatoms::io::read_poscar(zero);
    FA_CHECK(a.has_momenta());
    FA_CHECK((a.velocities() == std::vector<double>{0.0, 0.0, 0.0}));

    // Velocities right after the positions, with no blank line and no mode line.
    const std::string noblank = R"(noblank
1.0
  3.0 0.0 0.0
  0.0 3.0 0.0
  0.0 0.0 3.0
H
1
Direct
0.0 0.0 0.0
0.1 0.2 0.3
)";
    const Atoms b = fastatoms::io::read_poscar(noblank);
    FA_CHECK((b.velocities() == std::vector<double>{0.1, 0.2, 0.3}));
}

// Writer velocities are read back bit-for-bit.
FA_TEST(writer_velocity_block_round_trip) {
    Atoms a(std::vector<int>{26, 8}, std::vector<double>{0, 0, 0, 1.5, 1.5, 1.5},
            Cell::cubic(3.0));
    a.set_velocities({0.1, -1e-7, 0.123456789012345678, 1.0 / 3.0, -2.5e10, 7.0});
    const std::string text = fastatoms::io::write_poscar(a);
    FA_CHECK(text.find("\n\n") != std::string::npos);  // the blank line before velocities
    const Atoms b = fastatoms::io::read_poscar(text);
    FA_CHECK(b.velocities() == a.velocities());
}

// sort=true reorders selective_dynamics and velocities with the atoms.
FA_TEST(sort_keeps_selective_dynamics_and_velocities_aligned) {
    Atoms a(std::vector<int>{26, 8, 26, 8}, std::vector<double>{1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0},
            Cell::cubic(10.0));
    a.set_string_array("labels", {"Fe1", "O", "Fe2", "O"});
    a.set_int_array("selective_dynamics", std::vector<std::int64_t>{1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1},
                    3);
    a.set_velocities({0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 1e-7, 2.0, 3.0, -0.7, 0.8, 0.9});

    PoscarWriteOptions sorted;
    sorted.sort = true;
    sorted.direct = false;
    const Atoms b = fastatoms::io::read_poscar(fastatoms::io::write_poscar(a, sorted));
    // Groups in first-appearance order: Fe1 (atom 0), O (atoms 1 and 3), Fe2 (atom 2).
    FA_CHECK((b.string_array("labels") == std::vector<std::string>{"Fe1", "O", "O", "Fe2"}));
    FA_CHECK((b.numbers() == std::vector<int>{26, 8, 8, 26}));
    FA_CHECK((b.positions() == std::vector<double>{1, 0, 0, 2, 0, 0, 4, 0, 0, 3, 0, 0}));
    FA_CHECK((flags_of(b) == std::vector<int>{1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1}));
    FA_CHECK((b.velocities() ==
              std::vector<double>{0.1, -0.2, 0.3, 0.4, 0.5, -0.6, -0.7, 0.8, 0.9, 1e-7, 2.0, 3.0}));
}

// labels=false writes symbols; vasp5=false drops the species line.
FA_TEST(writer_labels_false_and_vasp5_false) {
    Atoms a(std::vector<int>{26, 8}, std::vector<double>{0, 0, 0, 1, 1, 1}, Cell::cubic(10.0));
    a.set_string_array("labels", {"Fe1", "O2-"});

    PoscarWriteOptions symbols;
    symbols.write_labels = false;
    const std::string with_symbols = fastatoms::io::write_poscar(a, symbols);
    FA_CHECK(with_symbols.find("\nFe O\n1 1\n") != std::string::npos);
    FA_CHECK((fastatoms::io::read_poscar(with_symbols).string_array("labels") ==
              std::vector<std::string>{"Fe", "O"}));

    PoscarWriteOptions v4;
    v4.vasp5 = false;
    const std::string four = fastatoms::io::write_poscar(a, v4);
    FA_CHECK(four.find("\n1 1\nDirect\n") != std::string::npos);
    // The species ride on the comment line, so the file reads back as is.
    FA_CHECK(four.rfind("Fe O\n", 0) == 0);
    FA_CHECK((fastatoms::io::read_poscar(four).numbers() == std::vector<int>{26, 8}));
    PoscarReadOptions species;
    species.species = {"Fe", "O"};
    FA_CHECK((fastatoms::io::read_poscar(four, species).numbers() == std::vector<int>{26, 8}));
}

// Labels that name another element, or are not one token, fall back to the symbol.
FA_TEST(writer_replaces_labels_that_do_not_name_the_element) {
    Atoms a(std::vector<int>{26, 8, 26}, std::vector<double>{0, 0, 0, 1, 1, 1, 2, 2, 2},
            Cell::cubic(10.0));
    a.set_string_array("labels", {"1", "2", "Fe2"});
    const std::string text = fastatoms::io::write_poscar(a);
    FA_CHECK(text.find("\nFe O Fe2\n1 1 1\n") != std::string::npos);
    const Atoms back = fastatoms::io::read_poscar(text);
    FA_CHECK((back.string_array("labels") == std::vector<std::string>{"Fe", "O", "Fe2"}));
    FA_CHECK((back.numbers() == std::vector<int>{26, 8, 26}));

    Atoms b(std::vector<int>{26, 8}, std::vector<double>{0, 0, 0, 1, 1, 1}, Cell::cubic(10.0));
    b.set_string_array("labels", {"O", "O1"});  // "O" on an iron atom
    FA_CHECK(fastatoms::io::write_poscar(b).find("\nFe O1\n1 1\n") != std::string::npos);
}

// The comment line: info["comment"], then the comment option, then the formula.
FA_TEST(writer_comment_sources) {
    Atoms a(std::vector<int>{1, 1}, std::vector<double>{0, 0, 0, 1, 1, 1}, Cell::cubic(10.0));
    a.info()[std::string("comment")] = fastatoms::Value(std::string("from info"));
    FA_CHECK(fastatoms::io::write_poscar(a).find("from info\n1.0\n") == 0);

    PoscarWriteOptions explicit_comment;
    explicit_comment.comment = "explicit\nand more";
    FA_CHECK(fastatoms::io::write_poscar(a, explicit_comment).find("explicit and more\n1.0\n") == 0);

    Atoms b(std::vector<int>{1, 1}, std::vector<double>{0, 0, 0, 1, 1, 1}, Cell::cubic(10.0));
    FA_CHECK(fastatoms::io::write_poscar(b).find(b.formula() + "\n1.0\n") == 0);
}

// With vasp5=false the comment line is the species line, and the reader takes
// every comment token as a species. So a custom comment cannot be written.
FA_TEST(writer_vasp4_comment_must_be_the_species_line) {
    Atoms a(std::vector<int>{26, 8}, std::vector<double>{0, 0, 0, 1, 1, 1}, Cell::cubic(10.0));
    PoscarWriteOptions v4;
    v4.vasp5 = false;

    v4.comment = "my structure";
    const std::string message = format_error([&] { (void)fastatoms::io::write_poscar(a, v4); });
    FA_CHECK(message.find("vasp5=true") != std::string::npos);

    // The species themselves are an acceptable comment, and read back.
    v4.comment = "Fe O";
    const std::string text = fastatoms::io::write_poscar(a, v4);
    FA_CHECK(text.rfind("Fe O\n", 0) == 0);
    FA_CHECK((fastatoms::io::read_poscar(text).numbers() == std::vector<int>{26, 8}));

    // Without a comment option the species line is written, not info["comment"].
    a.info()[std::string("comment")] = fastatoms::Value(std::string("from info"));
    PoscarWriteOptions no_comment;
    no_comment.vasp5 = false;
    FA_CHECK(fastatoms::io::write_poscar(a, no_comment).rfind("Fe O\n", 0) == 0);
}

// Byte-order mark, lowercase mode words.
FA_TEST(leading_utf8_bom_is_ignored) {
    const std::string text = "\xEF\xBB\xBF"
                             R"(bom
1.0
  3.0 0.0 0.0
  0.0 3.0 0.0
  0.0 0.0 3.0
Fe O
1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
)";
    const Atoms a = fastatoms::io::read_poscar(text);
    FA_CHECK(a.info()["comment"].as_string() == "bom");
    FA_CHECK((a.string_array("labels") == std::vector<std::string>{"Fe", "O"}));

    // The same BOM in front of a VASP 4 comment that names the species.
    const std::string v4 = "\xEF\xBB\xBF" R"(Fe
1.0
  3.0 0.0 0.0
  0.0 3.0 0.0
  0.0 0.0 3.0
1
Direct
0.0 0.0 0.0
)";
    FA_CHECK((fastatoms::io::read_poscar(v4).string_array("labels") == std::vector<std::string>{"Fe"}));
}

FA_TEST(lowercase_direct_and_cartesian_words) {
    const std::string direct = R"(lower
1.0
  4.0 0.0 0.0
  0.0 4.0 0.0
  0.0 0.0 4.0
H
1
direct
0.5 0.5 0.5
)";
    FA_CHECK(fastatoms::io::read_poscar(direct).position(0).x == 2.0);
    const std::string cartesian = R"(lower
1.0
  4.0 0.0 0.0
  0.0 4.0 0.0
  0.0 0.0 4.0
H
1
cartesian
0.5 0.5 0.5
)";
    FA_CHECK(fastatoms::io::read_poscar(cartesian).position(0).x == 0.5);
}

// Underflow reads as zero, overflow / nan / inf are errors naming the line.
FA_TEST(underflow_is_zero_and_overflow_nan_inf_are_errors) {
    const std::string head = R"(range
1.0
  4.0 0.0 0.0
  0.0 4.0 0.0
  0.0 0.0 4.0
H
1
Cartesian
)";
    const Atoms a = fastatoms::io::read_poscar(head + "1e-400 1D-400 -1e-400\n");
    FA_CHECK(a.position(0).x == 0.0);
    FA_CHECK(a.position(0).y == 0.0);
    FA_CHECK(a.position(0).z == 0.0);

    // The position is line 9 of the text.
    for (const char* bad : {"1e400 0 0", "1D+400 0 0", "-1e400 0 0", "nan 0 0", "inf 0 0", "0 -inf 0"}) {
        const std::string message =
            format_error([&] { (void)fastatoms::io::read_poscar(head + bad + "\n"); });
        FA_CHECK(message.find("line 9") != std::string::npos);
    }
}

// ---------------------------------------------------------------- XDATCAR edge cases

// The comment may contain "configuration"; only configuration lines count.
FA_TEST(xdatcar_configuration_line_matching) {
    const std::string fixed = R"(my configuration test
1.0
3 0 0
0 3 0
0 0 3
Fe O
1 1
direct configuration=     1
0.0 0.0 0.0
0.5 0.5 0.5
Direct configuration=     2
0.1 0.0 0.0
0.5 0.5 0.5
)";
    const auto frames = fastatoms::io::read_xdatcar(fixed);
    FA_CHECK(frames.size() == 2);
    FA_CHECK_CLOSE(frames[1].position(0).x, 0.3, 1e-12);

    // A variable-cell header whose comment is "configuration 2" is still a header.
    const std::string variable = R"(configuration 2
1.0
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
Fe O
1 1
Direct configuration=     1
0.0 0.0 0.0
0.5 0.5 0.5
configuration 2
1.0
3.1 0.0 0.0
0.0 3.1 0.0
0.0 0.0 3.1
Fe O
1 1
Direct configuration=     2
0.0 0.0 0.0
0.5 0.5 0.5
)";
    const auto vframes = fastatoms::io::read_xdatcar(variable);
    FA_CHECK(vframes.size() == 2);
    FA_CHECK_CLOSE(vframes[1].cell().vector(0).x, 3.1, 1e-12);
}

// The VASP 4 layout: no species line, species from the comment line.
FA_TEST(xdatcar_vasp4_layout) {
    const std::string text = R"(Fe O
1.0
3 0 0
0 3 0
0 0 3
1 1
Direct configuration=     1
0 0 0
.5 .5 .5
Direct configuration=     2
0 0 0
.5 .5 .5
)";
    const auto frames = fastatoms::io::read_xdatcar(text);
    FA_CHECK(frames.size() == 2);
    FA_CHECK((frames[1].string_array("labels") == std::vector<std::string>{"Fe", "O"}));
    FA_CHECK_CLOSE(frames[1].position(1).x, 1.5, 1e-12);
    FA_CHECK(!frames[1].info().find("truncated"));
}

// An incomplete last frame is dropped and the last returned frame is marked.
FA_TEST(xdatcar_truncated_last_frame_is_dropped) {
    const std::string head = R"(Fe O cut
1.0
  3.0 0.0 0.0
  0.0 3.0 0.0
  0.0 0.0 3.0
Fe O
1 1
Direct configuration=     1
0.0 0.0 0.0
0.5 0.5 0.5
Direct configuration=     2
0.1 0.0 0.0
)";
    // A whole row missing.
    const auto missing_row = fastatoms::io::read_xdatcar(head);
    FA_CHECK(missing_row.size() == 1);
    FA_CHECK(missing_row[0].info().find("truncated") != nullptr);
    FA_CHECK(missing_row[0].info()["truncated"].as_bool());

    // The file is cut mid-row, with no newline at the end.
    const auto mid_row = fastatoms::io::read_xdatcar(head + "0.5 0.5");
    FA_CHECK(mid_row.size() == 1);
    FA_CHECK(mid_row[0].info().find("truncated") != nullptr);

    // Cut right after the configuration line.
    const auto no_rows = fastatoms::io::read_xdatcar(head.substr(0, head.rfind("0.1")));
    FA_CHECK(no_rows.size() == 1);

    // A complete file has no mark.
    const auto complete = fastatoms::io::read_xdatcar(head + "0.5 0.5 0.5\n");
    FA_CHECK(complete.size() == 2);
    FA_CHECK(complete.back().info().find("truncated") == nullptr);

    // A variable-cell file cut inside the second frame's header.
    const std::string variable = fixture("XDATCAR_variable");
    const auto cut_header = fastatoms::io::read_xdatcar(
        variable.substr(0, variable.rfind("Fe O XDATCAR variable cell") + 40));
    FA_CHECK(cut_header.size() == 1);
    FA_CHECK(cut_header[0].info().find("truncated") != nullptr);
}

// Cartesian frames are Cartesian times the scale, as in POSCAR. Direct frames
// are fractional, on the scaled lattice.
FA_TEST(xdatcar_cartesian_configuration_is_not_fractional) {
    const std::string text = R"(cart frames
2.0
3 0 0
0 3 0
0 0 3
Fe O
1 1
Cartesian configuration=     1
0.25 0 0
0.5 0.5 0.5
Direct configuration=     2
0.25 0 0
0.5 0.5 0.5
)";
    const auto frames = fastatoms::io::read_xdatcar(text);
    FA_CHECK(frames.size() == 2);
    FA_CHECK_CLOSE(frames[0].position(0).x, 0.5, 1e-12);
    FA_CHECK_CLOSE(frames[0].position(1).x, 1.0, 1e-12);
    FA_CHECK_CLOSE(frames[1].position(0).x, 1.5, 1e-12);
    FA_CHECK_CLOSE(frames[1].position(1).x, 3.0, 1e-12);
    FA_CHECK_CLOSE(frames[0].cell().vector(0).x, 6.0, 1e-12);
}

// Atom counts are checked against the lines left before anything is allocated.
// A count that wraps when multiplied by 3 (6148914691236517206 * 3 == 2 mod 2^64),
// a count past the int64 range, a sum that exceeds the file and a single count
// that exceeds it are all FormatErrors naming the counts line (line 7).
FA_TEST(poscar_atom_counts_are_checked_before_allocation) {
    const auto poscar = [](const std::string& counts) {
        return std::string(R"(huge counts
1.0
  3.0 0.0 0.0
  0.0 3.0 0.0
  0.0 0.0 3.0
Fe O
)") + counts + "\nDirect\n0.0 0.0 0.0\n";
    };
    // Only one coordinate line follows, so "2 2" (sum 4) and "2 1" (sum 3) do not fit.
    for (const char* counts : {"6148914691236517206", "9223372036854775807 9223372036854775807",
                               "18446744073709551615", "3 0", "2 2", "2 1"}) {
        const std::string message =
            format_error([&] { (void)fastatoms::io::read_poscar(poscar(counts)); });
        FA_CHECK(message.find("line 7") != std::string::npos);
    }
    // Counts that fit the lines left are accepted.
    FA_CHECK(fastatoms::io::read_poscar(poscar("1 1") + "0.5 0.5 0.5\n").size() == 2);
}

// The same checks on the XDATCAR header, which is parsed the same way.
FA_TEST(xdatcar_atom_counts_are_checked_before_allocation) {
    const std::string text = R"(huge counts
1.0
  3.0 0.0 0.0
  0.0 3.0 0.0
  0.0 0.0 3.0
Fe O
6148914691236517206 1
Direct configuration=     1
0.0 0.0 0.0
)";
    const std::string message = format_error([&] { (void)fastatoms::io::read_xdatcar(text); });
    FA_CHECK(message.find("line 7") != std::string::npos);
}

FA_TEST_MAIN()
