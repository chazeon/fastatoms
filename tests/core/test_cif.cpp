// SPDX-License-Identifier: MIT
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "fastatoms/io/cif.hpp"
#include "fastatoms/io/format.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::io;

namespace {

std::string read_text(const std::string& relative) {
    std::ifstream in(std::string(FASTATOMS_TEST_DATA) + "/cif/" + relative);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

/// One structure as ASE reads it (tests/data/cif/<name>.ase, made by ASE).
struct Reference {
    std::array<double, 6> cell{};
    std::vector<std::string> symbols;
    std::vector<std::array<double, 3>> frac;
};

std::vector<Reference> read_reference(const std::string& name) {
    std::ifstream in(std::string(FASTATOMS_TEST_DATA) + "/cif/" + name + ".ase");
    std::vector<Reference> out;
    std::string line;
    std::size_t remaining = 0;
    while (std::getline(in, line)) {
        std::istringstream row(line);
        std::string key;
        row >> key;
        if (key == "structure") {
            out.emplace_back();
        } else if (key == "cell") {
            for (double& v : out.back().cell) {
                row >> v;
            }
        } else if (key == "count") {
            row >> remaining;
        } else if (!key.empty() && remaining > 0) {
            std::array<double, 3> f{};
            row >> f[0] >> f[1] >> f[2];
            out.back().symbols.push_back(key);
            out.back().frac.push_back(f);
            --remaining;
        }
    }
    return out;
}

/// True when read_cif throws a FormatError whose message contains `needle`.
bool throws_with(const std::string& text, const std::string& needle,
                 const CifReadOptions& options = {}) {
    try {
        (void)read_cif(text, options);
    } catch (const FormatError& error) {
        return std::string(error.what()).find(needle) != std::string::npos;
    } catch (...) {
    }
    return false;
}

/// A data block with a 3 Å cubic cell, the given symmetry lines (may be
/// empty), and a loop with `header` and `rows`.
std::string cubic_block(const std::string& symmetry, const std::string& header,
                        const std::string& rows) {
    return "data_t\n_cell_length_a 3\n_cell_length_b 3\n_cell_length_c 3\n"
           "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n" +
           symmetry + "loop_\n" + header + rows;
}

const char* const kSiteHeader =
    "_atom_site_label\n_atom_site_type_symbol\n"
    "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n";

/// Squared fractional distance, periodic in each direction.
double periodic_distance2(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    double d2 = 0.0;
    for (std::size_t k = 0; k < 3; ++k) {
        double d = a[k] - b[k];
        d -= std::round(d);
        d2 += d * d;
    }
    return d2;
}

/// Compare with ASE: cell, atom count, and every ASE atom matched to a
/// distinct atom of the same element at the same periodic fractional spot.
void check_against_ase(const Atoms& atoms, const Reference& ref) {
    const Vec3 lengths = atoms.cell().lengths();
    const Vec3 angles = atoms.cell().angles();
    FA_CHECK_CLOSE(lengths.x, ref.cell[0], 1e-7);
    FA_CHECK_CLOSE(lengths.y, ref.cell[1], 1e-7);
    FA_CHECK_CLOSE(lengths.z, ref.cell[2], 1e-7);
    FA_CHECK_CLOSE(angles.x, ref.cell[3], 1e-7);
    FA_CHECK_CLOSE(angles.y, ref.cell[4], 1e-7);
    FA_CHECK_CLOSE(angles.z, ref.cell[5], 1e-7);
    FA_CHECK(atoms.size() == ref.symbols.size());
    if (atoms.size() != ref.symbols.size()) {
        return;
    }

    const std::vector<double> scaled = atoms.scaled_positions(false);
    const std::vector<std::string> symbols = atoms.symbols();
    std::vector<bool> used(atoms.size(), false);
    for (std::size_t i = 0; i < ref.symbols.size(); ++i) {
        bool found = false;
        for (std::size_t j = 0; j < atoms.size() && !found; ++j) {
            if (used[j] || symbols[j] != ref.symbols[i]) {
                continue;
            }
            const std::array<double, 3> f{scaled[3 * j], scaled[3 * j + 1], scaled[3 * j + 2]};
            if (periodic_distance2(f, ref.frac[i]) < 1e-10) {
                used[j] = true;
                found = true;
            }
        }
        FA_CHECK(found);
    }
}

}  // namespace

FA_TEST(cif_nacl_hm_symbol_only) {
    const auto structures = read_cif(read_text("nacl.cif"));
    FA_CHECK(structures.size() == 1);
    const Atoms& atoms = structures.at(0);
    FA_CHECK(atoms.size() == 8);
    FA_CHECK(atoms.formula() == "Cl4Na4");
    // Site order first (Na's four images, then Cl's), then the op order.
    const auto labels = atoms.labels();
    FA_CHECK(labels.size() == 8);
    FA_CHECK(labels.at(0) == "Na1" && labels.at(3) == "Na1");
    FA_CHECK(labels.at(4) == "Cl1" && labels.at(7) == "Cl1");
    FA_CHECK(atoms.info()["spacegroup"].as_string() == "F m -3 m");
    FA_CHECK(atoms.has_array("occupancy"));
    FA_CHECK_CLOSE(atoms.array("occupancy").data.at(0), 1.0, 1e-12);
    check_against_ase(atoms, read_reference("nacl").at(0));
}

FA_TEST(cif_rutile_symop_loop) {
    // No H-M symbol in this file: only the symop loop (plus an IT number that
    // ASE needs) can produce the 16 operations.
    const auto structures = read_cif(read_text("rutile.cif"));
    FA_CHECK(structures.size() == 1);
    const Atoms& atoms = structures.at(0);
    FA_CHECK(atoms.size() == 6);
    int ti = 0;
    int o = 0;
    for (const int z : atoms.numbers()) {
        ti += z == 22 ? 1 : 0;
        o += z == 8 ? 1 : 0;
    }
    FA_CHECK(ti == 2);
    FA_CHECK(o == 4);
    FA_CHECK(atoms.info().is_null());
    check_against_ase(atoms, read_reference("rutile").at(0));
}

FA_TEST(cif_wurtzite_hexagonal_table) {
    const auto structures = read_cif(read_text("wurtzite.cif"));
    FA_CHECK(structures.size() == 1);
    const Atoms& atoms = structures.at(0);
    FA_CHECK(atoms.size() == 4);
    FA_CHECK_CLOSE(atoms.cell().angles().z, 120.0, 1e-9);
    FA_CHECK(atoms.info()["spacegroup"].as_string() == "P 63 m c");
    check_against_ase(atoms, read_reference("wurtzite").at(0));
}

FA_TEST(cif_monoclinic_non_right_beta) {
    const auto structures = read_cif(read_text("monoclinic.cif"));
    FA_CHECK(structures.size() == 1);
    const Atoms& atoms = structures.at(0);
    FA_CHECK(atoms.size() == 8);
    FA_CHECK_CLOSE(atoms.cell().angles().y, 105.3, 1e-9);
    // Lattice vector a along x, b in the xy plane (Cell::from_parameters).
    FA_CHECK_CLOSE(atoms.cell().matrix().row(0).y, 0.0, 1e-12);
    FA_CHECK_CLOSE(atoms.cell().matrix().row(1).z, 0.0, 1e-12);
    check_against_ase(atoms, read_reference("monoclinic").at(0));
}

FA_TEST(cif_special_position_merged) {
    // Na on the inversion centre: its four ops give two distinct images.
    const auto structures = read_cif(read_text("special.cif"));
    const Atoms& atoms = structures.at(0);
    FA_CHECK(atoms.size() == 6);
    const auto labels = atoms.labels();
    FA_CHECK(labels.at(0) == "Na1" && labels.at(1) == "Na1");
    FA_CHECK(labels.at(2) == "C1");
    check_against_ase(atoms, read_reference("special").at(0));
}

FA_TEST(cif_disorder_keep_all) {
    const auto structures = read_cif(read_text("disorder.cif"));
    const Atoms& atoms = structures.at(0);
    FA_CHECK(atoms.size() == 4);
    const auto& occ = atoms.array("occupancy").data;
    FA_CHECK_CLOSE(occ.at(0), 0.6, 1e-12);
    FA_CHECK_CLOSE(occ.at(1), 0.4, 1e-12);
}

FA_TEST(cif_disorder_max_occupancy) {
    CifReadOptions options;
    options.disorder = CifDisorder::MaxOccupancy;
    const auto structures = read_cif(read_text("disorder.cif"), options);
    const Atoms& atoms = structures.at(0);
    FA_CHECK(atoms.size() == 2);
    const auto labels = atoms.labels();
    // Fe (0.6) beats Co (0.4); Ni and Cu tie at 0.5, so the first wins.
    FA_CHECK(labels.at(0) == "Fe1");
    FA_CHECK(labels.at(1) == "Ni1");
    FA_CHECK_CLOSE(atoms.array("occupancy").data.at(0), 0.6, 1e-12);
}

// ASE's info["occupancy"]: per asymmetric-unit row, every species at its
// position, its own first; atoms point at the first row there through
// spacegroup_kinds (Ni and Cu tie, so atom 1 keeps row 2 either way).
FA_TEST(cif_occupancy_info_matches_ase) {
    CifReadOptions options;
    options.disorder = CifDisorder::MaxOccupancy;
    const Atoms atoms = read_cif(read_text("disorder.cif"), options).at(0);
    FA_CHECK(atoms.int_array("spacegroup_kinds").data == (std::vector<std::int64_t>{0, 2}));
    const Value& occ = *atoms.info().find("occupancy");
    FA_CHECK(occ.as_map().size() == 4);
    const Value::Map& row1 = occ.find("1")->as_map();
    FA_CHECK(row1.at(0).first == "Co" && row1.at(1).first == "Fe");
    FA_CHECK_CLOSE(row1.at(0).second.as_double(), 0.4, 1e-12);
    FA_CHECK_CLOSE(row1.at(1).second.as_double(), 0.6, 1e-12);
    FA_CHECK(occ.find("2")->find("Cu") != nullptr);
    // keep_all keeps the same info and gives every atom its own row.
    const Atoms all = read_cif(read_text("disorder.cif")).at(0);
    FA_CHECK(all.int_array("spacegroup_kinds").data == (std::vector<std::int64_t>{0, 1, 2, 3}));
    FA_CHECK(all.info().find("occupancy") != nullptr);
}

FA_TEST(cif_disorder_error) {
    CifReadOptions options;
    options.disorder = CifDisorder::Error;
    FA_CHECK_THROWS(read_cif(read_text("disorder.cif"), options), FormatError);
    // An ordered file is fine under the same policy.
    options.disorder = CifDisorder::Error;
    FA_CHECK(read_cif(read_text("nacl.cif"), options).size() == 1);
}

FA_TEST(cif_charged_type_symbols) {
    const auto structures = read_cif(read_text("charges.cif"));
    const Atoms& atoms = structures.at(0);
    FA_CHECK(atoms.size() == 3);
    FA_CHECK(atoms.numbers() == (std::vector<int>{26, 8, 29}));
    const auto labels = atoms.labels();
    FA_CHECK(labels.at(0) == "Fe_1");
    FA_CHECK(labels.at(1) == "O_1");
    FA_CHECK(labels.at(2) == "Cu_2");
    check_against_ase(atoms, read_reference("charges").at(0));
}

FA_TEST(cif_quotes_text_fields_and_multiline_rows) {
    const auto structures = read_cif(read_text("syntax.cif"));
    const Atoms& atoms = structures.at(0);
    FA_CHECK(atoms.size() == 2);
    const auto labels = atoms.labels();
    FA_CHECK(labels.at(0) == "Na 1");
    FA_CHECK(labels.at(1) == "Cl 1");
    check_against_ase(atoms, read_reference("syntax").at(0));
}

FA_TEST(cif_two_data_blocks) {
    const auto structures = read_cif(read_text("two_blocks.cif"));
    // The "data_meta" block has no atoms and is skipped.
    FA_CHECK(structures.size() == 2);
    if (structures.size() != 2) {
        return;
    }
    FA_CHECK(structures[0].size() == 1);
    FA_CHECK(structures[1].size() == 2);
    const auto reference = read_reference("two_blocks");
    FA_CHECK(reference.size() == 2);
    if (reference.size() == 2) {
        check_against_ase(structures[0], reference[0]);
        check_against_ase(structures[1], reference[1]);
    }
}

FA_TEST(cif_element_from_label_without_type_symbol) {
    const std::string text =
        "data_x\n"
        "_cell_length_a 3.0\n_cell_length_b 3.0\n_cell_length_c 3.0\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_space_group_name_H-M 'P 1'\n"
        "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Fe1 0 0 0\nOw 0.5 0.5 0.5\n";
    const Atoms atoms = read_cif(text).at(0);
    FA_CHECK(atoms.numbers() == (std::vector<int>{26, 8}));
    FA_CHECK(atoms.labels().at(0) == "Fe1");
}

FA_TEST(cif_round_trip_writer) {
    for (const char* name : {"nacl.cif", "monoclinic.cif", "charges.cif"}) {
        const Atoms original = read_cif(read_text(name)).at(0);
        const std::string written = write_cif(original);
        const Atoms again = read_cif(written).at(0);

        FA_CHECK(again.size() == original.size());
        FA_CHECK(again.numbers() == original.numbers());
        FA_CHECK(again.labels() == original.labels());
        FA_CHECK(again.symbols() == original.symbols());
        const Vec3 la = original.cell().lengths();
        const Vec3 lb = again.cell().lengths();
        FA_CHECK_CLOSE(la.x, lb.x, 1e-12);
        FA_CHECK_CLOSE(la.y, lb.y, 1e-12);
        FA_CHECK_CLOSE(la.z, lb.z, 1e-12);
        const auto fa = original.scaled_positions(false);
        const auto fb = again.scaled_positions(false);
        FA_CHECK(fa.size() == fb.size());
        for (std::size_t i = 0; i < fa.size() && i < fb.size(); ++i) {
            FA_CHECK_CLOSE(fa[i], fb[i], 1e-12);
        }
        FA_CHECK(again.array("occupancy").data == original.array("occupancy").data);
    }
}

FA_TEST(cif_writer_format) {
    const Atoms atoms = read_cif(read_text("nacl.cif")).at(0);
    const std::string text = write_cif(atoms);
    FA_CHECK(text.find("_symmetry_space_group_name_H-M 'P 1'") != std::string::npos);
    FA_CHECK(text.find("_atom_site_occupancy") != std::string::npos);
    FA_CHECK(text.find("_cell_length_a 5.64") != std::string::npos);
    // The writer emits one row per atom, P1 only.
    const Atoms again = read_cif(text).at(0);
    FA_CHECK(again.size() == 8);
    FA_CHECK(again.info()["spacegroup"].as_string() == "P 1");
}

FA_TEST(cif_writer_quotes_odd_labels) {
    Atoms atoms(std::vector<int>{26}, std::vector<double>{0.0, 0.0, 0.0}, Cell::cubic(3.0));
    atoms.set_string_array("labels", {"Fe 1"});
    const std::string text = write_cif(atoms);
    FA_CHECK(text.find("'Fe 1'") != std::string::npos);
    const Atoms again = read_cif(text).at(0);
    FA_CHECK(again.labels().at(0) == "Fe 1");
}

FA_TEST(cif_missing_cell_throws) {
    FA_CHECK_THROWS(read_cif(read_text("no_cell.cif")), FormatError);
}

FA_TEST(cif_no_atoms_throws) {
    FA_CHECK_THROWS(read_cif(read_text("no_atoms.cif")), FormatError);
}

FA_TEST(cif_unknown_space_group_throws) {
    FA_CHECK_THROWS(read_cif(read_text("unknown_sg.cif")), FormatError);
}

FA_TEST(cif_syntax_error_throws) {
    FA_CHECK_THROWS(read_cif("data_x\nloop_\n_atom_site_label\n'unterminated\n"), FormatError);
}

FA_TEST(cif_negative_tolerance_throws) {
    CifReadOptions options;
    options.merge_tolerance = -1.0;
    FA_CHECK_THROWS(read_cif(read_text("nacl.cif"), options), FormatError);
}

FA_TEST(cif_hex_third_positions_merge) {
    // Truncated 1/3, 2/3 coordinates: the images differ by about 1e-4 in
    // fractional units and must merge (ASE gives 2).
    const auto structures = read_cif(read_text("hex_third_4digit.cif"));
    FA_CHECK(structures.size() == 1);
    const Atoms& atoms = structures.at(0);
    FA_CHECK(atoms.size() == 2);
    FA_CHECK(atoms.symbols() == (std::vector<std::string>{"C", "C"}));
}

FA_TEST(cif_hex_third_positions_three_digits) {
    // Two images differ by exactly 1e-3 in fractional units (0.333 and 0.334),
    // which is ASE's symprec, so ASE keeps both and reports 4. Within 0.01 A
    // they are one atom, so the answer is the same 2 atoms per cell.
    const auto structures = read_cif(read_text("hex_third_3digit.cif"));
    FA_CHECK(structures.size() == 1);
    FA_CHECK(structures.at(0).size() == 2);
}

FA_TEST(cif_merge_is_cartesian_and_periodic) {
    // Inversion maps (1e-7, 1/2, 1/2) to (-1e-7, -1/2, -1/2): the two images
    // sit 2e-7 apart across the cell boundary, so they are one atom.
    const std::string text = cubic_block("_symmetry_space_group_name_H-M 'P -1'\n", kSiteHeader,
                                         "Fe1 Fe 0.0000001 0.5 0.5\n");
    FA_CHECK(read_cif(text).at(0).size() == 1);
    CifReadOptions exact;
    exact.merge_tolerance = 0.0;
    FA_CHECK(read_cif(text, exact).at(0).size() == 2);
}

FA_TEST(cif_merge_tolerance_is_a_distance) {
    // The 1/3, 2/3 images are about 2e-4 A apart: a 1e-5 A tolerance keeps them.
    const std::string text = read_text("hex_third_4digit.cif");
    CifReadOptions tight;
    tight.merge_tolerance = 1e-5;
    FA_CHECK(read_cif(text, tight).at(0).size() > 2);
}

FA_TEST(cif_no_symmetry_is_p1) {
    const std::string text = cubic_block("", kSiteHeader, "Fe1 Fe 0 0 0\nOw O 0.5 0.5 0.5\n");
    const Atoms atoms = read_cif(text).at(0);
    FA_CHECK(atoms.size() == 2);
    FA_CHECK(atoms.info()["spacegroup"].as_string() == "P 1");
}

FA_TEST(cif_malformed_symop_throws_naming_it) {
    const std::string text =
        cubic_block("loop_\n_symmetry_equiv_pos_as_xyz\n'x, y, z'\n'x+zz, y'\n", kSiteHeader,
                    "Fe1 Fe 0 0 0\n");
    FA_CHECK(throws_with(text, "x+zz, y"));
}

FA_TEST(cif_null_fractional_coordinate_throws_naming_site) {
    FA_CHECK(throws_with(cubic_block("", kSiteHeader, "Fe1 Fe ? 0.5 0.5\n"), "'Fe1'"));
    FA_CHECK(throws_with(cubic_block("", kSiteHeader, "Co2 Co 0.5 . 0.5\n"), "'Co2'"));
}

FA_TEST(cif_missing_fractional_column_throws) {
    const std::string header =
        "_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n";
    FA_CHECK(throws_with(cubic_block("", header, "Fe1 Fe 0 0\n"), "_atom_site_fract_z"));
}

FA_TEST(cif_cartesian_only_sites_throw_specific_message) {
    const std::string text = cubic_block(
        "", "_atom_site_label\n_atom_site_Cartn_x\n_atom_site_Cartn_y\n_atom_site_Cartn_z\n",
        "Fe1 0 0 0\n");
    FA_CHECK(throws_with(text, "Cartesian-only atom sites are not supported"));
}

FA_TEST(cif_mmcif_throws_specific_message) {
    const std::string text =
        "data_m\nloop_\n_atom_site.group_PDB\n_atom_site.label_atom_id\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        "ATOM CA 1.0 2.0 3.0\n";
    FA_CHECK(throws_with(text, "mmCIF"));
}

FA_TEST(cif_wrapped_coordinate_stays_below_one) {
    // -1e-17 wraps to exactly 1.0 in floating point; it must fold to 0.
    const std::string text = cubic_block("", kSiteHeader, "Fe1 Fe -1e-17 0.5 0.5\n");
    const Atoms atoms = read_cif(text).at(0);
    const std::vector<double> scaled = atoms.scaled_positions(false);
    FA_CHECK(scaled[0] >= 0.0 && scaled[0] < 1.0);
    FA_CHECK_CLOSE(scaled[0], 0.0, 1e-12);
}

FA_TEST(cif_writer_quotes_reserved_labels) {
    const std::vector<std::string> labels = {"x'y\"z", "loop_", "DATA_1", "save_s", "global_",
                                             "stop_",  "_a",    "#c",    "$d",     "[e]",
                                             ";f",     "a b",   "it's",  "x\ny"};
    const std::vector<std::string> expected = {"x_y_z", "loop_", "DATA_1", "save_s", "global_",
                                               "stop_", "_a",    "#c",     "$d",     "[e]",
                                               ";f",    "a b",   "it's",   "x_y"};
    std::vector<int> numbers(labels.size(), 26);
    std::vector<double> positions;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        positions.insert(positions.end(), {1.0 + 1.5 * static_cast<double>(i), 1.0, 1.0});
    }
    Atoms atoms(numbers, positions, Cell::cubic(40.0));
    atoms.set_string_array("labels", labels);
    const Atoms again = read_cif(write_cif(atoms)).at(0);
    FA_CHECK(again.size() == labels.size());
    FA_CHECK(again.labels() == expected);
}

FA_TEST(cif_max_occupancy_many_sites) {
    // 2000 positions, each given twice: a 0.4 copy and a 0.6 copy, the second
    // one shifted by a lattice vector. The index must merge them across the
    // boundary and keep the more occupied one.
    std::ostringstream text;
    text << "data_many\n_cell_length_a 10\n_cell_length_b 10\n_cell_length_c 20\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
            "_symmetry_space_group_name_H-M 'P 1'\n"
            "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
            "_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\n";
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 20; ++k) {
                const double x = i / 10.0;
                const double y = j / 10.0;
                const double z = k / 20.0;
                text << "A" << i << "_" << j << "_" << k << " Fe " << x << ' ' << y << ' ' << z
                     << " 0.4\n";
                text << "B" << i << "_" << j << "_" << k << " Fe " << x + 1.0 << ' ' << y << ' '
                     << z << " 0.6\n";
            }
        }
    }
    CifReadOptions options;
    options.disorder = CifDisorder::MaxOccupancy;
    const Atoms atoms = read_cif(text.str(), options).at(0);
    FA_CHECK(atoms.size() == 2000);
    std::size_t kept_more = 0;
    for (const std::string& label : atoms.labels()) {
        kept_more += label.front() == 'B' ? 1 : 0;
    }
    FA_CHECK(kept_more == 2000);
    FA_CHECK_CLOSE(atoms.array("occupancy").data.at(0), 0.6, 1e-12);
}

FA_TEST(cif_deterministic) {
    const auto first = read_cif(read_text("rutile.cif")).at(0);
    const auto second = read_cif(read_text("rutile.cif")).at(0);
    FA_CHECK(first.labels() == second.labels());
    FA_CHECK(first.positions() == second.positions());
}

FA_TEST(cif_merge_finds_image_in_unreduced_cell) {
    // Cell rows (10,0,0), (1000,0.01,0), (0,0,10). The sites at fractional
    // (0,0,0) and (0,0.1,0) differ by (100,0.001,0), i.e. 0.001 Å along y once
    // the lattice vector (-100,0,0) is added: the nearest image has an offset
    // of -10 in the first fractional coordinate, which a search around the
    // fractional difference does not reach.
    const std::string text =
        "data_t\n_cell_length_a 10\n_cell_length_b 1000.0000000499999\n"
        "_cell_length_c 10\n_cell_angle_alpha 90\n_cell_angle_beta 90\n"
        "_cell_angle_gamma 0.0005729578188365804\n"
        "_symmetry_space_group_name_H-M 'P 1'\nloop_\n" +
        std::string(kSiteHeader) + "Fe1 Fe 0 0 0\nFe2 Fe 0 0.1 0\n";
    CifReadOptions merged;
    merged.disorder = CifDisorder::MaxOccupancy;
    FA_CHECK(read_cif(text, merged).at(0).size() == 1);
    // Keep-all does not merge across sites, so both stay.
    FA_CHECK(read_cif(text).at(0).size() == 2);
}

FA_TEST(cif_writer_quotes_missing_value_labels) {
    // A bare ? or . is a missing value; a label spelled that way must stay quoted.
    Atoms atoms(std::vector<int>{26, 26}, std::vector<double>{1.0, 1.0, 1.0, 2.0, 2.0, 2.0},
                Cell::cubic(10.0));
    atoms.set_string_array("labels", {"?", "."});
    const std::string text = write_cif(atoms);
    FA_CHECK(text.find("'?'") != std::string::npos);
    FA_CHECK(text.find("'.'") != std::string::npos);
    const Atoms again = read_cif(text).at(0);
    FA_CHECK(again.labels() == (std::vector<std::string>{"?", "."}));
}

FA_TEST(cif_deuterium_and_tritium_masses) {
    // tests/data/cif/deuterium.cif, read by ASE: both images of each D site
    // get 2.01355 (ase.io.cif), the H sites keep 1.008. A D2+ label counts as D.
    const Atoms atoms = read_cif(read_text("deuterium.cif")).at(0);
    FA_CHECK(atoms.size() == 6);
    FA_CHECK(atoms.has_masses());
    const std::vector<double> masses = atoms.masses();
    const std::vector<std::string> labels = atoms.labels();
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        FA_CHECK_CLOSE(masses[i], labels[i][0] == 'D' ? 2.01355 : 1.008, 1e-12);
    }

    const Atoms tritium =
        read_cif(cubic_block("", kSiteHeader, "T1 T 0.5 0.5 0.5\nH1 H 0.25 0.25 0.25\n")).at(0);
    FA_CHECK(tritium.size() == 2);
    FA_CHECK_CLOSE(tritium.masses().at(0), 3.0160492, 1e-12);
    FA_CHECK_CLOSE(tritium.masses().at(1), 1.008, 1e-12);

    // Without isotope sites no per-atom masses are stored.
    FA_CHECK(!read_cif(read_text("nacl.cif")).at(0).has_masses());
}

FA_TEST_MAIN()
