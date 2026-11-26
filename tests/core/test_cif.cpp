// SPDX-License-Identifier: MIT
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "fastatoms/formats/cif.hpp"
#include "fastatoms/formats/format.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::formats;

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

FA_TEST(cif_deterministic) {
    const auto first = read_cif(read_text("rutile.cif")).at(0);
    const auto second = read_cif(read_text("rutile.cif")).at(0);
    FA_CHECK(first.labels() == second.labels());
    FA_CHECK(first.positions() == second.positions());
}

FA_TEST_MAIN()
