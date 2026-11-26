// SPDX-License-Identifier: MIT
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "fastatoms/core/cell.hpp"
#include "fastatoms/formats/format.hpp"
#include "fastatoms/formats/lammps.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::formats;

namespace {

std::string read_fixture(const std::string& name) {
    std::ifstream in(std::string(FASTATOMS_TEST_DATA) + "/lammps/" + name, std::ios::binary);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void check_positions(const Atoms& atoms, const double* expected, double tol = 1e-9) {
    const auto& p = atoms.positions();
    for (std::size_t j = 0; j < p.size(); ++j) {
        FA_CHECK_CLOSE(p[j], expected[j], tol);
    }
}

void check_cell(const Atoms& atoms, const double* expected, double tol = 1e-9) {
    const Mat3& m = atoms.cell().matrix();
    for (std::size_t j = 0; j < 9; ++j) {
        FA_CHECK_CLOSE(m.m[j], expected[j], tol);
    }
}

// ASE 3.29 reference values, read from tests/data/lammps/*.data with
// ase.io.read(format='lammps-data'). ASE keeps the box origin in celldisp
// and does not subtract it, so the values below are positions - celldisp,
// which is the origin-relative convention used here.
constexpr double kOrthoPos[] = {3.750572799628002, 6.280496606787027, 6.205485521961548, 1.3512431399435512, 2.1011639943785783, 6.988427563170095, 0.031591827393448346, 5.748598928679364, 6.37655543001637, 2.8076097170623244, 2.121226987735194, 2.2274048968061866, 1.5292175259247476, 3.115534141178526, 4.036386071663626, 3.320984112446955, 6.968501984040749, 6.3412953537100245};
constexpr double kFullPos[] = {4.448364413188994, 4.553230880461008, 1.3133830592371543, 1.566867099989828, 2.9006422933453027, 0.26804624856443865, 5.422612195921328, 2.9518809145098177, -3.2561432455157346, 5.700792723973131, 2.5575801396172877, 3.1361176442570353, 0.18964090128007305, 10.777457002742931, 0.0719435558092858, 1.9328739717756562, 3.1412081286445326, 1.2237010163206707};
constexpr double kFullVel[] = {-0.7932166646886639, -0.03194507816803312, 0.8687091741315129, -0.5732528929253005, -0.10972141580416005, 0.10850555642088211, 0.06265089015741884, -1.2033349481730613, 0.07479022441166906, 1.334730774318351, -1.5197129966852554, 0.8441453850925993, 0.11723781661920145, -0.6300967897150993, 1.9649481184515003, 0.7487444501350616, -1.1780248848571766, 0.07319501720094497};
constexpr double kFullCharges[] = {0.05766895836701853, -0.018878212535074934, 0.0682910267195206, -0.006651732014941557, 0.0667247560834328, 0.1438522591656152};
constexpr double kFullCell[] = {5.0, 0.0, 0.0, 1.2, 4.8, 0.0, 0.7, -0.9000000000000001, 6.099999999999999};
constexpr double kGenPos[] = {1.0, -0.5, -0.25, 6.5, 1.5, 2.75, 3.5, -3.5000000000000004, 9.75};
constexpr double kGenCell[] = {4.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.5, 1.0, 6.0};
constexpr double kTypesPos[] = {0.5, 0.5, 0.5, 2.0, 2.5, 3.0, 4.0, 1.0, 5.5};
constexpr double kScaledFrame0[] = {0.0, 3.0, 1.0, 3.0, 0.0, 3.0, 1.5, 4.5, 2.0, 4.5, 1.5, 0.5};
constexpr double kScaledFrame1[] = {0.6000000000000001, 3.0, 1.0, 3.0, 0.6000000000000001, 3.0, 1.5, 4.5, 2.4, 4.5, 1.5, 0.5};
constexpr double kUnsortedPos[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 1.0, 2.0};
constexpr double kElemPos[] = {0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0};

/// The message of the FormatError that `f` throws, or "" when it throws none.
template <typename F>
std::string format_error_of(F&& f) {
    try {
        f();
    } catch (const FormatError& error) {
        return error.what();
    }
    return {};
}

std::string with_bom(const std::string& text) { return std::string("\xEF\xBB\xBF") + text; }

/// A one-atom dump with the given box lines, ATOMS header and row.
std::string one_atom_dump(const std::string& box, const std::string& columns,
                          const std::string& row) {
    return "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n" + box + "\nITEM: ATOMS " + columns +
           "\n" + row + "\n";
}

constexpr double kCommentPos[] = {0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5};
constexpr double kCommentVel[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
constexpr double kRotatedPos[] = {1.5, 4.0, 2.0, 1.5, 1.75, -5.5};
constexpr double kRotatedCell[] = {3.0, 1.0, 0.0, -1.0, 4.0, 0.5, 0.5, 0.5, 5.0};

}  // namespace

FA_TEST(orthogonal_atomic_matches_ase) {
    const Atoms atoms = read_lammps_data(read_fixture("orthogonal_atomic.data"));
    FA_CHECK(atoms.size() == 6);
    const double cell[] = {6.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 8.0};
    check_cell(atoms, cell);
    FA_CHECK(atoms.numbers() == std::vector<int>({26, 26, 8, 8, 26, 8}));
    check_positions(atoms, kOrthoPos);
    FA_CHECK(atoms.labels() == std::vector<std::string>({"Fe", "Fe", "O", "O", "Fe", "O"}));
    FA_CHECK(atoms.int_array("type").data == std::vector<std::int64_t>({1, 1, 2, 2, 1, 2}));
    FA_CHECK(atoms.int_array("id").data == std::vector<std::int64_t>({1, 2, 3, 4, 5, 6}));
    FA_CHECK(atoms.pbc()[0] && atoms.pbc()[1] && atoms.pbc()[2]);
    FA_CHECK(!atoms.has_charges());
    FA_CHECK(atoms.info()["comment"].as_string() == "(written by ASE)");
}

FA_TEST(triclinic_full_matches_ase_with_image_flags) {
    LammpsDataReadOptions options;
    options.unwrap = true;
    const Atoms atoms = read_lammps_data(read_fixture("triclinic_full.data"), options);
    FA_CHECK(atoms.size() == 6);
    check_cell(atoms, kFullCell);
    check_positions(atoms, kFullPos);
    FA_CHECK(atoms.numbers() == std::vector<int>({26, 8, 26, 8, 8, 26}));
    // Masses come from the file, so mass(i) matches the Masses section.
    FA_CHECK_CLOSE(atoms.mass(0), 55.845, 1e-6);
    FA_CHECK_CLOSE(atoms.mass(1), 15.999, 1e-6);
    // Charges and velocities (native units) match ASE.
    for (std::size_t i = 0; i < 6; ++i) {
        FA_CHECK_CLOSE(atoms.charges()[i], kFullCharges[i], 1e-12);
    }
    const auto velocities = atoms.velocities();
    for (std::size_t j = 0; j < velocities.size(); ++j) {
        FA_CHECK_CLOSE(velocities[j], kFullVel[j], 1e-12);
    }
    FA_CHECK(atoms.int_array("mol").data == std::vector<std::int64_t>({1, 1, 2, 2, 3, 3}));
    const auto& image = atoms.int_array("image").data;
    FA_CHECK(image.size() == 18);
    FA_CHECK(image[6] == 1 && image[7] == 0 && image[8] == -1);   // atom 3
    FA_CHECK(image[12] == -1 && image[13] == 2 && image[14] == 0);  // atom 5
    FA_CHECK(atoms.labels()[0] == "Fe");
    FA_CHECK(atoms.labels()[1] == "O");
}

FA_TEST(image_flags_are_kept_without_unwrap) {
    const Atoms atoms = read_lammps_data(read_fixture("triclinic_full.data"));
    // Atom 3 keeps the x written in the file: no lattice vector is added.
    FA_CHECK_CLOSE(atoms.position(2).x, 1.1226121959213273, 1e-12);
    FA_CHECK(atoms.int_array("image").data[6] == 1);
}

FA_TEST(mass_only_infers_elements) {
    // Same file without the Masses comments: the element comes from the mass.
    const Atoms atoms = read_lammps_data(read_fixture("mass_only.data"));
    FA_CHECK(atoms.numbers() == std::vector<int>({26, 8, 26, 8, 8, 26}));
    FA_CHECK(atoms.labels() == std::vector<std::string>({"Fe", "O", "Fe", "O", "O", "Fe"}));
    const auto t = atoms.int_array("type").data;
    FA_CHECK(t == std::vector<std::int64_t>({1, 2, 1, 2, 2, 1}));
}

FA_TEST(mass_inference_without_match_throws) {
    const std::string text =
        "title\n\n1 atoms\n1 atom types\n\n0 5 xlo xhi\n0 5 ylo yhi\n0 5 zlo zhi\n\n"
        "Masses\n\n1 100.0\n\nAtoms # atomic\n\n1 1 0 0 0\n";
    FA_CHECK_THROWS(read_lammps_data(text), FormatError);
    const std::string no_masses =
        "title\n\n1 atoms\n1 atom types\n\n0 5 xlo xhi\n0 5 ylo yhi\n0 5 zlo zhi\n\n"
        "Atoms # atomic\n\n1 1 0 0 0\n";
    FA_CHECK_THROWS(read_lammps_data(no_masses), FormatError);
}

FA_TEST(types_map_gives_both_types_iron) {
    LammpsDataReadOptions options;
    options.types = {{1, "Fe1"}, {2, "Fe2"}};
    const Atoms atoms = read_lammps_data(read_fixture("types_map.data"), options);
    FA_CHECK(atoms.numbers() == std::vector<int>({26, 26, 26}));
    FA_CHECK(atoms.labels() == std::vector<std::string>({"Fe1", "Fe2", "Fe2"}));
    FA_CHECK(atoms.int_array("type").data == std::vector<std::int64_t>({1, 2, 2}));
    check_positions(atoms, kTypesPos);
}

FA_TEST(general_triclinic_matches_ase) {
    LammpsDataReadOptions options;
    options.unwrap = true;
    const Atoms atoms = read_lammps_data(read_fixture("general_triclinic.data"), options);
    check_cell(atoms, kGenCell);
    check_positions(atoms, kGenPos);
    FA_CHECK(atoms.numbers() == std::vector<int>({26, 8, 26}));
}

FA_TEST(data_positions_without_unwrap_are_origin_relative) {
    const Atoms atoms = read_lammps_data(read_fixture("general_triclinic.data"));
    // Atom 1 is at the origin of the box, so the origin-relative position is 0
    // only after the abc origin (-1, 0.5, 0.25) is subtracted.
    FA_CHECK_CLOSE(atoms.position(0).x, 1.0, 1e-12);
    FA_CHECK_CLOSE(atoms.position(0).y, -0.5, 1e-12);
    FA_CHECK_CLOSE(atoms.position(0).z, -0.25, 1e-12);
}

FA_TEST(data_writer_rejects_bad_style_and_missing_label) {
    Atoms atoms(std::vector<int>{26}, {0.0, 0.0, 0.0}, Cell::cubic(4.0));
    LammpsDataWriteOptions bad;
    bad.atom_style = "sphere";
    FA_CHECK_THROWS(write_lammps_data(atoms, bad), FormatError);
    LammpsDataWriteOptions spec;
    spec.specorder = {"O"};
    FA_CHECK_THROWS(write_lammps_data(atoms, spec), FormatError);
}

FA_TEST(data_writer_cubic_cell_has_no_tilt_line) {
    Atoms atoms(std::vector<int>{26, 8}, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}, Cell::cubic(4.0));
    const std::string text = write_lammps_data(atoms);
    FA_CHECK(text.find("xy xz yz") == std::string::npos);
    FA_CHECK(text.find("Masses") != std::string::npos);
    FA_CHECK(text.find("# Fe") != std::string::npos);
    LammpsDataWriteOptions forced;
    forced.force_skew = true;
    FA_CHECK(write_lammps_data(atoms, forced).find("0 0 0 xy xz yz") != std::string::npos);
}

FA_TEST(data_round_trip_triclinic_cell) {
    // An arbitrary (not LAMMPS-oriented) triclinic cell with mixed species.
    const Cell cell = Cell::from_parameters(4.2, 5.1, 6.3, 80.0, 95.0, 110.0);
    Atoms atoms(std::vector<int>{26, 8, 8, 26},
                {0.3, 0.2, 0.1, 2.1, 1.7, 2.9, 3.5, 4.0, 1.2, 1.0, 2.5, 5.0}, cell);
    atoms.set_velocities({0.01, -0.02, 0.03, 0.0, 0.1, -0.1, 0.2, 0.0, 0.05, -0.3, 0.1, 0.0});

    for (const bool full : {false, true}) {
        LammpsDataWriteOptions options;
        options.atom_style = full ? "full" : "atomic";
        const std::string text = write_lammps_data(atoms, options);
        FA_CHECK(text.find("xy xz yz") != std::string::npos);  // triclinic box

        const Atoms back = read_lammps_data(text);
        FA_CHECK(back.size() == atoms.size());
        const Vec3 l0 = cell.lengths();
        const Vec3 l1 = back.cell().lengths();
        const Vec3 a0 = cell.angles();
        const Vec3 a1 = back.cell().angles();
        for (std::size_t k = 0; k < 3; ++k) {
            FA_CHECK_CLOSE(l1[k], l0[k], 1e-9);
            FA_CHECK_CLOSE(a1[k], a0[k], 1e-9);
        }
        const auto s0 = atoms.scaled_positions();
        const auto s1 = back.scaled_positions();
        for (std::size_t j = 0; j < s0.size(); ++j) {
            FA_CHECK_CLOSE(s1[j], s0[j], 1e-9);
        }
        FA_CHECK(back.numbers() == atoms.numbers());
        const auto v0 = atoms.velocities();
        const auto v1 = back.velocities();
        for (std::size_t j = 0; j < v0.size(); ++j) {
            FA_CHECK_CLOSE(v1[j], v0[j], 1e-9);
        }
    }
}

FA_TEST(data_species_order_from_specorder) {
    Atoms atoms(std::vector<int>{26, 8}, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}, Cell::cubic(4.0));
    LammpsDataWriteOptions options;
    options.specorder = {"O", "Fe"};
    const Atoms back = read_lammps_data(write_lammps_data(atoms, options));
    FA_CHECK(back.int_array("type").data == std::vector<std::int64_t>({2, 1}));
    FA_CHECK(back.numbers() == std::vector<int>({26, 8}));
}

FA_TEST(dump_multi_frame_scaled_coordinates) {
    const auto frames = read_lammps_dump(read_fixture("orthogonal_scaled.lammpstrj"));
    FA_CHECK(frames.size() == 2);
    FA_CHECK(frames[0].info()["timestep"].as_int() == 0);
    FA_CHECK(frames[1].info()["timestep"].as_int() == 100);
    const double cell[] = {6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 4.0};
    check_cell(frames[0], cell);
    check_cell(frames[1], cell);
    check_positions(frames[0], kScaledFrame0);
    check_positions(frames[1], kScaledFrame1);
    // No types map and no element column: dummy X with the type as label.
    FA_CHECK(frames[0].numbers() == std::vector<int>({0, 0, 0, 0}));
    FA_CHECK(frames[0].labels() == std::vector<std::string>({"1", "2", "1", "2"}));
    FA_CHECK(frames[0].pbc()[0] && frames[0].pbc()[1] && frames[0].pbc()[2]);
}

FA_TEST(dump_triclinic_with_image_flags) {
    const std::string text = read_fixture("triclinic_image.lammpstrj");
    const double cell[] = {6.0, 0.0, 0.0, 1.5, 5.0, 0.0, -2.0, 1.0, 7.0};

    // Unwrapped (default): x + ix*a + iy*b + iz*c.
    const auto unwrapped = read_lammps_dump(text);
    FA_CHECK(unwrapped.size() == 1);
    check_cell(unwrapped[0], cell);
    const double u[] = {1.25, 3.0, 3.5, 3.15, 1.4, 6.3, 2.45, 3.7, 1.4};
    check_positions(unwrapped[0], u);

    // Wrapped: the x column as written. ASE reads these values too.
    LammpsDumpReadOptions no_unwrap;
    no_unwrap.unwrap = false;
    const auto wrapped = read_lammps_dump(text, no_unwrap);
    const double w[] = {1.25, 3.0, 3.5, -4.85, 2.4, 13.3, 3.95, 8.7, 1.4};
    check_positions(wrapped[0], w);
    FA_CHECK(wrapped[0].int_array("image").data == std::vector<std::int64_t>({0, 0, 0, 1, 0, -1, 0, -1, 0}));
    FA_CHECK(wrapped[0].info()["timestep"].as_int() == 7);
}

FA_TEST(dump_element_column_overrides_types) {
    LammpsDumpReadOptions options;
    options.types = {{1, "O"}, {2, "O"}};  // the element column must win
    const auto frames = read_lammps_dump(read_fixture("element_column.lammpstrj"), options);
    FA_CHECK(frames.size() == 1);
    const Atoms& atoms = frames[0];
    FA_CHECK(atoms.numbers() == std::vector<int>({26, 8, 8}));
    FA_CHECK(atoms.labels() == std::vector<std::string>({"Fe", "O", "O"}));
    FA_CHECK(atoms.pbc()[0] && atoms.pbc()[1] && !atoms.pbc()[2]);
    // Rows are sorted by id: id 1 is the Fe atom at 0.5.
    check_positions(atoms, kElemPos);
    const auto charges = atoms.charges();
    FA_CHECK_CLOSE(charges[0], -0.25, 1e-12);
    FA_CHECK_CLOSE(charges[1], 0.25, 1e-12);
    FA_CHECK_CLOSE(charges[2], 0.0, 1e-12);
    FA_CHECK(atoms.has_array("c_pe"));
    FA_CHECK_CLOSE(atoms.array("c_pe").data[0], -3.0, 1e-12);
    FA_CHECK_CLOSE(atoms.array("c_pe").data[2], -2.0, 1e-12);
    FA_CHECK(atoms.has_array("forces"));
    FA_CHECK_CLOSE(atoms.array("forces").data[3 * 1 + 2], 0.3, 1e-12);
    FA_CHECK_CLOSE(atoms.array("forces").data[3 * 2 + 1], 2.0, 1e-12);
}

FA_TEST(dump_unsorted_ids_without_types_is_dummy) {
    const auto frames = read_lammps_dump(read_fixture("unsorted_ids.lammpstrj"));
    FA_CHECK(frames.size() == 1);
    const Atoms& atoms = frames[0];
    FA_CHECK(atoms.int_array("id").data == std::vector<std::int64_t>({1, 2, 3}));
    check_positions(atoms, kUnsortedPos);
    // No types map and no element column: type k is the dummy X with label "k".
    FA_CHECK(atoms.numbers() == std::vector<int>({0, 0, 0}));
    FA_CHECK(atoms.labels() == std::vector<std::string>({"1", "2", "2"}));
}

FA_TEST(dump_errors) {
    const std::string no_coords =
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\n"
        "ITEM: ATOMS id type q\n1 1 0.5\n";
    FA_CHECK_THROWS(read_lammps_dump(no_coords), FormatError);
    FA_CHECK_THROWS(read_lammps_data("t\n1 atoms\n0 5 xlo xhi\n0 5 ylo yhi\n0 5 zlo zhi\n"
                                     "Masses\n\n1 55.845\n\nAtoms # bogus\n\n1 1 0 0 0\n"),
                    FormatError);
}

FA_TEST(data_writer_round_trip_then_read_text) {
    // Reading what the writer produced for a cubic full-style box keeps the
    // charges and molecule ids.
    Atoms atoms(std::vector<int>{26, 8}, {0.5, 0.5, 0.5, 1.0, 1.0, 1.0}, Cell::cubic(3.0));
    atoms.set_charges({0.75, -0.75});
    atoms.set_int_array("mol", {4, 7});
    LammpsDataWriteOptions options;
    options.atom_style = "full";
    const Atoms back = read_lammps_data(write_lammps_data(atoms, options));
    FA_CHECK(back.int_array("mol").data == std::vector<std::int64_t>({4, 7}));
    FA_CHECK_CLOSE(back.charges()[0], 0.75, 1e-12);
    FA_CHECK_CLOSE(back.charges()[1], -0.75, 1e-12);
}

FA_TEST(atom_styles_map_their_columns) {
    LammpsDataReadOptions options;
    options.types = {{1, "Fe"}, {2, "O"}};
    options.unwrap = true;
    const std::string header = "title\n\n2 atoms\n2 atom types\n\n"
                               "0 5 xlo xhi\n0 5 ylo yhi\n0 5 zlo zhi\n\n";

    // charge: id type q x y z [ix iy iz]
    const Atoms charge = read_lammps_data(
        header + "Atoms # charge\n\n1 1 0.5 1.0 2.0 3.0\n2 2 -0.5 0.5 0.5 0.5 1 0 -1\n", options);
    FA_CHECK(charge.numbers() == std::vector<int>({26, 8}));
    FA_CHECK_CLOSE(charge.charges()[0], 0.5, 1e-12);
    FA_CHECK_CLOSE(charge.charges()[1], -0.5, 1e-12);
    FA_CHECK_CLOSE(charge.position(1).x, 5.5, 1e-12);
    FA_CHECK_CLOSE(charge.position(1).z, -4.5, 1e-12);

    // molecular and bond share the layout: id mol type x y z.
    for (const char* style : {"molecular", "bond"}) {
        const Atoms mol = read_lammps_data(
            header + std::string("Atoms # ") + style + "\n\n1 7 1 1.0 2.0 3.0\n2 7 2 0 0 0\n", options);
        FA_CHECK(mol.int_array("mol").data == std::vector<std::int64_t>({7, 7}));
        FA_CHECK(mol.int_array("type").data == std::vector<std::int64_t>({1, 2}));
        FA_CHECK_CLOSE(mol.position(0).z, 3.0, 1e-12);
        FA_CHECK(!mol.has_charges() || mol.charges()[0] == 0.0);
    }

    // sphere: id type diameter density x y z
    const std::string one_atom = "title\n\n1 atoms\n1 atom types\n\n"
                                 "0 5 xlo xhi\n0 5 ylo yhi\n0 5 zlo zhi\n\n";
    const Atoms sphere = read_lammps_data(
        one_atom + "Atoms # sphere\n\n1 1 0.5 2.0 1.0 2.0 3.0\n", options);
    FA_CHECK(sphere.size() == 1);
    FA_CHECK_CLOSE(sphere.position(0).x, 1.0, 1e-12);
    FA_CHECK_CLOSE(sphere.position(0).y, 2.0, 1e-12);
    FA_CHECK_CLOSE(sphere.position(0).z, 3.0, 1e-12);
}

FA_TEST(dump_velocities_native_units) {
    const std::string text =
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n0 5\n0 5\n0 5\n"
        "ITEM: ATOMS id type x y z vx vy vz\n2 2 1 1 1 0.5 -0.5 0.25\n1 1 0 0 0 1 2 3\n";
    // Velocities are stored verbatim, so a dummy atom (no mass) keeps them too.
    const Atoms dummy = read_lammps_dump(text).at(0);
    FA_CHECK(dummy.has_momenta());
    FA_CHECK(!dummy.has_array("velocity"));
    const auto d = dummy.velocities();
    FA_CHECK_CLOSE(d[0], 1.0, 1e-12);
    FA_CHECK_CLOSE(d[5], 0.25, 1e-12);

    LammpsDumpReadOptions options;
    options.types = {{1, "Fe"}, {2, "O"}};
    const Atoms typed = read_lammps_dump(text, options).at(0);
    const auto v = typed.velocities();
    FA_CHECK_CLOSE(v[0], 1.0, 1e-12);
    FA_CHECK_CLOSE(v[4], -0.5, 1e-12);
    FA_CHECK_CLOSE(v[5], 0.25, 1e-12);
}

FA_TEST(masses_comment_must_agree_with_mass) {
    // "Carbon" is not an element label; "Ca" is an element but its mass
    // disagrees with 12.011, so both types fall back to the mass.
    const Atoms atoms = read_lammps_data(read_fixture("masses_labels.data"));
    FA_CHECK(atoms.numbers() == std::vector<int>({6, 11, 26, 6}));
    FA_CHECK(atoms.labels() == std::vector<std::string>({"C", "Na", "Fe1", "C"}));
}

FA_TEST(masses_comment_element_with_matching_mass_keeps_label) {
    const std::string text =
        "t\n\n1 atoms\n1 atom types\n\n0 5 xlo xhi\n0 5 ylo yhi\n0 5 zlo zhi\n\n"
        "Masses\n\n1 55.845 # Fe1\n\nAtoms # atomic\n\n1 1 0 0 0\n";
    const Atoms atoms = read_lammps_data(text);
    FA_CHECK(atoms.numbers() == std::vector<int>({26}));
    FA_CHECK(atoms.labels() == std::vector<std::string>({"Fe1"}));
}

FA_TEST(comments_throughout_data_file) {
    const Atoms atoms = read_lammps_data(read_fixture("comments_everywhere.data"));
    FA_CHECK(atoms.size() == 3);
    FA_CHECK(atoms.numbers() == std::vector<int>({1, 8, 1}));
    FA_CHECK(atoms.labels() == std::vector<std::string>({"H", "O", "H"}));
    check_positions(atoms, kCommentPos);
    const auto v = atoms.velocities();
    for (std::size_t j = 0; j < v.size(); ++j) {
        FA_CHECK_CLOSE(v[j], kCommentVel[j], 1e-12);
    }
}

FA_TEST(non_contiguous_ids_are_sorted_and_kept) {
    const Atoms atoms = read_lammps_data(read_fixture("noncontiguous_ids.data"));
    FA_CHECK(atoms.int_array("id").data == std::vector<std::int64_t>({3, 7, 10}));
    const double pos[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 9.0, 9.0};
    check_positions(atoms, pos);
    const double vel[] = {2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0};
    const auto v = atoms.velocities();
    for (std::size_t j = 0; j < v.size(); ++j) {
        FA_CHECK_CLOSE(v[j], vel[j], 1e-12);
    }
}

FA_TEST(atom_type_labels_section) {
    const Atoms atoms = read_lammps_data(read_fixture("type_labels.data"));
    FA_CHECK(atoms.numbers() == std::vector<int>({26, 8, 11}));
    FA_CHECK(atoms.labels() == std::vector<std::string>({"Fe", "O", "Sodium"}));
    FA_CHECK(atoms.int_array("type").data == std::vector<std::int64_t>({1, 2, 3}));
    FA_CHECK_CLOSE(atoms.mass(2), 22.99, 1e-9);
}

FA_TEST(dump_type_labels_as_species) {
    const auto frames = read_lammps_dump(read_fixture("type_labels.lammpstrj"));
    FA_CHECK(frames.size() == 1);
    FA_CHECK(frames[0].numbers() == std::vector<int>({26, 8, 26}));
    FA_CHECK(frames[0].labels() == std::vector<std::string>({"Fe", "O", "Fe2"}));
    FA_CHECK(!frames[0].has_int_array("type"));
}

FA_TEST(nohint_atomic_and_charge_are_guessed_from_fields) {
    const Atoms atomic = read_lammps_data(read_fixture("nohint_atomic.data"));
    FA_CHECK(atomic.numbers() == std::vector<int>({26, 8}));
    FA_CHECK(!atomic.has_charges());
    FA_CHECK_CLOSE(atomic.position(1).x, 1.0, 1e-12);
    FA_CHECK(atomic.int_array("image").data == std::vector<std::int64_t>({0, 0, 0, 0, 0, 1}));

    const Atoms charge = read_lammps_data(read_fixture("nohint_charge.data"));
    FA_CHECK(charge.numbers() == std::vector<int>({26, 8}));
    FA_CHECK_CLOSE(charge.charges()[0], 0.5, 1e-12);
    FA_CHECK_CLOSE(charge.charges()[1], -0.5, 1e-12);
    FA_CHECK_CLOSE(charge.position(1).z, 1.0, 1e-12);
}

FA_TEST(general_triclinic_rotated_frame_maps_positions) {
    // Hand calculation: origin (0.5, -1, 2); atom 1 at (2, 3, 4) gives
    // (1.5, 4, 2). Atom 2 at (-0.5, 0.25, 1.5) with image (1, 0, -1) gets
    // avec - cvec = (2.5, 0.5, -5) before the origin is removed, which gives
    // (1.5, 1.75, -5.5). ASE 3.29 agrees with both values.
    const Atoms raw = read_lammps_data(read_fixture("general_rotated.data"));
    check_cell(raw, kRotatedCell);
    const double origin_relative[] = {1.5, 4.0, 2.0, -1.0, 1.25, -0.5};
    check_positions(raw, origin_relative);

    LammpsDataReadOptions options;
    options.unwrap = true;
    const Atoms unwrapped = read_lammps_data(read_fixture("general_rotated.data"), options);
    check_positions(unwrapped, kRotatedPos);

    // Written in the restricted frame and read back: same lengths, angles
    // and fractional positions.
    const Atoms back = read_lammps_data(write_lammps_data(unwrapped));
    const Vec3 l0 = unwrapped.cell().lengths();
    const Vec3 l1 = back.cell().lengths();
    const Vec3 a0 = unwrapped.cell().angles();
    const Vec3 a1 = back.cell().angles();
    for (std::size_t k = 0; k < 3; ++k) {
        FA_CHECK_CLOSE(l1[k], l0[k], 1e-9);
        FA_CHECK_CLOSE(a1[k], a0[k], 1e-9);
    }
    const auto s0 = unwrapped.scaled_positions();
    const auto s1 = back.scaled_positions();
    for (std::size_t j = 0; j < s0.size(); ++j) {
        FA_CHECK_CLOSE(s1[j], s0[j], 1e-9);
    }
}

FA_TEST(data_comments_and_bom_are_ignored) {
    const Atoms atoms = read_lammps_data(with_bom(read_fixture("nohint_charge.data")));
    FA_CHECK(atoms.info()["comment"].as_string() == "no Atoms style hint, charge rows");
    FA_CHECK(atoms.numbers() == std::vector<int>({26, 8}));
}

FA_TEST(data_rejects_bad_counts_and_ids) {
    const std::string head = "t\n\n";
    const std::string box = "0 5 xlo xhi\n0 5 ylo yhi\n0 5 zlo zhi\n\n";
    const std::string masses = "Masses\n\n1 55.845\n\n";
    // Huge and negative atom counts are FormatErrors, not allocation failures.
    FA_CHECK(!format_error_of([&] {
                  (void)read_lammps_data(head + "99999999999999 atoms\n" + box + masses +
                                   "Atoms # atomic\n\n1 1 0 0 0\n");
              }).empty());
    FA_CHECK(!format_error_of([&] {
                  (void)read_lammps_data(head + "-3 atoms\n" + box + masses + "Atoms # atomic\n\n");
              }).empty());
    // Duplicate ids, a type above 'atom types', and type 0.
    FA_CHECK(!format_error_of([&] {
                  (void)read_lammps_data(head + "2 atoms\n1 atom types\n" + box + masses +
                                   "Atoms # atomic\n\n1 1 0 0 0\n1 1 1 1 1\n");
              }).empty());
    FA_CHECK(!format_error_of([&] {
                  (void)read_lammps_data(head + "1 atoms\n1 atom types\n" + box + masses +
                                   "Atoms # atomic\n\n1 2 0 0 0\n");
              }).empty());
    FA_CHECK(!format_error_of([&] {
                  (void)read_lammps_data(head + "1 atoms\n1 atom types\n" + box + masses +
                                   "Atoms # atomic\n\n1 0 0 0 0\n");
              }).empty());
}

FA_TEST(writer_full_style_round_trips_charge_velocity_and_mol) {
    Atoms atoms(std::vector<int>{26, 8}, {0.5, 0.5, 0.5, 1.0, 1.0, 1.0}, Cell::cubic(3.0));
    atoms.set_charges({0.75, -0.75});
    atoms.set_int_array("mol", {4, 7});
    atoms.set_velocities({0.1, 0.2, 0.3, -0.4, 0.5, -0.6});
    LammpsDataWriteOptions options;
    options.atom_style = "full";
    const std::string text = write_lammps_data(atoms, options);
    FA_CHECK(text.find("Atoms # full") != std::string::npos);
    FA_CHECK(text.find("Velocities") != std::string::npos);
    const Atoms back = read_lammps_data(text);
    FA_CHECK(back.int_array("mol").data == std::vector<std::int64_t>({4, 7}));
    FA_CHECK_CLOSE(back.charges()[0], 0.75, 1e-12);
    FA_CHECK_CLOSE(back.charges()[1], -0.75, 1e-12);
    const auto v = back.velocities();
    const double expected[] = {0.1, 0.2, 0.3, -0.4, 0.5, -0.6};
    for (std::size_t j = 0; j < 6; ++j) {
        FA_CHECK_CLOSE(v[j], expected[j], 1e-12);
    }
}

FA_TEST(writer_masses_comment_is_the_label_and_type_array_is_used) {
    Atoms atoms(std::vector<int>{26, 26, 8}, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0},
                Cell::cubic(4.0));
    atoms.set_string_array("labels", {"Fe1", "Fe2", "O"});
    atoms.set_int_array("type", {1, 2, 3});
    const std::string text = write_lammps_data(atoms);
    FA_CHECK(text.find("# Fe1") != std::string::npos);
    FA_CHECK(text.find("# Fe2") != std::string::npos);
    FA_CHECK(text.find("# O") != std::string::npos);
    FA_CHECK(text.find("3 atom types") != std::string::npos);

    const Atoms back = read_lammps_data(text);
    FA_CHECK(back.int_array("type").data == std::vector<std::int64_t>({1, 2, 3}));
    FA_CHECK(back.labels() == std::vector<std::string>({"Fe1", "Fe2", "O"}));
    FA_CHECK(back.numbers() == std::vector<int>({26, 26, 8}));
}

FA_TEST(writer_specorder_falls_back_to_symbols) {
    Atoms atoms(std::vector<int>{26, 26, 8}, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0},
                Cell::cubic(4.0));
    atoms.set_string_array("labels", {"Fe1", "Fe2", "O"});
    LammpsDataWriteOptions options;
    options.specorder = {"Fe", "O"};
    const Atoms back = read_lammps_data(write_lammps_data(atoms, options));
    FA_CHECK(back.int_array("type").data == std::vector<std::int64_t>({1, 1, 2}));
    FA_CHECK(back.numbers() == std::vector<int>({26, 26, 8}));
}

FA_TEST(writer_warns_on_left_handed_and_large_tilt) {
    const Atoms mirror(std::vector<int>{26}, {1.0, 1.0, 1.0},
                       Cell(Mat3::from_rows({5.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, -5.0}),
                            {true, true, true}));
    const std::string text = write_lammps_data(mirror);
    FA_CHECK(text.find("mirror") != std::string::npos);
    FA_CHECK(read_lammps_data(text).size() == 1);

    const Atoms skewed(std::vector<int>{26}, {1.0, 1.0, 1.0},
                       Cell(Mat3::from_rows({4.0, 0.0, 0.0}, {3.0, 4.0, 0.0}, {0.0, 0.0, 4.0}),
                            {true, true, true}));
    FA_CHECK(write_lammps_data(skewed).find("box tilt large") != std::string::npos);
    // A cubic cell has no note.
    FA_CHECK(write_lammps_data(Atoms(std::vector<int>{26}, {1.0, 1.0, 1.0}, Cell::cubic(4.0)))
                 .find("note") == std::string::npos);
}

FA_TEST(dump_xs_with_image_flags_unwraps) {
    // Box 10 x 10 x 10. xs = 0.5 with ix = 1 is x = 15; xu and xsu are already
    // unwrapped, so their image flags must not be added again.
    const std::string box = "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10";
    const auto xs = read_lammps_dump(one_atom_dump(box, "id type xs ys zs ix iy iz",
                                                   "1 1 0.5 0.25 0.75 1 0 -1"));
    const double unwrapped[] = {15.0, 2.5, -2.5};
    check_positions(xs[0], unwrapped);
    LammpsDumpReadOptions wrapped;
    wrapped.unwrap = false;
    const double raw[] = {5.0, 2.5, 7.5};
    check_positions(read_lammps_dump(one_atom_dump(box, "id type xs ys zs ix iy iz",
                                                   "1 1 0.5 0.25 0.75 1 0 -1"),
                                     wrapped)[0],
                    raw);

    const auto xu = read_lammps_dump(
        one_atom_dump(box, "id type xu yu zu ix iy iz", "1 1 15 2.5 -2.5 1 0 -1"));
    check_positions(xu[0], unwrapped);
    const auto xsu = read_lammps_dump(
        one_atom_dump(box, "id type xsu ysu zsu ix iy iz", "1 1 1.5 0.25 -0.25 1 0 -1"));
    check_positions(xsu[0], unwrapped);
}

FA_TEST(dump_boundary_flags_give_pbc) {
    const std::string text =
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp fs\n0 5\n0 5\n0 5\n"
        "ITEM: ATOMS id type x y z\n1 1 1 1 1\n";
    const auto frames = read_lammps_dump(text);
    FA_CHECK(frames[0].pbc()[0] && frames[0].pbc()[1] && !frames[0].pbc()[2]);
}

FA_TEST(dump_negative_tilts) {
    // LAMMPS writes the enclosing bounds: xlo_bound = xlo + min(0, xy, xz, xy+xz)
    // = 0 - 3.5, and so on. Undoing that gives the box back.
    const std::string text =
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS xy xz yz pp pp pp\n-3.5 10 -1.5\n-0.5 10 -2\n0 10 -0.5\n"
        "ITEM: ATOMS id type x y z\n1 1 1 2 3\n";
    const auto frames = read_lammps_dump(text);
    const double cell[] = {10.0, 0.0, 0.0, -1.5, 10.0, 0.0, -2.0, -0.5, 10.0};
    check_cell(frames[0], cell);
    const double pos[] = {1.0, 2.0, 3.0};
    check_positions(frames[0], pos);
}

FA_TEST(dump_mol_q_cpe_and_forces_columns) {
    const std::string text =
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n0 5\n0 5\n0 5\n"
        "ITEM: ATOMS id mol type q x y z c_pe fx fy fz\n"
        "2 5 2 0.5 1 1 1 -1.5 0.1 0.2 0.3\n"
        "1 4 1 -0.5 0 0 0 -2.5 0 0 0\n";
    LammpsDumpReadOptions options;
    options.types = {{1, "Fe"}, {2, "O"}};
    const Atoms atoms = read_lammps_dump(text, options).at(0);
    FA_CHECK(atoms.int_array("mol").data == std::vector<std::int64_t>({4, 5}));
    FA_CHECK_CLOSE(atoms.charges()[0], -0.5, 1e-12);
    FA_CHECK_CLOSE(atoms.charges()[1], 0.5, 1e-12);
    FA_CHECK_CLOSE(atoms.array("c_pe").data[0], -2.5, 1e-12);
    FA_CHECK_CLOSE(atoms.array("forces").data[3], 0.1, 1e-12);
    FA_CHECK_CLOSE(atoms.array("forces").data[5], 0.3, 1e-12);
}

FA_TEST(dump_frames_with_different_atom_counts) {
    const std::string text =
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n3\nITEM: BOX BOUNDS pp pp pp\n0 5\n0 5\n0 5\n"
        "ITEM: ATOMS id type x y z\n1 1 0 0 0\n2 1 1 1 1\n3 1 2 2 2\n"
        "ITEM: TIMESTEP\n10\nITEM: NUMBER OF ATOMS\n5\nITEM: BOX BOUNDS pp pp pp\n0 5\n0 5\n0 5\n"
        "ITEM: ATOMS id type x y z\n1 1 0 0 0\n2 1 1 1 1\n3 1 2 2 2\n4 1 3 3 3\n5 1 4 4 4\n";
    const auto frames = read_lammps_dump(text);
    FA_CHECK(frames.size() == 2);
    FA_CHECK(frames[0].size() == 3);
    FA_CHECK(frames[1].size() == 5);
    FA_CHECK(frames[1].info()["timestep"].as_int() == 10);
}

FA_TEST(dump_empty_atoms_block) {
    const std::string text =
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n0\nITEM: BOX BOUNDS pp pp pp\n0 5\n0 5\n0 5\n"
        "ITEM: ATOMS id type x y z\n";
    const auto frames = read_lammps_dump(text);
    FA_CHECK(frames.size() == 1);
    FA_CHECK(frames[0].size() == 0);
}

FA_TEST(dump_truncated_last_frame_is_dropped) {
    const auto frames = read_lammps_dump(read_fixture("truncated.lammpstrj"));
    FA_CHECK(frames.size() == 1);
    FA_CHECK(frames[0].size() == 2);
    FA_CHECK(frames[0].info()["timestep"].as_int() == 0);
    FA_CHECK(frames[0].info()["truncated"].as_bool());

    // Through formats::read, index 0 picks the surviving frame.
    Value::Map options;
    options.emplace_back("index", Value(std::int64_t{0}));
    const Atoms first = fastatoms::formats::read(read_fixture("truncated.lammpstrj"), "lammps-dump",
                                                 Value(std::move(options)));
    FA_CHECK(first.size() == 2);
}

FA_TEST(dump_truncated_only_frame_is_error) {
    const std::string text =
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n3\nITEM: BOX BOUNDS pp pp pp\n0 5\n0 5\n0 5\n"
        "ITEM: ATOMS id type x y z\n1 1 0 0 0\n";
    FA_CHECK(!format_error_of([&] { (void)read_lammps_dump(text); }).empty());
}

FA_TEST(dump_errors_carry_line_numbers_and_reject_bad_input) {
    const std::string bad_number =
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n0 5\n0 5\n0 5\n"
        "ITEM: ATOMS id type x y z\n1 1 0 0 0\n2 2 1 oops 1\n";
    const std::string message = format_error_of([&] { (void)read_lammps_dump(bad_number); });
    FA_CHECK(message.find("line 11") != std::string::npos);

    const std::string negative =
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n-4\nITEM: BOX BOUNDS pp pp pp\n0 5\n0 5\n0 5\n"
        "ITEM: ATOMS id type x y z\n";
    FA_CHECK(format_error_of([&] { (void)read_lammps_dump(negative); }).find("line 4") !=
             std::string::npos);

    const std::string huge =
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n99999999999999\nITEM: BOX BOUNDS pp pp pp\n0 5\n"
        "0 5\n0 5\nITEM: ATOMS id type x y z\n1 1 0 0 0\n";
    FA_CHECK(!format_error_of([&] { (void)read_lammps_dump(huge); }).empty());

    const std::string no_box =
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\nITEM: ATOMS id type x y z\n1 1 0 0 0\n";
    FA_CHECK(!format_error_of([&] { (void)read_lammps_dump(no_box); }).empty());

    FA_CHECK(!format_error_of([&] { (void)read_lammps_dump("hello\nworld\n"); }).empty());
    FA_CHECK(!format_error_of([&] { (void)read_lammps_dump(""); }).empty());
}

FA_TEST(dump_strips_bom) {
    const auto frames = read_lammps_dump(with_bom(read_fixture("type_labels.lammpstrj")));
    FA_CHECK(frames.size() == 1);
    FA_CHECK(frames[0].labels() == std::vector<std::string>({"Fe", "O", "Fe2"}));
}

FA_TEST_MAIN()
