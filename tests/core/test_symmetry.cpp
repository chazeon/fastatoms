// SPDX-License-Identifier: MIT
// Reference numbers come from the Python spglib 2.x bindings, run on the same
// structures (see the scratchpad of the port).

#include <cmath>
#include <string>
#include <vector>

#include "fastatoms/symmetry/spacegroup.hpp"
#include "fastatoms/symmetry/standardize.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::symmetry;

namespace {

/// A crystal from a cell (rows are lattice vectors) and fractional sites.
Atoms crystal(const Mat3& cell, const std::vector<int>& z, const std::vector<Vec3>& frac) {
    std::vector<double> cart;
    for (const Vec3& f : frac) {
        const Vec3 r = f * cell;
        cart.insert(cart.end(), {r.x, r.y, r.z});
    }
    return Atoms(z, cart, Cell(cell, {true, true, true}));
}

Atoms copper(double a = 3.61) {
    return crystal(Mat3::diagonal(a, a, a), std::vector<int>(4, 29),
                   {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}});
}

Atoms rocksalt(bool chlorine_first = false) {
    const std::vector<Vec3> na{{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
    const std::vector<Vec3> cl{{0.5, 0, 0}, {0, 0.5, 0}, {0, 0, 0.5}, {0.5, 0.5, 0.5}};
    std::vector<Vec3> sites = chlorine_first ? cl : na;
    const std::vector<Vec3>& other = chlorine_first ? na : cl;
    sites.insert(sites.end(), other.begin(), other.end());
    std::vector<int> z(4, chlorine_first ? 17 : 11);
    z.insert(z.end(), 4, chlorine_first ? 11 : 17);
    return crystal(Mat3::diagonal(5.64, 5.64, 5.64), z, sites);
}

Atoms wurtzite() {
    const double a = 3.25;
    const double c = 5.20;
    const double u = 0.382;
    const Mat3 cell = Mat3::from_rows({a, 0, 0}, {-a / 2, a * std::sqrt(3.0) / 2, 0}, {0, 0, c});
    return crystal(cell, {30, 30, 8, 8},
                   {{1.0 / 3, 2.0 / 3, 0}, {2.0 / 3, 1.0 / 3, 0.5},
                    {1.0 / 3, 2.0 / 3, u}, {2.0 / 3, 1.0 / 3, 0.5 + u}});
}

Atoms rutile() {
    const double u = 0.305;
    return crystal(Mat3::diagonal(4.594, 4.594, 2.959), {22, 22, 8, 8, 8, 8},
                   {{0, 0, 0}, {0.5, 0.5, 0.5}, {u, u, 0}, {-u, -u, 0},
                    {0.5 - u, 0.5 + u, 0.5}, {0.5 + u, 0.5 - u, 0.5}});
}

Atoms monoclinic() {
    constexpr double kPi = 3.14159265358979323846;
    const double beta = 110.0 * kPi / 180.0;
    const Mat3 cell = Mat3::from_rows({5, 0, 0}, {0, 6, 0}, {7 * std::cos(beta), 0, 7 * std::sin(beta)});
    return crystal(cell, {14, 14}, {{0.1, 0.2, 0.3}, {0.9, 0.2, 0.7}});
}

/// A rotation that changes every Cartesian component, so a transposed
/// lattice cannot pass by accident.
const Mat3 kRotation = rotation_matrix(Vec3{1.0, 2.0, 3.0}, 0.7);

}  // namespace

FA_TEST(copper_is_fm3m) {
    const SymmetryDataset ds = get_symmetry_dataset(copper());
    FA_CHECK(ds.number == 225);
    FA_CHECK(ds.international == "Fm-3m");
    FA_CHECK(ds.hall_number == 523);
    FA_CHECK(ds.crystal_system == CrystalSystem::Cubic);
    FA_CHECK(ds.pointgroup.symbol == "m-3m");
    FA_CHECK(ds.pointgroup.order == 48);
    FA_CHECK(ds.rotations.size() == 192);
    FA_CHECK(ds.translations.size() == ds.rotations.size());
    FA_CHECK(ds.wyckoffs.size() == 4 && ds.wyckoffs[0] == 'a');
    FA_CHECK(ds.mapping_to_primitive == std::vector<int>(4, 0));
}

FA_TEST(bcc_iron_is_im3m) {
    const Atoms fe = crystal(Mat3::diagonal(2.87, 2.87, 2.87), {26, 26}, {{0, 0, 0}, {0.5, 0.5, 0.5}});
    FA_CHECK(get_symmetry_dataset(fe).number == 229);
    FA_CHECK(analyse(fe).international == "Im-3m");
}

FA_TEST(diamond_silicon_is_fd3m) {
    const Atoms si = crystal(Mat3::diagonal(5.43, 5.43, 5.43), std::vector<int>(8, 14),
                             {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0},
                              {0.25, 0.25, 0.25}, {0.25, 0.75, 0.75}, {0.75, 0.25, 0.75},
                              {0.75, 0.75, 0.25}});
    FA_CHECK(get_symmetry_dataset(si).number == 227);
    FA_CHECK(get_symmetry_dataset(si).choice == "1");
}

FA_TEST(rocksalt_is_fm3m) {
    FA_CHECK(get_symmetry_dataset(rocksalt()).number == 225);
}

FA_TEST(wurtzite_hexagonal_lattice_convention) {
    const SymmetryDataset ds = get_symmetry_dataset(wurtzite());
    FA_CHECK(ds.number == 186);
    FA_CHECK(ds.international == "P6_3mc");
    FA_CHECK(ds.crystal_system == CrystalSystem::Hexagonal);
    // Lattice vectors are rows: two of length a, one of length c. Transposing
    // the lattice would break these lengths.
    FA_CHECK_CLOSE(ds.std_lattice.row(0).norm(), 3.25, 1e-6);
    FA_CHECK_CLOSE(ds.std_lattice.row(1).norm(), 3.25, 1e-6);
    FA_CHECK_CLOSE(ds.std_lattice.row(2).norm(), 5.20, 1e-6);
    // The same crystal, rotated in space, keeps its symmetry.
    const Atoms rotated = crystal(wurtzite().cell().matrix() * kRotation, {30, 30, 8, 8},
                                  {{1.0 / 3, 2.0 / 3, 0}, {2.0 / 3, 1.0 / 3, 0.5},
                                   {1.0 / 3, 2.0 / 3, 0.382}, {2.0 / 3, 1.0 / 3, 0.882}});
    FA_CHECK(get_symmetry_dataset(rotated).number == 186);
}

FA_TEST(rutile_is_p42mnm_with_wyckoff_letters) {
    const Atoms tio2 = rutile();
    const SpaceGroup sg = analyse(tio2);
    FA_CHECK(sg.number == 136);
    FA_CHECK(sg.international == "P4_2/mnm");
    FA_CHECK(sg.pointgroup.symbol == "4/mmm");
    FA_CHECK(sg.crystal_system == CrystalSystem::Tetragonal);
    FA_CHECK(sg.wyckoff.size() == 2);
    if (sg.wyckoff.size() == 2) {
        // Ti 2a and O 4f, with their site symmetries.
        FA_CHECK(sg.wyckoff[0].letter == 'a');
        FA_CHECK(sg.wyckoff[0].site_symmetry == "m.mm");
        FA_CHECK(sg.wyckoff[0].atoms == (std::vector<std::size_t>{0, 1}));
        FA_CHECK(sg.wyckoff[1].letter == 'f');
        FA_CHECK(sg.wyckoff[1].site_symmetry == "m.2m");
        FA_CHECK(sg.wyckoff[1].atoms == (std::vector<std::size_t>{2, 3, 4, 5}));
    }
}

FA_TEST(monoclinic_cell) {
    const SymmetryDataset ds = get_symmetry_dataset(monoclinic());
    FA_CHECK(ds.number == 10);
    FA_CHECK(ds.international == "P2/m");
    FA_CHECK(ds.crystal_system == CrystalSystem::Monoclinic);
    FA_CHECK(ds.pointgroup.symbol == "2/m");
    FA_CHECK(ds.pointgroup.order == 4);
}

FA_TEST(rattled_structure_depends_on_symprec) {
    // Four copper sites displaced by a few hundredths of an angstrom.
    Atoms cu = copper();
    const double offsets[4][3] = {{0.03, -0.02, 0.01},
                                  {-0.04, 0.01, -0.03},
                                  {0.02, 0.04, -0.01},
                                  {-0.01, -0.03, 0.02}};
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t k = 0; k < 3; ++k) {
            cu.positions()[3 * i + k] += offsets[i][k];
        }
    }
    FA_CHECK(get_symmetry_dataset(cu, 1e-5).number == 1);
    FA_CHECK(get_symmetry_dataset(cu, 0.2).number == 225);
}

FA_TEST(labels_lower_the_symmetry_when_used) {
    Atoms fe = crystal(Mat3::diagonal(2.87, 2.87, 2.87), {26, 26}, {{0, 0, 0}, {0.5, 0.5, 0.5}});
    fe.set_string_array("labels", {"Fe1", "Fe2"});
    FA_CHECK(get_symmetry_dataset(fe).number == 229);
    FA_CHECK(get_symmetry_dataset(fe, 1e-5, -1.0, true).number == 221);
    FA_CHECK(analyse(fe, 1e-5, -1.0, true).international == "Pm-3m");
}

FA_TEST(primitive_fcc_has_one_atom) {
    const Atoms conventional = copper();
    const Atoms prim = primitive_cell(conventional);
    FA_CHECK(prim.size() == 1);
    FA_CHECK_CLOSE(prim.volume(), conventional.volume() / 4.0, 1e-8);
    FA_CHECK(prim.cell().all_pbc());
    FA_CHECK(find_primitive(rocksalt()).size() == 2);
    FA_CHECK(find_primitive(rutile()).size() == 6);
}

FA_TEST(rotated_primitive_cell_standardises_to_conventional) {
    // The primitive fcc basis, rotated in space: standardising gives back the
    // four-atom cubic cell.
    const double a = 3.61;
    const Mat3 primitive = Mat3::from_rows({0, a / 2, a / 2}, {a / 2, 0, a / 2}, {a / 2, a / 2, 0});
    const Atoms rotated = crystal(primitive * kRotation, {29}, {{0, 0, 0}});
    const Atoms conventional = conventional_cell(rotated);
    FA_CHECK(conventional.size() == 4);
    FA_CHECK_CLOSE(conventional.volume(), std::pow(a, 3), 1e-6);
    FA_CHECK(get_symmetry_dataset(conventional).number == 225);
    FA_CHECK_CLOSE(conventional.cell().vector(0).norm(), a, 1e-6);
    FA_CHECK_CLOSE(conventional.cell().vector(2).norm(), a, 1e-6);

    // Permuting the atoms of a rotated conventional cell changes nothing.
    const Atoms permuted = crystal(copper().cell().matrix() * kRotation, std::vector<int>(4, 29),
                                   {{0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}, {0, 0, 0}});
    FA_CHECK(standardize_cell(permuted).size() == 4);
    FA_CHECK_CLOSE(standardize_cell(permuted).volume(), std::pow(a, 3), 1e-6);
}

FA_TEST(analyse_cell_gives_the_primitive_basis) {
    const CellAnalysis info = analyse_cell(copper());
    FA_CHECK(info.number == 225);
    FA_CHECK(info.primitive_atoms == 1);
    // Each primitive vector is a face diagonal of the cubic cell.
    const Mat3 prim = info.to_primitive * copper().cell().matrix();
    FA_CHECK_CLOSE(prim.row(0).norm(), 3.61 / std::sqrt(2.0), 1e-6);

    const CellAnalysis rotated = analyse_cell(crystal(copper().cell().matrix() * kRotation,
                                                      std::vector<int>(4, 29),
                                                      {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}}));
    FA_CHECK(rotated.primitive_atoms == 1);
}

FA_TEST(niggli_matches_spglib) {
    // Reference: spglib.niggli_reduce of the same basis.
    const Atoms skewed = crystal(Mat3::from_rows({1, 0, 0}, {3, 1, 0}, {0.5, 2, 4}), {1},
                                 {{0.2, 0.1, 0.3}});
    const Atoms reduced = niggli_reduce(skewed);
    const Mat3& m = reduced.cell().matrix();
    FA_CHECK_CLOSE(m(0, 0), -1.0, 1e-6);
    FA_CHECK_CLOSE(m(1, 1), 1.0, 1e-6);
    FA_CHECK_CLOSE(m(2, 0), 0.5, 1e-6);
    FA_CHECK_CLOSE(m(2, 2), -4.0, 1e-6);
    FA_CHECK_CLOSE(reduced.volume(), 4.0, 1e-8);
    // The atoms stay where they were in space.
    FA_CHECK_CLOSE(reduced.position(0).x, skewed.position(0).x, 1e-8);
    FA_CHECK_CLOSE(reduced.position(0).z, skewed.position(0).z, 1e-8);
}

FA_TEST(labels_follow_atoms_through_standardisation) {
    Atoms labelled = rocksalt(true);
    std::vector<std::string> labels;
    for (std::size_t i = 0; i < labelled.size(); ++i) {
        labels.push_back(labelled.number(i) == 17 ? "Cl1" : "Na1");
    }
    labelled.set_string_array("labels", labels);

    const Atoms prim = find_primitive(labelled);
    FA_CHECK(prim.size() == 2);
    FA_CHECK(prim.has_string_array("labels"));
    FA_CHECK(prim.cell().all_pbc());
    for (std::size_t i = 0; i < prim.size(); ++i) {
        const std::string label = prim.labels()[i];
        FA_CHECK(label == (prim.number(i) == 17 ? "Cl1" : "Na1"));
    }

    const Atoms conventional = standardize_cell(labelled);
    FA_CHECK(conventional.size() == 8);
    for (std::size_t i = 0; i < conventional.size(); ++i) {
        const std::string label = conventional.labels()[i];
        FA_CHECK(label == (conventional.number(i) == 17 ? "Cl1" : "Na1"));
    }
}

FA_TEST(space_group_table_round_trips) {
    const auto& table = space_group_table();
    FA_CHECK(table.size() == 230);
    FA_CHECK(table.front().first == 1 && table.front().second == "P1");
    FA_CHECK(table.back().first == 230 && table.back().second == "Ia-3d");
    FA_CHECK(space_group_number("Fm-3m") == 225);
    FA_CHECK(space_group_number("P6_3mc") == 186);
    FA_CHECK(space_group_number("not a symbol") == 0);
    FA_CHECK(space_group_symbol(227) == "Fd-3m");
    FA_CHECK_THROWS(space_group_symbol(0), SymmetryError);
    FA_CHECK_THROWS(space_group_symbol(231), SymmetryError);
}

FA_TEST(unusable_structures_throw) {
    FA_CHECK_THROWS(get_symmetry_dataset(Atoms{}), SymmetryError);
    const Atoms isolated({29}, {0.0, 0.0, 0.0});
    FA_CHECK_THROWS(get_symmetry_dataset(isolated), SymmetryError);
    FA_CHECK_THROWS(niggli_reduce(isolated), SymmetryError);
}

FA_TEST_MAIN()
