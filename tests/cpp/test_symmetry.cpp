// SPDX-License-Identifier: MIT
///
/// Crystallographic symmetry: point groups, space groups, Wyckoff positions
/// and cell standardization. Checked against structures whose symmetry is
/// known independently of anything here.

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <string>
#include <vector>

#include "fastatoms/fastatoms.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::symmetry;

namespace {

/// The distinct rotations of a structure, as the point-group finder wants them.
std::vector<Mat3> rotations_of(const Atoms& atoms, double tolerance = 1e-5) {
    return phonon::find_symmetry(atoms, tolerance).point_group;
}

}  // namespace

// ---------------------------------------------------------------------------
// The thirty-two point groups
// ---------------------------------------------------------------------------

FA_TEST(the_thirty_two_are_thirty_two_and_all_different) {
    const std::vector<PointGroup>& groups = crystallographic_point_groups();
    FA_CHECK(groups.size() == 32);

    // Distinct symbols, distinct Schoenflies names, and - the property the
    // identification rests on - no two with the same operation counts.
    std::set<std::string> symbols;
    std::set<std::string> schoenflies;
    for (const PointGroup& group : groups) {
        FA_CHECK(symbols.insert(group.symbol).second);
        FA_CHECK(schoenflies.insert(group.schoenflies).second);
    }

    // The orders are the known ones, and the systems hold the known numbers of
    // groups: 2 triclinic, 3 monoclinic, 3 orthorhombic, 7 tetragonal,
    // 5 trigonal, 7 hexagonal, 5 cubic.
    const std::map<CrystalSystem, int> expected{
        {CrystalSystem::Triclinic, 2},  {CrystalSystem::Monoclinic, 3},
        {CrystalSystem::Orthorhombic, 3}, {CrystalSystem::Tetragonal, 7},
        {CrystalSystem::Trigonal, 5},   {CrystalSystem::Hexagonal, 7},
        {CrystalSystem::Cubic, 5},
    };
    std::map<CrystalSystem, int> counted;
    for (const PointGroup& group : groups) {
        ++counted[group.system];
    }
    FA_CHECK(counted == expected);
}

FA_TEST(every_rotation_is_one_of_ten_kinds) {
    // Determinant and trace between them fix the kind, and the ten pairs are
    // distinct - which is why counting them identifies a group at all.
    std::set<std::pair<int, int>> seen;
    for (const RotationType type : {RotationType::Identity, RotationType::Two,
                                    RotationType::Three, RotationType::Four,
                                    RotationType::Six, RotationType::Inversion,
                                    RotationType::Mirror, RotationType::BarThree,
                                    RotationType::BarFour, RotationType::BarSix}) {
        FA_CHECK(!rotation_type_name(type).empty());
        FA_CHECK(rotation_order(type) != 0);
    }

    // The identity and the inversion, checked directly.
    FA_CHECK(classify_rotation(Mat3::identity()) == RotationType::Identity);
    FA_CHECK(classify_rotation(Mat3::diagonal(-1, -1, -1)) == RotationType::Inversion);
    FA_CHECK(classify_rotation(Mat3::diagonal(1, 1, -1)) == RotationType::Mirror);
    FA_CHECK(classify_rotation(Mat3::diagonal(1, -1, -1)) == RotationType::Two);

    // Something that is not a rotation at all.
    FA_CHECK_THROWS(classify_rotation(Mat3::diagonal(2, 1, 1)), SymmetryError);
}

FA_TEST(the_lattices_have_the_point_groups_they_should) {
    // Known answers, none of which depend on this module: the cubic lattices
    // are m-3m, zinc blende loses the inversion, hcp and graphene are 6/mmm.
    struct Case { const char* expected; Atoms atoms; };
    const std::vector<Case> cases{
        {"m-3m", build::sc("Po", 3.35)},
        {"m-3m", build::fcc("Al", 4.05)},
        {"m-3m", build::bcc("Fe", 2.87)},
        {"m-3m", build::fcc("Al", 4.05, true)},
        {"m-3m", build::diamond("Si", 5.43)},
        {"m-3m", build::rocksalt("Na", "Cl", 5.64)},
        {"m-3m", build::cscl("Cs", "Cl", 4.12)},
        {"-43m", build::zincblende("Ga", "As", 5.65)},
        {"6/mmm", build::hcp("Mg", 3.21)},
        {"6/mmm", build::graphene()},
        {"6/mmm", build::honeycomb({"C"}, 2.46)},
    };
    for (const Case& one : cases) {
        const PointGroup group = identify_point_group(rotations_of(one.atoms));
        FA_CHECK(group.symbol == one.expected);
    }
}

FA_TEST(lowering_the_symmetry_shows_in_the_point_group) {
    // Boron nitride is graphene with the two sites made different, and the
    // inversion goes with it: 6/mmm becomes -6m2.
    FA_CHECK(identify_point_group(rotations_of(build::honeycomb({"B", "N"}, 2.51))).symbol ==
             "-6m2");

    // The two dichalcogenide phases differ in exactly this way, and it is why
    // one is a semiconductor and the other is not.
    FA_CHECK(identify_point_group(rotations_of(build::tmd("Mo", "S", build::Phase::H)))
                 .symbol == "-6m2");
    FA_CHECK(identify_point_group(rotations_of(build::tmd("Mo", "S", build::Phase::T)))
                 .symbol == "-3m");
}

FA_TEST(point_groups_report_what_they_are) {
    const PointGroup cubic = identify_point_group(rotations_of(build::fcc("Al", 4.05)));
    FA_CHECK(cubic.order == 48);
    FA_CHECK(cubic.centrosymmetric);
    FA_CHECK(!cubic.enantiomorphic);
    FA_CHECK(cubic.system == CrystalSystem::Cubic);
    FA_CHECK(cubic.schoenflies == "Oh");

    const PointGroup zincblende =
        identify_point_group(rotations_of(build::zincblende("Ga", "As", 5.65)));
    FA_CHECK(zincblende.order == 24);
    FA_CHECK(!zincblende.centrosymmetric);
    FA_CHECK(!zincblende.enantiomorphic);   // -43m has mirrors, so it is not chiral
}

FA_TEST(nonsense_is_refused) {
    FA_CHECK_THROWS(identify_point_group({}), SymmetryError);
    // The identity plus a single three-fold is not a group, and its counts
    // match none of the thirty-two.
    Mat3 three;
    three(0, 0) = 0; three(0, 1) = -1; three(0, 2) = 0;
    three(1, 0) = 1; three(1, 1) = -1; three(1, 2) = 0;
    three(2, 0) = 0; three(2, 1) = 0;  three(2, 2) = 1;
    FA_CHECK_THROWS(identify_point_group({Mat3::identity(), three}), SymmetryError);
}

// ---------------------------------------------------------------------------
// Cells: primitive, conventional, Niggli
// ---------------------------------------------------------------------------

FA_TEST(a_conventional_cell_reduces_to_its_primitive_one) {
    struct Case { const char* name; Atoms conventional; std::size_t primitive; };
    const std::vector<Case> cases{
        {"fcc",        build::fcc("Al", 4.05, true),            1},
        {"bcc",        build::bcc("Fe", 2.87, true),            1},
        {"diamond",    build::diamond("Si", 5.43, true),        2},
        {"rocksalt",   build::rocksalt("Na", "Cl", 5.64, true), 2},
        {"zincblende", build::zincblende("Ga", "As", 5.65, true), 2},
        {"sc",         build::sc("Po", 3.35),                   1},
        {"hcp",        build::hcp("Mg", 3.21),                  2},
    };
    for (const Case& one : cases) {
        const Atoms reduced = primitive_cell(one.conventional);
        FA_CHECK(reduced.size() == one.primitive);
        // The density is unchanged, which is the real check: atoms per volume
        // cannot change when the same crystal is described differently.
        FA_CHECK_CLOSE(static_cast<double>(reduced.size()) / reduced.volume(),
                       static_cast<double>(one.conventional.size()) /
                           one.conventional.volume(),
                       1e-9);
    }
}

FA_TEST(a_primitive_cell_is_already_primitive) {
    // Reducing twice must change nothing, or the reduction is not idempotent
    // and cannot be trusted.
    for (const Atoms& atoms : {build::fcc("Al", 4.05), build::bcc("Fe", 2.87),
                               build::diamond("Si", 5.43), build::graphene()}) {
        const Atoms once = primitive_cell(atoms);
        const Atoms twice = primitive_cell(once);
        FA_CHECK(once.size() == twice.size());
        FA_CHECK_CLOSE(once.volume(), twice.volume(), 1e-9);
    }
}

FA_TEST(a_supercell_reduces_back_to_what_it_is_a_supercell_of) {
    const Atoms unit = build::diamond("Si", 5.43);
    for (const int n : {2, 3}) {
        const Atoms supercell = unit.repeat(n, n, n);
        const Atoms reduced = primitive_cell(supercell);
        FA_CHECK(reduced.size() == unit.size());
        FA_CHECK_CLOSE(reduced.volume(), unit.volume(), 1e-8);
    }
}

FA_TEST(a_primitive_cell_grows_back_into_its_conventional_one) {
    struct Case { Atoms primitive; std::size_t conventional; Centring centring; };
    const std::vector<Case> cases{
        {build::fcc("Al", 4.05), 4, Centring::F},
        {build::bcc("Fe", 2.87), 2, Centring::I},
        {build::diamond("Si", 5.43), 8, Centring::F},
        {build::sc("Po", 3.35), 1, Centring::P},
        {build::rocksalt("Na", "Cl", 5.64), 8, Centring::F},
    };
    for (const Case& one : cases) {
        const CellAnalysis analysis = analyse_cell(one.primitive);
        FA_CHECK(analysis.centring == one.centring);
        FA_CHECK(analysis.conventional_atoms == one.conventional);

        const Atoms grown = conventional_cell(one.primitive);
        FA_CHECK(grown.size() == one.conventional);
        // Same density again, in the other direction.
        FA_CHECK_CLOSE(static_cast<double>(grown.size()) / grown.volume(),
                       static_cast<double>(one.primitive.size()) / one.primitive.volume(),
                       1e-9);
        // And the conventional cell of a cubic crystal is a cube.
        const Vec3 lengths = grown.cell().lengths();
        const Vec3 angles = grown.cell().angles();
        FA_CHECK_CLOSE(lengths.x, lengths.y, 1e-8);
        FA_CHECK_CLOSE(lengths.y, lengths.z, 1e-8);
        FA_CHECK_CLOSE(angles.x, 90.0, 1e-6);
    }
}

FA_TEST(the_round_trip_is_a_round_trip) {
    // primitive -> conventional -> primitive must return the same cell.
    for (const Atoms& atoms : {build::fcc("Al", 4.05), build::bcc("Fe", 2.87),
                               build::diamond("Si", 5.43),
                               build::rocksalt("Na", "Cl", 5.64)}) {
        const Atoms conventional = conventional_cell(atoms);
        const Atoms back = primitive_cell(conventional);
        FA_CHECK(back.size() == atoms.size());
        FA_CHECK_CLOSE(back.volume(), atoms.volume(), 1e-8);
    }
}

FA_TEST(the_analysis_reports_what_the_cell_is) {
    const CellAnalysis conventional = analyse_cell(build::fcc("Al", 4.05, true));
    FA_CHECK(conventional.input_multiplicity == 4);   // four primitive cells in it
    FA_CHECK(conventional.primitive_atoms == 1);
    FA_CHECK(conventional.centring == Centring::F);
    FA_CHECK(conventional.point_group.symbol == "m-3m");

    const CellAnalysis primitive = analyse_cell(build::fcc("Al", 4.05));
    FA_CHECK(primitive.input_multiplicity == 1);      // already primitive
    FA_CHECK(primitive.centring == Centring::F);
}

FA_TEST(niggli_reduction_finds_the_short_cell) {
    // A badly skewed description of a simple cubic lattice: adding a lattice
    // vector to another leaves the lattice alone but ruins the cell.
    Atoms skewed = build::sc("Po", 3.0);
    Mat3 skew = Mat3::identity();
    skew(1, 0) = 3.0;
    skew(2, 0) = -5.0;
    skew(2, 1) = 4.0;
    skewed.set_cell(Cell(skew * skewed.cell().matrix(), {true, true, true}), false);

    const Atoms reduced = niggli_reduce(skewed);
    const Vec3 lengths = reduced.cell().lengths();
    FA_CHECK_CLOSE(lengths.x, 3.0, 1e-8);
    FA_CHECK_CLOSE(lengths.y, 3.0, 1e-8);
    FA_CHECK_CLOSE(lengths.z, 3.0, 1e-8);
    FA_CHECK_CLOSE(reduced.volume(), 27.0, 1e-8);
}

// ---------------------------------------------------------------------------
// Space groups
// ---------------------------------------------------------------------------

FA_TEST(the_table_holds_two_hundred_and_thirty_distinct_groups) {
    const auto& table = space_group_table();
    FA_CHECK(table.size() == 230);

    std::set<std::string> symbols;
    for (int k = 0; k < 230; ++k) {
        FA_CHECK(table[static_cast<std::size_t>(k)].first == k + 1);
        const std::string& symbol = table[static_cast<std::size_t>(k)].second;
        FA_CHECK(symbols.insert(symbol).second);              // no duplicates
        FA_CHECK(space_group_number(symbol) == k + 1);        // round trips
        FA_CHECK(space_group_symbol(k + 1) == symbol);
        // Every symbol starts with a centring letter.
        FA_CHECK(std::string("PABCIFR").find(symbol.front()) != std::string::npos);
    }
    FA_CHECK(space_group_number("not a space group") == 0);
    FA_CHECK_THROWS(space_group_symbol(0), SymmetryError);
    FA_CHECK_THROWS(space_group_symbol(231), SymmetryError);
}

FA_TEST(the_crystal_system_boundaries_fall_where_they_must) {
    // The Tables order the groups by system, and the boundaries are fixed:
    // 2 triclinic, 13 monoclinic, 59 orthorhombic, 68 tetragonal, 25 trigonal,
    // 27 hexagonal, 36 cubic.
    FA_CHECK(space_group_symbol(2) == "P-1");
    FA_CHECK(space_group_symbol(15) == "C2/c");
    FA_CHECK(space_group_symbol(74) == "Imma");
    FA_CHECK(space_group_symbol(142) == "I4_1/acd");
    FA_CHECK(space_group_symbol(167) == "R-3c");
    FA_CHECK(space_group_symbol(194) == "P6_3/mmc");
    FA_CHECK(space_group_symbol(230) == "Ia-3d");
}

FA_TEST(the_famous_structures_come_out_right) {
    // Every one of these is in any textbook, and none of them depends on this
    // code being right.
    struct Case { const char* symbol; int number; Atoms atoms; };
    const std::vector<Case> cases{
        {"Pm-3m", 221, build::sc("Po", 3.35)},
        {"Fm-3m", 225, build::fcc("Cu", 3.61)},
        {"Im-3m", 229, build::bcc("Fe", 2.87)},
        {"Fd-3m", 227, build::diamond("Si", 5.43)},
        {"Fm-3m", 225, build::rocksalt("Na", "Cl", 5.64)},
        {"F-43m", 216, build::zincblende("Ga", "As", 5.65)},
        {"Pm-3m", 221, build::cscl("Cs", "Cl", 4.12)},
        {"P6_3/mmc", 194, build::hcp("Mg", 3.21)},
        {"P6/mmm", 191, build::graphene()},
        {"P-6m2", 187, build::tmd("Mo", "S", build::Phase::H)},
        {"P-3m1", 164, build::tmd("Mo", "S", build::Phase::T)},
    };
    for (const Case& one : cases) {
        const SpaceGroup group = analyse(one.atoms);
        FA_CHECK(group.hermann_mauguin == one.symbol);
        FA_CHECK(group.number == one.number);
    }
}

FA_TEST(the_answer_does_not_depend_on_the_cell_it_was_given_in) {
    // The same crystal in a primitive cell, a conventional cell and a
    // supercell is the same space group. This is the property that makes the
    // whole thing worth having.
    const Atoms primitive = build::diamond("Si", 5.43);
    for (const Atoms& description : {primitive, build::diamond("Si", 5.43, true),
                                     primitive.repeat(2, 1, 1),
                                     primitive.repeat(2, 2, 2)}) {
        const SpaceGroup group = analyse(description);
        FA_CHECK(group.number == 227);
        FA_CHECK(group.hermann_mauguin == "Fd-3m");
        FA_CHECK(group.primitive_atoms == 2);
        FA_CHECK(group.conventional_atoms == 8);
    }
}

FA_TEST(a_space_group_reports_its_own_consistency) {
    const SpaceGroup group = analyse(build::fcc("Cu", 3.61));
    FA_CHECK(group.point_group.symbol == "m-3m");
    FA_CHECK(group.centring == Centring::F);
    FA_CHECK(group.system == CrystalSystem::Cubic);
    // 48 point-group operations on a primitive cell holding one lattice point.
    FA_CHECK(group.order() == 48);
    FA_CHECK(group.schoenflies == "Oh-5");     // Fm-3m is the fifth m-3m group
    FA_CHECK(!group.hermann_mauguin_full.empty());
    FA_CHECK(!group.summary().empty());
}

FA_TEST(screw_axes_and_glides_are_read_off_the_translations) {
    // The two things a point group cannot tell you. Hexagonal close packing is
    // P6_3/mmc and not P6/mmm because of the 6_3 screw; diamond is Fd-3m and
    // not Fm-3m because of the d glide. Both come from translation parts.
    const SpaceGroup hcp = analyse(build::hcp("Mg", 3.21));
    FA_CHECK(hcp.hermann_mauguin == "P6_3/mmc");
    FA_CHECK(hcp.point_group.symbol == "6/mmm");   // same point group as graphene

    const SpaceGroup graphene = analyse(build::graphene());
    FA_CHECK(graphene.point_group.symbol == "6/mmm");
    FA_CHECK(graphene.hermann_mauguin == "P6/mmm");   // but a different space group

    const SpaceGroup diamond = analyse(build::diamond("Si", 5.43));
    const SpaceGroup copper = analyse(build::fcc("Cu", 3.61));
    FA_CHECK(diamond.point_group.symbol == copper.point_group.symbol);
    FA_CHECK(diamond.hermann_mauguin == "Fd-3m");
    FA_CHECK(copper.hermann_mauguin == "Fm-3m");
}

FA_TEST(wyckoff_orbits_group_the_atoms_symmetry_relates) {
    // Rock salt: two orbits, one per species, each of multiplicity four in the
    // conventional cell.
    const SpaceGroup salt = analyse(build::rocksalt("Na", "Cl", 5.64, true));
    FA_CHECK(salt.wyckoff.size() == 2);
    for (const WyckoffPosition& position : salt.wyckoff) {
        FA_CHECK(position.multiplicity == 4);
        FA_CHECK(position.site_symmetry == "m-3m");   // both sites are octahedral
    }

    // Diamond: one orbit, because the two atoms of the primitive cell are
    // related by symmetry even though they are not related by a translation.
    const SpaceGroup diamond = analyse(build::diamond("Si", 5.43, true));
    FA_CHECK(diamond.wyckoff.size() == 1);
    FA_CHECK(diamond.wyckoff.front().multiplicity == 8);
    FA_CHECK(diamond.wyckoff.front().site_symmetry == "-43m");
    FA_CHECK(diamond.wyckoff.front().letter == 'a');

    // Molybdenum disulphide: the metal and the two chalcogens are two orbits.
    const SpaceGroup mos2 = analyse(build::tmd("Mo", "S", build::Phase::H));
    FA_CHECK(mos2.wyckoff.size() == 2);
}

FA_TEST(analysing_nonsense_is_refused) {
    FA_CHECK_THROWS(analyse(Atoms{}), SymmetryError);
    Atoms no_cell(std::vector<std::string>{"Ar"}, {0.0, 0.0, 0.0});
    FA_CHECK_THROWS(analyse(no_cell), SymmetryError);
}

FA_TEST_MAIN()
