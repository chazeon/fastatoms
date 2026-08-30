// SPDX-License-Identifier: MIT
//
// Extended XYZ: the comment line, the Properties spec, frames and appending.
//
// The format is loose enough that most of the work is tolerating what other
// programs write - unquoted values, a missing Properties, three lattice
// numbers instead of nine - so most of these cases are things a reader meets
// in the wild rather than things FastAtoms itself produces.

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

#include "fastatoms/core/elements.hpp"
#include "fastatoms/io/extxyz.hpp"
#include "fastatoms/io/formats.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::io;

namespace {

/// A scratch directory, removed when it goes out of scope, so a test never
/// leaves a file behind.
class Scratch {
public:
    Scratch() {
        static int counter = 0;
        path_ = std::filesystem::temp_directory_path() /
                ("fastatoms-extxyz-" + std::to_string(++counter) + "-" +
                 std::to_string(::getpid()));
        std::filesystem::create_directories(path_);
    }
    ~Scratch() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;

    [[nodiscard]] std::string file(const std::string& name) const {
        return (path_ / name).string();
    }

private:
    std::filesystem::path path_;
};

Atoms water() {
    Atoms atoms;
    atoms.append(8, Vec3{0.0, 0.0, 0.0});
    atoms.append(1, Vec3{0.958, 0.0, 0.0});
    atoms.append(1, Vec3{-0.24, 0.927, 0.0});
    atoms.set_cell(Mat3::diagonal(10.0, 11.0, 12.0));
    atoms.set_pbc({true, true, true});
    return atoms;
}

}  // namespace

// ---------------------------------------------------------------------------
// The comment line
// ---------------------------------------------------------------------------
FA_TEST(comment_line_key_value_pairs) {
    const Value info = parse_comment_line(
        R"(Lattice="1.0 0 0 0 2.0 0 0 0 3.0" Properties=species:S:1:pos:R:3 )"
        R"(pbc="T T F" energy=-1.25 step=7 name="a frame" flag)");

    FA_CHECK(info.at("Lattice").as_string() == "1.0 0 0 0 2.0 0 0 0 3.0");
    FA_CHECK(info.at("Properties").as_string() == "species:S:1:pos:R:3");
    FA_CHECK(info.at("pbc").as_string() == "T T F");
    // Typed where it can be: a reader that hands back "-1.25" as text has not
    // parsed anything.
    FA_CHECK(info.at("energy").is_float());
    FA_CHECK_CLOSE(info.at("energy").as_double(), -1.25, 1e-15);
    FA_CHECK(info.at("step").is_int());
    FA_CHECK(info.at("step").as_int() == 7);
    FA_CHECK(info.at("name").as_string() == "a frame");
    // A bare word is a flag.
    FA_CHECK(info.at("flag").as_bool());
}

FA_TEST(comment_line_tolerates_what_other_writers_emit) {
    // Single quotes, no quotes around a multi-word key, and extra spacing.
    const Value info = parse_comment_line("  a='one two'   b=3   c=T  d=F ");
    FA_CHECK(info.at("a").as_string() == "one two");
    FA_CHECK(info.at("b").as_int() == 3);
    FA_CHECK(info.at("c").as_bool());
    FA_CHECK(!info.at("d").as_bool());

    // An empty comment line is legal and means no information.
    FA_CHECK(parse_comment_line("").as_map().empty());
}

FA_TEST(properties_spec_round_trips) {
    const std::vector<PropertyColumn> columns =
        parse_properties("species:S:1:pos:R:3:forces:R:3:tags:I:1");
    FA_CHECK(columns.size() == 4);
    FA_CHECK(columns[0].name == "species" && columns[0].type == 'S' && columns[0].count == 1);
    FA_CHECK(columns[1].name == "pos" && columns[1].type == 'R' && columns[1].count == 3);
    FA_CHECK(columns[3].type == 'I');
    FA_CHECK(format_properties(columns) == "species:S:1:pos:R:3:forces:R:3:tags:I:1");

    FA_CHECK_THROWS(parse_properties("species:S"), ExtxyzError);
    FA_CHECK_THROWS(parse_properties("species:X:1"), ExtxyzError);
    FA_CHECK_THROWS(parse_properties("species:S:0"), ExtxyzError);
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------
FA_TEST(a_frame_carries_cell_pbc_energy_and_columns) {
    const Atoms atoms = parse_extxyz(
        "3\n"
        "Lattice=\"10 0 0 0 11 0 0 0 12\" "
        "Properties=species:S:1:pos:R:3:forces:R:3:charges:R:1 pbc=\"T T F\" energy=-1.5\n"
        "O  0.0 0.0 0.0   0.1 -0.2 0.3  -0.834\n"
        "H  0.958 0.0 0.0  -0.05 0.1 0.0  0.417\n"
        "H -0.24 0.927 0.0 -0.05 0.1 0.0  0.417\n");

    FA_CHECK(atoms.size() == 3);
    FA_CHECK(atoms.number(0) == 8 && atoms.number(1) == 1);
    FA_CHECK_CLOSE(atoms.cell().lengths().x, 10.0, 1e-12);
    FA_CHECK_CLOSE(atoms.cell().lengths().z, 12.0, 1e-12);
    FA_CHECK(atoms.pbc()[0] && atoms.pbc()[1] && !atoms.pbc()[2]);
    FA_CHECK_CLOSE(atoms.info().at("energy").as_double(), -1.5, 1e-15);

    FA_CHECK(atoms.has_array("forces"));
    FA_CHECK_CLOSE(atoms.array("forces").data[2], 0.3, 1e-15);
    FA_CHECK_CLOSE(atoms.charges()[0], -0.834, 1e-15);

    // Lattice, Properties and pbc describe the structure; they are not left
    // lying in info to be written out twice.
    FA_CHECK(atoms.info().find("Lattice") == nullptr);
    FA_CHECK(atoms.info().find("Properties") == nullptr);
}

FA_TEST(a_plain_xyz_file_without_properties_still_reads) {
    // No comment line keys at all: the columns are the historical default.
    const Atoms atoms = parse_extxyz("2\nsome free-form comment\nH 0 0 0\nH 0 0 0.74\n");
    FA_CHECK(atoms.size() == 2);
    FA_CHECK(atoms.number(0) == 1);
    FA_CHECK_CLOSE(atoms.position(1).z, 0.74, 1e-12);
    // Without a Lattice there is nothing to be periodic about.
    FA_CHECK(!atoms.pbc()[0]);
}

FA_TEST(a_three_number_lattice_means_a_diagonal_cell) {
    const Atoms atoms = parse_extxyz(
        "1\nLattice=\"5 6 7\" Properties=species:S:1:pos:R:3\nAr 0 0 0\n");
    FA_CHECK_CLOSE(atoms.cell().lengths().x, 5.0, 1e-12);
    FA_CHECK_CLOSE(atoms.cell().lengths().y, 6.0, 1e-12);
    FA_CHECK_CLOSE(atoms.cell().lengths().z, 7.0, 1e-12);
}

FA_TEST(velocities_masses_and_custom_columns) {
    const Atoms atoms = parse_extxyz(
        "1\nProperties=species:S:1:pos:R:3:velo:R:3:masses:R:1:tags:I:1:stress:R:6\n"
        "Ar 0 0 0  1.0 2.0 3.0  39.948  4  1 2 3 4 5 6\n");
    FA_CHECK_CLOSE(atoms.masses()[0], 39.948, 1e-12);
    FA_CHECK_CLOSE(atoms.velocities()[1], 2.0, 1e-12);
    FA_CHECK(atoms.tags()[0] == 4);
    // A column with no built-in home becomes an array of its own name.
    FA_CHECK(atoms.has_array("stress"));
    FA_CHECK(atoms.array("stress").ncols == 6);
    FA_CHECK_CLOSE(atoms.array("stress").data[5], 6.0, 1e-12);
}

FA_TEST(every_element_survives_a_round_trip) {
    // A hash table with no collision handling works for most elements and not
    // for argon, which is the kind of bug that hides until someone uses it.
    for (int z = 1; z <= 118; ++z) {
        const std::string symbol(symbol_from_number(z));
        if (symbol.empty()) {
            continue;
        }
        const Atoms atoms = parse_extxyz(
            "1\nProperties=species:S:1:pos:R:3\n" + symbol + " 0 0 0\n");
        FA_CHECK(atoms.number(0) == z);
    }
}

FA_TEST(malformed_frames_are_reported_with_their_line) {
    FA_CHECK_THROWS(parse_extxyz("not a number\ncomment\n"), ExtxyzError);
    // Fewer atoms than the count promises.
    FA_CHECK_THROWS(parse_extxyz("3\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"), ExtxyzError);
    // A column short.
    FA_CHECK_THROWS(parse_extxyz("1\nProperties=species:S:1:pos:R:3\nH 0 0\n"), ExtxyzError);
    // Not an element.
    FA_CHECK_THROWS(parse_extxyz("1\nProperties=species:S:1:pos:R:3\nXx 0 0 0\n"), ExtxyzError);
    // Properties with no positions.
    FA_CHECK_THROWS(parse_extxyz("1\nProperties=species:S:1\nH\n"), ExtxyzError);

    try {
        (void)parse_extxyz("2\nProperties=species:S:1:pos:R:3\nH 0 0 0\nH 0 0 x\n");
        FA_CHECK(false);
    } catch (const ExtxyzError& error) {
        // The message names the line, which is the only thing that helps in a
        // file with a million of them.
        FA_CHECK(std::string(error.what()).find("line 4") != std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// Writing and appending
// ---------------------------------------------------------------------------
FA_TEST(a_written_frame_reads_back_exactly) {
    Atoms atoms = water();
    atoms.set_array("forces", {0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1e-17, 2.5e8, -0.5}, 3);
    atoms.info().insert("energy", Value(-1.2345678901234567));

    const Atoms back = parse_extxyz(extxyz_string(atoms));
    FA_CHECK(back.size() == atoms.size());
    // Bit for bit: the default is the shortest text that reads back as the
    // same double, so nothing is lost to a fixed number of digits.
    FA_CHECK(back.positions() == atoms.positions());
    FA_CHECK(back.array("forces").data == atoms.array("forces").data);
    FA_CHECK(back.info().at("energy").as_double() == atoms.info().at("energy").as_double());
    FA_CHECK(back.cell().matrix() == atoms.cell().matrix());
}

FA_TEST(frames_can_be_appended_to_an_existing_file) {
    const Scratch scratch;
    const std::string path = scratch.file("traj.extxyz");

    {
        ExtxyzWriter writer(path);
        for (int step = 0; step < 3; ++step) {
            Atoms atoms = water();
            Value info(Value::Map{});
            info.insert("step", Value(static_cast<std::int64_t>(step)));
            writer.write(atoms, info);
        }
    }
    // A second writer, opened later, must extend the file rather than replace
    // it: this is what a restarted run does.
    {
        ExtxyzOptions options;
        options.append = true;
        ExtxyzWriter writer(path, options);
        for (int step = 3; step < 5; ++step) {
            Atoms atoms = water();
            Value info(Value::Map{});
            info.insert("step", Value(static_cast<std::int64_t>(step)));
            writer.write(atoms, info);
        }
    }

    ExtxyzReader reader(path);
    FA_CHECK(reader.size() == 5);
    for (std::size_t frame = 0; frame < 5; ++frame) {
        FA_CHECK(reader.at(static_cast<std::ptrdiff_t>(frame)).info().at("step").as_int() ==
                 static_cast<std::int64_t>(frame));
    }
    // Negative indices count from the end, and the default read is the last.
    FA_CHECK(reader.at(-1).info().at("step").as_int() == 4);
    FA_CHECK(read_extxyz(path).info().at("step").as_int() == 4);
    FA_CHECK_THROWS(reader.at(5), ExtxyzError);
    FA_CHECK_THROWS(reader.at(-6), ExtxyzError);

    // Without append, the file starts again.
    write_extxyz(path, water());
    FA_CHECK(ExtxyzReader(path).size() == 1);
}

FA_TEST(streaming_reads_one_frame_at_a_time) {
    const Scratch scratch;
    const std::string path = scratch.file("stream.extxyz");
    std::vector<Atoms> frames;
    for (int step = 0; step < 4; ++step) {
        Atoms atoms = water();
        atoms.set_position(0, Vec3{0.1 * step, 0.0, 0.0});
        frames.push_back(atoms);
    }
    write_extxyz(path, frames);

    ExtxyzReader reader(path);
    Atoms atoms;
    std::size_t seen = 0;
    while (reader.next(atoms)) {
        FA_CHECK_CLOSE(atoms.position(0).x, 0.1 * static_cast<double>(seen), 1e-12);
        ++seen;
    }
    FA_CHECK(seen == 4);
    FA_CHECK(reader.frames_read() == 4);

    reader.rewind();
    FA_CHECK(reader.next(atoms));
    FA_CHECK_CLOSE(atoms.position(0).x, 0.0, 1e-12);
}

FA_TEST(what_gets_written_can_be_chosen) {
    Atoms atoms = water();
    atoms.set_array("forces", std::vector<double>(9, 1.0), 3);
    atoms.set_velocities(std::vector<double>(9, 0.5));
    atoms.set_charges({-0.8, 0.4, 0.4});

    FA_CHECK(extxyz_string(atoms).find("forces:R:3") != std::string::npos);
    // Velocities come before forces, which is the order this project has
    // always written and the order files on disk already use.
    const std::string text = extxyz_string(atoms);
    FA_CHECK(text.find("velo:R:3") < text.find("forces:R:3"));

    ExtxyzOptions bare;
    bare.forces = false;
    bare.velocities = false;
    const std::string minimal = extxyz_string(atoms, bare);
    FA_CHECK(minimal.find("forces") == std::string::npos);
    FA_CHECK(minimal.find("Properties=species:S:1:pos:R:3") != std::string::npos);

    ExtxyzOptions with_charges;
    with_charges.charges = true;
    FA_CHECK(extxyz_string(atoms, with_charges).find("charges:R:1") != std::string::npos);
}

FA_TEST(a_non_periodic_structure_writes_no_lattice) {
    Atoms molecule;
    molecule.append(1, Vec3{0.0, 0.0, 0.0});
    molecule.append(1, Vec3{0.0, 0.0, 0.74});
    const std::string text = extxyz_string(molecule);
    FA_CHECK(text.find("Lattice") == std::string::npos);
    FA_CHECK(text.find("pbc=\"F F F\"") != std::string::npos);
    FA_CHECK(parse_extxyz(text).size() == 2);
}

// ---------------------------------------------------------------------------
// The default format
// ---------------------------------------------------------------------------
FA_TEST(the_extension_picks_the_format_and_extxyz_is_the_default) {
    FA_CHECK(format_from_path("a.extxyz") == Format::Extxyz);
    FA_CHECK(format_from_path("a.xyz") == Format::Extxyz);
    FA_CHECK(format_from_path("/tmp/a.yaml") == Format::Yaml);
    FA_CHECK(format_from_path("a.yml") == Format::Yaml);
    // Anything unrecognised, including no extension at all.
    FA_CHECK(format_from_path("dump") == Format::Extxyz);
    FA_CHECK(format_from_path("trajectory.out") == Format::Extxyz);
    FA_CHECK(format_from_name("XYZ") == Format::Extxyz);
    FA_CHECK_THROWS(format_from_name("pdb"), std::invalid_argument);
}

FA_TEST(read_and_write_go_through_the_extension) {
    const Scratch scratch;
    Atoms atoms = water();

    const std::string xyz = scratch.file("s.extxyz");
    io::write(xyz, atoms);
    FA_CHECK(io::read(xyz).size() == 3);

    const std::string yaml = scratch.file("s.yaml");
    io::write(yaml, atoms);
    FA_CHECK(io::read(yaml).size() == 3);

    // A trajectory needs a format that can hold one.
    FA_CHECK_THROWS(io::write(yaml, std::vector<Atoms>{atoms, atoms}), ExtxyzError);

    io::append(xyz, atoms);
    FA_CHECK(io::read_frames(xyz).size() == 2);
}

FA_TEST_MAIN()
