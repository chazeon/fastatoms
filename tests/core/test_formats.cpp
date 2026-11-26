// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cstdint>
#include <string>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/io/format.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::io;

namespace {

bool have_format(const char* name) {
    const auto available = available_formats();
    return std::find(available.begin(), available.end(), name) != available.end();
}

}  // namespace

FA_TEST(guess_format_from_names) {
    const auto available = available_formats();
    const auto have = [&](const char* f) {
        return std::find(available.begin(), available.end(), f) != available.end();
    };
    if (have("vasp")) {
        FA_CHECK(guess_format("POSCAR") == "vasp");
        FA_CHECK(guess_format("run/CONTCAR") == "vasp");
        FA_CHECK(guess_format("POSCAR-001") == "vasp");
        FA_CHECK(guess_format("SPOSCAR") == "vasp");
        FA_CHECK(guess_format("Fe.vasp") == "vasp");
        FA_CHECK(guess_format("XDATCAR") == "xdatcar");
    }
    if (have("extxyz")) {
        FA_CHECK(guess_format("a.xyz") == "extxyz");
        FA_CHECK(guess_format("a.extxyz") == "extxyz");
    }
    if (have("cif")) {
        FA_CHECK(guess_format("NaCl.CIF") == "cif");
    }
    if (have("lammps-data")) {
        FA_CHECK(guess_format("in.data") == "lammps-data");
        FA_CHECK(guess_format("x.lmp") == "lammps-data");
        FA_CHECK(guess_format("traj.lammpstrj") == "lammps-dump");
        FA_CHECK(guess_format("", "ITEM: TIMESTEP\n0\n") == "lammps-dump");
    }
    FA_CHECK(guess_format("README.md").empty());
}

FA_TEST(unknown_format_is_an_error) {
    FA_CHECK_THROWS(read_all("", "pdb"), FormatError);
    FA_CHECK(!can_write("pdb"));
}

FA_TEST(xyz_plain_and_extended_are_read_differently) {
    if (!have_format("xyz")) {
        return;
    }
    Atoms atoms({6}, {0.0, 0.0, 0.0});
    atoms.set_info(Value(Value::Map{{"comment", Value("step=1")}}));
    const std::string text = write(atoms, "xyz");
    // format "xyz" is plain: the comment survives verbatim and is not parsed.
    const Atoms plain = read(text, "xyz");
    FA_CHECK(plain.info()["comment"].as_string() == "step=1");
    FA_CHECK(plain.info().find("step") == nullptr);
    // format "extxyz" parses the same comment as key=value.
    const Atoms extended = read(text, "extxyz");
    FA_CHECK(extended.info().find("step") != nullptr);
    FA_CHECK(extended.info()["step"].as_int() == 1);
    // The guessed format for *.xyz is extxyz, as before.
    FA_CHECK(guess_format("run.xyz") == "extxyz");
}

FA_TEST(xyz_precision_option_is_range_checked) {
    if (!have_format("xyz")) {
        return;
    }
    Atoms atoms({6}, {1e-100, 0.0, 0.0});
    FA_CHECK_THROWS(write(atoms, "xyz", Value(Value::Map{{"precision", Value(100)}})),
                    FormatError);
    // A value that would wrap into range when narrowed is rejected too.
    FA_CHECK_THROWS(write(atoms, "extxyz",
                          Value(Value::Map{{"precision", Value(std::int64_t{4294967313})}})),
                    FormatError);
    FA_CHECK_THROWS(write(atoms, "extxyz", Value(Value::Map{{"precision", Value("6")}})),
                    FormatError);
    FA_CHECK(write(atoms, "extxyz", Value(Value::Map{{"precision", Value(17)}}))
                 .find("\nC 1e-100 0 0\n") != std::string::npos);
}

FA_TEST(index_option_of_the_wrong_type_is_a_format_error) {
    if (!have_format("xyz")) {
        return;
    }
    const std::string text = "1\nc\nH 0 0 0\n";
    const auto with_index = [](Value index) {
        return Value(Value::Map{{"index", std::move(index)}});
    };
    FA_CHECK_THROWS(read(text, "xyz", with_index(Value("a"))), FormatError);
    FA_CHECK_THROWS(read(text, "xyz", with_index(Value(true))), FormatError);
    FA_CHECK_THROWS(read(text, "xyz", with_index(Value(1.5))), FormatError);
    FA_CHECK_THROWS(read(text, "xyz", with_index(Value(1e30))), FormatError);
    FA_CHECK(read(text, "xyz", with_index(Value(0.0))).size() == 1);
    FA_CHECK(read(text, "xyz", with_index(Value(-1))).size() == 1);
}

FA_TEST(lammps_type_keys_must_be_whole_integers) {
    if (!have_format("lammps-dump")) {
        return;
    }
    const auto with_types = [](const char* key) {
        return Value(Value::Map{{"types", Value(Value::Map{{key, Value("C")}})}});
    };
    FA_CHECK_THROWS(read_all("", "lammps-dump", with_types("1abc")), FormatError);
    FA_CHECK_THROWS(read_all("", "lammps-dump", with_types(" 1")), FormatError);
    FA_CHECK_THROWS(read_all("", "lammps-dump", with_types("")), FormatError);
    FA_CHECK_THROWS(read_all("", "lammps-dump", with_types("99999999999999999999")),
                    FormatError);
    FA_CHECK_THROWS(read_all("", "lammps-data", with_types("2x")), FormatError);
}

FA_TEST_MAIN()
