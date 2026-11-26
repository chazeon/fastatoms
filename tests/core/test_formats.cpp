// SPDX-License-Identifier: MIT
#include <algorithm>

#include "fastatoms/formats/format.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::formats;

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

FA_TEST_MAIN()
