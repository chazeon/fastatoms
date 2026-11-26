// SPDX-License-Identifier: MIT
#pragma once

/// \file format.hpp
/// \brief Format-independent entry points: read/write text in a named format.
///
/// Every reader takes the file *contents* (not a path), so the same code runs
/// natively, under pybind11 and in the browser. Paths are a convenience on
/// top (`read_file`).
///
/// Conventions every reader follows, so downstream code can rely on them:
///
///   string array "labels"           species label as written ("Fe1", "Fe_pv",
///                                   "O2-"); always present after a read.
///   int array "selective_dynamics"  3 per atom, 1 = free (VASP 'T').
///   int array "type", "id", "mol"   LAMMPS type, atom id, molecule id.
///   int array "image"               3 per atom, LAMMPS image flags.
///   double array "occupancy"        CIF site occupancy.
///   velocities (Atoms::set_velocities) in the file's native units; nothing
///                                   is converted.
///   info["comment"]                 POSCAR comment line / xyz comment.
///   info["timestep"]                LAMMPS dump timestep.
///   info["spacegroup"]              CIF H-M symbol, when given.

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/core/value.hpp"

namespace fastatoms::formats {

/// Malformed or unsupported input. The message names the line when known.
class FormatError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// Every frame in `text`. `options` is a map of format-specific options
/// (see each format header for the keys); unknown keys are an error.
[[nodiscard]] std::vector<Atoms> read_all(std::string_view text, const std::string& format,
                                          const Value& options = {});

/// One frame: `options["index"]` (default -1, the last frame, like ASE).
[[nodiscard]] Atoms read(std::string_view text, const std::string& format,
                         const Value& options = {});

/// Serialise one structure.
[[nodiscard]] std::string write(const Atoms& atoms, const std::string& format,
                                const Value& options = {});

/// Format from a file name (`POSCAR`, `CONTCAR`, `*.vasp`, `*.xyz`,
/// `*.extxyz`, `*.cif`, `*.data`/`*.lmp`, `*.dump`/`*.lammpstrj`), falling
/// back to sniffing the first bytes of `head`. Empty if unknown.
[[nodiscard]] std::string guess_format(std::string_view filename, std::string_view head = {});

/// Names of the formats compiled into this build, e.g.
/// {"vasp", "xyz", "extxyz", "cif", "lammps-data", "lammps-dump"}.
[[nodiscard]] std::vector<std::string> available_formats();

/// Whether `format` can be written (not every reader has a writer).
[[nodiscard]] bool can_write(const std::string& format);

}  // namespace fastatoms::formats
