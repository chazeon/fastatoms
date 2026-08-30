// SPDX-License-Identifier: MIT
#pragma once

/// \file formats.hpp
/// \brief Reading and writing structures, whatever the file is.
///
/// Extended XYZ is the default. It is the format that carries what a
/// simulation produces - many frames, per-atom forces and velocities, a cell,
/// per-frame energies - in a form every other atomistic code already reads,
/// and it is what `read` and `write` choose when nothing says otherwise.
///
/// YAML remains, and remains necessary: it is the only one of the two that can
/// describe a *run*. An extended XYZ file has no place to say which calculator
/// to attach or how long to integrate for, so an input deck is still YAML and
/// `read_input` is still how you load one. What changed is the default for
/// structures, which is what most reading and nearly all writing is.

#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/io/extxyz.hpp"

namespace fastatoms::io {

/// The file formats FastAtoms reads and writes structures in.
enum class Format {
    Extxyz,   ///< .extxyz, .xyz - the default
    Yaml,     ///< .yaml, .yml - structures, and the only format for input decks
};

[[nodiscard]] std::string_view format_name(Format format);

/// Which format a path names, from its extension.
///
/// Anything unrecognised is extended XYZ, which is what "the default format"
/// means in practice: a file called `dump` or `trajectory.out` is read as
/// extxyz rather than refused.
[[nodiscard]] Format format_from_path(std::string_view path);

/// Parse a format by name: "extxyz", "xyz", "yaml", "yml".
/// \throws std::invalid_argument if the name is not one of those.
[[nodiscard]] Format format_from_name(std::string_view name);

/// Read one structure.
///
/// `frame` selects which, counting from the end when negative, so the default
/// gives the last frame of a trajectory - the relaxed geometry, the final
/// configuration. YAML files hold one structure and ignore it.
///
/// \throws ExtxyzError or InputError on a malformed file.
[[nodiscard]] Atoms read(const std::string& path, std::ptrdiff_t frame = -1);

/// Read every frame. A YAML file yields one.
[[nodiscard]] std::vector<Atoms> read_frames(const std::string& path);

/// Write one structure, replacing the file.
///
/// The format follows the extension, so `write("relaxed.extxyz", atoms)` and
/// `write("relaxed.yaml", atoms)` both do what they look like.
void write(const std::string& path, const Atoms& atoms, ExtxyzOptions options = {});

/// Write a trajectory. Only extended XYZ can hold more than one frame; asking
/// YAML to is an error rather than a silent loss of every frame but one.
void write(const std::string& path, const std::vector<Atoms>& frames,
           ExtxyzOptions options = {});

/// Append one frame to a file, creating it when it does not exist.
///
/// This is the trajectory case in one call. For a long run, hold an
/// `ExtxyzWriter` open instead: this reopens the file each time.
void append(const std::string& path, const Atoms& atoms, ExtxyzOptions options = {});

}  // namespace fastatoms::io
