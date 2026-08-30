// SPDX-License-Identifier: MIT
#include "fastatoms/io/formats.hpp"

#include "fastatoms/io/cif.hpp"
#include "fastatoms/io/vasp.hpp"

#include <algorithm>
#include <stdexcept>

#include "fastatoms/io/yaml_input.hpp"
#include "fastatoms/util/text.hpp"

namespace fastatoms::io {
namespace {

std::string extension_of(std::string_view path) {
    const std::size_t slash = path.find_last_of("/\\");
    const std::string_view name =
        slash == std::string_view::npos ? path : path.substr(slash + 1);
    const std::size_t dot = name.find_last_of('.');
    if (dot == std::string_view::npos || dot + 1 >= name.size()) {
        return {};
    }
    return util::lowercase(name.substr(dot + 1));
}

}  // namespace

std::string_view format_name(Format format) {
    switch (format) {
        case Format::Yaml:
            return "yaml";
        case Format::Poscar:
            return "poscar";
        case Format::Xdatcar:
            return "xdatcar";
        case Format::Outcar:
            return "outcar";
        case Format::Cif:
            return "cif";
        case Format::Extxyz:
            break;
    }
    return "extxyz";
}

std::string_view unsupported_format_name(std::string_view path) {
    // Extensions that name a real format this build does not read or write.
    // Kept deliberately short: only names whose meaning is unambiguous, so
    // that a file called `dump` or `run.out` still falls through to extended
    // XYZ rather than being refused for guessing.
    static const std::vector<std::pair<const char*, const char*>> known{
        {"pdb", "PDB"},
        {"traj", "ASE trajectory"}, {"json", "JSON"},     {"cube", "Gaussian cube"},
        {"xsf", "XCrySDen"},      {"lmp", "LAMMPS data"}, {"gro", "GROMACS"},
        {"mol", "MDL molfile"},   {"sdf", "MDL SDF"},     {"cml", "Chemical Markup"},
        {"pwi", "Quantum ESPRESSO input"},
        {"pwo", "Quantum ESPRESSO output"},
        {"fdf", "SIESTA"},        {"res", "SHELX"},       {"cfg", "AtomEye"},
        {"gen", "DFTB+"},         {"nc", "NetCDF"},       {"db", "ASE database"},
    };
    const std::string extension = extension_of(path);
    for (const auto& [suffix, name] : known) {
        if (extension == suffix) {
            return name;
        }
    }

    // Some formats are named by the whole file, not an extension.
    static const std::vector<std::pair<const char*, const char*>> by_name{
        {"CHGCAR", "VASP CHGCAR"},   {"LOCPOT", "VASP LOCPOT"},
        {"WAVECAR", "VASP WAVECAR"}, {"PROCAR", "VASP PROCAR"},
    };
    std::string base(path);
    const std::size_t slash = base.find_last_of("/\\");
    if (slash != std::string::npos) {
        base = base.substr(slash + 1);
    }
    for (const auto& [stem, name] : by_name) {
        if (base == stem) {
            return name;
        }
    }
    return {};
}

Format format_from_path(std::string_view path) {
    const std::string extension = extension_of(path);
    if (extension == "yaml" || extension == "yml") {
        return Format::Yaml;
    }
    if (extension == "vasp" || extension == "poscar" || extension == "contcar") {
        return Format::Poscar;
    }
    if (extension == "cif") {
        return Format::Cif;
    }
    // VASP names its files rather than extending them, and a run directory
    // holds all of them at once - so `POSCAR`, `CONTCAR.relaxed` and
    // `OUTCAR.2` all have to be recognised by their stem.
    std::string base(path);
    const std::size_t slash = base.find_last_of("/\\");
    if (slash != std::string::npos) {
        base = base.substr(slash + 1);
    }
    const auto starts_with = [&base](const char* stem) {
        const std::string prefix(stem);
        return base.size() >= prefix.size() && base.compare(0, prefix.size(), prefix) == 0;
    };
    if (starts_with("POSCAR") || starts_with("CONTCAR")) {
        return Format::Poscar;
    }
    if (starts_with("XDATCAR")) {
        return Format::Xdatcar;
    }
    if (starts_with("OUTCAR")) {
        return Format::Outcar;
    }
    // Everything else, including no extension at all, is extended XYZ.
    return Format::Extxyz;
}

Format format_from_name(std::string_view name) {
    const std::string key = util::lowercase(name);
    if (key == "extxyz" || key == "xyz") {
        return Format::Extxyz;
    }
    if (key == "yaml" || key == "yml") {
        return Format::Yaml;
    }
    if (key == "poscar" || key == "contcar" || key == "vasp") {
        return Format::Poscar;
    }
    if (key == "xdatcar") {
        return Format::Xdatcar;
    }
    if (key == "outcar") {
        return Format::Outcar;
    }
    if (key == "cif") {
        return Format::Cif;
    }
    throw std::invalid_argument("unknown format '" + std::string(name) +
                                "'; expected extxyz, yaml, poscar, xdatcar, outcar or cif");
}

/// Pick a frame out of a list the way a negative index does elsewhere.
Atoms pick(std::vector<Atoms> frames, std::ptrdiff_t frame, const std::string& path) {
    const auto count = static_cast<std::ptrdiff_t>(frames.size());
    const std::ptrdiff_t index = frame < 0 ? count + frame : frame;
    if (index < 0 || index >= count) {
        throw ExtxyzError("io: '" + path + "' has " + std::to_string(count) +
                          " frames, so there is no frame " + std::to_string(frame));
    }
    return std::move(frames[static_cast<std::size_t>(index)]);
}

Atoms read(const std::string& path, std::ptrdiff_t frame) {
    switch (format_from_path(path)) {
        case Format::Yaml:
            return read_yaml_file(path).atoms;
        case Format::Poscar:
            return read_poscar(path);
        case Format::Xdatcar:
            return pick(read_xdatcar(path), frame, path);
        case Format::Outcar:
            return pick(read_outcar(path), frame, path);
        case Format::Cif:
            return pick(read_cif_frames(path), frame, path);
        case Format::Extxyz:
            break;
    }
    return read_extxyz(path, frame);
}

std::vector<Atoms> read_frames(const std::string& path) {
    switch (format_from_path(path)) {
        case Format::Yaml:
            return {read_yaml_file(path).atoms};
        case Format::Poscar:
            return {read_poscar(path)};
        case Format::Xdatcar:
            return read_xdatcar(path);
        case Format::Outcar:
            return read_outcar(path);
        case Format::Cif:
            return read_cif_frames(path);
        case Format::Extxyz:
            break;
    }
    return read_extxyz_frames(path);
}

namespace {

/// Refuse to write a file whose name promises a format this build does not
/// produce.
///
/// The alternative is what used to happen: extended XYZ written into a file
/// called `structure.cif`, which nothing detects until it reaches whatever was
/// meant to read it. A wrong answer that looks like a right one is the worst
/// outcome available here, so it is the one case where refusing is kinder.
void refuse_foreign_format(const std::string& path) {
    const std::string_view name = unsupported_format_name(path);
    if (name.empty()) {
        return;
    }
    throw ExtxyzError("io: '" + path + "' names a " + std::string(name) +
                      " file, which this build cannot write. Writing extended XYZ under that "
                      "name would produce a file nothing can read as what it claims to be. "
                      "Use .extxyz or .xyz, or give the file a name that does not promise a "
                      "format.");
}

}  // namespace

void write(const std::string& path, const Atoms& atoms, ExtxyzOptions options) {
    refuse_foreign_format(path);
    switch (format_from_path(path)) {
        case Format::Yaml:
            write_yaml_file(path, atoms);
            return;
        case Format::Poscar:
            write_poscar(path, atoms);
            return;
        case Format::Cif:
            write_cif(path, atoms);
            return;
        case Format::Xdatcar:
        case Format::Outcar:
            // Both are things VASP produces, not things it is given. Writing
            // one would be writing a fake record of a run that never happened.
            throw ExtxyzError("io: " + std::string(format_name(format_from_path(path))) +
                              " is output from a VASP run, not input to one; write a POSCAR, "
                              "or an .extxyz trajectory");
        case Format::Extxyz:
            break;
    }
    write_extxyz(path, atoms, std::move(options));
}

void write(const std::string& path, const std::vector<Atoms>& frames, ExtxyzOptions options) {
    refuse_foreign_format(path);
    if (format_from_path(path) == Format::Cif) {
        if (frames.size() != 1) {
            throw ExtxyzError("io: a CIF data block holds one structure, so a " +
                              std::to_string(frames.size()) +
                              "-frame trajectory cannot be written to '" + path + "'");
        }
        write_cif(path, frames.front());
        return;
    }
    if (format_from_path(path) == Format::Poscar) {
        if (frames.size() != 1) {
            throw ExtxyzError("io: a POSCAR holds one structure, so a " +
                              std::to_string(frames.size()) +
                              "-frame trajectory cannot be written to '" + path + "'");
        }
        write_poscar(path, frames.front());
        return;
    }
    if (format_from_path(path) == Format::Yaml) {
        throw ExtxyzError(
            "io: YAML holds one structure, so a " + std::to_string(frames.size()) +
            "-frame trajectory cannot be written to '" + path +
            "'. Use an .extxyz file, which is what the format is for.");
    }
    write_extxyz(path, frames, std::move(options));
}

void append(const std::string& path, const Atoms& atoms, ExtxyzOptions options) {
    refuse_foreign_format(path);
    if (format_from_path(path) == Format::Yaml) {
        throw ExtxyzError("io: a YAML file cannot be appended to; use .extxyz");
    }
    options.append = true;
    write_extxyz(path, atoms, std::move(options));
}

}  // namespace fastatoms::io
