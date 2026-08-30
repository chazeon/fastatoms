// SPDX-License-Identifier: MIT
#include "fastatoms/io/formats.hpp"

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
    return format == Format::Yaml ? "yaml" : "extxyz";
}

Format format_from_path(std::string_view path) {
    const std::string extension = extension_of(path);
    if (extension == "yaml" || extension == "yml") {
        return Format::Yaml;
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
    throw std::invalid_argument("unknown format '" + std::string(name) +
                                "'; expected extxyz or yaml");
}

Atoms read(const std::string& path, std::ptrdiff_t frame) {
    if (format_from_path(path) == Format::Yaml) {
        return read_yaml_file(path).atoms;
    }
    return read_extxyz(path, frame);
}

std::vector<Atoms> read_frames(const std::string& path) {
    if (format_from_path(path) == Format::Yaml) {
        return {read_yaml_file(path).atoms};
    }
    return read_extxyz_frames(path);
}

void write(const std::string& path, const Atoms& atoms, ExtxyzOptions options) {
    if (format_from_path(path) == Format::Yaml) {
        write_yaml_file(path, atoms);
        return;
    }
    write_extxyz(path, atoms, std::move(options));
}

void write(const std::string& path, const std::vector<Atoms>& frames, ExtxyzOptions options) {
    if (format_from_path(path) == Format::Yaml) {
        throw ExtxyzError(
            "io: YAML holds one structure, so a " + std::to_string(frames.size()) +
            "-frame trajectory cannot be written to '" + path +
            "'. Use an .extxyz file, which is what the format is for.");
    }
    write_extxyz(path, frames, std::move(options));
}

void append(const std::string& path, const Atoms& atoms, ExtxyzOptions options) {
    if (format_from_path(path) == Format::Yaml) {
        throw ExtxyzError("io: a YAML file cannot be appended to; use .extxyz");
    }
    options.append = true;
    write_extxyz(path, atoms, std::move(options));
}

}  // namespace fastatoms::io
