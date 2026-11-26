// SPDX-License-Identifier: MIT
#include "fastatoms/io/format.hpp"

#include <algorithm>
#include <cctype>
#include <initializer_list>

#ifdef FASTATOMS_WITH_VASP
#  include "fastatoms/io/vasp.hpp"
#endif
#ifdef FASTATOMS_WITH_XYZ
#  include "fastatoms/io/xyz.hpp"
#endif
#ifdef FASTATOMS_WITH_CIF
#  include "fastatoms/io/cif.hpp"
#endif
#ifdef FASTATOMS_WITH_LAMMPS
#  include "fastatoms/io/lammps.hpp"
#endif

namespace fastatoms::io {
namespace {

std::string lower(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

/// Canonical name for the aliases people (and ASE) use.
std::string canonical(const std::string& format) {
    const std::string f = lower(format);
    if (f == "poscar" || f == "contcar" || f == "vasp-poscar") {
        return "vasp";
    }
    if (f == "vasp-xdatcar") {
        return "xdatcar";
    }
    if (f == "lammps" || f == "lammpsdata" || f == "lmp") {
        return "lammps-data";
    }
    if (f == "lammps-dump-text" || f == "lammpstrj" || f == "dump") {
        return "lammps-dump";
    }
    return f;
}

/// Reject option keys a format does not know, so a typo is not silently
/// ignored. "index" is consumed by `read` itself.
[[maybe_unused]] void check_keys(const Value& options, const std::string& format,
                std::initializer_list<std::string_view> known) {
    if (options.is_null()) {
        return;
    }
    if (!options.is_map()) {
        throw FormatError("options for '" + format + "' must be a mapping");
    }
    for (const auto& [key, value] : options.as_map()) {
        if (key == "index") {
            continue;
        }
        if (std::find(known.begin(), known.end(), key) == known.end()) {
            throw FormatError("unknown option '" + key + "' for format '" + format + "'");
        }
    }
}

[[maybe_unused]] bool get_bool(const Value& options, std::string_view key, bool fallback) {
    const Value* v = options.is_map() ? options.find(key) : nullptr;
    return v == nullptr ? fallback : v->value_or(fallback);
}

[[maybe_unused]] std::string get_string(const Value& options, std::string_view key,
                                        std::string fallback) {
    const Value* v = options.is_map() ? options.find(key) : nullptr;
    return v == nullptr || v->is_null() ? fallback : v->as_string();
}

[[maybe_unused]] std::vector<std::string> get_strings(const Value& options, std::string_view key) {
    const Value* v = options.is_map() ? options.find(key) : nullptr;
    if (v == nullptr || v->is_null()) {
        return {};
    }
    if (v->is_string()) {
        // "Fe O" is accepted as shorthand for ["Fe", "O"].
        std::vector<std::string> out;
        std::string token;
        for (const char c : v->as_string()) {
            if (std::isspace(static_cast<unsigned char>(c)) != 0 || c == ',') {
                if (!token.empty()) {
                    out.push_back(token);
                    token.clear();
                }
            } else {
                token.push_back(c);
            }
        }
        if (!token.empty()) {
            out.push_back(token);
        }
        return out;
    }
    return v->as_strings();
}

#ifdef FASTATOMS_WITH_LAMMPS
LammpsTypeMap get_types(const Value& options) {
    LammpsTypeMap types;
    const Value* v = options.is_map() ? options.find("types") : nullptr;
    if (v == nullptr || v->is_null()) {
        return types;
    }
    if (v->is_list() || v->is_string()) {
        const auto labels = get_strings(options, "types");
        for (std::size_t k = 0; k < labels.size(); ++k) {
            types[static_cast<int>(k) + 1] = labels[k];
        }
        return types;
    }
    for (const auto& [key, label] : v->as_map()) {
        int type = 0;
        try {
            type = std::stoi(key);
        } catch (const std::exception&) {
            throw FormatError("LAMMPS 'types' keys must be integers, got '" + key + "'");
        }
        types[type] = label.as_string();
    }
    return types;
}
#endif

std::vector<Atoms> dispatch_read(std::string_view text, const std::string& format,
                                 const Value& options) {
#ifdef FASTATOMS_WITH_VASP
    if (format == "vasp" || format == "xdatcar") {
        check_keys(options, format, {"species"});
        PoscarReadOptions o;
        o.species = get_strings(options, "species");
        if (format == "xdatcar") {
            return read_xdatcar(text, o);
        }
        return {read_poscar(text, o)};
    }
#endif
#ifdef FASTATOMS_WITH_XYZ
    if (format == "xyz" || format == "extxyz") {
        check_keys(options, format, {});
        return read_xyz(text);
    }
#endif
#ifdef FASTATOMS_WITH_CIF
    if (format == "cif") {
        check_keys(options, format, {"disorder", "merge_tolerance"});
        CifReadOptions o;
        const std::string disorder = get_string(options, "disorder", "keep_all");
        if (disorder == "keep_all") {
            o.disorder = CifDisorder::KeepAll;
        } else if (disorder == "max_occupancy") {
            o.disorder = CifDisorder::MaxOccupancy;
        } else if (disorder == "error") {
            o.disorder = CifDisorder::Error;
        } else {
            throw FormatError("cif: 'disorder' must be keep_all, max_occupancy or error");
        }
        if (const Value* tol = options.is_map() ? options.find("merge_tolerance") : nullptr) {
            o.merge_tolerance = tol->as_double();
        }
        return read_cif(text, o);
    }
#endif
#ifdef FASTATOMS_WITH_LAMMPS
    if (format == "lammps-data") {
        check_keys(options, format, {"atom_style", "types", "unwrap"});
        LammpsDataReadOptions o;
        o.atom_style = get_string(options, "atom_style", "");
        o.types = get_types(options);
        o.unwrap = get_bool(options, "unwrap", false);
        return {read_lammps_data(text, o)};
    }
    if (format == "lammps-dump") {
        check_keys(options, format, {"types", "unwrap"});
        LammpsDumpReadOptions o;
        o.types = get_types(options);
        o.unwrap = get_bool(options, "unwrap", false);
        return read_lammps_dump(text, o);
    }
#endif
    (void)text;
    (void)options;
    throw FormatError("format '" + format + "' is not available in this build");
}

}  // namespace

std::vector<Atoms> read_all(std::string_view text, const std::string& format,
                            const Value& options) {
    return dispatch_read(text, canonical(format), options);
}

Atoms read(std::string_view text, const std::string& format, const Value& options) {
    std::vector<Atoms> frames = read_all(text, format, options);
    if (frames.empty()) {
        throw FormatError("no structure found in the input");
    }
    std::int64_t index = -1;
    if (const Value* v = options.is_map() ? options.find("index") : nullptr) {
        index = v->as_int();
    }
    const auto n = static_cast<std::int64_t>(frames.size());
    const std::int64_t k = index < 0 ? n + index : index;
    if (k < 0 || k >= n) {
        throw FormatError("frame index " + std::to_string(index) + " is out of range for " +
                          std::to_string(n) + " frames");
    }
    return std::move(frames[static_cast<std::size_t>(k)]);
}

std::string write(const Atoms& atoms, const std::string& format_in, const Value& options) {
    const std::string format = canonical(format_in);
#ifdef FASTATOMS_WITH_VASP
    if (format == "vasp") {
        check_keys(options, format, {"direct", "labels", "vasp5", "sort", "comment"});
        PoscarWriteOptions o;
        o.direct = get_bool(options, "direct", o.direct);
        o.write_labels = get_bool(options, "labels", o.write_labels);
        o.vasp5 = get_bool(options, "vasp5", o.vasp5);
        o.sort = get_bool(options, "sort", o.sort);
        o.comment = get_string(options, "comment", "");
        return write_poscar(atoms, o);
    }
#endif
#ifdef FASTATOMS_WITH_XYZ
    if (format == "xyz" || format == "extxyz") {
        check_keys(options, format, {"precision"});
        XyzWriteOptions o;
        o.extended = format == "extxyz";
        if (const Value* p = options.is_map() ? options.find("precision") : nullptr) {
            o.precision = static_cast<int>(p->as_int());
        }
        return write_xyz(atoms, o);
    }
#endif
#ifdef FASTATOMS_WITH_CIF
    if (format == "cif") {
        check_keys(options, format, {});
        return write_cif(atoms);
    }
#endif
#ifdef FASTATOMS_WITH_LAMMPS
    if (format == "lammps-data") {
        check_keys(options, format, {"atom_style", "specorder", "masses", "force_skew"});
        LammpsDataWriteOptions o;
        o.atom_style = get_string(options, "atom_style", o.atom_style);
        o.specorder = get_strings(options, "specorder");
        o.masses = get_bool(options, "masses", o.masses);
        o.force_skew = get_bool(options, "force_skew", o.force_skew);
        return write_lammps_data(atoms, o);
    }
#endif
    (void)atoms;
    (void)options;
    throw FormatError("format '" + format + "' cannot be written in this build");
}

std::string guess_format(std::string_view filename, std::string_view head) {
    std::string name = lower(filename);
    if (const auto slash = name.find_last_of("/\\"); slash != std::string::npos) {
        name = name.substr(slash + 1);
    }
    const auto ends_with = [&name](std::string_view suffix) {
        return name.size() >= suffix.size() &&
               name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    const auto has = [&name](std::string_view part) {
        return name.find(part) != std::string::npos;
    };
    std::string guess;
    if (has("xdatcar")) {
        guess = "xdatcar";
    } else if (has("poscar") || has("contcar") || ends_with(".vasp")) {
        guess = "vasp";
    } else if (ends_with(".extxyz")) {
        guess = "extxyz";
    } else if (ends_with(".xyz")) {
        guess = "extxyz";  // the extxyz reader also reads plain XYZ
    } else if (ends_with(".cif")) {
        guess = "cif";
    } else if (ends_with(".lammpstrj") || ends_with(".dump") || has("dump.")) {
        guess = "lammps-dump";
    } else if (ends_with(".data") || ends_with(".lmp") || has("data.")) {
        guess = "lammps-data";
    }
    if (guess.empty() && !head.empty()) {
        if (head.find("ITEM: TIMESTEP") != std::string_view::npos) {
            guess = "lammps-dump";
        } else if (head.find("data_") != std::string_view::npos &&
                   head.find("_cell_length_a") != std::string_view::npos) {
            guess = "cif";
        } else if (head.find("atoms") != std::string_view::npos &&
                   head.find("xlo xhi") != std::string_view::npos) {
            guess = "lammps-data";
        }
    }
    const auto available = available_formats();
    if (!guess.empty() &&
        std::find(available.begin(), available.end(), guess) == available.end()) {
        return {};
    }
    return guess;
}

std::vector<std::string> available_formats() {
    std::vector<std::string> out;
#ifdef FASTATOMS_WITH_VASP
    out.insert(out.end(), {"vasp", "xdatcar"});
#endif
#ifdef FASTATOMS_WITH_XYZ
    out.insert(out.end(), {"xyz", "extxyz"});
#endif
#ifdef FASTATOMS_WITH_CIF
    out.emplace_back("cif");
#endif
#ifdef FASTATOMS_WITH_LAMMPS
    out.insert(out.end(), {"lammps-data", "lammps-dump"});
#endif
    return out;
}

bool can_write(const std::string& format_in) {
    const std::string format = canonical(format_in);
    for (const char* writable : {"vasp", "xyz", "extxyz", "cif", "lammps-data"}) {
        if (format == writable) {
            const auto available = available_formats();
            return std::find(available.begin(), available.end(), format) != available.end();
        }
    }
    return false;
}

}  // namespace fastatoms::io
