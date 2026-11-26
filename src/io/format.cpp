// SPDX-License-Identifier: MIT
#include "fastatoms/io/format.hpp"

#include "fastatoms/core/elements.hpp"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <initializer_list>
#include <system_error>

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

/// An integer option. Only a Value holding an integer, or a float with an exact
/// integer value that fits in int64, is accepted. Anything else is a FormatError.
[[maybe_unused]] std::int64_t get_integer(const Value& options, std::string_view key) {
    const Value* v = options.is_map() ? options.find(key) : nullptr;
    if (v == nullptr) {
        throw FormatError("option '" + std::string(key) + "' is missing");
    }
    if (v->is_int()) {
        return v->as_int();
    }
    // [-2^63, 2^63) is the range a double converts to int64 without overflow.
    if (v->is_float()) {
        const double number = v->as_double();
        if (std::isfinite(number) && number == std::trunc(number) &&
            number >= -9223372036854775808.0 && number < 9223372036854775808.0) {
            return static_cast<std::int64_t>(number);
        }
    }
    throw FormatError("option '" + std::string(key) + "' must be an integer");
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
        // The whole key must be an integer: "1abc" is not type 1.
        std::string_view digits = key;
        if (!digits.empty() && digits.front() == '+') {
            digits.remove_prefix(1);
        }
        int type = 0;
        const char* last = digits.data() + digits.size();
        const auto [ptr, ec] = std::from_chars(digits.data(), last, type);
        if (digits.empty() || ec != std::errc{} || ptr != last) {
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
        // "xyz" is plain XYZ: the comment line is never parsed. "extxyz" parses it.
        XyzReadOptions o;
        o.extended = format == "extxyz";
        return read_xyz(text, o);
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
        check_keys(options, format, {"atom_style", "types", "unwrap", "units"});
        LammpsDataReadOptions o;
        o.atom_style = get_string(options, "atom_style", "");
        o.types = get_types(options);
        o.unwrap = get_bool(options, "unwrap", false);
        o.units = get_string(options, "units", "metal");
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
    if (options.is_map() && options.find("index") != nullptr) {
        index = get_integer(options, "index");
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
        if (options.is_map() && options.find("precision") != nullptr) {
            // Range-checked before the narrowing cast, so no value wraps into range.
            const std::int64_t precision = get_integer(options, "precision");
            if (precision < 0 || precision > 17) {
                throw FormatError("xyz: precision must be 0 (shortest) or 1 to 17, got " +
                                  std::to_string(precision));
            }
            o.precision = static_cast<int>(precision);
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

bool can_write_frames(const std::string& format_in) {
    const std::string format = canonical(format_in);
    return (format == "xyz" || format == "extxyz" || format == "cif") && can_write(format);
}

std::string write_all(const std::vector<Atoms>& frames, const std::string& format_in,
                      const Value& options) {
    const std::string format = canonical(format_in);
    if (frames.empty()) {
        throw FormatError("no frames to write");
    }
    if (frames.size() == 1) {
        return write(frames.front(), format, options);
    }
    if (!can_write_frames(format)) {
        throw FormatError("format '" + format + "' holds one structure; got " +
                          std::to_string(frames.size()) + " frames");
    }
#ifdef FASTATOMS_WITH_CIF
    if (format == "cif") {
        check_keys(options, format, {});
        return write_cif(frames);
    }
#endif
    // xyz/extxyz: frames are simply concatenated.
    std::string out;
    for (const Atoms& frame : frames) {
        out += write(frame, format, options);
    }
    return out;
}

namespace {

/// Trims ASCII whitespace from both ends.
std::string_view trim_view(std::string_view s) {
    const auto space = [](char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; };
    while (!s.empty() && space(s.front())) {
        s.remove_prefix(1);
    }
    while (!s.empty() && space(s.back())) {
        s.remove_suffix(1);
    }
    return s;
}

/// Lines of `text`, without their terminators (a trailing '\r' is dropped).
std::vector<std::string_view> lines_of(std::string_view text) {
    std::vector<std::string_view> lines;
    while (!text.empty()) {
        const auto nl = text.find('\n');
        std::string_view line = text.substr(0, nl);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        lines.push_back(line);
        if (nl == std::string_view::npos) {
            break;
        }
        text.remove_prefix(nl + 1);
    }
    return lines;
}

/// Whitespace-separated tokens of `line`.
std::vector<std::string_view> tokens_of(std::string_view line) {
    std::vector<std::string_view> tokens;
    while (true) {
        line = trim_view(line);
        if (line.empty()) {
            return tokens;
        }
        const auto end = std::min(line.find_first_of(" \t"), line.size());
        tokens.push_back(line.substr(0, end));
        line.remove_prefix(end);
    }
}

bool is_real(std::string_view token) {
    double value = 0.0;
    const char* last = token.data() + token.size();
    const auto [ptr, ec] = std::from_chars(token.data(), last, value);
    return !token.empty() && ec == std::errc{} && ptr == last;
}

/// LAMMPS dump magic, as ASE matches it: a line that is exactly "ITEM: TIMESTEP".
bool has_lammps_dump_magic(const std::vector<std::string_view>& lines) {
    return std::any_of(lines.begin(), lines.end(), [](std::string_view line) {
        return trim_view(line) == "ITEM: TIMESTEP";
    });
}

bool has_cif_signature(std::string_view head) {
    return head.find("data_") != std::string_view::npos &&
           head.find("_cell_length") != std::string_view::npos;
}

/// A LAMMPS data header: an "atoms" count line and a "<lo> <hi> xlo xhi" box line.
bool has_lammps_data_signature(const std::vector<std::string_view>& lines) {
    bool atoms = false;
    bool box = false;
    for (const std::string_view line : lines) {
        atoms = atoms || line.find("atoms") != std::string_view::npos;
        const std::string_view t = trim_view(line);
        box = box || (t.size() >= 7 && t.substr(t.size() - 7) == " xlo xhi");
    }
    return atoms && box;
}

/// The plain-XYZ shape: the first non-blank line is an atom count, and the
/// third line (counted from that one) is an atom row, a label then three numbers.
/// Only consulted for input with no name or extension, as ASE does.
bool looks_like_xyz(std::string_view head) {
    const auto lines = lines_of(head);
    std::size_t i = 0;
    while (i < lines.size() && trim_view(lines[i]).empty()) {
        ++i;
    }
    if (i + 2 >= lines.size()) {
        return false;
    }
    const std::string_view count = trim_view(lines[i]);
    if (!std::all_of(count.begin(), count.end(),
                     [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; })) {
        return false;
    }
    const auto row = tokens_of(lines[i + 2]);
    return row.size() >= 4 && try_number_from_label(row[0]) >= 0 && is_real(row[1]) &&
           is_real(row[2]) && is_real(row[3]);
}

}  // namespace

std::string guess_format(std::string_view filename, std::string_view head) {
    std::string name = lower(filename);
    if (const auto slash = name.find_last_of("/\\"); slash != std::string::npos) {
        name = name.substr(slash + 1);
    }
    // A compression suffix is transparent: "traj.xyz.gz" is an XYZ file.
    for (const char* compression : {".gz", ".bz2", ".xz"}) {
        const std::string_view suffix(compression);
        if (name.size() > suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            name.resize(name.size() - suffix.size());
            break;
        }
    }
    const auto has = [&name](std::string_view part) {
        return name.find(part) != std::string::npos;
    };
    // The extension of the basename, if it has one.
    const auto dot = name.find_last_of('.');
    const bool has_ext = dot != std::string::npos;
    const std::string ext = has_ext ? name.substr(dot + 1) : std::string();

    // ASE order: special file names, then content signatures, then the extension,
    // then an XYZ sniff for input that has neither a name nor an extension.
    std::string guess;
    if (has("xdatcar")) {
        guess = "xdatcar";
    } else if (has("poscar") || has("contcar") || has("centcar")) {
        guess = "vasp";
    } else if (!head.empty()) {
        const auto lines = lines_of(head);
        if (has_lammps_dump_magic(lines)) {
            guess = "lammps-dump";
        } else if (has_cif_signature(head)) {
            guess = "cif";
        } else if (has_lammps_data_signature(lines)) {
            guess = "lammps-data";
        }
    }
    if (guess.empty() && has_ext) {
        if (ext == "xyz" || ext == "extxyz") {
            guess = "extxyz";  // the extxyz reader also reads plain XYZ
        } else if (ext == "cif") {
            guess = "cif";
        } else if (ext == "vasp" || ext == "poscar") {
            guess = "vasp";
        } else if (ext == "lammpstrj" || ext == "dump") {
            guess = "lammps-dump";
        } else if (ext == "data" || ext == "lmp") {
            guess = "lammps-data";
        }
    }
    if (guess.empty()) {
        // LAMMPS naming convention without an extension: dump.1000, data.box.
        if (name.rfind("dump.", 0) == 0) {
            guess = "lammps-dump";
        } else if (name.rfind("data.", 0) == 0) {
            guess = "lammps-data";
        }
    }
    if (guess.empty() && !has_ext && looks_like_xyz(head)) {
        guess = "extxyz";
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
