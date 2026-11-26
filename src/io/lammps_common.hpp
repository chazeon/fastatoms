// SPDX-License-Identifier: MIT
#pragma once

/// \file lammps_common.hpp
/// \brief Private helpers shared by the LAMMPS data and dump implementations.

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "fastatoms/core/cell.hpp"
#include "fastatoms/core/elements.hpp"
#include "fastatoms/core/vec3.hpp"
#include "fastatoms/io/format.hpp"

namespace fastatoms::io::lammps_common {

inline bool is_blank(char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; }

inline std::string_view trim(std::string_view s) {
    while (!s.empty() && is_blank(s.front())) {
        s.remove_prefix(1);
    }
    while (!s.empty() && is_blank(s.back())) {
        s.remove_suffix(1);
    }
    return s;
}

/// The line up to the first '#'.
inline std::string_view without_comment(std::string_view line) {
    return line.substr(0, line.find('#'));
}

/// The trimmed text after the first '#', or "" when there is none.
inline std::string_view comment_of(std::string_view line) {
    const auto hash = line.find('#');
    return hash == std::string_view::npos ? std::string_view{} : trim(line.substr(hash + 1));
}

/// Whitespace-separated tokens.
inline std::vector<std::string_view> fields(std::string_view line) {
    std::vector<std::string_view> out;
    std::size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && is_blank(line[i])) {
            ++i;
        }
        std::size_t j = i;
        while (j < line.size() && !is_blank(line[j])) {
            ++j;
        }
        if (j > i) {
            out.push_back(line.substr(i, j - i));
        }
        i = j;
    }
    return out;
}

/// Split on '\n', dropping a trailing '\r' from each line.
inline std::vector<std::string_view> split_lines(std::string_view text) {
    std::vector<std::string_view> lines;
    std::size_t start = 0;
    while (start <= text.size()) {
        std::size_t end = text.find('\n', start);
        if (end == std::string_view::npos) {
            end = text.size();
        }
        std::string_view line = text.substr(start, end - start);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        lines.push_back(line);
        if (end == text.size()) {
            break;
        }
        start = end + 1;
    }
    return lines;
}

/// Throws with the 1-based line number, or without one when it is unknown (0).
[[noreturn]] inline void line_error(std::size_t line_number, const std::string& what) {
    if (line_number == 0) {
        throw FormatError("LAMMPS: " + what);
    }
    throw FormatError("LAMMPS line " + std::to_string(line_number) + ": " + what);
}

inline double parse_double(std::string_view token, std::size_t line_number) {
    const std::string s(token);
    char* end = nullptr;
    const double value = std::strtod(s.c_str(), &end);
    if (s.empty() || end != s.c_str() + s.size()) {
        line_error(line_number, "expected a number, got '" + s + "'");
    }
    return value;
}

/// True when the whole token is a decimal integer; stores it in `value`.
inline bool try_parse_int(std::string_view token, std::int64_t& value) {
    std::int64_t parsed = 0;
    const char* first = token.data();
    const char* last = token.data() + token.size();
    const auto [ptr, ec] = std::from_chars(first, last, parsed);
    if (token.empty() || ec != std::errc{} || ptr != last) {
        return false;
    }
    value = parsed;
    return true;
}

inline std::int64_t parse_int(std::string_view token, std::size_t line_number) {
    std::int64_t value = 0;
    if (!try_parse_int(token, value)) {
        line_error(line_number, "expected an integer, got '" + std::string(token) + "'");
    }
    return value;
}

/// Drop a leading UTF-8 byte order mark, which some editors write.
inline std::string_view strip_bom(std::string_view text) {
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBB &&
        static_cast<unsigned char>(text[2]) == 0xBF) {
        text.remove_prefix(3);
    }
    return text;
}

/// Shortest decimal text that reads back to the same double.
inline std::string format_double(double value) {
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof buffer, value);
    return std::string(buffer, result.ptr);
}

/// Element of a species label ("Fe1", "Fe_pv", "Fe2"), or FormatError naming
/// the line (0 when unknown).
inline int element_of_label(std::string_view label, std::size_t line_number = 0) {
    const int z = try_number_from_label(label);
    if (z < 0) {
        line_error(line_number, "cannot tell which element the label '" + std::string(label) +
                                    "' names");
    }
    return z;
}

/// Whether a stated mass is compatible with element `z`: within 1 u, or within
/// 5 % of the tabulated weight for heavier elements.
inline bool mass_agrees(int z, double mass) {
    const double reference = atomic_mass(z);
    return std::abs(reference - mass) <= std::max(1.0, 0.05 * reference);
}

/// Element whose standard atomic weight is nearest `mass` (amu), provided it
/// lies within `tolerance`; -1 otherwise.
inline int element_from_mass(double mass, double tolerance = 0.1) {
    int best = -1;
    double best_distance = std::numeric_limits<double>::infinity();
    for (int z = 1; z <= kMaxAtomicNumber; ++z) {
        const double distance = std::abs(atomic_mass(z) - mass);
        if (distance <= tolerance && distance < best_distance) {
            best = z;
            best_distance = distance;
        }
    }
    return best;
}

/// Restricted-triclinic cell: a = (lx,0,0), b = (xy,ly,0), c = (xz,yz,lz).
inline Cell restricted_cell(double lx, double ly, double lz, double xy, double xz, double yz) {
    return Cell(Mat3::from_rows({lx, 0.0, 0.0}, {xy, ly, 0.0}, {xz, yz, lz}), {true, true, true});
}

}  // namespace fastatoms::io::lammps_common
