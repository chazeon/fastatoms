// SPDX-License-Identifier: MIT
#include "fastatoms/util/text.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fastatoms::util {

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open '" + path.string() + "' for reading");
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

void write_text_file(const std::filesystem::path& path, std::string_view text) {
    if (path.has_parent_path() && !path.parent_path().empty()) {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot open '" + path.string() + "' for writing");
    }
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!stream) {
        throw std::runtime_error("failed while writing '" + path.string() + "'");
    }
}

std::vector<std::string> split_lines(std::string_view text) {
    std::vector<std::string> lines;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = text.find('\n', start);
        const std::size_t stop = end == std::string_view::npos ? text.size() : end;
        std::string_view line = text.substr(start, stop - start);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        lines.emplace_back(line);
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    // A trailing newline produces one empty line that carries no information.
    if (!lines.empty() && lines.back().empty()) {
        lines.pop_back();
    }
    return lines;
}

std::vector<std::string> split_whitespace(std::string_view text) {
    std::vector<std::string> parts;
    std::size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) != 0) {
            ++i;
        }
        const std::size_t start = i;
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) == 0) {
            ++i;
        }
        if (i > start) {
            parts.emplace_back(text.substr(start, i - start));
        }
    }
    return parts;
}

std::string_view trim(std::string_view text) {
    std::size_t start = 0;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start])) != 0) {
        ++start;
    }
    std::size_t stop = text.size();
    while (stop > start && std::isspace(static_cast<unsigned char>(text[stop - 1])) != 0) {
        --stop;
    }
    return text.substr(start, stop - start);
}

std::string lowercase(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

std::string uppercase(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return out;
}

bool contains_ignore_case(std::string_view haystack, std::string_view needle) {
    if (needle.empty()) {
        return true;
    }
    const std::string lower_haystack = lowercase(haystack);
    const std::string lower_needle = lowercase(needle);
    return lower_haystack.find(lower_needle) != std::string::npos;
}

std::vector<double> scan_numbers(std::string_view text) {
    std::vector<double> values;
    // strtod needs a NUL-terminated buffer, and it also stops at the second
    // sign in a run such as "-1234.5-6789.0", which is exactly the recovery we
    // want for fixed-width Fortran output.
    const std::string buffer(text);
    const char* cursor = buffer.c_str();
    const char* const end = cursor + buffer.size();

    while (cursor < end) {
        // Skip to something that could start a number.
        while (cursor < end && std::isdigit(static_cast<unsigned char>(*cursor)) == 0 &&
               *cursor != '+' && *cursor != '-' && *cursor != '.') {
            ++cursor;
        }
        if (cursor >= end) {
            break;
        }
        char* stop = nullptr;
        const double value = std::strtod(cursor, &stop);
        if (stop == cursor) {
            ++cursor;  // a lone sign or dot
            continue;
        }
        values.push_back(value);
        cursor = stop;
    }
    return values;
}

std::ptrdiff_t find_last_line(const std::vector<std::string>& lines, std::string_view needle) {
    for (std::size_t k = lines.size(); k > 0; --k) {
        if (lines[k - 1].find(needle) != std::string::npos) {
            return static_cast<std::ptrdiff_t>(k - 1);
        }
    }
    return -1;
}

std::ptrdiff_t find_line(const std::vector<std::string>& lines, std::string_view needle,
                         std::size_t from) {
    for (std::size_t k = from; k < lines.size(); ++k) {
        if (lines[k].find(needle) != std::string::npos) {
            return static_cast<std::ptrdiff_t>(k);
        }
    }
    return -1;
}

std::string tail(std::string_view text, std::size_t max_lines) {
    const std::vector<std::string> lines = split_lines(text);
    const std::size_t start = lines.size() > max_lines ? lines.size() - max_lines : 0;
    std::string out;
    for (std::size_t k = start; k < lines.size(); ++k) {
        out += lines[k];
        out += '\n';
    }
    return out;
}

std::string format_real(double value, int precision) {
    if (std::isnan(value)) {
        return "NaN";
    }
    std::ostringstream out;
    out << std::setprecision(precision) << value;
    std::string text = out.str();
    // Fortran namelists and fdf files want a decimal point on real values.
    if (text.find_first_of(".eEnN") == std::string::npos) {
        text += ".0";
    }
    return text;
}

}  // namespace fastatoms::util
