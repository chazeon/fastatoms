// SPDX-License-Identifier: MIT
#pragma once

/// \file text.hpp
/// \brief Small text helpers shared by the input writers and output parsers.

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace fastatoms::util {

/// Read a whole file into memory.
/// \throws std::runtime_error if the file cannot be opened.
[[nodiscard]] std::string read_text_file(const std::filesystem::path& path);

/// Write `text` to `path`, creating parent directories as needed.
/// \throws std::runtime_error if the file cannot be written.
void write_text_file(const std::filesystem::path& path, std::string_view text);

/// Split into lines, dropping a trailing empty line and any '\r'.
[[nodiscard]] std::vector<std::string> split_lines(std::string_view text);

/// Split on runs of whitespace.
[[nodiscard]] std::vector<std::string> split_whitespace(std::string_view text);

/// Remove leading and trailing whitespace.
[[nodiscard]] std::string_view trim(std::string_view text);

[[nodiscard]] std::string lowercase(std::string_view text);
[[nodiscard]] std::string uppercase(std::string_view text);

/// Case-insensitive substring test.
[[nodiscard]] bool contains_ignore_case(std::string_view haystack, std::string_view needle);

/// Every floating-point number in `text`, in order.
///
/// The scanner walks the string with `strtod` rather than splitting on
/// whitespace, so it also recovers values from fixed-width output where large
/// numbers run together - VASP's stress line can print `-1234.56789-1234.56789`
/// with no separator.
[[nodiscard]] std::vector<double> scan_numbers(std::string_view text);

/// Index of the last line containing `needle`, or -1.
[[nodiscard]] std::ptrdiff_t find_last_line(const std::vector<std::string>& lines,
                                            std::string_view needle);

/// Index of the first line at or after `from` containing `needle`, or -1.
[[nodiscard]] std::ptrdiff_t find_line(const std::vector<std::string>& lines,
                                       std::string_view needle, std::size_t from = 0);

/// The last `max_lines` lines of `text`, for quoting in an error message.
[[nodiscard]] std::string tail(std::string_view text, std::size_t max_lines = 20);

/// Format a double with `precision` significant digits, always with a decimal
/// point so that Fortran readers see a real rather than an integer.
[[nodiscard]] std::string format_real(double value, int precision = 12);

}  // namespace fastatoms::util
