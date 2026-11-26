// SPDX-License-Identifier: MIT
#include "fastatoms/io/lammps.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "fastatoms/core/cell.hpp"
#include "fastatoms/core/elements.hpp"
#include "lammps_common.hpp"

namespace fastatoms::io {
namespace {

using namespace lammps_common;

/// The box of one frame, already converted to the origin-shifted cell.
struct FrameBox {
    Mat3 matrix{};
    Vec3 origin{};
    std::array<bool, 3> pbc{false, false, false};
};

bool is_tilt_name(std::string_view s) { return s == "xy" || s == "xz" || s == "yz"; }

/// True when no line from `i` on has any token.
bool blank_from(const std::vector<std::string_view>& lines, std::size_t i) {
    for (; i < lines.size(); ++i) {
        if (!fields(lines[i]).empty()) {
            return false;
        }
    }
    return true;
}

/// Parse "ITEM: BOX BOUNDS ..." (line index `i`) and the three lines after it.
/// On return `i` is the last of those lines. Nullopt when the input ends first.
std::optional<FrameBox> read_box(const std::vector<std::string_view>& lines, std::size_t& i) {
    const auto header = fields(lines[i]);
    std::vector<std::string_view> tail(header.begin() + std::min<std::size_t>(3, header.size()),
                                       header.end());
    FrameBox box;
    if (tail.size() >= 3) {
        const std::size_t k = tail.size() - 3;
        for (std::size_t d = 0; d < 3; ++d) {
            // "pp", "ff", "fs", "ss" or "mm" per axis: periodic when it has a 'p'.
            box.pbc[d] = tail[k + d].find_first_of("pP") != std::string_view::npos;
        }
    }

    std::vector<std::vector<std::string_view>> rows;
    std::vector<std::size_t> row_lines;
    for (std::size_t r = 0; r < 3; ++r) {
        // Blank lines do not count as rows; running out of input is EOF.
        while (i + 1 < lines.size() && fields(lines[i + 1]).empty()) {
            ++i;
        }
        if (i + 1 >= lines.size()) {
            return std::nullopt;
        }
        ++i;
        rows.push_back(fields(lines[i]));
        row_lines.push_back(i + 1);
    }

    const bool general = tail.size() >= 2 && tail[0] == "abc" && tail[1] == "origin";
    if (general) {
        Mat3 m{};
        Vec3 origin{};
        for (std::size_t r = 0; r < 3; ++r) {
            if (rows[r].size() < 4) {
                line_error(row_lines[r], "general triclinic BOX BOUNDS rows need 4 numbers");
            }
            m.set_row(r, Vec3(parse_double(rows[r][0], row_lines[r]),
                              parse_double(rows[r][1], row_lines[r]),
                              parse_double(rows[r][2], row_lines[r])));
            origin[r] = parse_double(rows[r][3], row_lines[r]);
        }
        box.matrix = m;
        box.origin = origin;
        return box;
    }

    for (std::size_t r = 0; r < 3; ++r) {
        if (rows[r].size() < 2) {
            line_error(row_lines[r], "BOX BOUNDS rows need lo and hi");
        }
    }
    double lo[3], hi[3];
    for (std::size_t d = 0; d < 3; ++d) {
        lo[d] = parse_double(rows[d][0], row_lines[d]);
        hi[d] = parse_double(rows[d][1], row_lines[d]);
    }

    const bool tilted = tail.size() >= 6 && is_tilt_name(tail[0]) && is_tilt_name(tail[1]) &&
                        is_tilt_name(tail[2]);
    if (!tilted) {
        box.origin = Vec3(lo[0], lo[1], lo[2]);
        box.matrix = restricted_cell(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 0.0, 0.0, 0.0)
                         .matrix();
        return box;
    }

    // Tilt factors: row r carries the tilt named by header label r.
    double tilt_xy = 0.0, tilt_xz = 0.0, tilt_yz = 0.0;
    for (std::size_t r = 0; r < 3; ++r) {
        if (rows[r].size() < 3) {
            line_error(row_lines[r], "tilted BOX BOUNDS rows need lo, hi and a tilt");
        }
        const double tilt = parse_double(rows[r][2], row_lines[r]);
        if (tail[r] == "xy") {
            tilt_xy = tilt;
        } else if (tail[r] == "xz") {
            tilt_xz = tilt;
        } else {
            tilt_yz = tilt;
        }
    }

    // LAMMPS writes the bounds that enclose the tilted box (see
    // https://docs.lammps.org/dump.html); undo them.
    const double xlo = lo[0] - std::min({0.0, tilt_xy, tilt_xz, tilt_xy + tilt_xz});
    const double xhi = hi[0] - std::max({0.0, tilt_xy, tilt_xz, tilt_xy + tilt_xz});
    const double ylo = lo[1] - std::min(0.0, tilt_yz);
    const double yhi = hi[1] - std::max(0.0, tilt_yz);
    const double zlo = lo[2];
    const double zhi = hi[2];
    box.origin = Vec3(xlo, ylo, zlo);
    box.matrix = restricted_cell(xhi - xlo, yhi - ylo, zhi - zlo, tilt_xy, tilt_xz, tilt_yz)
                     .matrix();
    return box;
}

/// Parse one frame starting at the `ITEM: TIMESTEP` line `i`. On return `i`
/// is the first line of the next frame, or the end of input.
///
/// Nullopt when the input ends inside the frame before its ITEM: ATOMS block
/// is complete: a truncated frame, which the caller drops.
std::optional<Atoms> read_frame(const std::vector<std::string_view>& lines, std::size_t& i,
                                const LammpsDumpReadOptions& options, std::size_t& cut_line) {
    const std::size_t start_line = i + 1;
    std::int64_t timestep = 0;
    std::int64_t natoms = -1;
    std::optional<FrameBox> box;
    std::vector<std::string> names;
    std::vector<std::vector<std::string_view>> columns;
    std::vector<std::size_t> row_lines;  // 1-based line of each atom row
    bool have_atoms = false;
    bool have_natoms = false;
    bool have_box_block = false;
    bool started = false;
    std::size_t item_line = start_line;  // the last ITEM: line, for error messages

    // The input ends inside this frame: report the last block it reached.
    const auto cut = [&]() -> std::optional<Atoms> {
        cut_line = item_line;
        return std::nullopt;
    };

    // Skip blank lines; false at the end of input.
    const auto skip_blank = [&]() {
        while (i < lines.size() && fields(lines[i]).empty()) {
            ++i;
        }
        return i < lines.size();
    };

    while (i < lines.size()) {
        const std::size_t lineno = i + 1;
        const auto f = fields(lines[i]);
        if (f.empty()) {
            ++i;
            continue;
        }
        if (f[0] != "ITEM:") {
            ++i;  // stray text between blocks
            continue;
        }
        const std::string_view item = f.size() > 1 ? f[1] : std::string_view{};
        item_line = lineno;
        if (item == "TIMESTEP") {
            if (started) {
                break;  // next frame
            }
            started = true;
            ++i;
            if (!skip_blank()) {
                return cut();
            }
            timestep = parse_int(fields(lines[i]).front(), i + 1);
            ++i;
        } else if (item == "NUMBER" && f.size() > 2 && f[2] == "OF") {
            if (have_natoms) {
                line_error(lineno, "repeated ITEM: NUMBER OF ATOMS in one frame");
            }
            have_natoms = true;
            ++i;
            if (!skip_blank()) {
                return cut();
            }
            natoms = parse_int(fields(lines[i]).front(), i + 1);
            if (natoms < 0) {
                line_error(i + 1, "NUMBER OF ATOMS is negative");
            }
            ++i;
        } else if (item == "BOX" && f.size() > 2 && f[2] == "BOUNDS") {
            if (have_box_block) {
                line_error(lineno, "repeated ITEM: BOX BOUNDS in one frame");
            }
            have_box_block = true;
            auto parsed = read_box(lines, i);
            if (!parsed) {
                return cut();
            }
            box = *parsed;
            ++i;
        } else if (item == "ATOMS") {
            if (natoms < 0) {
                line_error(lineno, "ITEM: ATOMS before NUMBER OF ATOMS");
            }
            if (have_atoms) {
                line_error(lineno, "repeated ITEM: ATOMS block in one frame");
            }
            names.clear();
            for (std::size_t k = 2; k < f.size(); ++k) {
                names.emplace_back(f[k]);
            }
            // Every row takes a line, so a count above the lines left cannot
            // be finished: the input ends inside this frame. Checked as 64-bit
            // before any narrowing to size_t.
            if (static_cast<std::uint64_t>(natoms) > lines.size() - (i + 1)) {
                return cut();
            }
            const auto want = static_cast<std::size_t>(natoms);
            // Rows are appended only once validated, so memory follows the
            // input actually read, never the header's shape alone.
            columns.assign(names.size(), std::vector<std::string_view>{});
            row_lines.clear();
            ++i;
            std::size_t read = 0;
            while (read < want) {
                if (i >= lines.size()) {
                    return cut();
                }
                const auto row = fields(lines[i]);
                const std::size_t row_line = i + 1;
                ++i;
                if (row.empty()) {
                    continue;
                }
                if (row.size() != names.size()) {
                    // A short last line is a row cut off by EOF; anything else is malformed.
                    if (row.size() < names.size() && blank_from(lines, i)) {
                        return cut();
                    }
                    line_error(row_line, "expected " + std::to_string(names.size()) +
                                             " columns, got " + std::to_string(row.size()));
                }
                for (std::size_t k = 0; k < row.size(); ++k) {
                    columns[k].push_back(row[k]);
                }
                row_lines.push_back(row_line);
                ++read;
            }
            have_atoms = true;
        } else {
            ++i;  // unknown ITEM block
        }
    }

    if (!have_atoms) {
        if (i >= lines.size()) {
            return cut();
        }
        line_error(start_line, "frame has no ITEM: ATOMS block");
    }
    if (!box) {
        line_error(start_line, "frame has no ITEM: BOX BOUNDS block");
    }

    const auto n = static_cast<std::size_t>(natoms);
    for (const auto& column : columns) {
        if (column.size() != n) {
            line_error(start_line, "ITEM: ATOMS column length differs from NUMBER OF ATOMS");
        }
    }
    const auto column_index = [&names](const std::string& name) -> int {
        const auto it = std::find(names.begin(), names.end(), name);
        return it == names.end() ? -1 : static_cast<int>(it - names.begin());
    };
    const auto has_all = [&](std::initializer_list<const char*> cols) {
        return std::all_of(cols.begin(), cols.end(),
                           [&](const char* c) { return column_index(c) >= 0; });
    };
    const auto col = [&](const std::string& name) -> const std::vector<std::string_view>& {
        return columns[static_cast<std::size_t>(column_index(name))];
    };

    // Atoms sorted by id when there is an id column; file order otherwise.
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    const int id_col = column_index("id");
    if (id_col >= 0) {
        std::vector<std::int64_t> file_ids(n);
        for (std::size_t r = 0; r < n; ++r) {
            file_ids[r] = parse_int(col("id")[r], row_lines[r]);
        }
        std::stable_sort(order.begin(), order.end(), [&file_ids](std::size_t a, std::size_t b) {
            return file_ids[a] < file_ids[b];
        });
    }
    // Accessors by sorted position k; errors name the row's own line.
    const auto numbers_of = [&](const std::string& name) {
        std::vector<double> out(n);
        for (std::size_t k = 0; k < n; ++k) {
            const std::size_t r = order[k];
            out[k] = parse_double(col(name)[r], row_lines[r]);
        }
        return out;
    };
    const auto ints_of = [&](const std::string& name) {
        std::vector<std::int64_t> out(n);
        for (std::size_t k = 0; k < n; ++k) {
            const std::size_t r = order[k];
            out[k] = parse_int(col(name)[r], row_lines[r]);
        }
        return out;
    };

    // Coordinates. Columns are x, y, z with a base of "", "s" (scaled), "u"
    // (unwrapped) or "su" (scaled and unwrapped), tried in the order of the
    // PLAN.md priority list.
    const Mat3 M = box->matrix;
    const Vec3 origin = box->origin;
    const bool has_image = has_all({"ix", "iy", "iz"});
    const std::vector<std::string> priority =
        options.unwrap ? std::vector<std::string>{"u", "su", "", "s"}
                       : std::vector<std::string>{"", "s", "u", "su"};

    std::vector<double> positions(3 * n, 0.0);
    bool found = false;
    for (const std::string& base : priority) {
        const std::string names3[3] = {std::string("x") + base, std::string("y") + base,
                                       std::string("z") + base};
        if (!has_all({names3[0].c_str(), names3[1].c_str(), names3[2].c_str()})) {
            continue;
        }
        found = true;
        const bool scaled = base == "s" || base == "su";
        // Image flags shift the wrapped coordinates (x, or xs as a fraction of
        // the box). xu and xsu are unwrapped already, so they are left alone.
        const bool use_image = options.unwrap && has_image && (base.empty() || base == "s");
        const std::vector<double> x = numbers_of(names3[0]);
        const std::vector<double> y = numbers_of(names3[1]);
        const std::vector<double> z = numbers_of(names3[2]);
        std::vector<std::int64_t> ix, iy, iz;
        if (use_image) {
            ix = ints_of("ix");
            iy = ints_of("iy");
            iz = ints_of("iz");
        }
        for (std::size_t k = 0; k < n; ++k) {
            // Fractional -> Cartesian with the rows of the cell; otherwise
            // shift to the box origin.
            Vec3 p = scaled ? Vec3(x[k], y[k], z[k]) * M : Vec3(x[k], y[k], z[k]) - origin;
            if (use_image) {
                p += static_cast<double>(ix[k]) * M.row(0) + static_cast<double>(iy[k]) * M.row(1) +
                     static_cast<double>(iz[k]) * M.row(2);
            }
            p.store(positions.data() + 3 * k);
        }
        break;
    }
    if (!found) {
        throw FormatError("LAMMPS dump: no coordinate columns (x, xs, xu or xsu)");
    }

    // Species. 'element' wins; otherwise 'type' holds an integer type or a
    // label (a label is its own species name). The 'type' array is decoded
    // on its own, so it survives an element column too.
    std::vector<int> numbers(n);
    std::vector<std::string> labels(n);
    const int element_col = column_index("element");
    const int type_col = column_index("type");
    if (element_col >= 0) {
        for (std::size_t k = 0; k < n; ++k) {
            const std::size_t r = order[k];
            const std::string label(col("element")[r]);
            labels[k] = label;
            numbers[k] = label == "X" ? kDummyAtomicNumber
                                      : element_of_label(label, row_lines[r]);
        }
    } else if (type_col >= 0) {
        for (std::size_t k = 0; k < n; ++k) {
            const std::size_t r = order[k];
            const std::size_t line = row_lines[r];
            const std::string_view token = col("type")[r];
            std::int64_t t = 0;
            if (try_parse_int(token, t)) {
                // A type with no label, or not in the map, is the dummy X named by its number.
                const auto it = options.types.find(static_cast<int>(t));
                if (it != options.types.end()) {
                    labels[k] = it->second;
                    numbers[k] = element_of_label(it->second, line);
                } else {
                    labels[k] = std::to_string(t);
                    numbers[k] = kDummyAtomicNumber;
                }
            } else {
                // A type label such as "Fe1" or "O": the element when it names one.
                labels[k] = std::string(token);
                const int z = try_number_from_label(token);
                numbers[k] = z >= 0 ? z : kDummyAtomicNumber;
            }
        }
    } else {
        throw FormatError("LAMMPS dump: needs a 'type' or 'element' column");
    }
    // The 'type' array holds integers only, and only when every row has one.
    bool type_numeric = type_col >= 0;
    std::vector<std::int64_t> type_ints(n, 0);
    if (type_numeric) {
        for (std::size_t k = 0; k < n && type_numeric; ++k) {
            type_numeric = try_parse_int(col("type")[order[k]], type_ints[k]);
        }
    }

    Atoms atoms(numbers, positions, Cell(box->matrix, box->pbc));
    atoms.set_string_array("labels", labels);
    if (id_col >= 0) {
        atoms.set_int_array("id", ints_of("id"));
    }
    if (type_numeric) {
        atoms.set_int_array("type", type_ints);
    }
    if (column_index("mol") >= 0) {
        atoms.set_int_array("mol", ints_of("mol"));
    }
    if (has_image) {
        std::vector<std::int64_t> img(3 * n);
        const auto ix = ints_of("ix");
        const auto iy = ints_of("iy");
        const auto iz = ints_of("iz");
        for (std::size_t k = 0; k < n; ++k) {
            img[3 * k] = ix[k];
            img[3 * k + 1] = iy[k];
            img[3 * k + 2] = iz[k];
        }
        atoms.set_int_array("image", img, 3);
    }
    if (column_index("q") >= 0) {
        atoms.set_charges(numbers_of("q"));
    }
    if (has_all({"vx", "vy", "vz"})) {
        const auto vx = numbers_of("vx");
        const auto vy = numbers_of("vy");
        const auto vz = numbers_of("vz");
        std::vector<double> v(3 * n);
        for (std::size_t k = 0; k < n; ++k) {
            v[3 * k] = vx[k];
            v[3 * k + 1] = vy[k];
            v[3 * k + 2] = vz[k];
        }
        atoms.set_velocities(v);
    }
    if (has_all({"fx", "fy", "fz"})) {
        const auto fx = numbers_of("fx");
        const auto fy = numbers_of("fy");
        const auto fz = numbers_of("fz");
        std::vector<double> force(3 * n);
        for (std::size_t k = 0; k < n; ++k) {
            force[3 * k] = fx[k];
            force[3 * k + 1] = fy[k];
            force[3 * k + 2] = fz[k];
        }
        atoms.set_array("forces", force, 3);
    }

    // Every column not consumed above becomes an array named after it. A
    // component that does not complete a triplet (a lone vx, say) is kept
    // on its own.
    std::vector<std::string> consumed = {"id", "type", "element", "mol", "q"};
    const auto consume = [&consumed](std::initializer_list<const char*> cols) {
        consumed.insert(consumed.end(), cols.begin(), cols.end());
    };
    if (has_image) {
        consume({"ix", "iy", "iz"});
    }
    if (has_all({"vx", "vy", "vz"})) {
        consume({"vx", "vy", "vz"});
    }
    if (has_all({"fx", "fy", "fz"})) {
        consume({"fx", "fy", "fz"});
    }
    for (const char* base : {"", "s", "u", "su"}) {
        const std::string names3[3] = {std::string("x") + base, std::string("y") + base,
                                       std::string("z") + base};
        if (has_all({names3[0].c_str(), names3[1].c_str(), names3[2].c_str()})) {
            consumed.insert(consumed.end(), names3, names3 + 3);
        }
    }
    for (std::size_t c = 0; c < names.size(); ++c) {
        const std::string& name = names[c];
        if (std::find(consumed.begin(), consumed.end(), name) != consumed.end()) {
            continue;
        }
        bool numeric = true;
        std::vector<double> values(n);
        for (std::size_t k = 0; k < n && numeric; ++k) {
            numeric = try_parse_double(columns[c][order[k]], values[k]);
        }
        if (numeric) {
            atoms.set_array(name, values, 1);
        } else {
            std::vector<std::string> text(n);
            for (std::size_t k = 0; k < n; ++k) {
                text[k] = std::string(columns[c][order[k]]);
            }
            atoms.set_string_array(name, text);
        }
    }

    Value::Map info;
    info.emplace_back("timestep", Value(timestep));
    atoms.set_info(Value(std::move(info)));
    return atoms;
}

}  // namespace

std::vector<Atoms> read_lammps_dump(std::string_view text, const LammpsDumpReadOptions& options) {
    text = strip_bom(text);
    const auto lines = split_lines(text);
    std::vector<Atoms> frames;
    bool saw_timestep = false;
    std::optional<std::size_t> cut_line;  // 1-based line where a frame was cut short
    std::size_t i = 0;
    while (i < lines.size()) {
        const auto f = fields(lines[i]);
        if (f.size() >= 2 && f[0] == "ITEM:" && f[1] == "TIMESTEP") {
            saw_timestep = true;
            const std::size_t start = i;
            std::size_t where = start + 1;
            auto frame = read_frame(lines, i, options, where);
            if (!frame) {
                cut_line = where;  // the input ends inside this frame
                break;
            }
            frames.push_back(std::move(*frame));
        } else {
            ++i;
        }
    }
    if (!saw_timestep) {
        throw FormatError("LAMMPS dump: no 'ITEM: TIMESTEP' block found");
    }
    if (cut_line) {
        if (frames.empty()) {
            line_error(*cut_line, "truncated frame: the input ends before its atom rows are complete");
        }
        // A truncated last frame is dropped; the last complete one is marked.
        Value::Map info = frames.back().info().as_map();
        info.emplace_back("truncated", Value(true));
        frames.back().set_info(Value(std::move(info)));
    }
    return frames;
}

}  // namespace fastatoms::io
