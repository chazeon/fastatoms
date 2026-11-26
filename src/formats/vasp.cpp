// SPDX-License-Identifier: MIT
#include "fastatoms/formats/vasp.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "fastatoms/core/cell.hpp"
#include "fastatoms/core/elements.hpp"
#include "fastatoms/core/value.hpp"
#include "fastatoms/core/vec3.hpp"
#include "fastatoms/formats/format.hpp"

namespace fastatoms::formats {
namespace {

using Lines = std::vector<std::string_view>;
using Tokens = std::vector<std::string_view>;

bool is_blank_char(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f';
}

/// Lines without their terminator. CRLF is handled by dropping the '\r'; a
/// leading UTF-8 byte-order mark is dropped.
Lines split_lines(std::string_view text) {
    if (text.starts_with("\xEF\xBB\xBF")) {
        text.remove_prefix(3);
    }
    Lines lines;
    std::size_t start = 0;
    while (start < text.size()) {
        std::size_t end = text.find('\n', start);
        if (end == std::string_view::npos) {
            end = text.size();
        }
        std::string_view line = text.substr(start, end - start);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        lines.push_back(line);
        start = end + 1;
    }
    return lines;
}

Tokens tokens(std::string_view line) {
    Tokens out;
    std::size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && is_blank_char(line[i])) {
            ++i;
        }
        const std::size_t start = i;
        while (i < line.size() && !is_blank_char(line[i])) {
            ++i;
        }
        if (i > start) {
            out.push_back(line.substr(start, i - start));
        }
    }
    return out;
}

/// Tokens before a `#` or `!` comment (the species, counts and scale lines).
Tokens code_tokens(std::string_view line) {
    const std::size_t cut = line.find_first_of("#!");
    return tokens(cut == std::string_view::npos ? line : line.substr(0, cut));
}

/// The first non-blank character of a line, or '\0' for a blank line.
char first_char(std::string_view line) noexcept {
    for (const char c : line) {
        if (!is_blank_char(c)) {
            return c;
        }
    }
    return '\0';
}

bool is_blank(std::string_view line) noexcept { return first_char(line) == '\0'; }

[[noreturn]] void fail(std::size_t index, const std::string& what) {
    throw FormatError("vasp: line " + std::to_string(index + 1) + ": " + what);
}

std::string_view line_at(const Lines& lines, std::size_t index, const char* what) {
    if (index >= lines.size()) {
        throw FormatError("vasp: unexpected end of file, expected " + std::string(what) +
                          " at line " + std::to_string(index + 1));
    }
    return lines[index];
}

std::size_t skip_blank(const Lines& lines, std::size_t i) noexcept {
    while (i < lines.size() && is_blank(lines[i])) {
        ++i;
    }
    return i;
}

/// Drops an optional leading '+' so from_chars (which rejects it) can parse.
/// A second sign after the '+' is rejected.
std::optional<std::string_view> strip_plus(std::string_view s) {
    if (!s.empty() && s.front() == '+') {
        s.remove_prefix(1);
        if (!s.empty() && (s.front() == '+' || s.front() == '-')) {
            return std::nullopt;
        }
    }
    return s;
}

/// A finite real number. Fortran exponents (`1.0D-03`) are accepted. A number
/// too small for a double (`1e-400`) reads as zero; too large (`1e400`), nan
/// and inf are rejected.
std::optional<double> parse_double(std::string_view token) {
    std::string text(token);
    for (char& c : text) {
        if (c == 'd' || c == 'D') {
            c = 'e';
        }
    }
    const auto body = strip_plus(text);
    if (!body || body->empty()) {
        return std::nullopt;
    }
    double value = 0.0;
    const char* end = body->data() + body->size();
    const auto [ptr, ec] = std::from_chars(body->data(), end, value);
    if (ptr != end) {
        return std::nullopt;
    }
    if (ec == std::errc::result_out_of_range) {
        // The exponent decides: a negative one is underflow, anything else overflow.
        const std::size_t e = body->find_first_of("eE");
        if (e == std::string_view::npos || e + 1 >= body->size() || (*body)[e + 1] != '-') {
            return std::nullopt;
        }
        return 0.0;
    }
    if (ec != std::errc{} || !std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

/// A whole-token integer, with an optional sign.
std::optional<std::int64_t> parse_integer(std::string_view token) {
    const auto body = strip_plus(token);
    if (!body || body->empty()) {
        return std::nullopt;
    }
    std::int64_t value = 0;
    const char* end = body->data() + body->size();
    const auto [ptr, ec] = std::from_chars(body->data(), end, value);
    if (ec != std::errc{} || ptr != end) {
        return std::nullopt;
    }
    return value;
}

/// Parsed structure header: everything before the coordinates.
struct Header {
    std::string comment;
    std::array<Vec3, 3> lattice{};        ///< lattice vectors, scale applied
    std::array<double, 3> factor{1, 1, 1};  ///< per-axis scale for Cartesian coordinates
    std::vector<std::string> species;     ///< one label per species block
    std::vector<int> numbers;             ///< atomic number per species block
    std::vector<std::size_t> counts;      ///< atoms per species block
    std::size_t next{0};                  ///< first line after the counts line

    [[nodiscard]] std::size_t size() const noexcept {
        std::size_t n = 0;
        for (const std::size_t c : counts) {
            n += c;
        }
        return n;
    }
};

/// Parses a header starting at `first` (the comment line). Handles VASP 4 and
/// VASP 5 layouts, scale lines and atom counts.
Header parse_header(const Lines& lines, std::size_t first, const PoscarReadOptions& options) {
    Header h;
    h.comment = std::string(line_at(lines, first, "comment line"));

    const std::size_t scale_index = first + 1;
    const Tokens scale_tokens = code_tokens(line_at(lines, scale_index, "scale line"));
    std::vector<double> scale;
    for (std::size_t k = 0; k < scale_tokens.size() && k < 3; ++k) {
        const auto value = parse_double(scale_tokens[k]);
        if (!value) {
            break;
        }
        scale.push_back(*value);
    }
    if (scale.size() != 1 && scale.size() != 3) {
        fail(scale_index, "the scale line needs one or three numbers");
    }

    std::array<Vec3, 3> lattice{};
    for (std::size_t r = 0; r < 3; ++r) {
        const std::size_t index = first + 2 + r;
        const Tokens t = tokens(line_at(lines, index, "lattice vector"));
        if (t.size() < 3) {
            fail(index, "a lattice vector needs three numbers");
        }
        std::array<double, 3> v{};
        for (std::size_t k = 0; k < 3; ++k) {
            const auto value = parse_double(t[k]);
            if (!value) {
                fail(index, "bad lattice number '" + std::string(t[k]) + "'");
            }
            v[k] = *value;
        }
        lattice[r] = Vec3(v[0], v[1], v[2]);
    }

    // Scale semantics follow ASE (read_vasp): a single positive value scales
    // everything; a negative value is a target volume; three values are
    // per-axis factors applied to the components of the lattice vectors (and
    // of Cartesian coordinates) along x, y and z.
    if (scale.size() == 1) {
        const double s = scale[0];
        if (s < 0.0) {
            const double det = Mat3::from_rows(lattice[0], lattice[1], lattice[2]).determinant();
            if (det == 0.0) {
                fail(scale_index, "a negative scale (target volume) needs a non-singular cell");
            }
            const double f = std::cbrt(-s / std::abs(det));
            h.factor = {f, f, f};
        } else {
            if (s == 0.0) {
                fail(scale_index, "the scale factor must be nonzero");
            }
            h.factor = {s, s, s};
        }
    } else {
        for (std::size_t k = 0; k < 3; ++k) {
            if (!(scale[k] > 0.0)) {
                fail(scale_index, "per-axis scale factors must be positive");
            }
            h.factor[k] = scale[k];
        }
    }
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t k = 0; k < 3; ++k) {
            lattice[r][k] *= h.factor[k];
        }
    }
    h.lattice = lattice;

    // Line 6 is either the VASP 5 species line or the VASP 4 counts line.
    std::size_t counts_at = first + 5;
    const Tokens line5 = code_tokens(line_at(lines, counts_at, "species or counts line"));
    if (line5.empty()) {
        fail(counts_at, "blank line where the species or counts line is expected");
    }
    Tokens labels;
    if (!parse_integer(line5[0])) {
        labels = line5;
        counts_at = first + 6;
    }
    const Tokens count_tokens = code_tokens(line_at(lines, counts_at, "atom counts line"));
    h.next = counts_at + 1;
    if (count_tokens.empty()) {
        fail(counts_at, "no atom counts");
    }
    for (const std::string_view t : count_tokens) {
        const auto c = parse_integer(t);
        if (!c || *c < 0) {
            fail(counts_at, "bad atom count '" + std::string(t) + "'");
        }
        h.counts.push_back(static_cast<std::size_t>(*c));
    }

    auto resolve = [&](const std::vector<std::string>& names, std::size_t index) {
        for (const std::string& name : names) {
            const int z = try_number_from_label(name);
            if (z < 0) {
                fail(index, "cannot tell the element of species '" + name + "'");
            }
            h.numbers.push_back(z);
        }
        h.species = names;
    };

    if (!labels.empty()) {
        if (labels.size() != count_tokens.size()) {
            fail(first + 5, "the species line names " + std::to_string(labels.size()) +
                                " species but the counts line has " +
                                std::to_string(count_tokens.size()));
        }
        std::vector<std::string> names;
        for (const std::string_view l : labels) {
            names.emplace_back(l);
        }
        resolve(names, first + 5);
    } else {
        // VASP 4: no species line. Use the option, else comment-line tokens
        // that all resolve as labels.
        std::vector<std::string> names = options.species;
        if (names.empty()) {
            const Tokens comment_tokens = tokens(lines[first]);
            const bool all_labels =
                !comment_tokens.empty() &&
                std::all_of(comment_tokens.begin(), comment_tokens.end(),
                            [](std::string_view t) { return try_number_from_label(t) >= 0; });
            if (!all_labels) {
                fail(first, "VASP 4 file without a species line: pass the species option "
                            "or name the species in the comment line");
            }
            for (const std::string_view t : comment_tokens) {
                names.emplace_back(t);
            }
        }
        if (names.size() != count_tokens.size()) {
            fail(counts_at, std::to_string(names.size()) + " species given but the counts line has " +
                                std::to_string(count_tokens.size()) + " entries");
        }
        resolve(names, counts_at);
    }

    if (h.size() == 0) {
        fail(counts_at, "the structure has no atoms");
    }
    return h;
}

/// Fractional -> Cartesian with the header's lattice rows.
Vec3 to_cartesian(const Header& h, const Vec3& frac) {
    Vec3 r = h.lattice[0] * frac.x;
    r += h.lattice[1] * frac.y;
    r += h.lattice[2] * frac.z;
    return r;
}

/// One coordinate line: three numbers, then T/F flags if selective dynamics
/// is on. Trailing tokens (e.g. a `! Fe1` site label) are ignored.
void parse_position(std::string_view line, std::size_t index, bool selective, double* xyz,
                    std::int64_t* flags) {
    const Tokens t = tokens(line);
    if (t.size() < 3) {
        fail(index, "a position needs three coordinates");
    }
    for (std::size_t k = 0; k < 3; ++k) {
        const auto value = parse_double(t[k]);
        if (!value) {
            fail(index, "bad coordinate '" + std::string(t[k]) + "'");
        }
        xyz[k] = *value;
    }
    if (!selective) {
        return;
    }
    if (t.size() < 6) {
        fail(index, "selective dynamics is on but the position has no T/F flags");
    }
    for (std::size_t k = 0; k < 3; ++k) {
        const char c = t[3 + k].size() == 1 ? t[3 + k][0] : '\0';
        if (c == 'T' || c == 't') {
            flags[k] = 1;
        } else if (c == 'F' || c == 'f') {
            flags[k] = 0;
        } else {
            fail(index, "bad selective-dynamics flag '" + std::string(t[3 + k]) + "'");
        }
    }
}

/// Three numbers from a line, or false. Never throws, for the optional block.
bool try_vec3(std::string_view line, double* out) {
    const Tokens t = tokens(line);
    if (t.size() < 3) {
        return false;
    }
    for (std::size_t k = 0; k < 3; ++k) {
        const auto value = parse_double(t[k]);
        if (!value) {
            return false;
        }
        out[k] = *value;
    }
    return true;
}

/// Optional velocity block after the coordinates. Returns 3N values in
/// Cartesian units, or empty when there is none. A predictor-corrector block
/// (or anything else that does not start with numbers or a mode line) is
/// skipped silently, and so is a block that is malformed or short: the read
/// still succeeds, without velocities.
std::vector<double> read_velocities(const Lines& lines, std::size_t i, const Header& h,
                                    std::size_t n) {
    i = skip_blank(lines, i);
    // Lattice-velocity block from VASP MD runs: skip it up to the next blank line.
    if (i < lines.size() && (first_char(lines[i]) == 'l' || first_char(lines[i]) == 'L')) {
        while (i < lines.size() && !is_blank(lines[i])) {
            ++i;
        }
        i = skip_blank(lines, i);
    }
    if (i >= lines.size()) {
        return {};
    }

    bool direct = false;
    if (!parse_double(tokens(lines[i]).front())) {
        // A mode line: c/C/k/K is Cartesian, d/D is Direct, anything else is not ours.
        switch (first_char(lines[i])) {
            case 'c':
            case 'C':
            case 'k':
            case 'K':
                break;
            case 'd':
            case 'D':
                direct = true;
                break;
            default:
                return {};
        }
        ++i;
    }

    std::vector<double> v(3 * n);
    for (std::size_t a = 0; a < n; ++a) {
        if (i >= lines.size() || !try_vec3(lines[i], v.data() + 3 * a)) {
            return {};
        }
        ++i;
    }
    if (direct) {
        for (std::size_t a = 0; a < n; ++a) {
            const Vec3 cart = to_cartesian(h, Vec3::load(v.data() + 3 * a));
            cart.store(v.data() + 3 * a);
        }
    }
    return v;
}

/// Builds the Atoms for one frame. Species blocks expand to per-atom labels,
/// atomic numbers and (optionally) selective-dynamics flags.
Atoms make_atoms(const Header& h, const std::vector<double>& coords, bool cartesian,
                 const std::vector<std::int64_t>& flags) {
    std::vector<int> numbers;
    std::vector<std::string> labels;
    numbers.reserve(h.size());
    labels.reserve(h.size());
    for (std::size_t s = 0; s < h.species.size(); ++s) {
        for (std::size_t c = 0; c < h.counts[s]; ++c) {
            numbers.push_back(h.numbers[s]);
            labels.push_back(h.species[s]);
        }
    }

    const std::size_t n = numbers.size();
    std::vector<double> positions(3 * n);
    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 x = Vec3::load(coords.data() + 3 * i);
        const Vec3 r = cartesian ? Vec3(x.x * h.factor[0], x.y * h.factor[1], x.z * h.factor[2])
                                 : to_cartesian(h, x);
        r.store(positions.data() + 3 * i);
    }

    const Mat3 lattice = Mat3::from_rows(h.lattice[0], h.lattice[1], h.lattice[2]);
    Atoms atoms(std::move(numbers), std::move(positions), Cell(lattice, {true, true, true}));
    atoms.set_pbc(true);
    atoms.set_string_array("labels", labels);
    if (!flags.empty()) {
        atoms.set_int_array("selective_dynamics", flags, 3);
    }
    atoms.info()[std::string("comment")] = Value(h.comment);
    return atoms;
}

/// Shortest decimal that reads back to the same double.
void append_double(std::string& out, double value) {
    char buf[64];
    const auto [end, ec] = std::to_chars(buf, buf + sizeof buf, value);
    if (ec == std::errc{}) {
        out.append(buf, end);
    }
}

void append_vec(std::string& out, const double* p) {
    for (std::size_t k = 0; k < 3; ++k) {
        out += ' ';
        append_double(out, p[k]);
    }
}

/// An XDATCAR frame line: `Direct configuration=     1` (VASP writes this,
/// also `Cartesian configuration`), or the bare `configuration=N`. Matched on
/// the leading words, case-insensitively, so a comment line that merely
/// contains the word is not one. `configuration 2` (no '=') is a comment.
bool is_configuration_line(std::string_view line) {
    const Tokens t = tokens(line);
    if (t.empty()) {
        return false;
    }
    std::string first(t[0]);
    for (char& c : first) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (first.starts_with("configuration=")) {
        return true;
    }
    if (t.size() < 2 || (first != "direct" && first != "cartesian")) {
        return false;
    }
    std::string second(t[1]);
    for (char& c : second) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return second.starts_with("configuration");
}

/// Atoms grouped for writing: each group is one species line entry.
struct Group {
    std::string name;
    std::vector<std::size_t> atoms;
};

}  // namespace

Atoms read_poscar(std::string_view text, const PoscarReadOptions& options) {
    const Lines lines = split_lines(text);
    const Header h = parse_header(lines, 0, options);
    const std::size_t n = h.size();

    std::size_t i = h.next;
    bool selective = false;
    const char first = first_char(line_at(lines, i, "coordinate mode line"));
    if (first == 's' || first == 'S') {
        selective = true;
        ++i;
    }

    const std::string_view mode_line = line_at(lines, i, "coordinate mode line");
    const char mode = first_char(mode_line);
    if (mode == '\0') {
        fail(i, "empty coordinate mode line");
    }
    const bool cartesian = mode == 'c' || mode == 'C' || mode == 'k' || mode == 'K';
    ++i;

    std::vector<double> coords(3 * n);
    std::vector<std::int64_t> flags(selective ? 3 * n : 0);
    for (std::size_t a = 0; a < n; ++a) {
        if (i >= lines.size()) {
            fail(lines.size() - 1, "expected " + std::to_string(n) + " position lines, found " +
                                       std::to_string(a));
        }
        parse_position(lines[i], i, selective, coords.data() + 3 * a,
                       selective ? flags.data() + 3 * a : nullptr);
        ++i;
    }

    Atoms atoms = make_atoms(h, coords, cartesian, flags);
    const std::vector<double> velocities = read_velocities(lines, i, h, n);
    if (!velocities.empty()) {
        atoms.set_velocities(velocities);
    }
    return atoms;
}

std::string write_poscar(const Atoms& atoms, const PoscarWriteOptions& options) {
    const std::size_t n = atoms.size();
    // A label is written only if it resolves to the atom's own element and
    // would survive the species line as one token; otherwise the symbol is.
    const std::vector<std::string> symbols = atoms.symbols();
    std::vector<std::string> names = symbols;
    if (options.write_labels) {
        const std::vector<std::string> labels = atoms.labels();
        const std::vector<int> numbers = atoms.numbers();
        for (std::size_t i = 0; i < n; ++i) {
            const std::string& label = labels[i];
            const bool one_token = !label.empty() && tokens(label).size() == 1 &&
                                   label.find_first_of("#!") == std::string::npos;
            if (one_token && try_number_from_label(label) == numbers[i]) {
                names[i] = label;
            }
        }
    }

    std::vector<Group> groups;
    if (options.sort) {
        std::map<std::string, std::size_t> index;
        for (std::size_t i = 0; i < n; ++i) {
            const auto [it, inserted] = index.try_emplace(names[i], groups.size());
            if (inserted) {
                groups.push_back({names[i], {}});
            }
            groups[it->second].atoms.push_back(i);
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            if (groups.empty() || groups.back().name != names[i]) {
                groups.push_back({names[i], {}});
            }
            groups.back().atoms.push_back(i);
        }
    }
    std::vector<std::size_t> order;
    order.reserve(n);
    for (const Group& g : groups) {
        order.insert(order.end(), g.atoms.begin(), g.atoms.end());
    }

    const bool selective = atoms.has_int_array("selective_dynamics");
    if (selective && atoms.int_array("selective_dynamics").ncols != 3) {
        throw FormatError("vasp: the selective_dynamics array must have 3 columns");
    }
    if (options.direct && !atoms.cell().is_invertible()) {
        throw FormatError("vasp: Direct coordinates need an invertible cell; use direct=false");
    }

    std::string comment = options.comment;
    if (comment.empty() && !options.vasp5) {
        // A VASP 4 file has no species line; by convention (and for our own
        // reader) the element symbols go on the comment line instead, one per
        // count. Plain symbols, since VASP 4 era tools know nothing of labels.
        for (std::size_t g = 0; g < groups.size(); ++g) {
            comment += (g == 0 ? "" : " ") + atoms.symbol(groups[g].atoms.front());
        }
    }
    if (comment.empty()) {
        const Value* c = atoms.info().find("comment");
        if (c != nullptr && c->is_string()) {
            comment = c->as_string();
        }
    }
    if (comment.empty()) {
        comment = atoms.formula();
    }
    std::replace(comment.begin(), comment.end(), '\n', ' ');
    std::replace(comment.begin(), comment.end(), '\r', ' ');

    std::string out = comment + "\n1.0\n";
    for (std::size_t r = 0; r < 3; ++r) {
        const Vec3 v = atoms.cell().vector(r);
        const double row[3] = {v.x, v.y, v.z};
        append_vec(out, row);
        out += '\n';
    }
    if (options.vasp5) {
        for (std::size_t g = 0; g < groups.size(); ++g) {
            out += (g == 0 ? "" : " ") + groups[g].name;
        }
        out += '\n';
    }
    for (std::size_t g = 0; g < groups.size(); ++g) {
        out += (g == 0 ? "" : " ") + std::to_string(groups[g].atoms.size());
    }
    out += '\n';

    if (selective) {
        out += "Selective dynamics\n";
    }
    out += options.direct ? "Direct\n" : "Cartesian\n";

    const std::vector<double> frac = options.direct ? atoms.scaled_positions() : std::vector<double>{};
    const std::vector<double>& cart = atoms.positions();
    const std::vector<std::int64_t>* flags = selective ? &atoms.int_array("selective_dynamics").data : nullptr;
    for (const std::size_t i : order) {
        append_vec(out, options.direct ? frac.data() + 3 * i : cart.data() + 3 * i);
        if (flags != nullptr) {
            for (std::size_t k = 0; k < 3; ++k) {
                out += (*flags)[3 * i + k] != 0 ? " T" : " F";
            }
        }
        out += '\n';
    }

    if (atoms.has_momenta()) {
        const std::vector<double> velocities = atoms.velocities();
        out += '\n';
        for (const std::size_t i : order) {
            append_vec(out, velocities.data() + 3 * i);
            out += '\n';
        }
    }
    return out;
}

std::vector<Atoms> read_xdatcar(std::string_view text, const PoscarReadOptions& options) {
    const Lines lines = split_lines(text);
    // A file cut off mid-write ends in a line without its newline.
    const bool may_be_cut = !text.empty() && text.back() != '\n';
    std::vector<Atoms> frames;
    std::optional<Header> header;
    bool truncated = false;
    std::size_t i = skip_blank(lines, 0);

    while (i < lines.size() && !truncated) {
        // Fixed-cell files have one header and many configuration blocks;
        // variable-cell files repeat the header before each block.
        const std::size_t start = i;
        if (!header || !is_configuration_line(lines[i])) {
            // A header is at least six lines, then the configuration line.
            if (header && lines.size() - start < 7 + header->size()) {
                truncated = true;
                break;
            }
            header = parse_header(lines, i, options);
            i = skip_blank(lines, header->next);
            if (i >= lines.size()) {
                truncated = true;
                break;
            }
            if (!is_configuration_line(lines[i])) {
                fail(i, "expected a 'Direct configuration=' line");
            }
        }
        ++i;

        const std::size_t n = header->size();
        std::vector<double> coords(3 * n);
        for (std::size_t a = 0; a < n; ++a) {
            if (i >= lines.size()) {
                truncated = true;
                break;
            }
            double* xyz = coords.data() + 3 * a;
            if (!try_vec3(lines[i], xyz)) {
                if (may_be_cut && i + 1 == lines.size()) {
                    truncated = true;
                    break;
                }
                parse_position(lines[i], i, false, xyz, nullptr);  // throws, naming the line
            }
            ++i;
        }
        if (truncated) {
            break;  // an incomplete last frame is dropped
        }
        frames.push_back(make_atoms(*header, coords, false, {}));
        i = skip_blank(lines, i);
    }

    if (frames.empty()) {
        throw FormatError("vasp: XDATCAR has no configurations");
    }
    if (truncated) {
        frames.back().info()[std::string("truncated")] = Value(true);
    }
    return frames;
}

}  // namespace fastatoms::formats
