// SPDX-License-Identifier: MIT
/// \file bindings.cpp
/// \brief Embind layer behind the `fastatoms` npm package.
///
/// Conventions, mirrored by js/src/index.ts:
///  - Arrays are copies, never views into wasm memory: positions, cell and
///    doubles come back as Float64Array, numbers and neighbour indices as
///    Int32Array, string arrays as string[]. Integer per-atom arrays (LAMMPS
///    type/id, image flags) are converted to Float64Array because embind has
///    no BigInt64Array mapping.
///  - Options cross as JSON text and are parsed into fastatoms::Value.
///  - Every entry point catches std::exception and rethrows it as a JS Error
///    carrying the same message.

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/core/cell.hpp"
#include "fastatoms/core/value.hpp"
#include "fastatoms/io/format.hpp"
#ifdef FASTATOMS_WITH_NEIGHBORS
#  include "fastatoms/core/neighbor_list.hpp"
#endif
#ifdef FASTATOMS_WITH_SYMMETRY
#  include "fastatoms/symmetry/spacegroup.hpp"
#  include "fastatoms/symmetry/standardize.hpp"
#endif

namespace fa = fastatoms;
using emscripten::class_;
using emscripten::function;
using emscripten::typed_memory_view;
using emscripten::val;

namespace {

// ---------------------------------------------------------------------------
// Errors: C++ exceptions -> JS Error(message)
// ---------------------------------------------------------------------------

template <typename F>
decltype(auto) guarded(F&& body) {
    try {
        return std::forward<F>(body)();
    } catch (const std::exception& e) {
        val::global("Error").new_(std::string(e.what())).throw_();
        __builtin_unreachable();
    }
}

// ---------------------------------------------------------------------------
// JSON <-> fa::Value (objects, arrays, strings, numbers, bools, null)
// ---------------------------------------------------------------------------

class JsonReader {
public:
    explicit JsonReader(std::string_view text) : s_(text) {}

    fa::Value document() {
        fa::Value v = value(0);
        skip_ws();
        if (p_ != s_.size()) {
            fail("trailing characters");
        }
        return v;
    }

private:
    static constexpr int kMaxDepth = 64;

    [[noreturn]] void fail(const std::string& what) const {
        throw fa::InputError("invalid options JSON: " + what + " at offset " + std::to_string(p_));
    }

    void skip_ws() {
        while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\n' || s_[p_] == '\r')) {
            ++p_;
        }
    }

    char peek() const { return p_ < s_.size() ? s_[p_] : '\0'; }

    void expect(char c) {
        skip_ws();
        if (peek() != c) {
            fail(std::string("expected '") + c + "'");
        }
        ++p_;
    }

    bool literal(std::string_view word) {
        if (s_.substr(p_, word.size()) == word) {
            p_ += word.size();
            return true;
        }
        return false;
    }

    fa::Value value(int depth) {
        if (depth > kMaxDepth) {
            fail("nesting too deep");
        }
        skip_ws();
        const char c = peek();
        if (c == '{') {
            return object(depth);
        }
        if (c == '[') {
            return array(depth);
        }
        if (c == '"') {
            return fa::Value(string());
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            return number();
        }
        if (literal("true")) {
            return fa::Value(true);
        }
        if (literal("false")) {
            return fa::Value(false);
        }
        if (literal("null")) {
            return fa::Value();
        }
        fail("unexpected character");
    }

    fa::Value object(int depth) {
        expect('{');
        fa::Value::Map map;
        skip_ws();
        if (peek() == '}') {
            ++p_;
            return fa::Value(std::move(map));
        }
        for (;;) {
            skip_ws();
            if (peek() != '"') {
                fail("object key must be a string");
            }
            std::string key = string();
            expect(':');
            map.emplace_back(std::move(key), value(depth + 1));
            skip_ws();
            if (peek() == ',') {
                ++p_;
                continue;
            }
            expect('}');
            return fa::Value(std::move(map));
        }
    }

    fa::Value array(int depth) {
        expect('[');
        fa::Value::List list;
        skip_ws();
        if (peek() == ']') {
            ++p_;
            return fa::Value(std::move(list));
        }
        for (;;) {
            list.push_back(value(depth + 1));
            skip_ws();
            if (peek() == ',') {
                ++p_;
                continue;
            }
            expect(']');
            return fa::Value(std::move(list));
        }
    }

    fa::Value number() {
        const std::size_t start = p_;
        bool floating = false;
        while (p_ < s_.size()) {
            const char c = s_[p_];
            if (c == '.' || c == 'e' || c == 'E') {
                floating = true;
            } else if (!(c == '-' || c == '+' || (c >= '0' && c <= '9'))) {
                break;
            }
            ++p_;
        }
        const std::string token(s_.substr(start, p_ - start));
        char* end = nullptr;
        if (!floating) {
            errno = 0;
            const long long n = std::strtoll(token.c_str(), &end, 10);
            if (end == token.c_str() + token.size() && errno == 0) {
                return fa::Value(static_cast<std::int64_t>(n));
            }
        }
        const double d = std::strtod(token.c_str(), &end);
        if (token.empty() || end != token.c_str() + token.size()) {
            fail("malformed number");
        }
        return fa::Value(d);
    }

    static void append_utf8(std::string& out, std::uint32_t cp) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    std::uint32_t hex4() {
        if (p_ + 4 > s_.size()) {
            fail("truncated \\u escape");
        }
        std::uint32_t cp = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s_[p_++];
            cp <<= 4;
            if (c >= '0' && c <= '9') {
                cp |= static_cast<std::uint32_t>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                cp |= static_cast<std::uint32_t>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                cp |= static_cast<std::uint32_t>(c - 'A' + 10);
            } else {
                fail("bad hex digit in \\u escape");
            }
        }
        return cp;
    }

    std::string string() {
        expect('"');
        std::string out;
        for (;;) {
            if (p_ >= s_.size()) {
                fail("unterminated string");
            }
            const char c = s_[p_++];
            if (c == '"') {
                return out;
            }
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (p_ >= s_.size()) {
                fail("unterminated escape");
            }
            const char e = s_[p_++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    std::uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && s_.substr(p_, 2) == "\\u") {
                        p_ += 2;
                        const std::uint32_t lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default:
                    fail("bad escape");
            }
        }
    }

    std::string_view s_;
    std::size_t p_{0};
};

fa::Value parse_options(const std::string& json) {
    if (json.find_first_not_of(" \t\r\n") == std::string::npos) {
        return fa::Value();
    }
    return JsonReader(json).document();
}

void json_escape(std::string& out, std::string_view s) {
    out.push_back('"');
    for (const char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
}

void json_write(std::string& out, const fa::Value& v) {
    switch (v.type()) {
        case fa::Value::Type::Null:
            out += "null";
            break;
        case fa::Value::Type::Bool:
            out += v.as_bool() ? "true" : "false";
            break;
        case fa::Value::Type::Int:
            out += std::to_string(v.as_int());
            break;
        case fa::Value::Type::Float: {
            const double d = v.as_double();
            if (!std::isfinite(d)) {
                out += "null";  // JSON has no NaN or Infinity
            } else {
                // Shortest of 15..17 significant digits that reads back exactly.
                char buf[32];
                for (int precision = 15; precision <= 17; ++precision) {
                    std::snprintf(buf, sizeof buf, "%.*g", precision, d);
                    if (std::strtod(buf, nullptr) == d) {
                        break;
                    }
                }
                out += buf;
            }
            break;
        }
        case fa::Value::Type::String:
            json_escape(out, v.as_string());
            break;
        case fa::Value::Type::List: {
            out.push_back('[');
            bool first = true;
            for (const auto& item : v.as_list()) {
                if (!first) {
                    out.push_back(',');
                }
                first = false;
                json_write(out, item);
            }
            out.push_back(']');
            break;
        }
        case fa::Value::Type::Map: {
            out.push_back('{');
            bool first = true;
            for (const auto& [key, item] : v.as_map()) {
                if (!first) {
                    out.push_back(',');
                }
                first = false;
                json_escape(out, key);
                out.push_back(':');
                json_write(out, item);
            }
            out.push_back('}');
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Typed-array and JS array helpers
// ---------------------------------------------------------------------------

/// Copy `n` doubles out of wasm memory into a fresh Float64Array.
val float64_copy(const double* data, std::size_t n) {
    val out = val::global("Float64Array").new_(n);
    if (n > 0) {
        out.call<void>("set", typed_memory_view(n, data));
    }
    return out;
}

/// Copy `n` int32 values into a fresh Int32Array.
val int32_copy(const std::int32_t* data, std::size_t n) {
    val out = val::global("Int32Array").new_(n);
    if (n > 0) {
        out.call<void>("set", typed_memory_view(n, data));
    }
    return out;
}

val string_list(const std::vector<std::string>& items) {
    val out = val::array();
    for (const auto& s : items) {
        out.call<void>("push", s);
    }
    return out;
}

std::size_t js_length(const val& arr) {
    return arr.isUndefined() || arr.isNull() ? 0 : arr["length"].as<std::size_t>();
}

std::vector<double> doubles_from(const val& arr) {
    const std::size_t n = js_length(arr);
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = arr[i].as<double>();
    }
    return out;
}

std::vector<int> ints_from(const val& arr) {
    const std::size_t n = js_length(arr);
    std::vector<int> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = arr[i].as<int>();
    }
    return out;
}

std::vector<std::size_t> indices_from(const val& arr) {
    const std::size_t n = js_length(arr);
    std::vector<std::size_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double d = arr[i].as<double>();
        if (!(d >= 0.0) || d != std::floor(d)) {
            throw std::invalid_argument("subset: indices must be non-negative integers");
        }
        out[i] = static_cast<std::size_t>(d);
    }
    return out;
}

std::vector<std::string> strings_from(const val& arr) {
    const std::size_t n = js_length(arr);
    std::vector<std::string> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = arr[i].as<std::string>();
    }
    return out;
}

std::array<bool, 3> pbc_from(const val& arr) {
    if (arr.isUndefined() || arr.isNull()) {
        return {false, false, false};
    }
    if (js_length(arr) != 3) {
        throw std::invalid_argument("pbc must have exactly 3 entries");
    }
    return {arr[0].as<bool>(), arr[1].as<bool>(), arr[2].as<bool>()};
}

fa::Mat3 cell_from(const val& arr) {
    const auto m = doubles_from(arr);
    if (m.size() != 9) {
        throw std::invalid_argument("cell must have 9 entries (row-major, rows are lattice vectors)");
    }
    fa::Mat3 out;
    for (std::size_t k = 0; k < 9; ++k) {
        out.m[k] = m[k];
    }
    return out;
}

// ---------------------------------------------------------------------------
// Atoms
// ---------------------------------------------------------------------------

std::size_t atoms_length(const fa::Atoms& a) { return a.size(); }

val get_numbers(const fa::Atoms& a) { return int32_copy(a.numbers().data(), a.size()); }

val get_positions(const fa::Atoms& a) { return float64_copy(a.positions().data(), 3 * a.size()); }

val get_cell(const fa::Atoms& a) { return float64_copy(a.cell().matrix().m.data(), 9); }

val get_pbc(const fa::Atoms& a) {
    val out = val::array();
    for (const bool flag : a.pbc()) {
        out.call<void>("push", val(flag));
    }
    return out;
}

void set_positions_js(fa::Atoms& a, const val& positions) {
    guarded([&] { a.set_positions(doubles_from(positions)); });
}

void set_cell_js(fa::Atoms& a, const val& cell, bool scale_atoms) {
    guarded([&] { a.set_cell(cell_from(cell), scale_atoms); });
}

void set_pbc_js(fa::Atoms& a, const val& pbc) {
    guarded([&] { a.set_pbc(pbc_from(pbc)); });
}

val get_scaled_positions(const fa::Atoms& a, bool wrap) {
    return guarded([&] {
        const auto s = a.scaled_positions(wrap);
        return float64_copy(s.data(), s.size());
    });
}

val get_symbols(const fa::Atoms& a) { return string_list(a.symbols()); }

val get_labels(const fa::Atoms& a) { return string_list(a.labels()); }

/// Named per-atom array. Double and integer arrays are returned as Float64Array
/// (integers converted); string arrays as string[]. The built-in optional
/// arrays (masses, momenta, charges, tags) come first, then double, integer
/// and string arrays.
val get_array(const fa::Atoms& a, const std::string& name) {
    return guarded([&]() -> val {
        if (name == "masses" && a.has_masses()) {
            const auto m = a.masses();
            return float64_copy(m.data(), m.size());
        }
        if (name == "momenta" && a.has_momenta()) {
            const auto p = a.momenta();
            return float64_copy(p.data(), p.size());
        }
        if (name == "charges" && a.has_charges()) {
            const auto q = a.charges();
            return float64_copy(q.data(), q.size());
        }
        if (name == "tags" && a.has_tags()) {
            const auto t = a.tags();
            std::vector<double> converted(t.begin(), t.end());
            return float64_copy(converted.data(), converted.size());
        }
        if (a.has_array(name)) {
            const auto& arr = a.array(name);
            return float64_copy(arr.data.data(), arr.data.size());
        }
        if (a.has_int_array(name)) {
            const auto& arr = a.int_array(name);
            std::vector<double> converted(arr.data.begin(), arr.data.end());
            return float64_copy(converted.data(), converted.size());
        }
        if (a.has_string_array(name)) {
            return string_list(a.string_array(name));
        }
        throw std::out_of_range("no per-atom array named '" + name + "'");
    });
}

/// [{name, kind: "float" | "int" | "string", ncols}] for every per-atom array.
val array_names(const fa::Atoms& a) {
    val out = val::array();
    auto add = [&out](const std::string& name, const char* kind, std::size_t ncols) {
        val item = val::object();
        item.set("name", name);
        item.set("kind", std::string(kind));
        item.set("ncols", static_cast<double>(ncols));
        out.call<void>("push", item);
    };
    if (a.has_masses()) {
        add("masses", "float", 1);
    }
    if (a.has_momenta()) {
        add("momenta", "float", 3);
    }
    if (a.has_charges()) {
        add("charges", "float", 1);
    }
    if (a.has_tags()) {
        add("tags", "int", 1);
    }
    for (const auto& name : a.array_names()) {
        add(name, "float", a.array(name).ncols);
    }
    for (const auto& name : a.int_array_names()) {
        add(name, "int", a.int_array(name).ncols);
    }
    for (const auto& name : a.string_array_names()) {
        add(name, "string", 1);
    }
    return out;
}

std::string info_json(const fa::Atoms& a) {
    std::string out;
    json_write(out, a.info());
    return out;
}

fa::Atoms repeat_js(const fa::Atoms& a, int nx, int ny, int nz) {
    return guarded([&] { return a.repeat(nx, ny, nz); });
}

fa::Atoms subset_js(const fa::Atoms& a, const val& indices) {
    return guarded([&] { return a.subset(indices_from(indices)); });
}

void wrap_js(fa::Atoms& a) {
    guarded([&] { a.wrap(); });
}

fa::Atoms copy_js(const fa::Atoms& a) { return a; }

double volume_js(const fa::Atoms& a) { return a.volume(); }

// ---------------------------------------------------------------------------
// Module functions
// ---------------------------------------------------------------------------

val read_all_js(const std::string& text, const std::string& format, const std::string& options) {
    return guarded([&] {
        auto frames = fa::io::read_all(text, format, parse_options(options));
        val out = val::array();
        for (auto& frame : frames) {
            out.call<void>("push", val(std::move(frame)));
        }
        return out;
    });
}

fa::Atoms read_js(const std::string& text, const std::string& format, const std::string& options) {
    return guarded([&] { return fa::io::read(text, format, parse_options(options)); });
}

std::string write_js(const fa::Atoms& atoms, const std::string& format, const std::string& options) {
    return guarded([&] { return fa::io::write(atoms, format, parse_options(options)); });
}

std::string guess_format_js(const std::string& name, const std::string& head) {
    return fa::io::guess_format(name, head);
}

val available_formats_js() { return string_list(fa::io::available_formats()); }

bool can_write_js(const std::string& format) { return fa::io::can_write(format); }

fa::Atoms from_arrays_js(const val& numbers, const val& positions, const val& cell, const val& pbc) {
    return guarded([&]() -> fa::Atoms {
        fa::Cell c;
        if (!cell.isUndefined() && !cell.isNull()) {
            c = fa::Cell(cell_from(cell), pbc_from(pbc));
        } else {
            c.set_pbc(pbc_from(pbc));
        }
        auto pos = doubles_from(positions);
        const bool by_symbol = js_length(numbers) > 0 && numbers[0].typeOf().as<std::string>() == "string";
        if (by_symbol) {
            return fa::Atoms(strings_from(numbers), std::move(pos), c);
        }
        return fa::Atoms(ints_from(numbers), std::move(pos), c);
    });
}

#ifdef FASTATOMS_WITH_NEIGHBORS
/// Neighbour pairs as an object holding only the requested keys
/// (letters of `quantities`, e.g. "ijSdD").
val neighbor_list_js(const fa::Atoms& atoms, double cutoff, bool full, const std::string& quantities) {
    return guarded([&] {
        fa::NeighborOptions options;
        options.cutoff = cutoff;
        options.full = full;
        options.sorted = true;
        const fa::NeighborPairs pairs = fa::neighbor_pairs(atoms, options);
        const std::size_t n = pairs.size();
        val out = val::object();
        for (const char q : quantities) {
            switch (q) {
                case 'i': out.set("i", int32_copy(pairs.i.data(), n)); break;
                case 'j': out.set("j", int32_copy(pairs.j.data(), n)); break;
                case 'S': out.set("S", int32_copy(pairs.S.data(), 3 * n)); break;
                case 'd': out.set("d", float64_copy(pairs.d.data(), n)); break;
                case 'D': out.set("D", float64_copy(pairs.D.data(), 3 * n)); break;
                default:
                    throw std::invalid_argument(std::string("unknown neighbour quantity '") + q +
                                                "' (use letters from i, j, S, d, D)");
            }
        }
        return out;
    });
}
#endif

// ---------------------------------------------------------------------------
// Symmetry (spglib). Compiled only with FASTATOMS_WITH_SYMMETRY; hasSymmetry()
// is always bound so JS can tell the two builds apart.
// ---------------------------------------------------------------------------

bool has_symmetry_js() {
#ifdef FASTATOMS_WITH_SYMMETRY
    return true;
#else
    return false;
#endif
}

#ifdef FASTATOMS_WITH_SYMMETRY
namespace sym = fa::symmetry;

val flat_vec3_copy(const std::vector<fa::Vec3>& points) {
    std::vector<double> flat;
    flat.reserve(3 * points.size());
    for (const auto& p : points) {
        flat.push_back(p.x);
        flat.push_back(p.y);
        flat.push_back(p.z);
    }
    return float64_copy(flat.data(), flat.size());
}

/// Dataset as a plain object with camelCase fields. Per-atom arrays are one
/// entry per input atom; lattices are row-major 3x3 with rows as vectors.
val dataset_js(const fa::Atoms& atoms, double symprec, double angle_tolerance, bool use_labels) {
    return guarded([&]() -> val {
        const sym::SymmetryDataset d =
            sym::get_symmetry_dataset(atoms, symprec, angle_tolerance, use_labels);

        std::vector<std::int32_t> rotations;
        rotations.reserve(9 * d.rotations.size());
        for (const auto& r : d.rotations) {
            for (const auto& row : r) {
                for (const int v : row) {
                    rotations.push_back(v);
                }
            }
        }

        std::vector<std::string> wyckoffs;
        wyckoffs.reserve(d.wyckoffs.size());
        for (const char c : d.wyckoffs) {
            wyckoffs.emplace_back(1, c);
        }

        val out = val::object();
        out.set("number", d.number);
        out.set("international", d.international);
        out.set("hall", d.hall);
        out.set("hallNumber", d.hall_number);
        out.set("choice", d.choice);
        out.set("pointgroup", d.pointgroup.symbol);
        out.set("rotations", int32_copy(rotations.data(), rotations.size()));
        out.set("translations", flat_vec3_copy(d.translations));
        out.set("wyckoffs", string_list(wyckoffs));
        out.set("siteSymmetrySymbols", string_list(d.site_symmetry_symbols));
        out.set("equivalentAtoms", int32_copy(d.equivalent_atoms.data(), d.equivalent_atoms.size()));
        out.set("crystallographicOrbits",
                int32_copy(d.crystallographic_orbits.data(), d.crystallographic_orbits.size()));
        out.set("stdLattice", float64_copy(d.std_lattice.m.data(), 9));
        out.set("stdTypes", int32_copy(d.std_types.data(), d.std_types.size()));
        out.set("stdPositions", flat_vec3_copy(d.std_positions));
        out.set("transformationMatrix", float64_copy(d.transformation_matrix.m.data(), 9));
        out.set("originShift", float64_copy(&d.origin_shift.x, 3));
        out.set("primitiveLattice", float64_copy(d.primitive_lattice.m.data(), 9));
        out.set("mappingToPrimitive",
                int32_copy(d.mapping_to_primitive.data(), d.mapping_to_primitive.size()));
        out.set("stdMappingToPrimitive",
                int32_copy(d.std_mapping_to_primitive.data(), d.std_mapping_to_primitive.size()));
        return out;
    });
}

fa::Atoms standardize_js(const fa::Atoms& atoms, bool to_primitive, bool no_idealize, double symprec,
                         double angle_tolerance, bool use_labels) {
    return guarded([&] {
        return sym::standardize_cell(atoms, to_primitive, no_idealize, symprec, angle_tolerance,
                                     use_labels);
    });
}

fa::Atoms find_primitive_js(const fa::Atoms& atoms, double symprec, double angle_tolerance,
                            bool use_labels) {
    return guarded([&] {
        return sym::find_primitive(atoms, symprec, angle_tolerance, use_labels);
    });
}

fa::Atoms niggli_reduce_js(const fa::Atoms& atoms, double symprec) {
    return guarded([&] { return sym::niggli_reduce(atoms, symprec); });
}
#endif

}  // namespace

EMSCRIPTEN_BINDINGS(fastatoms) {
    class_<fa::Atoms>("Atoms")
        .property("length", &atoms_length)
        .function("getNumbers", &get_numbers)
        .function("getPositions", &get_positions)
        .function("setPositions", &set_positions_js)
        .function("getCell", &get_cell)
        .function("setCell", &set_cell_js)
        .function("getPbc", &get_pbc)
        .function("setPbc", &set_pbc_js)
        .function("getScaledPositions", &get_scaled_positions)
        .function("getSymbols", &get_symbols)
        .function("getLabels", &get_labels)
        .function("getArray", &get_array)
        .function("arrayNames", &array_names)
        .function("getInfoJSON", &info_json)
        .function("repeat", &repeat_js)
        .function("subset", &subset_js)
        .function("wrap", &wrap_js)
        .function("copy", &copy_js)
        .function("volume", &volume_js);

    function("readAll", &read_all_js);
    function("read", &read_js);
    function("write", &write_js);
    function("guessFormat", &guess_format_js);
    function("availableFormats", &available_formats_js);
    function("canWrite", &can_write_js);
    function("fromArrays", &from_arrays_js);
#ifdef FASTATOMS_WITH_NEIGHBORS
    function("neighborList", &neighbor_list_js);
#endif
    function("hasSymmetry", &has_symmetry_js);
#ifdef FASTATOMS_WITH_SYMMETRY
    function("getSymmetryDataset", &dataset_js);
    function("standardizeCell", &standardize_js);
    function("findPrimitive", &find_primitive_js);
    function("niggliReduce", &niggli_reduce_js);
#endif
}
