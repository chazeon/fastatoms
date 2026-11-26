// SPDX-License-Identifier: MIT
#include "fastatoms/formats/xyz.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "fastatoms/core/elements.hpp"
#include "fastatoms/core/value.hpp"
#include "fastatoms/formats/format.hpp"

namespace fastatoms::formats {
namespace {

// ---------------------------------------------------------------------------
// Scanning
// ---------------------------------------------------------------------------

constexpr bool is_space(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f';
}

/// Lines without their terminators. A final newline does not start a new line.
std::vector<std::string_view> split_lines(std::string_view text) {
    std::vector<std::string_view> lines;
    std::size_t start = 0;
    while (start < text.size()) {
        std::size_t stop = text.find('\n', start);
        if (stop == std::string_view::npos) {
            stop = text.size();
        }
        std::string_view line = text.substr(start, stop - start);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        lines.push_back(line);
        start = stop + 1;
    }
    return lines;
}

void split_fields(std::string_view line, std::vector<std::string_view>& fields) {
    fields.clear();
    std::size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && is_space(line[i])) {
            ++i;
        }
        if (i >= line.size()) {
            break;
        }
        const std::size_t start = i;
        while (i < line.size() && !is_space(line[i])) {
            ++i;
        }
        fields.push_back(line.substr(start, i - start));
    }
}

bool parse_long(std::string_view text, long long& value) noexcept {
    // from_chars does not accept a leading '+', which some writers emit.
    if (!text.empty() && text.front() == '+') {
        text.remove_prefix(1);
    }
    if (text.empty()) {
        return false;
    }
    const char* last = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), last, value);
    return ec == std::errc{} && ptr == last;
}

bool parse_double(std::string_view text, double& value) noexcept {
    if (!text.empty() && text.front() == '+') {
        text.remove_prefix(1);
    }
    if (text.empty()) {
        return false;
    }
    const char* last = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), last, value);
    // "nan" and "inf" are legal in a trajectory that went unstable.
    return ec == std::errc{} && ptr == last;
}

/// Atomic number for a species column entry: a symbol, a label such as
/// "Fe1" or "Fe_pv", or a bare atomic number. -1 when none of those.
int species_number(std::string_view token) noexcept {
    const int z = try_number_from_label(token);
    if (z >= 0) {
        return z;
    }
    long long number = 0;
    if (parse_long(token, number) && number >= 0 && number <= kMaxAtomicNumber) {
        return static_cast<int>(number);
    }
    return -1;
}

std::string lowercase(std::string_view text) {
    std::string out(text);
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

[[noreturn]] void fail_at(std::size_t line_no, const std::string& message) {
    throw FormatError("xyz: line " + std::to_string(line_no) + ": " + message);
}

const Value* find_key(const Value::Map& map, std::string_view key) {
    for (const auto& [name, value] : map) {
        if (name == key) {
            return &value;
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// The comment line
// ---------------------------------------------------------------------------

/// The comment-line key that remembers which int arrays were L columns. An int
/// array cannot say it was logical, so the reader keeps the names under this
/// key and the writer writes those arrays back as L. The key is never written
/// to the file itself.
constexpr const char* kLogicalKey = "_logical_columns";

/// Names of the int arrays that the writer must write as L columns.
std::vector<std::string> logical_names(const Atoms& atoms) {
    std::vector<std::string> names;
    const Value* value = atoms.info().find(kLogicalKey);
    if (value != nullptr && value->is_list()) {
        for (const Value& item : value->as_list()) {
            if (item.is_string()) {
                names.push_back(item.as_string());
            }
        }
    }
    return names;
}

/// Words that read as a T/F flag, as ASE reads them.
bool parse_flag(std::string_view token, bool& value) noexcept {
    if (token == "T" || token == "True" || token == "TRUE" || token == "true") {
        value = true;
        return true;
    }
    if (token == "F" || token == "False" || token == "FALSE" || token == "false") {
        value = false;
        return true;
    }
    return false;
}

/// The items of a value: runs of characters that are neither whitespace nor
/// commas, which is how ASE splits a value.
std::vector<std::string_view> value_items(std::string_view text) {
    const auto is_break = [](char c) { return is_space(c) || c == ','; };
    std::vector<std::string_view> items;
    std::size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && is_break(text[i])) {
            ++i;
        }
        const std::size_t start = i;
        while (i < text.size() && !is_break(text[i])) {
            ++i;
        }
        if (i > start) {
            items.push_back(text.substr(start, i - start));
        }
    }
    return items;
}

/// A value typed the way ASE types it, whatever its quoting: if every item is
/// an integer the value is integers, else if every item is a number it is
/// floats, else if every item is a T/F word it is flags. One item is a scalar,
/// several are a list. Anything else, an empty value included, stays a string.
/// The "uid" key is never typed, as in ASE.
Value typed_value(const std::string& key, const std::string& text) {
    const std::vector<std::string_view> items = value_items(text);
    if (items.empty() || lowercase(key) == "uid") {
        return Value(text);
    }
    std::vector<std::int64_t> integers;
    std::vector<double> reals;
    std::vector<bool> flags;
    bool all_integers = true;
    bool all_reals = true;
    bool all_flags = true;
    for (const std::string_view item : items) {
        long long integer = 0;
        double real = 0.0;
        bool flag = false;
        if (parse_long(item, integer)) {
            integers.push_back(integer);
        } else {
            all_integers = false;
        }
        if (parse_double(item, real)) {
            reals.push_back(real);
        } else {
            all_reals = false;
        }
        if (parse_flag(item, flag)) {
            flags.push_back(flag);
        } else {
            all_flags = false;
        }
    }

    const bool scalar = items.size() == 1;
    if (all_integers) {
        if (scalar) {
            return Value(integers.front());
        }
        Value::List list;
        for (const std::int64_t integer : integers) {
            list.emplace_back(integer);
        }
        return Value(std::move(list));
    }
    if (all_reals) {
        if (scalar) {
            return Value(reals.front());
        }
        Value::List list;
        for (const double real : reals) {
            list.emplace_back(real);
        }
        return Value(std::move(list));
    }
    if (all_flags) {
        if (scalar) {
            return Value(static_cast<bool>(flags.front()));
        }
        Value::List list;
        for (const bool flag : flags) {
            list.emplace_back(flag);
        }
        return Value(std::move(list));
    }
    return Value(text);
}

/// The key=value pairs of a comment line, tokenized the way ASE tokenizes it. A
/// key may be followed by '=' and a value. Whitespace ends a pair, except right
/// after a bare '=', so "m = 7" is one pair. Quotes, brackets and braces group
/// text and are dropped; a backslash escapes the next character. A bare word is
/// a flag (true). A repeated key keeps its last value.
Value::Map parse_key_values(std::string_view line) {
    // Each pair is its key, then its value segments. A second '=' stays in the
    // value, so "x=a=b" reads as x = "a=b".
    using Segments = std::vector<std::string>;
    std::vector<Segments> pairs{Segments{std::string()}};
    char closer = '\0';
    bool escaped = false;
    // A quoted or bracketed group counts as content even when it is empty, so
    // that `e="" n=2` does not run n into e's value (ASE does; this does not).
    bool grouped = false;
    for (const char c : line) {
        std::string& text = pairs.back().back();
        if (escaped) {
            text += c;
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (closer != '\0') {
            if (c == closer) {
                closer = '\0';
            } else {
                text += c;
            }
        } else if (c == '"' || c == '\'' || c == '{' || c == '[') {
            closer = c == '{' ? '}' : c == '[' ? ']' : c;
            grouped = true;
        } else if (is_space(c)) {
            if (!text.empty() || grouped) {
                pairs.push_back(Segments{std::string()});
                grouped = false;
            }
        } else if (c == '=') {
            if (pairs.back().size() == 1 && text.empty() && !grouped) {
                pairs.pop_back();  // a pair that is only spaces, before this '='
            }
            if (pairs.empty()) {
                pairs.push_back(Segments{std::string()});
            }
            pairs.back().push_back(std::string());
            grouped = false;
        } else {
            text += c;
        }
    }

    Value::Map fields;
    for (const Segments& pair : pairs) {
        const std::string& key = pair.front();
        if (key.empty()) {
            continue;
        }
        std::string text = "T";  // a bare word is a flag
        if (pair.size() > 1) {
            text = pair[1];
            for (std::size_t k = 2; k < pair.size(); ++k) {
                text += '=';
                text += pair[k];
            }
        }
        Value value = typed_value(key, text);
        const auto existing = std::find_if(fields.begin(), fields.end(),
                                           [&key](const auto& field) { return field.first == key; });
        if (existing != fields.end()) {
            existing->second = std::move(value);
        } else {
            fields.emplace_back(key, std::move(value));
        }
    }
    return fields;
}

// ---------------------------------------------------------------------------
// Columns
// ---------------------------------------------------------------------------

/// What a Properties column means for the structure.
enum class Role {
    Species,     ///< species, Z, numbers, symbol
    Position,    ///< pos
    Forces,      ///< forces (R:3) -> array "forces"
    Velocities,  ///< velo, velocities (R:3): set_velocities, verbatim
    Momenta,     ///< momenta (R:3): set_momenta, velocity = momentum / mass
    Masses,      ///< masses (R:1)
    Charges,     ///< charges or initial_charges (R:1)
    Tags,        ///< tags (I:1)
    Real,        ///< any other R column -> double array
    Integer,     ///< any other I column -> int array
    Logical,     ///< L column: int array of 0/1, named under kLogicalKey
    Text,        ///< S column -> string array
};

struct Column {
    std::string name;
    char type{'R'};
    std::size_t count{1};
    std::size_t offset{0};  ///< first token of this column within an atom line
    Role role{Role::Real};
};

/// The values of one column across all atoms of a frame.
struct Buffer {
    std::vector<double> reals;
    std::vector<std::int64_t> ints;
    std::vector<std::string> texts;
};

std::vector<Column> parse_properties(std::string_view spec, std::size_t line_no) {
    std::vector<std::string_view> parts;
    std::size_t start = 0;
    while (true) {
        const std::size_t colon = spec.find(':', start);
        parts.push_back(spec.substr(start, colon == std::string_view::npos
                                               ? std::string_view::npos
                                               : colon - start));
        if (colon == std::string_view::npos) {
            break;
        }
        start = colon + 1;
    }
    if (parts.size() % 3 != 0) {
        fail_at(line_no, "Properties '" + std::string(spec) + "' is not name:type:count triples");
    }

    std::vector<Column> columns;
    for (std::size_t i = 0; i < parts.size(); i += 3) {
        const std::string_view name = parts[i];
        const std::string_view type = parts[i + 1];
        const std::string_view count = parts[i + 2];
        if (name.empty()) {
            fail_at(line_no, "Properties '" + std::string(spec) + "' has an empty column name");
        }
        if (type.size() != 1) {
            fail_at(line_no, "Properties type '" + std::string(type) + "' is not S, R, I or L");
        }
        const char kind = static_cast<char>(std::toupper(static_cast<unsigned char>(type[0])));
        if (kind != 'S' && kind != 'R' && kind != 'I' && kind != 'L') {
            fail_at(line_no, "Properties type '" + std::string(type) + "' is not S, R, I or L");
        }
        long long width = 0;
        if (!parse_long(count, width) || width <= 0 || width > 1000000) {
            fail_at(line_no, "Properties count '" + std::string(count) + "' is not a positive integer");
        }
        columns.push_back(Column{std::string(name), kind, static_cast<std::size_t>(width), 0,
                                 Role::Real});
    }
    if (columns.empty()) {
        fail_at(line_no, "Properties is empty");
    }
    return columns;
}

Role role_of(const Column& column) {
    const std::string name = lowercase(column.name);
    const std::size_t count = column.count;
    if (name == "species" || name == "z" || name == "numbers" || name == "symbol") {
        return Role::Species;
    }
    if (name == "pos" || name == "positions") {
        return Role::Position;
    }
    if (column.type == 'S') {
        return Role::Text;
    }
    if (column.type == 'L') {
        return Role::Logical;
    }
    if (column.type == 'I') {
        return (name == "tags" || name == "tag") && count == 1 ? Role::Tags : Role::Integer;
    }
    if ((name == "forces" || name == "force") && count == 3) {
        return Role::Forces;
    }
    if ((name == "velo" || name == "velocities" || name == "vel") && count == 3) {
        return Role::Velocities;
    }
    if ((name == "momenta" || name == "momentum") && count == 3) {
        return Role::Momenta;
    }
    if ((name == "masses" || name == "mass") && count == 1) {
        return Role::Masses;
    }
    if ((name == "charges" || name == "charge" || name == "initial_charges") && count == 1) {
        return Role::Charges;
    }
    return Role::Real;
}

/// Assign token offsets and roles, and check that species and pos are present.
/// Returns the number of tokens each atom line must carry.
std::size_t finalize_columns(std::vector<Column>& columns, std::size_t line_no) {
    std::size_t offset = 0;
    bool has_species = false;
    bool has_position = false;
    for (Column& column : columns) {
        column.role = role_of(column);
        column.offset = offset;
        offset += column.count;
        if (column.role == Role::Species) {
            if (has_species) {
                fail_at(line_no, "Properties names more than one species column");
            }
            if (column.count != 1) {
                fail_at(line_no, "the species column must have count 1");
            }
            has_species = true;
        } else if (column.role == Role::Position) {
            if (has_position) {
                fail_at(line_no, "Properties names more than one pos column");
            }
            if (column.type != 'R' || column.count != 3) {
                fail_at(line_no, "the pos column must be R:3");
            }
            has_position = true;
        } else if (column.role == Role::Text && column.count != 1) {
            fail_at(line_no, "string column '" + column.name + "' must have count 1");
        }
    }
    if (!has_species) {
        fail_at(line_no, "Properties names no species column");
    }
    if (!has_position) {
        fail_at(line_no, "Properties names no pos column");
    }
    return offset;
}

std::int64_t parse_integer(const Column& column, std::string_view token, std::size_t line_no) {
    long long value = 0;
    if (parse_long(token, value)) {
        return value;
    }
    fail_at(line_no, "column '" + column.name + "' value '" + std::string(token) +
                         "' is not an integer");
}

std::int64_t parse_logical(const Column& column, std::string_view token, std::size_t line_no) {
    if (token == "T" || token == "True" || token == "true" || token == "1") {
        return 1;
    }
    if (token == "F" || token == "False" || token == "false" || token == "0") {
        return 0;
    }
    fail_at(line_no, "column '" + column.name + "' value '" + std::string(token) +
                         "' is not a logical (T or F)");
}

double parse_real(const Column& column, std::string_view token, std::size_t line_no) {
    double value = 0.0;
    if (parse_double(token, value)) {
        return value;
    }
    fail_at(line_no, "column '" + column.name + "' value '" + std::string(token) +
                         "' is not a number");
}

/// Consume one column's tokens from an atom line.
void read_column(const Column& column, Buffer& buffer, const std::vector<std::string_view>& fields,
                 std::vector<int>& numbers, std::size_t line_no) {
    const std::string_view* tokens = fields.data() + column.offset;
    switch (column.role) {
        case Role::Species: {
            const int z = species_number(tokens[0]);
            if (z < 0) {
                fail_at(line_no, "'" + std::string(tokens[0]) +
                                     "' is not a chemical element or species label");
            }
            numbers.push_back(z);
            buffer.texts.emplace_back(tokens[0]);
            break;
        }
        case Role::Text:
            buffer.texts.emplace_back(tokens[0]);
            break;
        case Role::Integer:
        case Role::Tags:
            for (std::size_t k = 0; k < column.count; ++k) {
                buffer.ints.push_back(parse_integer(column, tokens[k], line_no));
            }
            break;
        case Role::Logical:
            for (std::size_t k = 0; k < column.count; ++k) {
                buffer.ints.push_back(parse_logical(column, tokens[k], line_no));
            }
            break;
        default:
            for (std::size_t k = 0; k < column.count; ++k) {
                buffer.reals.push_back(parse_real(column, tokens[k], line_no));
            }
            break;
    }
}

/// Numbers in a Lattice-like value: one number, or a list of them.
std::vector<double> numbers_of(const Value& value, std::size_t line_no, const std::string& key) {
    std::vector<double> out;
    if (value.is_number()) {
        out.push_back(value.as_double());
    } else if (value.is_list()) {
        for (const Value& item : value.as_list()) {
            if (!item.is_number()) {
                fail_at(line_no, "'" + key + "' must hold only numbers");
            }
            out.push_back(item.as_double());
        }
    } else {
        fail_at(line_no, "'" + key + "' must hold numbers");
    }
    return out;
}

/// The pbc flags: one T or F for all axes, or one per axis. "T F T" reads as a
/// list of bools.
std::array<bool, 3> pbc_of(const Value& value, std::size_t line_no) {
    const auto flag = [line_no](const Value& item) -> bool {
        if (item.is_bool()) {
            return item.as_bool();
        }
        if (item.is_number()) {
            return item.as_double() != 0.0;
        }
        fail_at(line_no, "pbc must be T or F");
    };
    std::vector<bool> flags;
    if (value.is_list()) {
        for (const Value& item : value.as_list()) {
            flags.push_back(flag(item));
        }
    } else {
        flags.push_back(flag(value));
    }
    if (flags.size() == 1) {
        const bool on = flags.front();
        return {on, on, on};
    }
    if (flags.size() != 3) {
        fail_at(line_no, "pbc must give three flags");
    }
    return {static_cast<bool>(flags[0]), static_cast<bool>(flags[1]), static_cast<bool>(flags[2])};
}

/// Hand a finished column to the structure. Velocities and momenta come after
/// the masses, because converting velocities to momenta needs them.
void apply_column(Atoms& atoms, const Column& column, Buffer& buffer) {
    switch (column.role) {
        case Role::Forces:
            atoms.set_array("forces", buffer.reals, 3);
            break;
        case Role::Velocities:
            atoms.set_velocities(buffer.reals);
            break;
        case Role::Momenta:
            atoms.set_momenta(buffer.reals);
            break;
        case Role::Masses:
            atoms.set_masses(buffer.reals);
            break;
        case Role::Charges:
            atoms.set_charges(buffer.reals);
            break;
        case Role::Tags:
            atoms.set_tags(std::vector<int>(buffer.ints.begin(), buffer.ints.end()));
            break;
        case Role::Real:
            atoms.set_array(column.name, buffer.reals, column.count);
            break;
        case Role::Integer:
        case Role::Logical:
            atoms.set_int_array(column.name, buffer.ints, column.count);
            break;
        case Role::Text:
            atoms.set_string_array(column.name, buffer.texts);
            break;
        default:
            break;
    }
}

/// Order in which columns are applied: masses before velocities, and the
/// string columns last, so that a `labels` column overrides the species labels.
int apply_stage(Role role) {
    switch (role) {
        case Role::Velocities:
        case Role::Momenta:
            return 1;
        case Role::Text:
            return 2;
        case Role::Species:
        case Role::Position:
            return -1;
        default:
            return 0;
    }
}

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

/// Read the frame whose atom count is on line `next`, and advance `next` past
/// it. `fields` is scratch space.
Atoms read_frame(const std::vector<std::string_view>& lines, std::size_t& next,
                 std::vector<std::string_view>& fields) {
    const std::size_t count_index = next;
    split_fields(lines[count_index], fields);
    long long natoms = 0;
    if (fields.empty() || !parse_long(fields.front(), natoms) || natoms < 0) {
        fail_at(count_index + 1,
                "expected an atom count, found '" +
                    std::string(fields.empty() ? std::string_view{} : fields.front()) + "'");
    }
    const auto count = static_cast<std::size_t>(natoms);

    const std::size_t comment_index = count_index + 1;
    if (comment_index >= lines.size()) {
        fail_at(count_index + 1, "the frame ends before its comment line");
    }
    const std::string_view comment = lines[comment_index];
    const std::size_t comment_line = comment_index + 1;

    // A comment with key=value pairs is extended XYZ; anything else is plain.
    const bool extended = comment.find('=') != std::string_view::npos;
    Value::Map info;
    std::vector<Column> columns;
    if (extended) {
        info = parse_key_values(comment);
        if (const Value* properties = find_key(info, "Properties"); properties != nullptr) {
            if (!properties->is_string()) {
                fail_at(comment_line, "Properties must be a string");
            }
            columns = parse_properties(properties->as_string(), comment_line);
        } else {
            columns = {Column{"species", 'S', 1, 0, Role::Species},
                       Column{"pos", 'R', 3, 0, Role::Position}};
        }
    } else {
        info.emplace_back("comment", Value(std::string(comment)));
        columns = {Column{"species", 'S', 1, 0, Role::Species},
                   Column{"pos", 'R', 3, 0, Role::Position}};
    }

    const std::size_t tokens_per_atom = finalize_columns(columns, comment_line);
    std::vector<Buffer> buffers(columns.size());
    std::vector<int> numbers;
    numbers.reserve(std::min(count, lines.size()));

    std::size_t row = comment_index + 1;
    for (std::size_t atom = 0; atom < count; ++atom, ++row) {
        if (row >= lines.size()) {
            fail_at(count_index + 1, "the frame declares " + std::to_string(count) +
                                         " atoms but the text ends after " +
                                         std::to_string(atom));
        }
        const std::size_t line_no = row + 1;
        split_fields(lines[row], fields);
        if (fields.size() < tokens_per_atom) {
            fail_at(line_no, "atom line has " + std::to_string(fields.size()) +
                                 " columns, Properties declares " +
                                 std::to_string(tokens_per_atom));
        }
        for (std::size_t c = 0; c < columns.size(); ++c) {
            read_column(columns[c], buffers[c], fields, numbers, line_no);
        }
    }
    next = row;

    // Positions and the species labels come from their own columns.
    std::vector<double> positions;
    std::vector<std::string> labels;
    for (std::size_t c = 0; c < columns.size(); ++c) {
        if (columns[c].role == Role::Position) {
            positions = std::move(buffers[c].reals);
        } else if (columns[c].role == Role::Species) {
            labels = std::move(buffers[c].texts);
        }
    }

    Atoms atoms;
    try {
        atoms = Atoms(std::move(numbers), std::move(positions));
        atoms.set_string_array("labels", labels);

        // The cell first, then pbc: a Lattice makes the file periodic unless
        // a pbc key says otherwise.
        if (const Value* lattice = find_key(info, "Lattice"); lattice != nullptr) {
            const std::vector<double> entries = numbers_of(*lattice, comment_line, "Lattice");
            if (entries.size() == 9) {
                atoms.set_cell(Mat3(entries[0], entries[1], entries[2], entries[3], entries[4],
                                    entries[5], entries[6], entries[7], entries[8]));
            } else if (entries.size() == 3) {
                atoms.set_cell(Mat3::diagonal(entries[0], entries[1], entries[2]));
            } else {
                fail_at(comment_line, "Lattice has " + std::to_string(entries.size()) +
                                          " numbers, expected 9 or 3");
            }
            atoms.set_pbc(true);
        }
        if (extended) {
            if (const Value* pbc = find_key(info, "pbc"); pbc != nullptr) {
                atoms.set_pbc(pbc_of(*pbc, comment_line));
            }
        }

        for (int stage = 0; stage <= 2; ++stage) {
            for (std::size_t c = 0; c < columns.size(); ++c) {
                if (apply_stage(columns[c].role) == stage) {
                    apply_column(atoms, columns[c], buffers[c]);
                }
            }
        }
    } catch (const std::invalid_argument& error) {
        fail_at(comment_line, error.what());
    }

    Value::Map carried;
    for (const auto& [key, value] : info) {
        if (key == kLogicalKey ||
            (extended && (key == "Lattice" || key == "Properties" || key == "pbc"))) {
            continue;
        }
        carried.emplace_back(key, value);
    }
    // Remember the L columns, so the writer can write them back as L.
    Value::List logical;
    for (const Column& column : columns) {
        if (column.role == Role::Logical) {
            logical.emplace_back(column.name);
        }
    }
    if (!logical.empty()) {
        carried.emplace_back(kLogicalKey, Value(std::move(logical)));
    }
    atoms.set_info(Value(std::move(carried)));
    return atoms;
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Shortest text that reads back as the same double (precision 0), or
/// `precision` significant digits. to_chars never looks at the locale, so a
/// file written here is readable by every other reader.
void append_real(std::string& out, double value, int precision) {
    char buffer[64];
    const auto result =
        precision > 0 ? std::to_chars(buffer, buffer + sizeof buffer, value,
                                      std::chars_format::general, precision)
                      : std::to_chars(buffer, buffer + sizeof buffer, value);
    if (result.ec == std::errc{}) {
        out.append(buffer, static_cast<std::size_t>(result.ptr - buffer));
    } else {
        out += '0';
    }
}

/// A float in info. "2" would read back as an integer, so a float that prints
/// without a point or an exponent gets ".0".
void append_float(std::string& out, double value, int precision) {
    const std::size_t start = out.size();
    append_real(out, value, precision);
    const std::string_view written(out.data() + start, out.size() - start);
    if (written.find_first_of(".eEni") == std::string_view::npos) {
        out += ".0";
    }
}

void append_quoted(std::string& out, std::string_view text) {
    out += '"';
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
        }
        out += c;
    }
    out += '"';
}

/// A value that must survive being one whitespace-delimited token.
void check_token(std::string_view text, const std::string& what) {
    if (text.empty() || text.find_first_of(" \t\r\n") != std::string_view::npos) {
        throw FormatError("xyz: " + what + " '" + std::string(text) +
                          "' must be one non-empty token without whitespace");
    }
}

void check_key(std::string_view key) {
    if (key.empty() || key.find_first_of(" \t\r\n=\"") != std::string_view::npos) {
        throw FormatError("xyz: info key '" + std::string(key) +
                          "' cannot be written on a comment line");
    }
}

void append_info_value(std::string& out, const std::string& key, const Value& value,
                       int precision) {
    switch (value.type()) {
        case Value::Type::Bool:
            out += value.as_bool() ? 'T' : 'F';
            break;
        case Value::Type::Int:
            out += std::to_string(value.as_int());
            break;
        case Value::Type::Float:
            append_float(out, value.as_double(), precision);
            break;
        case Value::Type::String:
            // ASE has no escape for a line break: it writes the break raw and
            // the file no longer reads back, so refuse it here.
            if (value.as_string().find_first_of("\r\n") != std::string::npos) {
                throw FormatError("xyz: info key '" + key +
                                  "' holds a line break, which a comment line cannot carry");
            }
            append_quoted(out, value.as_string());
            break;
        case Value::Type::List: {
            // All numbers, or all T/F flags, as typed_value reads them back.
            const Value::List& items = value.as_list();
            const bool flags = !items.empty() && items.front().is_bool();
            std::string body;
            for (const Value& item : items) {
                if (!body.empty()) {
                    body += ' ';
                }
                if (flags && item.is_bool()) {
                    body += item.as_bool() ? 'T' : 'F';
                } else if (!flags && item.is_int()) {
                    body += std::to_string(item.as_int());
                } else if (!flags && item.is_float()) {
                    append_float(body, item.as_double(), precision);
                } else {
                    throw FormatError("xyz: info key '" + key + "' holds a list that is not all "
                                      "numbers or all T/F flags, which a comment line cannot carry");
                }
            }
            append_quoted(out, body);
            break;
        }
        default:
            throw FormatError("xyz: info key '" + key + "' holds a mapping, which a comment line "
                              "cannot carry");
    }
}

/// One output column. Data is borrowed from the Atoms or from locals that
/// outlive the column list, so the pointers never dangle.
struct OutColumn {
    std::string name;
    char type{'R'};  ///< R, I, L or S
    std::size_t count{1};
    const double* reals{nullptr};
    const std::int64_t* ints{nullptr};
    const std::vector<std::string>* texts{nullptr};
};

void append_plain(std::string& out, const Atoms& atoms, const XyzWriteOptions& options) {
    out += std::to_string(atoms.size());
    out += '\n';

    std::string comment;
    if (const Value* value = atoms.info().find("comment"); value != nullptr) {
        comment = value->is_string() ? value->as_string() : value->to_string();
    }
    if (comment.find_first_of("\r\n") != std::string::npos) {
        throw FormatError("xyz: a comment line cannot contain a line break");
    }
    out += comment;
    out += '\n';

    const std::vector<std::string> labels = atoms.labels();
    const std::vector<double>& positions = atoms.positions();
    for (std::size_t atom = 0; atom < atoms.size(); ++atom) {
        check_token(labels[atom], "label");
        out += labels[atom];
        for (std::size_t k = 0; k < 3; ++k) {
            out += ' ';
            append_real(out, positions[3 * atom + k], options.precision);
        }
        out += '\n';
    }
}

void append_extended(std::string& out, const Atoms& atoms, const XyzWriteOptions& options) {
    const std::size_t count = atoms.size();
    const int precision = options.precision;

    // Columns are gathered before the header is written, so the header and the
    // body cannot disagree. Local vectors hold the values that the Atoms only
    // returns by value.
    const std::vector<std::string> symbol_list = atoms.symbols();
    const std::vector<std::string> label_list = atoms.labels();
    const bool write_labels = atoms.has_string_array("labels") && label_list != symbol_list;

    // Velocities are written verbatim as velo:R:3: exact, and it survives
    // massless (mass 0) atoms, where momenta would lose them. ASE has no
    // velocity column. It reads velo as a plain array named "velo", and it reads
    // momenta:R:3 as momenta (velocity = momentum / mass). So ASE sees velo as
    // data, not as momenta. We never write momenta.
    std::vector<double> velocities;
    if (atoms.has_velocities()) {
        velocities = atoms.velocities();
    }
    std::vector<double> masses;
    if (atoms.has_masses()) {
        masses = atoms.masses();
    }
    std::vector<double> charges;
    if (atoms.has_charges()) {
        charges = atoms.charges();
    }
    std::vector<std::int64_t> tags;
    if (atoms.has_tags()) {
        for (const int tag : atoms.tags()) {
            tags.push_back(tag);
        }
    }

    std::vector<OutColumn> columns;
    columns.push_back(OutColumn{"species", 'S', 1, nullptr, nullptr, &symbol_list});
    columns.push_back(OutColumn{"pos", 'R', 3, atoms.positions().data(), nullptr, nullptr});
    if (atoms.has_velocities()) {
        columns.push_back(OutColumn{"velo", 'R', 3, velocities.data(), nullptr, nullptr});
    }
    if (atoms.has_masses()) {
        columns.push_back(OutColumn{"masses", 'R', 1, masses.data(), nullptr, nullptr});
    }
    if (atoms.has_charges()) {
        // ASE's name for per-atom charges.
        columns.push_back(OutColumn{"initial_charges", 'R', 1, charges.data(), nullptr, nullptr});
    }
    if (atoms.has_tags()) {
        columns.push_back(OutColumn{"tags", 'I', 1, nullptr, tags.data(), nullptr});
    }

    std::vector<std::string> used{"species",  "pos",   "velo",         "momenta",
                                  "masses",   "charges", "initial_charges", "tags"};
    const auto claim = [&used](const std::string& name) {
        check_token(name, "array name");
        if (name.find(':') != std::string::npos) {
            throw FormatError("xyz: array name '" + name + "' cannot contain ':'");
        }
        if (std::find(used.begin(), used.end(), name) != used.end()) {
            throw FormatError("xyz: array name '" + name + "' clashes with another column");
        }
        used.push_back(name);
    };

    for (const std::string& name : atoms.array_names()) {
        claim(name);
        const Atoms::Array& array = atoms.array(name);
        columns.push_back(OutColumn{name, 'R', array.ncols, array.data.data(), nullptr, nullptr});
    }
    const std::vector<std::string> logical = logical_names(atoms);
    for (const std::string& name : atoms.int_array_names()) {
        claim(name);
        const Atoms::IntArray& array = atoms.int_array(name);
        const bool is_logical = std::find(logical.begin(), logical.end(), name) != logical.end();
        columns.push_back(OutColumn{name, is_logical ? 'L' : 'I', array.ncols, nullptr,
                                    array.data.data(), nullptr});
    }
    if (write_labels) {
        columns.push_back(OutColumn{"labels", 'S', 1, nullptr, nullptr, &label_list});
    }
    for (const std::string& name : atoms.string_array_names()) {
        if (name == "labels") {
            continue;
        }
        claim(name);
        columns.push_back(OutColumn{name, 'S', 1, nullptr, nullptr, &atoms.string_array(name)});
    }

    out += std::to_string(count);
    out += '\n';

    if (!atoms.cell().is_zero()) {
        const Mat3& matrix = atoms.cell().matrix();
        out += "Lattice=\"";
        for (std::size_t k = 0; k < 9; ++k) {
            if (k != 0) {
                out += ' ';
            }
            append_real(out, matrix.m[k], precision);
        }
        out += "\" ";
    }

    out += "Properties=";
    for (std::size_t c = 0; c < columns.size(); ++c) {
        if (c != 0) {
            out += ':';
        }
        out += columns[c].name;
        out += ':';
        out += columns[c].type;
        out += ':';
        out += std::to_string(columns[c].count);
    }

    const std::array<bool, 3> pbc = atoms.pbc();
    out += " pbc=\"";
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (axis != 0) {
            out += ' ';
        }
        out += pbc[axis] ? 'T' : 'F';
    }
    out += '"';

    if (atoms.info().is_map()) {
        for (const auto& [key, value] : atoms.info().as_map()) {
            if (key == "Lattice" || key == "Properties" || key == "pbc" || key == kLogicalKey ||
                value.is_null()) {
                continue;
            }
            check_key(key);
            out += ' ';
            out += key;
            out += '=';
            append_info_value(out, key, value, precision);
        }
    }
    out += '\n';

    for (std::size_t atom = 0; atom < count; ++atom) {
        for (std::size_t c = 0; c < columns.size(); ++c) {
            const OutColumn& column = columns[c];
            if (c != 0) {
                out += ' ';
            }
            if (column.type == 'S') {
                const std::string& text = (*column.texts)[atom];
                check_token(text, "value in column '" + column.name + "'");
                out += text;
                continue;
            }
            for (std::size_t k = 0; k < column.count; ++k) {
                if (k != 0) {
                    out += ' ';
                }
                const std::size_t index = atom * column.count + k;
                if (column.type == 'R') {
                    append_real(out, column.reals[index], precision);
                } else if (column.type == 'L') {
                    out += column.ints[index] != 0 ? 'T' : 'F';
                } else {
                    out += std::to_string(column.ints[index]);
                }
            }
        }
        out += '\n';
    }
}

void append_frame(std::string& out, const Atoms& atoms, const XyzWriteOptions& options) {
    if (options.extended) {
        append_extended(out, atoms, options);
    } else {
        append_plain(out, atoms, options);
    }
}

}  // namespace

std::vector<Atoms> read_xyz(std::string_view text) {
    const std::vector<std::string_view> lines = split_lines(text);
    std::vector<Atoms> frames;
    std::vector<std::string_view> fields;
    std::size_t next = 0;
    while (true) {
        // Blank lines between frames are tolerated.
        while (next < lines.size()) {
            split_fields(lines[next], fields);
            if (!fields.empty()) {
                break;
            }
            ++next;
        }
        if (next >= lines.size()) {
            break;
        }
        frames.push_back(read_frame(lines, next, fields));
    }
    return frames;
}

std::string write_xyz(const Atoms& atoms, const XyzWriteOptions& options) {
    std::string out;
    append_frame(out, atoms, options);
    return out;
}

std::string write_xyz(const std::vector<Atoms>& frames, const XyzWriteOptions& options) {
    std::string out;
    for (const Atoms& atoms : frames) {
        append_frame(out, atoms, options);
    }
    return out;
}

}  // namespace fastatoms::formats
