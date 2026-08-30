// SPDX-License-Identifier: MIT
#include "fastatoms/io/extxyz.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <utility>

#include "fastatoms/core/elements.hpp"
#include "fastatoms/util/text.hpp"

#if !defined(_WIN32)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace fastatoms::io {
namespace {

// ---------------------------------------------------------------------------
// Scanning primitives
//
// Everything here works on a (pointer, end) pair and never allocates. That is
// the whole performance story: the obvious implementation, a std::istringstream
// per line, spends its time in locale lookups and heap traffic.
// ---------------------------------------------------------------------------
constexpr bool is_space(char c) noexcept { return c == ' ' || c == '\t' || c == '\r'; }
constexpr bool is_line_end(char c) noexcept { return c == '\n'; }

inline const char* skip_blank(const char* p, const char* end) noexcept {
    while (p < end && is_space(*p)) {
        ++p;
    }
    return p;
}

inline const char* skip_to_line_end(const char* p, const char* end) noexcept {
    while (p < end && !is_line_end(*p)) {
        ++p;
    }
    return p;
}

/// Advance past the newline that ends the current line.
inline const char* next_line(const char* p, const char* end) noexcept {
    p = skip_to_line_end(p, end);
    return p < end ? p + 1 : end;
}

/// The next whitespace-delimited token, or an empty view at end of line.
inline std::string_view next_token(const char*& p, const char* end) noexcept {
    p = skip_blank(p, end);
    if (p >= end || is_line_end(*p)) {
        return {};
    }
    const char* start = p;
    while (p < end && !is_space(*p) && !is_line_end(*p)) {
        ++p;
    }
    return {start, static_cast<std::size_t>(p - start)};
}

[[nodiscard]] bool to_double(std::string_view text, double& value) noexcept {
    if (text.empty()) {
        return false;
    }
    const char* first = text.data();
    const char* last = first + text.size();
    // from_chars does not accept a leading '+', which some writers emit.
    if (*first == '+') {
        ++first;
    }
    const auto [ptr, ec] = std::from_chars(first, last, value);
    if (ec != std::errc{}) {
        // "nan" and "inf" are legal in a trajectory that went unstable, and
        // are worth reading back rather than refusing.
        return false;
    }
    return ptr == last;
}

[[nodiscard]] bool to_long(std::string_view text, long long& value) noexcept {
    if (text.empty()) {
        return false;
    }
    const char* first = text.data();
    const char* last = first + text.size();
    if (*first == '+') {
        ++first;
    }
    const auto [ptr, ec] = std::from_chars(first, last, value);
    return ec == std::errc{} && ptr == last;
}

/// Chemical symbol to atomic number, without a string allocation.
///
/// A trajectory of a million atoms asks this a million times per frame, and
/// the elements arrive in runs, so the last answer is remembered. Behind that
/// is a flat table keyed on the symbol packed into an integer, which turns the
/// lookup into one comparison instead of a walk through the element list.
class SymbolTable {
public:
    SymbolTable() {
        entries_.fill(0);
        keys_.fill(0);
        for (int z = 1; z <= kMaxElement; ++z) {
            const std::string_view symbol = symbol_from_number(z);
            if (symbol.empty()) {
                continue;
            }
            const std::uint32_t key = pack(symbol);
            if (key == 0) {
                continue;
            }
            // Open addressing with linear probing. Without the probe, two
            // symbols landing on the same slot silently lose one of them -
            // which is a table that works for most elements and not for Ar.
            std::size_t index = slot(key);
            while (keys_[index] != 0 && keys_[index] != key) {
                index = (index + 1) % kSlots;
            }
            keys_[index] = key;
            entries_[index] = static_cast<std::int32_t>(z);
        }
    }

    [[nodiscard]] int lookup(std::string_view symbol) const {
        const std::uint32_t key = pack(symbol);
        if (key != 0) {
            if (key == last_key_) {
                return last_z_;
            }
            std::size_t index = slot(key);
            for (std::size_t probe = 0; probe < kSlots; ++probe) {
                if (keys_[index] == 0) {
                    break;                       // absent
                }
                if (keys_[index] == key) {
                    last_key_ = key;
                    last_z_ = entries_[index];
                    return last_z_;
                }
                index = (index + 1) % kSlots;
            }
        }
        // A number in the species column, as some writers emit.
        long long z = 0;
        if (to_long(symbol, z) && z >= 0 && z <= kMaxElement) {
            return static_cast<int>(z);
        }
        return -1;
    }

private:
    static constexpr int kMaxElement = 118;
    static constexpr std::size_t kSlots = 512;   // 118 elements: a low load factor

    /// Up to three characters, case-normalised the way symbols are written.
    static std::uint32_t pack(std::string_view symbol) noexcept {
        if (symbol.empty() || symbol.size() > 3) {
            return 0;
        }
        std::uint32_t key = 0;
        for (std::size_t k = 0; k < symbol.size(); ++k) {
            const auto c = static_cast<unsigned char>(symbol[k]);
            if (std::isalpha(c) == 0) {
                return 0;
            }
            const auto normalised = static_cast<std::uint32_t>(
                k == 0 ? std::toupper(c) : std::tolower(c));
            key = (key << 8) | normalised;
        }
        return key | (static_cast<std::uint32_t>(symbol.size()) << 24);
    }

    static std::size_t slot(std::uint32_t key) noexcept {
        return (key * 2654435761u) % kSlots;
    }

    std::array<std::int32_t, kSlots> entries_{};
    std::array<std::uint32_t, kSlots> keys_{};
    mutable std::uint32_t last_key_{0};
    mutable int last_z_{-1};
};

const SymbolTable& symbols() {
    static const SymbolTable table;
    return table;
}

/// The numbers in a whitespace-separated string.
///
/// `Lattice` arrives as one quoted string, not as a list, so the nine numbers
/// have to be pulled out of it rather than asked for.
std::vector<double> numbers_in(std::string_view text) {
    std::vector<double> values;
    const char* p = text.data();
    const char* const end = p + text.size();
    while (p < end) {
        while (p < end && (is_space(*p) || is_line_end(*p) || *p == ',')) {
            ++p;
        }
        if (p >= end) {
            break;
        }
        const char* start = p;
        while (p < end && !is_space(*p) && !is_line_end(*p) && *p != ',') {
            ++p;
        }
        double value = 0.0;
        if (!to_double(std::string_view(start, static_cast<std::size_t>(p - start)), value)) {
            return {};
        }
        values.push_back(value);
    }
    return values;
}

/// The numbers a value holds, however it was written: a quoted string of them,
/// a list, or a single number.
std::vector<double> value_numbers(const Value& value) {
    if (value.is_string()) {
        return numbers_in(value.as_string());
    }
    if (value.is_number()) {
        return {value.as_double()};
    }
    if (value.is_list()) {
        std::vector<double> values;
        for (const Value& item : value.as_list()) {
            if (item.is_number()) {
                values.push_back(item.as_double());
            } else if (item.is_string()) {
                const std::vector<double> inner = numbers_in(item.as_string());
                values.insert(values.end(), inner.begin(), inner.end());
            }
        }
        return values;
    }
    return {};
}

/// Line number of an offset, for an error message worth reading.
std::size_t line_of(const char* begin, const char* position) {
    std::size_t line = 1;
    for (const char* p = begin; p < position; ++p) {
        if (*p == '\n') {
            ++line;
        }
    }
    return line;
}

[[noreturn]] void fail(const char* begin, const char* position, const std::string& message) {
    throw ExtxyzError("extxyz: line " + std::to_string(line_of(begin, position)) + ": " +
                      message);
}

}  // namespace

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------
std::string PropertyColumn::to_string() const {
    return name + ":" + std::string(1, type) + ":" + std::to_string(count);
}

std::vector<PropertyColumn> parse_properties(std::string_view spec) {
    std::vector<PropertyColumn> columns;
    std::size_t position = 0;
    while (position < spec.size()) {
        const auto take = [&spec, &position]() -> std::string_view {
            const std::size_t colon = spec.find(':', position);
            const std::size_t stop = colon == std::string_view::npos ? spec.size() : colon;
            std::string_view field = spec.substr(position, stop - position);
            position = stop == spec.size() ? spec.size() : stop + 1;
            return field;
        };

        const std::string_view name = take();
        if (name.empty()) {
            break;
        }
        const std::string_view type = take();
        const std::string_view count = take();
        if (type.size() != 1 || count.empty()) {
            throw ExtxyzError("extxyz: Properties entry '" + std::string(name) +
                              "' is not name:type:count");
        }
        const char kind = static_cast<char>(std::toupper(static_cast<unsigned char>(type[0])));
        if (kind != 'S' && kind != 'R' && kind != 'I' && kind != 'L') {
            throw ExtxyzError("extxyz: Properties type '" + std::string(type) +
                              "' is not one of S, R, I, L");
        }
        long long width = 0;
        if (!to_long(count, width) || width <= 0) {
            throw ExtxyzError("extxyz: Properties count '" + std::string(count) +
                              "' is not a positive integer");
        }
        columns.push_back(PropertyColumn{std::string(name), kind, static_cast<int>(width)});
    }
    if (columns.empty()) {
        throw ExtxyzError("extxyz: Properties is empty");
    }
    return columns;
}

std::string format_properties(const std::vector<PropertyColumn>& columns) {
    std::string out;
    for (const PropertyColumn& column : columns) {
        if (!out.empty()) {
            out += ':';
        }
        out += column.to_string();
    }
    return out;
}

// ---------------------------------------------------------------------------
// The comment line
// ---------------------------------------------------------------------------
Value parse_comment_line(std::string_view line) {
    Value::Map fields;
    std::size_t position = 0;

    const auto skip_spaces = [&line, &position]() {
        while (position < line.size() && is_space(line[position])) {
            ++position;
        }
    };

    while (true) {
        skip_spaces();
        if (position >= line.size()) {
            break;
        }

        // The key runs to '=' or to whitespace.
        const std::size_t key_start = position;
        while (position < line.size() && line[position] != '=' && !is_space(line[position])) {
            ++position;
        }
        const std::string key(line.substr(key_start, position - key_start));
        if (key.empty()) {
            ++position;
            continue;
        }

        if (position >= line.size() || line[position] != '=') {
            // A bare word is a flag.
            fields.emplace_back(key, Value(true));
            continue;
        }
        ++position;  // past '='

        std::string text;
        if (position < line.size() && (line[position] == '"' || line[position] == '\'')) {
            const char quote = line[position++];
            while (position < line.size() && line[position] != quote) {
                if (line[position] == '\\' && position + 1 < line.size()) {
                    ++position;
                }
                text.push_back(line[position++]);
            }
            if (position < line.size()) {
                ++position;  // past the closing quote
            }
        } else {
            const std::size_t value_start = position;
            while (position < line.size() && !is_space(line[position])) {
                ++position;
            }
            text = std::string(line.substr(value_start, position - value_start));
        }

        // Typed where it can be: a reader that hands back "-1.5" as a string
        // has not really parsed anything.
        double number = 0.0;
        long long integer = 0;
        if (to_long(text, integer)) {
            fields.emplace_back(key, Value(static_cast<std::int64_t>(integer)));
        } else if (to_double(text, number)) {
            fields.emplace_back(key, Value(number));
        } else if (text == "T" || text == "True" || text == "true") {
            fields.emplace_back(key, Value(true));
        } else if (text == "F" || text == "False" || text == "false") {
            fields.emplace_back(key, Value(false));
        } else {
            fields.emplace_back(key, Value(text));
        }
    }
    return Value(std::move(fields));
}

// ---------------------------------------------------------------------------
// The source a reader scans: a mapped file, or text held in memory
// ---------------------------------------------------------------------------
struct ExtxyzReader::Source {
    const char* begin{nullptr};
    const char* end{nullptr};
    std::string owned{};        ///< used when the text is not mapped
    void* mapping{nullptr};     ///< used when it is
    std::size_t mapped_size{0};
    std::string path{};

    ~Source() {
#if !defined(_WIN32)
        if (mapping != nullptr && mapped_size > 0) {
            ::munmap(mapping, mapped_size);
        }
#endif
    }
};

namespace {

/// Map the file, or read it. Mapping wins on a large trajectory: the pages
/// arrive as they are touched, so a frame near the end costs no more than one
/// near the start and the whole file is never resident at once.
std::unique_ptr<ExtxyzReader::Source> open_source(const std::string& path) {
    auto source = std::make_unique<ExtxyzReader::Source>();
    source->path = path;

#if !defined(_WIN32)
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor >= 0) {
        struct stat status {};
        if (::fstat(descriptor, &status) == 0 && S_ISREG(status.st_mode) && status.st_size > 0) {
            const auto size = static_cast<std::size_t>(status.st_size);
            void* mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, descriptor, 0);
            if (mapping != MAP_FAILED) {
                // The reader walks forward, so say so and let the kernel read
                // ahead rather than fault a page at a time.
                ::madvise(mapping, size, MADV_SEQUENTIAL);
                ::close(descriptor);
                source->mapping = mapping;
                source->mapped_size = size;
                source->begin = static_cast<const char*>(mapping);
                source->end = source->begin + size;
                return source;
            }
        }
        ::close(descriptor);
    }
#endif

    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw ExtxyzError("extxyz: cannot open '" + path + "': " + std::strerror(errno));
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    source->owned = buffer.str();
    source->begin = source->owned.data();
    source->end = source->begin + source->owned.size();
    return source;
}

/// Which built-in a Properties column maps onto.
enum class Destination {
    Skip, Species, Position, Force, Velocity, Momentum, Mass, Charge, Tag, Custom
};

Destination destination_of(const PropertyColumn& column) {
    const std::string name = util::lowercase(column.name);
    if (name == "species" || name == "z" || name == "numbers" || name == "symbol") {
        return Destination::Species;
    }
    if (name == "pos" || name == "positions") {
        return Destination::Position;
    }
    if (name == "forces" || name == "force") {
        return Destination::Force;
    }
    if (name == "velo" || name == "velocities" || name == "vel") {
        return Destination::Velocity;
    }
    if (name == "momenta" || name == "momentum") {
        return Destination::Momentum;
    }
    if (name == "masses" || name == "mass") {
        return Destination::Mass;
    }
    if (name == "charges" || name == "charge" || name == "initial_charges") {
        return Destination::Charge;
    }
    if (name == "tags" || name == "tag") {
        return Destination::Tag;
    }
    // A string column with no home: FastAtoms has no per-atom string array.
    if (column.type == 'S') {
        return Destination::Skip;
    }
    return Destination::Custom;
}

}  // namespace

// ---------------------------------------------------------------------------
// ExtxyzReader
// ---------------------------------------------------------------------------
ExtxyzReader::ExtxyzReader(const std::string& path) : source_(open_source(path)) {}

ExtxyzReader ExtxyzReader::from_string(std::string text) {
    ExtxyzReader reader;
    reader.source_ = std::make_unique<Source>();
    reader.source_->owned = std::move(text);
    reader.source_->begin = reader.source_->owned.data();
    reader.source_->end = reader.source_->begin + reader.source_->owned.size();
    reader.source_->path = "<string>";
    return reader;
}

ExtxyzReader::~ExtxyzReader() = default;
ExtxyzReader::ExtxyzReader(ExtxyzReader&&) noexcept = default;
ExtxyzReader& ExtxyzReader::operator=(ExtxyzReader&&) noexcept = default;

void ExtxyzReader::rewind() {
    cursor_ = 0;
    frames_read_ = 0;
}

bool ExtxyzReader::next(Atoms& atoms) {
    const char* const begin = source_->begin;
    const char* const end = source_->end;
    const char* p = begin + cursor_;

    // Blank lines between frames are tolerated: some writers leave one.
    while (p < end) {
        const char* probe = skip_blank(p, end);
        if (probe < end && !is_line_end(*probe)) {
            p = probe;
            break;
        }
        p = next_line(p, end);
    }
    if (p >= end) {
        cursor_ = static_cast<std::size_t>(end - begin);
        return false;
    }

    // --- the atom count ---------------------------------------------------
    const char* count_line = p;
    const std::string_view count_token = next_token(p, end);
    long long natoms = 0;
    if (!to_long(count_token, natoms) || natoms < 0) {
        fail(begin, count_line,
             "expected an atom count, found '" + std::string(count_token) + "'");
    }
    p = next_line(p, end);

    // --- the comment line -------------------------------------------------
    if (p >= end) {
        fail(begin, count_line, "the frame ends before its comment line");
    }
    const char* comment_start = p;
    const char* comment_stop = skip_to_line_end(p, end);
    const Value info = parse_comment_line(
        std::string_view(comment_start, static_cast<std::size_t>(comment_stop - comment_start)));
    p = comment_stop < end ? comment_stop + 1 : end;

    // --- what the columns mean --------------------------------------------
    std::vector<PropertyColumn> columns;
    if (const Value* properties = info.find("Properties");
        properties != nullptr && properties->is_string()) {
        columns = parse_properties(properties->as_string());
    } else {
        // The plain XYZ default, which is what a file without Properties is.
        columns = {PropertyColumn{"species", 'S', 1}, PropertyColumn{"pos", 'R', 3}};
    }

    const auto count = static_cast<std::size_t>(natoms);
    std::vector<int> numbers;
    std::vector<double> positions;
    numbers.reserve(count);
    positions.reserve(3 * count);

    // One buffer per column that needs one, sized up front. Growing these
    // inside the atom loop is what turns a linear read into a quadratic one.
    struct ColumnBuffer {
        Destination destination{Destination::Skip};
        const PropertyColumn* column{nullptr};
        std::vector<double> values{};
        int offset{0};      ///< first token index of this column
    };
    std::vector<ColumnBuffer> buffers;
    buffers.reserve(columns.size());
    int token_offset = 0;
    bool has_species = false;
    bool has_position = false;
    for (const PropertyColumn& column : columns) {
        ColumnBuffer buffer;
        buffer.destination = destination_of(column);
        buffer.column = &column;
        buffer.offset = token_offset;
        if (buffer.destination == Destination::Species) {
            has_species = true;
        } else if (buffer.destination == Destination::Position) {
            has_position = true;
        } else if (buffer.destination != Destination::Skip) {
            buffer.values.reserve(count * static_cast<std::size_t>(column.count));
        }
        buffers.push_back(std::move(buffer));
        token_offset += column.count;
    }
    const int tokens_per_atom = token_offset;

    if (!has_species) {
        fail(begin, comment_start, "Properties names no species column");
    }
    if (!has_position) {
        fail(begin, comment_start, "Properties names no pos column");
    }

    // --- the atoms --------------------------------------------------------
    std::vector<std::string_view> fields(static_cast<std::size_t>(tokens_per_atom));
    for (std::size_t atom = 0; atom < count; ++atom) {
        const char* line_start = p;
        if (p >= end) {
            fail(begin, line_start, "the frame declares " + std::to_string(count) +
                                        " atoms but ends after " + std::to_string(atom));
        }
        for (int k = 0; k < tokens_per_atom; ++k) {
            fields[static_cast<std::size_t>(k)] = next_token(p, end);
            if (fields[static_cast<std::size_t>(k)].empty()) {
                fail(begin, line_start,
                     "atom " + std::to_string(atom + 1) + " has fewer than the " +
                         std::to_string(tokens_per_atom) + " columns Properties declares");
            }
        }
        p = next_line(p, end);

        for (ColumnBuffer& buffer : buffers) {
            const int width = buffer.column->count;
            const auto base = static_cast<std::size_t>(buffer.offset);
            switch (buffer.destination) {
                case Destination::Skip:
                    break;
                case Destination::Species: {
                    const int z = symbols().lookup(fields[base]);
                    if (z < 0) {
                        fail(begin, line_start,
                             "'" + std::string(fields[base]) + "' is not a chemical symbol");
                    }
                    numbers.push_back(z);
                    break;
                }
                case Destination::Position: {
                    for (int k = 0; k < width && k < 3; ++k) {
                        double value = 0.0;
                        if (!to_double(fields[base + static_cast<std::size_t>(k)], value)) {
                            fail(begin, line_start, "position '" +
                                     std::string(fields[base + static_cast<std::size_t>(k)]) +
                                     "' is not a number");
                        }
                        positions.push_back(value);
                    }
                    break;
                }
                default: {
                    for (int k = 0; k < width; ++k) {
                        double value = 0.0;
                        const std::string_view text = fields[base + static_cast<std::size_t>(k)];
                        if (!to_double(text, value)) {
                            // Logical columns are written T/F.
                            if (text == "T" || text == "True") {
                                value = 1.0;
                            } else if (text == "F" || text == "False") {
                                value = 0.0;
                            } else {
                                fail(begin, line_start, "column '" + buffer.column->name +
                                         "' value '" + std::string(text) + "' is not a number");
                            }
                        }
                        buffer.values.push_back(value);
                    }
                    break;
                }
            }
        }
    }

    // --- assemble ---------------------------------------------------------
    atoms = Atoms(numbers, positions);

    if (const Value* lattice = info.find("Lattice"); lattice != nullptr) {
        const std::vector<double> entries = value_numbers(*lattice);
        if (entries.size() == 9) {
            atoms.set_cell(Mat3(entries[0], entries[1], entries[2], entries[3], entries[4],
                                entries[5], entries[6], entries[7], entries[8]));
        } else if (entries.size() == 3) {
            atoms.set_cell(Mat3::diagonal(entries[0], entries[1], entries[2]));
        } else if (!entries.empty()) {
            fail(begin, comment_start,
                 "Lattice has " + std::to_string(entries.size()) + " numbers, expected 9 or 3");
        }
        // A file that carries a cell is periodic unless it says otherwise.
        atoms.set_pbc({true, true, true});
    }

    if (const Value* pbc = info.find("pbc"); pbc != nullptr) {
        std::array<bool, 3> flags{{false, false, false}};
        if (pbc->is_string()) {
            const std::string text = pbc->as_string();
            std::size_t axis = 0;
            for (const char c : text) {
                if (is_space(c)) {
                    continue;
                }
                if (axis < 3) {
                    flags[axis++] = (c == 'T' || c == 't' || c == '1');
                }
            }
        } else if (pbc->is_bool()) {
            flags = {{pbc->as_bool(), pbc->as_bool(), pbc->as_bool()}};
        } else if (pbc->is_list()) {
            const std::vector<double> entries = value_numbers(*pbc);
            for (std::size_t axis = 0; axis < 3 && axis < entries.size(); ++axis) {
                flags[axis] = entries[axis] != 0.0;
            }
        }
        atoms.set_pbc(flags);
    }

    for (ColumnBuffer& buffer : buffers) {
        if (buffer.values.empty()) {
            continue;
        }
        switch (buffer.destination) {
            case Destination::Force:
                atoms.set_array("forces", buffer.values, 3);
                break;
            case Destination::Velocity:
                atoms.set_velocities(buffer.values);
                break;
            case Destination::Momentum:
                atoms.set_momenta(buffer.values);
                break;
            case Destination::Mass:
                atoms.set_masses(buffer.values);
                break;
            case Destination::Charge:
                atoms.set_charges(buffer.values);
                break;
            case Destination::Tag: {
                std::vector<int> tags(buffer.values.size());
                for (std::size_t k = 0; k < buffer.values.size(); ++k) {
                    tags[k] = static_cast<int>(buffer.values[k]);
                }
                atoms.set_tags(tags);
                break;
            }
            case Destination::Custom:
                atoms.set_array(buffer.column->name, buffer.values,
                                static_cast<std::size_t>(buffer.column->count));
                break;
            default:
                break;
        }
    }

    // Everything on the comment line that is not structure travels with the
    // frame, so `energy=` survives a read and comes back on a write.
    Value::Map carried;
    for (const auto& [key, value] : info.as_map()) {
        if (key == "Lattice" || key == "Properties" || key == "pbc") {
            continue;
        }
        carried.emplace_back(key, value);
    }
    atoms.set_info(Value(std::move(carried)));

    cursor_ = static_cast<std::size_t>(p - begin);
    ++frames_read_;
    return true;
}

const std::vector<std::size_t>& ExtxyzReader::index() {
    if (indexed_) {
        return index_;
    }
    // Read only each frame's atom count and step over the rest. Building an
    // index this way costs a scan of the newlines, not a parse of the numbers,
    // which is what makes seeking into a large trajectory affordable.
    const char* const begin = source_->begin;
    const char* const end = source_->end;
    const char* p = begin;

    while (p < end) {
        while (p < end) {
            const char* probe = skip_blank(p, end);
            if (probe < end && !is_line_end(*probe)) {
                p = probe;
                break;
            }
            p = next_line(p, end);
        }
        if (p >= end) {
            break;
        }

        const char* frame_start = p;
        const std::string_view token = next_token(p, end);
        long long natoms = 0;
        if (!to_long(token, natoms) || natoms < 0) {
            fail(begin, frame_start,
                 "expected an atom count, found '" + std::string(token) + "'");
        }
        index_.push_back(static_cast<std::size_t>(frame_start - begin));

        p = next_line(p, end);   // past the count
        p = next_line(p, end);   // past the comment
        for (long long atom = 0; atom < natoms && p < end; ++atom) {
            p = next_line(p, end);
        }
    }
    indexed_ = true;
    return index_;
}

std::size_t ExtxyzReader::size() { return index().size(); }

Atoms ExtxyzReader::at(std::ptrdiff_t frame) {
    const std::vector<std::size_t>& offsets = index();
    if (offsets.empty()) {
        throw ExtxyzError("extxyz: '" + source_->path + "' holds no frames");
    }
    const auto total = static_cast<std::ptrdiff_t>(offsets.size());
    const std::ptrdiff_t wanted = frame < 0 ? total + frame : frame;
    if (wanted < 0 || wanted >= total) {
        throw ExtxyzError("extxyz: frame " + std::to_string(frame) + " is out of range for " +
                          std::to_string(total) + " frames");
    }
    cursor_ = offsets[static_cast<std::size_t>(wanted)];
    Atoms atoms;
    if (!next(atoms)) {
        throw ExtxyzError("extxyz: frame " + std::to_string(frame) + " could not be read");
    }
    return atoms;
}

Atoms read_extxyz(const std::string& path, std::ptrdiff_t frame) {
    ExtxyzReader reader(path);
    // The common case, the last frame, needs no index: read forward and keep
    // the most recent. That is one pass and no offsets table.
    if (frame == -1) {
        Atoms current;
        Atoms latest;
        bool any = false;
        while (reader.next(current)) {
            latest = std::move(current);
            any = true;
        }
        if (!any) {
            throw ExtxyzError("extxyz: '" + path + "' holds no frames");
        }
        return latest;
    }
    return reader.at(frame);
}

std::vector<Atoms> read_extxyz_frames(const std::string& path) {
    ExtxyzReader reader(path);
    std::vector<Atoms> frames;
    Atoms atoms;
    while (reader.next(atoms)) {
        frames.push_back(std::move(atoms));
    }
    return frames;
}

Atoms parse_extxyz(const std::string& text, std::ptrdiff_t frame) {
    ExtxyzReader reader = ExtxyzReader::from_string(text);
    if (frame == -1) {
        Atoms current;
        Atoms latest;
        bool any = false;
        while (reader.next(current)) {
            latest = std::move(current);
            any = true;
        }
        if (!any) {
            throw ExtxyzError("extxyz: the text holds no frames");
        }
        return latest;
    }
    return reader.at(frame);
}

std::vector<Atoms> parse_extxyz_frames(const std::string& text) {
    ExtxyzReader reader = ExtxyzReader::from_string(text);
    std::vector<Atoms> frames;
    Atoms atoms;
    while (reader.next(atoms)) {
        frames.push_back(std::move(atoms));
    }
    return frames;
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------
namespace {

/// Append a double in the shortest form that reads back exactly.
///
/// `to_chars` is used rather than a stream because it neither allocates nor
/// looks at the locale - and a locale that writes a decimal comma silently
/// produces a file no other reader can parse.
void append_number(std::string& out, double value, int precision) {
    char buffer[64];
    // precision <= 0 means the shortest form that reads back as the same
    // double. That is not a compromise between size and accuracy: it is both.
    // Twelve fixed significant digits is longer than "0.1" and still cannot
    // represent every double, so a trajectory written that way does not come
    // back the way it went in.
    const auto [ptr, ec] =
        precision > 0
            ? std::to_chars(buffer, buffer + sizeof buffer, value,
                            std::chars_format::general, precision)
            : std::to_chars(buffer, buffer + sizeof buffer, value);
    if (ec == std::errc{}) {
        out.append(buffer, static_cast<std::size_t>(ptr - buffer));
    } else {
        out += "0";
    }
}

void append_value(std::string& out, const Value& value, int precision) {
    switch (value.type()) {
        case Value::Type::Bool:
            out += value.as_bool() ? 'T' : 'F';
            break;
        case Value::Type::Int:
            out += std::to_string(value.as_int());
            break;
        case Value::Type::Float:
            append_number(out, value.as_double(), precision);
            break;
        case Value::Type::List: {
            out += '"';
            bool first = true;
            for (const Value& item : value.as_list()) {
                if (!first) {
                    out += ' ';
                }
                first = false;
                append_value(out, item, precision);
            }
            out += '"';
            break;
        }
        default: {
            const std::string text = value.is_string() ? value.as_string() : value.to_string();
            // Quote anything a reader would otherwise split on.
            if (text.find_first_of(" \t\"'=") != std::string::npos || text.empty()) {
                out += '"';
                for (const char c : text) {
                    if (c == '"' || c == '\\') {
                        out += '\\';
                    }
                    out += c;
                }
                out += '"';
            } else {
                out += text;
            }
            break;
        }
    }
}

}  // namespace

void append_extxyz_frame(std::string& out, const Atoms& atoms, const ExtxyzOptions& options) {
    const std::size_t count = atoms.size();
    const int precision = options.precision;

    // Decide the columns before writing anything, so the header cannot
    // disagree with the body.
    struct OutputColumn {
        PropertyColumn spec;
        /// Columns the structure already holds are borrowed; ones that have to
        /// be computed (velocities from momenta) are owned.
        ///
        /// The pointer is resolved when it is used, never stored: a member
        /// pointing at a sibling member does not survive the move into the
        /// vector, nor the vector reallocating, and the result is a dangling
        /// read that looks like a working writer until it segfaults.
        const std::vector<double>* borrowed{nullptr};
        std::vector<double> owned{};

        [[nodiscard]] const std::vector<double>& values() const {
            return borrowed != nullptr ? *borrowed : owned;
        }
    };
    std::vector<OutputColumn> columns;
    columns.push_back(OutputColumn{PropertyColumn{"species", 'S', 1}, nullptr, {}});
    columns.push_back(OutputColumn{PropertyColumn{"pos", 'R', 3}, &atoms.positions(), {}});

    const auto wanted = [&options](const std::string& name) {
        return options.arrays.empty() ||
               std::find(options.arrays.begin(), options.arrays.end(), name) !=
                   options.arrays.end();
    };

    // velo before forces: that is the order this project has always written,
    // and reordering columns would invalidate every file already on disk that
    // a reader indexes positionally.
    if (options.velocities && atoms.has_momenta() && wanted("velo")) {
        columns.push_back(
            OutputColumn{PropertyColumn{"velo", 'R', 3}, nullptr, atoms.velocities()});
    }
    if (options.forces && atoms.has_array("forces") && wanted("forces")) {
        const Atoms::Array& array = atoms.array("forces");
        if (array.ncols == 3 && array.data.size() == 3 * count) {
            columns.push_back(OutputColumn{PropertyColumn{"forces", 'R', 3}, &array.data, {}});
        }
    }
    if (options.masses && atoms.has_masses() && wanted("masses")) {
        columns.push_back(
            OutputColumn{PropertyColumn{"masses", 'R', 1}, nullptr, atoms.masses()});
    }
    if (options.charges && atoms.has_charges() && wanted("charges")) {
        columns.push_back(
            OutputColumn{PropertyColumn{"charges", 'R', 1}, nullptr, atoms.charges()});
    }
    // Anything the caller named explicitly that is not one of the above.
    for (const std::string& name : options.arrays) {
        if (name == "forces" || name == "velo" || name == "masses" || name == "charges") {
            continue;
        }
        if (!atoms.has_array(name)) {
            continue;
        }
        const Atoms::Array& array = atoms.array(name);
        columns.push_back(OutputColumn{
            PropertyColumn{name, 'R', static_cast<int>(array.ncols)}, &array.data, {}});
    }

    std::vector<PropertyColumn> specs;
    specs.reserve(columns.size());
    for (const OutputColumn& column : columns) {
        specs.push_back(column.spec);
    }

    // Roughly what the frame will take, so the buffer grows once rather than
    // a dozen times while a million atoms are appended to it.
    out.reserve(out.size() + count * (specs.size() * 14 + 8) + 256);

    out += std::to_string(count);
    out += '\n';

    const Mat3& cell = atoms.cell().matrix();
    if (cell.determinant() != 0.0 || atoms.cell().any_pbc()) {
        out += "Lattice=\"";
        for (std::size_t k = 0; k < 9; ++k) {
            if (k != 0) {
                out += ' ';
            }
            append_number(out, cell.m[k], precision);
        }
        out += "\" ";
    }
    out += "Properties=";
    out += format_properties(specs);

    const std::array<bool, 3> pbc = atoms.pbc();
    out += " pbc=\"";
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (axis != 0) {
            out += ' ';
        }
        out += pbc[axis] ? 'T' : 'F';
    }
    out += '"';

    // The structure's own info, then the caller's, so a per-frame override
    // wins over what the structure happens to be carrying.
    Value::Map merged;
    const auto put = [&merged](const std::string& key, const Value& value) {
        for (auto& entry : merged) {
            if (entry.first == key) {
                entry.second = value;
                return;
            }
        }
        merged.emplace_back(key, value);
    };
    if (atoms.info().is_map()) {
        for (const auto& [key, value] : atoms.info().as_map()) {
            put(key, value);
        }
    }
    if (options.info.is_map()) {
        for (const auto& [key, value] : options.info.as_map()) {
            put(key, value);
        }
    }
    for (const auto& [key, value] : merged) {
        if (key == "Lattice" || key == "Properties" || key == "pbc") {
            continue;
        }
        out += ' ';
        out += key;
        out += '=';
        append_value(out, value, precision);
    }
    out += '\n';

    for (std::size_t atom = 0; atom < count; ++atom) {
        for (std::size_t c = 0; c < columns.size(); ++c) {
            if (c != 0) {
                out += ' ';
            }
            const OutputColumn& column = columns[c];
            if (column.spec.name == "species") {
                out += symbol_from_number(atoms.number(atom));
                continue;
            }
            const std::size_t width = static_cast<std::size_t>(column.spec.count);
            for (std::size_t k = 0; k < width; ++k) {
                if (k != 0) {
                    out += ' ';
                }
                append_number(out, column.values()[atom * width + k], precision);
            }
        }
        out += '\n';
    }
}

std::string extxyz_string(const Atoms& atoms, ExtxyzOptions options) {
    std::string out;
    append_extxyz_frame(out, atoms, options);
    return out;
}

struct ExtxyzWriter::Sink {
    std::ofstream stream;
    std::string buffer;
    std::string path;
};

ExtxyzWriter::ExtxyzWriter(const std::string& path, ExtxyzOptions options)
    : sink_(std::make_unique<Sink>()), options_(std::move(options)) {
    sink_->path = path;
    const auto mode = std::ios::binary |
                      (options_.append ? std::ios::app : std::ios::trunc);
    sink_->stream.open(path, mode);
    if (!sink_->stream) {
        throw ExtxyzError("extxyz: cannot open '" + path + "' for writing: " +
                          std::strerror(errno));
    }
}

ExtxyzWriter::~ExtxyzWriter() {
    if (sink_) {
        try {
            flush();
        } catch (...) {
            // A destructor that throws during stack unwinding terminates the
            // process; a trajectory frame is not worth that.
        }
    }
}

ExtxyzWriter::ExtxyzWriter(ExtxyzWriter&&) noexcept = default;
ExtxyzWriter& ExtxyzWriter::operator=(ExtxyzWriter&&) noexcept = default;

void ExtxyzWriter::write(const Atoms& atoms, const Value& info) {
    ExtxyzOptions options = options_;
    if (info.is_map()) {
        Value::Map merged = options.info.is_map() ? options.info.as_map() : Value::Map{};
        for (const auto& [key, value] : info.as_map()) {
            merged.emplace_back(key, value);
        }
        options.info = Value(std::move(merged));
    }
    append_extxyz_frame(sink_->buffer, atoms, options);
    ++frames_written_;
    // One write per few megabytes rather than one per frame: a long run of
    // small frames is otherwise dominated by the system call.
    if (sink_->buffer.size() >= (1u << 20)) {
        flush();
    }
}

void ExtxyzWriter::flush() {
    if (!sink_ || sink_->buffer.empty()) {
        return;
    }
    sink_->stream.write(sink_->buffer.data(),
                        static_cast<std::streamsize>(sink_->buffer.size()));
    sink_->stream.flush();
    if (!sink_->stream) {
        throw ExtxyzError("extxyz: writing '" + sink_->path + "' failed");
    }
    sink_->buffer.clear();
}

void write_extxyz(const std::string& path, const Atoms& atoms, ExtxyzOptions options) {
    ExtxyzWriter writer(path, std::move(options));
    writer.write(atoms);
}

void write_extxyz(const std::string& path, const std::vector<Atoms>& frames,
                  ExtxyzOptions options) {
    ExtxyzWriter writer(path, std::move(options));
    for (const Atoms& atoms : frames) {
        writer.write(atoms);
    }
}

}  // namespace fastatoms::io
