#include "fastatoms/io/cif.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

#include "fastatoms/core/elements.hpp"

namespace fastatoms::io {
namespace {

/// One symmetry operation, as a CIF spells it: `x, y, z` or `-y, x-y, z+1/3`.
struct Operation {
    Mat3 rotation{};
    Vec3 translation{};
};

std::string trim(std::string text) {
    const auto not_space = [](unsigned char c) { return std::isspace(c) == 0; };
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
    text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(), text.end());
    return text;
}

std::string lowered(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

/// Strip the quoting a CIF value may carry.
std::string unquote(std::string text) {
    text = trim(std::move(text));
    if (text.size() >= 2 && ((text.front() == '\'' && text.back() == '\'') ||
                             (text.front() == '"' && text.back() == '"'))) {
        return text.substr(1, text.size() - 2);
    }
    return text;
}

/// A number, with the `(3)` standard-uncertainty suffix that CIFs attach to
/// measured quantities removed.
double number(const std::string& text, const std::string& origin, const char* what) {
    std::string clean = unquote(text);
    const std::size_t bracket = clean.find('(');
    if (bracket != std::string::npos) {
        clean = clean.substr(0, bracket);
    }
    if (clean == "?" || clean == "." || clean.empty()) {
        throw ExtxyzError("cif: '" + origin + "': the " + what + " is not given");
    }
    try {
        return std::stod(clean);
    } catch (const std::exception&) {
        throw ExtxyzError("cif: '" + origin + "': the " + what + " should be a number, not '" +
                          text + "'");
    }
}

/// Split a CIF line into values, respecting quotes.
std::vector<std::string> split_values(const std::string& line) {
    std::vector<std::string> values;
    std::size_t index = 0;
    while (index < line.size()) {
        while (index < line.size() && std::isspace(static_cast<unsigned char>(line[index]))) {
            ++index;
        }
        if (index >= line.size() || line[index] == '#') {
            break;
        }
        if (line[index] == '\'' || line[index] == '"') {
            const char quote = line[index++];
            std::string value;
            while (index < line.size() && line[index] != quote) {
                value += line[index++];
            }
            ++index;
            values.push_back(value);
        } else {
            std::string value;
            while (index < line.size() && !std::isspace(static_cast<unsigned char>(line[index]))) {
                value += line[index++];
            }
            values.push_back(value);
        }
    }
    return values;
}

/// Parse one component of an operation, e.g. `-y+1/2` or `x-y` or `2/3+z`.
void parse_component(const std::string& text, std::size_t row, Operation& operation,
                     const std::string& origin) {
    std::string clean;
    for (const char c : lowered(text)) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            clean += c;
        }
    }

    double sign = 1.0;
    std::size_t index = 0;
    while (index < clean.size()) {
        if (clean[index] == '+') {
            sign = 1.0;
            ++index;
            continue;
        }
        if (clean[index] == '-') {
            sign = -1.0;
            ++index;
            continue;
        }
        if (clean[index] == 'x' || clean[index] == 'y' || clean[index] == 'z') {
            const std::size_t axis = static_cast<std::size_t>(clean[index] - 'x');
            operation.rotation.m[3 * row + axis] += sign;
            ++index;
            sign = 1.0;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(clean[index])) || clean[index] == '.') {
            // A fraction, a decimal, or an integer coefficient on an axis
            // ("2x" does not occur in a crystallographic operation, so a
            // number followed by an axis letter is a coefficient of one).
            std::size_t end = index;
            while (end < clean.size() && (std::isdigit(static_cast<unsigned char>(clean[end])) ||
                                          clean[end] == '.')) {
                ++end;
            }
            double value = std::stod(clean.substr(index, end - index));
            index = end;
            if (index < clean.size() && clean[index] == '/') {
                ++index;
                std::size_t denominator_end = index;
                while (denominator_end < clean.size() &&
                       std::isdigit(static_cast<unsigned char>(clean[denominator_end]))) {
                    ++denominator_end;
                }
                const double denominator =
                    std::stod(clean.substr(index, denominator_end - index));
                if (denominator == 0.0) {
                    throw ExtxyzError("cif: '" + origin + "': a symmetry operation divides by "
                                      "zero: '" + text + "'");
                }
                value /= denominator;
                index = denominator_end;
            }
            if (index < clean.size() &&
                (clean[index] == 'x' || clean[index] == 'y' || clean[index] == 'z')) {
                const std::size_t axis = static_cast<std::size_t>(clean[index] - 'x');
                operation.rotation.m[3 * row + axis] += sign * value;
                ++index;
            } else {
                operation.translation[row] += sign * value;
            }
            sign = 1.0;
            continue;
        }
        throw ExtxyzError("cif: '" + origin + "': '" + text +
                          "' is not a symmetry operation this can read");
    }
}

Operation parse_operation(const std::string& text, const std::string& origin) {
    std::vector<std::string> parts;
    std::string current;
    for (const char c : unquote(text)) {
        if (c == ',') {
            parts.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    parts.push_back(current);
    if (parts.size() != 3) {
        throw ExtxyzError("cif: '" + origin + "': a symmetry operation has three parts, and '" +
                          text + "' has " + std::to_string(parts.size()));
    }

    Operation operation;
    for (std::size_t row = 0; row < 3; ++row) {
        parse_component(parts[row], row, operation, origin);
    }
    return operation;
}

/// The element a CIF label names: `Cu1`, `O2-`, `Fe3+` all mean an element,
/// with the rest being the site number or the oxidation state.
std::string element_of(const std::string& label, const std::string& origin) {
    std::string letters;
    for (const char c : label) {
        if (std::isalpha(static_cast<unsigned char>(c))) {
            letters += c;
        } else {
            break;
        }
    }
    // Two letters first: "Cl" is chlorine, not carbon followed by rubbish.
    if (letters.size() >= 2) {
        const std::string two = letters.substr(0, 2);
        if (try_number_from_symbol(two) > 0) {
            return two;
        }
    }
    if (!letters.empty() && try_number_from_symbol(letters.substr(0, 1)) > 0) {
        return letters.substr(0, 1);
    }
    throw ExtxyzError("cif: '" + origin + "': '" + label + "' does not name an element");
}

double wrapped(double value) {
    value = std::fmod(value, 1.0);
    return value < 0.0 ? value + 1.0 : value;
}

/// The cell from its six parameters, in the orientation crystallography uses:
/// a along x, b in the xy plane.
Mat3 cell_from_parameters(double a, double b, double c, double alpha, double beta, double gamma) {
    const double to_radians = 3.14159265358979323846 / 180.0;
    const double ca = std::cos(alpha * to_radians);
    const double cb = std::cos(beta * to_radians);
    const double cg = std::cos(gamma * to_radians);
    const double sg = std::sin(gamma * to_radians);

    Mat3 cell{};
    cell.m[0] = a;
    cell.m[3] = b * cg;
    cell.m[4] = b * sg;
    cell.m[6] = c * cb;
    cell.m[7] = c * (ca - cb * cg) / sg;
    const double z = c * c - cell.m[6] * cell.m[6] - cell.m[7] * cell.m[7];
    cell.m[8] = z > 0.0 ? std::sqrt(z) : 0.0;
    return cell;
}

/// One `data_` block's worth of parsed content.
struct Block {
    bool started{false};
    std::map<std::string, std::string> values;
    std::vector<Operation> operations;
    std::vector<std::string> labels;
    std::vector<std::string> types;
    std::vector<std::array<double, 3>> sites;
    std::vector<double> occupancies;
};

Atoms build(const Block& block, const std::string& origin, double tolerance) {
    const auto value = [&](const char* key) -> const std::string* {
        const auto found = block.values.find(key);
        return found == block.values.end() ? nullptr : &found->second;
    };
    const auto required = [&](const char* key, const char* what) {
        const std::string* found = value(key);
        if (found == nullptr) {
            throw ExtxyzError("cif: '" + origin + "' does not give the " + std::string(what) +
                              " (" + key + ")");
        }
        return number(*found, origin, what);
    };

    const Mat3 cell = cell_from_parameters(
        required("_cell_length_a", "cell length a"), required("_cell_length_b", "cell length b"),
        required("_cell_length_c", "cell length c"),
        required("_cell_angle_alpha", "cell angle alpha"),
        required("_cell_angle_beta", "cell angle beta"),
        required("_cell_angle_gamma", "cell angle gamma"));

    if (block.sites.empty()) {
        throw ExtxyzError("cif: '" + origin + "' lists no atom sites");
    }

    std::vector<Operation> operations = block.operations;
    if (operations.empty()) {
        // No operation list. P1 needs none; anything else would need a lookup
        // this does not do, and returning the asymmetric unit as though it
        // were the structure would quietly drop most of the atoms.
        const std::string* number_text = value("_space_group_it_number");
        if (number_text == nullptr) {
            number_text = value("_symmetry_int_tables_number");
        }
        const std::string* symbol = value("_space_group_name_h-m_alt");
        if (symbol == nullptr) {
            symbol = value("_symmetry_space_group_name_h-m");
        }
        const bool is_p1 =
            (number_text != nullptr && trim(unquote(*number_text)) == "1") ||
            (symbol != nullptr && (unquote(*symbol) == "P 1" || unquote(*symbol) == "P1"));
        if (!is_p1) {
            throw ExtxyzError(
                "cif: '" + origin +
                "' gives no symmetry operations, only a space-group name. Reading it as a plain "
                "list of sites would drop every atom the symmetry was meant to generate. Re-export "
                "it with a _space_group_symop_operation_xyz loop, which is what published CIFs "
                "carry.");
        }
        operations.push_back(Operation{Mat3{1, 0, 0, 0, 1, 0, 0, 0, 1}, Vec3{0, 0, 0}});
    }

    std::vector<std::string> symbols;
    std::vector<double> scaled;
    for (std::size_t site = 0; site < block.sites.size(); ++site) {
        const std::string element =
            site < block.types.size() && !block.types[site].empty()
                ? element_of(block.types[site], origin)
                : element_of(block.labels[site], origin);
        const auto& position = block.sites[site];

        for (const Operation& operation : operations) {
            std::array<double, 3> image{};
            for (std::size_t row = 0; row < 3; ++row) {
                image[row] = operation.translation[row];
                for (std::size_t column = 0; column < 3; ++column) {
                    image[row] += operation.rotation.m[3 * row + column] * position[column];
                }
                image[row] = wrapped(image[row]);
            }

            // Different operations map a special position onto itself; without
            // merging, a site on a mirror plane comes back doubled.
            bool duplicate = false;
            for (std::size_t existing = 0; existing < symbols.size(); ++existing) {
                double worst = 0.0;
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    double delta = std::abs(scaled[3 * existing + axis] - image[axis]);
                    delta = std::min(delta, 1.0 - delta);  // across the periodic seam
                    worst = std::max(worst, delta);
                }
                if (worst < tolerance) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                continue;
            }

            symbols.push_back(element);
            scaled.insert(scaled.end(), image.begin(), image.end());
        }
    }

    Atoms atoms(symbols, std::vector<double>(scaled.size(), 0.0));
    atoms.set_cell(cell);
    atoms.set_pbc(true);
    atoms.set_scaled_positions(scaled);
    return atoms;
}

/// Read every data block. A CIF's loops are declared column by column, so the
/// column order has to be learnt before the rows can be read.
std::vector<Block> parse_blocks(std::istream& stream, const std::string& origin) {
    std::vector<Block> blocks;
    Block current;

    std::string line;
    std::vector<std::string> headers;
    bool in_loop = false;

    const auto finish_block = [&]() {
        if (current.started) {
            blocks.push_back(std::move(current));
        }
        current = Block{};
    };

    while (std::getline(stream, line)) {
        const std::string stripped = trim(line);
        if (stripped.empty() || stripped[0] == '#') {
            continue;
        }

        if (lowered(stripped).rfind("data_", 0) == 0) {
            finish_block();
            current.started = true;
            in_loop = false;
            headers.clear();
            continue;
        }

        if (lowered(stripped) == "loop_") {
            in_loop = true;
            headers.clear();
            continue;
        }

        if (stripped[0] == ';') {
            // A multi-line text value: skipped, since nothing structural is
            // ever written in one.
            while (std::getline(stream, line) && trim(line) != ";") {
            }
            continue;
        }

        if (stripped[0] == '_') {
            const auto values = split_values(stripped);
            if (in_loop && values.size() == 1) {
                headers.push_back(lowered(values[0]));
                continue;
            }
            in_loop = false;
            if (values.size() >= 2) {
                std::string joined = values[1];
                for (std::size_t k = 2; k < values.size(); ++k) {
                    joined += " " + values[k];
                }
                current.values[lowered(values[0])] = joined;
            } else if (values.size() == 1) {
                // The value is on the next line.
                std::string next;
                if (std::getline(stream, next)) {
                    const std::string trimmed = trim(next);
                    if (!trimmed.empty() && trimmed[0] == ';') {
                        while (std::getline(stream, next) && trim(next) != ";") {
                        }
                    } else {
                        current.values[lowered(values[0])] = trimmed;
                    }
                }
            }
            continue;
        }

        if (!in_loop || headers.empty()) {
            continue;
        }

        const auto row = split_values(stripped);
        if (row.size() != headers.size()) {
            continue;  // a wrapped or malformed row; not structural
        }

        const auto column = [&](const char* name) -> const std::string* {
            for (std::size_t k = 0; k < headers.size(); ++k) {
                if (headers[k] == name) {
                    return &row[k];
                }
            }
            return nullptr;
        };

        if (const std::string* xyz = column("_space_group_symop_operation_xyz")
                                         ? column("_space_group_symop_operation_xyz")
                                         : column("_symmetry_equiv_pos_as_xyz")) {
            current.operations.push_back(parse_operation(*xyz, origin));
            continue;
        }

        const std::string* fx = column("_atom_site_fract_x");
        const std::string* fy = column("_atom_site_fract_y");
        const std::string* fz = column("_atom_site_fract_z");
        if (fx != nullptr && fy != nullptr && fz != nullptr) {
            const std::string* label = column("_atom_site_label");
            const std::string* type = column("_atom_site_type_symbol");
            current.labels.push_back(label != nullptr ? unquote(*label) : "");
            current.types.push_back(type != nullptr ? unquote(*type) : "");
            current.sites.push_back({number(*fx, origin, "fractional x"),
                                     number(*fy, origin, "fractional y"),
                                     number(*fz, origin, "fractional z")});
            const std::string* occupancy = column("_atom_site_occupancy");
            current.occupancies.push_back(
                occupancy != nullptr ? number(*occupancy, origin, "occupancy") : 1.0);
        }
    }

    finish_block();
    return blocks;
}

void append_double(std::string& out, double value, int precision) {
    std::ostringstream stream;
    stream << std::setprecision(precision) << std::fixed << value;
    out += stream.str();
}

}  // namespace

Atoms parse_cif(std::istream& stream, const std::string& origin, double tolerance) {
    const std::vector<Block> blocks = parse_blocks(stream, origin);
    if (blocks.empty()) {
        throw ExtxyzError("cif: '" + origin + "' holds no data block");
    }
    return build(blocks.front(), origin, tolerance);
}

Atoms read_cif(const std::string& path, double tolerance) {
    std::ifstream stream(path);
    if (!stream) {
        throw ExtxyzError("cif: cannot open '" + path + "'");
    }
    return parse_cif(stream, path, tolerance);
}

std::vector<Atoms> read_cif_frames(const std::string& path, double tolerance) {
    std::ifstream stream(path);
    if (!stream) {
        throw ExtxyzError("cif: cannot open '" + path + "'");
    }
    const std::vector<Block> blocks = parse_blocks(stream, path);
    if (blocks.empty()) {
        throw ExtxyzError("cif: '" + path + "' holds no data block");
    }
    std::vector<Atoms> frames;
    frames.reserve(blocks.size());
    for (const Block& block : blocks) {
        frames.push_back(build(block, path, tolerance));
    }
    return frames;
}

std::string cif_to_string(const Atoms& atoms, CifOptions options) {
    if (atoms.size() == 0) {
        throw ExtxyzError("cif: a CIF needs at least one atom");
    }
    const Cell& cell = atoms.cell();
    if (std::abs(cell.matrix().determinant()) < 1e-12) {
        throw ExtxyzError("cif: a CIF needs a cell; this structure has none");
    }

    std::string name = options.name;
    if (name.empty()) {
        name = atoms.formula();
    }
    std::string out = "data_" + name + "\n";

    const auto lengths = cell.lengths();
    const auto angles = cell.angles();
    const char* length_keys[] = {"_cell_length_a", "_cell_length_b", "_cell_length_c"};
    const char* angle_keys[] = {"_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"};
    for (std::size_t k = 0; k < 3; ++k) {
        out += std::string(length_keys[k]) + "   ";
        append_double(out, lengths[k], options.precision);
        out += '\n';
    }
    for (std::size_t k = 0; k < 3; ++k) {
        out += std::string(angle_keys[k]) + "   ";
        append_double(out, angles[k], options.precision);
        out += '\n';
    }

    // P1 with every atom written out: see the header for why this does not
    // try to assert a symmetry.
    out += "_space_group_name_H-M_alt   'P 1'\n";
    out += "_space_group_IT_number      1\n";
    out += "\nloop_\n_space_group_symop_operation_xyz\n  'x, y, z'\n";

    out += "\nloop_\n"
           "  _atom_site_label\n"
           "  _atom_site_type_symbol\n"
           "  _atom_site_fract_x\n"
           "  _atom_site_fract_y\n"
           "  _atom_site_fract_z\n"
           "  _atom_site_occupancy\n";

    const std::vector<std::string> symbols = atoms.symbols();
    const std::vector<double> scaled = atoms.scaled_positions();
    std::map<std::string, std::size_t> seen;
    for (std::size_t atom = 0; atom < atoms.size(); ++atom) {
        const std::size_t ordinal = ++seen[symbols[atom]];
        out += "  " + symbols[atom] + std::to_string(ordinal) + "  " + symbols[atom] + "  ";
        for (std::size_t axis = 0; axis < 3; ++axis) {
            append_double(out, scaled[3 * atom + axis], options.precision);
            out += "  ";
        }
        out += "1.0\n";
    }
    return out;
}

void write_cif(const std::string& path, const Atoms& atoms, CifOptions options) {
    std::ofstream stream(path);
    if (!stream) {
        throw ExtxyzError("cif: cannot write '" + path + "'");
    }
    stream << cif_to_string(atoms, std::move(options));
    if (!stream) {
        throw ExtxyzError("cif: writing '" + path + "' failed");
    }
}

}  // namespace fastatoms::io
