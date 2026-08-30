#include "fastatoms/io/vasp.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>

#include "fastatoms/calculators/single_point.hpp"
#include "fastatoms/core/constraints.hpp"
#include "fastatoms/core/elements.hpp"
#include "fastatoms/core/units.hpp"

namespace fastatoms::io {
namespace {

std::vector<std::string> split(const std::string& line) {
    std::istringstream stream(line);
    std::vector<std::string> fields;
    std::string field;
    while (stream >> field) {
        fields.push_back(field);
    }
    return fields;
}

/// A line, or throw saying which file ran out and where.
std::string require_line(std::istream& stream, const std::string& origin, const char* what) {
    std::string line;
    if (!std::getline(stream, line)) {
        throw ExtxyzError("vasp: '" + origin + "' ends where the " + what + " should be");
    }
    return line;
}

double to_double(const std::string& text, const std::string& origin, const char* what) {
    try {
        std::size_t consumed = 0;
        const double value = std::stod(text, &consumed);
        if (consumed != text.size()) {
            throw std::invalid_argument("trailing");
        }
        return value;
    } catch (const std::exception&) {
        throw ExtxyzError("vasp: '" + origin + "': the " + what + " should be a number, not '" +
                          text + "'");
    }
}

long to_long(const std::string& text, const std::string& origin, const char* what) {
    try {
        std::size_t consumed = 0;
        const long value = std::stol(text, &consumed);
        if (consumed != text.size()) {
            throw std::invalid_argument("trailing");
        }
        return value;
    } catch (const std::exception&) {
        throw ExtxyzError("vasp: '" + origin + "': the " + what + " should be a whole number, "
                          "not '" + text + "'");
    }
}

/// True when every field names an element, which is how a version-4 file's
/// comment line can be pressed into service as the species list.
bool all_are_symbols(const std::vector<std::string>& fields) {
    if (fields.empty()) {
        return false;
    }
    return std::all_of(fields.begin(), fields.end(), [](const std::string& field) {
        return try_number_from_symbol(field) > 0;
    });
}

Mat3 read_cell(std::istream& stream, const std::string& origin, double scale) {
    Mat3 cell;
    for (std::size_t row = 0; row < 3; ++row) {
        const auto fields = split(require_line(stream, origin, "cell"));
        if (fields.size() < 3) {
            throw ExtxyzError("vasp: '" + origin + "': a cell vector is three numbers, not " +
                              std::to_string(fields.size()));
        }
        for (std::size_t column = 0; column < 3; ++column) {
            cell.m[3 * row + column] = scale * to_double(fields[column], origin, "cell vector");
        }
    }
    return cell;
}

/// The species list and the count list, expanded into one symbol per atom.
std::vector<std::string> expand_species(const std::vector<std::string>& names,
                                        const std::vector<std::string>& counts,
                                        const std::string& origin) {
    if (names.size() != counts.size()) {
        throw ExtxyzError("vasp: '" + origin + "': " + std::to_string(names.size()) +
                          " species but " + std::to_string(counts.size()) +
                          " counts; they have to line up");
    }
    std::vector<std::string> symbols;
    for (std::size_t k = 0; k < names.size(); ++k) {
        const long count = to_long(counts[k], origin, "atom count");
        if (count < 0) {
            throw ExtxyzError("vasp: '" + origin + "': a negative atom count");
        }
        symbols.insert(symbols.end(), static_cast<std::size_t>(count), names[k]);
    }
    return symbols;
}

void append_double(std::string& out, double value, int precision) {
    std::ostringstream stream;
    stream << std::setprecision(precision) << std::fixed << value;
    out += stream.str();
}

}  // namespace

// ---------------------------------------------------------------------------
// POSCAR / CONTCAR
// ---------------------------------------------------------------------------

Atoms parse_poscar(std::istream& stream, const std::string& origin) {
    const std::string comment = require_line(stream, origin, "comment line");

    const auto scale_fields = split(require_line(stream, origin, "scale factor"));
    if (scale_fields.empty()) {
        throw ExtxyzError("vasp: '" + origin + "': no scale factor on line 2");
    }
    double scale = to_double(scale_fields[0], origin, "scale factor");

    Mat3 cell = read_cell(stream, origin, 1.0);
    if (scale < 0.0) {
        // A negative scale is a target volume, not a length: VASP reads it as
        // the volume the cell should be scaled to have.
        const double volume = std::abs(cell.determinant());
        scale = std::cbrt(-scale / volume);
    }
    for (double& element : cell.m) {
        element *= scale;
    }

    auto first = split(require_line(stream, origin, "species or counts"));
    std::vector<std::string> names;
    std::vector<std::string> counts;
    if (all_are_symbols(first)) {
        names = std::move(first);
        counts = split(require_line(stream, origin, "atom counts"));
    } else {
        // Version 4: no species line. The comment is the only place left that
        // might name them, and every code that writes such a file puts them
        // there - but guessing beyond that would invent an answer.
        counts = std::move(first);
        const auto from_comment = split(comment);
        if (!all_are_symbols(from_comment)) {
            throw ExtxyzError(
                "vasp: '" + origin +
                "' has no species line, and its comment line does not name the elements either. "
                "This is a VASP 4 file whose species only exist in its POTCAR; add a species "
                "line (VASP 5 style) above the counts.");
        }
        names = from_comment;
    }

    std::vector<std::string> symbols = expand_species(names, counts, origin);
    const std::size_t count = symbols.size();

    std::string mode = require_line(stream, origin, "coordinate mode");
    bool selective = false;
    if (!mode.empty() && (mode[0] == 'S' || mode[0] == 's')) {
        selective = true;
        mode = require_line(stream, origin, "coordinate mode");
    }
    const bool direct = mode.empty() || mode[0] == 'D' || mode[0] == 'd' || mode[0] == '\r';

    std::vector<double> coordinates(3 * count, 0.0);
    std::vector<std::size_t> fixed;
    for (std::size_t atom = 0; atom < count; ++atom) {
        const auto fields = split(require_line(stream, origin, "position"));
        if (fields.size() < 3) {
            throw ExtxyzError("vasp: '" + origin + "': position " + std::to_string(atom + 1) +
                              " has " + std::to_string(fields.size()) + " numbers, not three");
        }
        for (std::size_t axis = 0; axis < 3; ++axis) {
            coordinates[3 * atom + axis] = to_double(fields[axis], origin, "position");
        }
        // Selective dynamics: three T/F flags. An atom free along some axes but
        // not others is a FixCartesian rather than a FixAtoms; until that
        // exists here, anything not wholly free is held wholly fixed, which is
        // the conservative reading - it never lets an atom move that VASP was
        // holding still.
        if (selective && fields.size() >= 6) {
            const bool free = fields[3][0] == 'T' || fields[3][0] == 't';
            const bool free_y = fields[4][0] == 'T' || fields[4][0] == 't';
            const bool free_z = fields[5][0] == 'T' || fields[5][0] == 't';
            if (!(free && free_y && free_z)) {
                fixed.push_back(atom);
            }
        }
    }

    Atoms atoms(symbols, std::vector<double>(3 * count, 0.0));
    atoms.set_cell(cell);
    atoms.set_pbc(true);
    if (direct) {
        atoms.set_scaled_positions(coordinates);
    } else {
        for (double& value : coordinates) {
            value *= scale;
        }
        atoms.set_positions(coordinates);
    }
    if (!fixed.empty()) {
        atoms.add_constraint(std::make_shared<FixAtoms>(std::move(fixed)));
    }
    return atoms;
}

Atoms read_poscar(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw ExtxyzError("vasp: cannot open '" + path + "'");
    }
    return parse_poscar(stream, path);
}

std::string poscar_to_string(const Atoms& atoms, PoscarOptions options) {
    const std::size_t count = atoms.size();
    if (count == 0) {
        throw ExtxyzError("vasp: a POSCAR needs at least one atom");
    }
    if (std::abs(atoms.cell().matrix().determinant()) < 1e-12) {
        throw ExtxyzError("vasp: a POSCAR needs a cell; this structure has none");
    }

    // VASP groups atoms by species, and the counts line says how many of each.
    // The order on file therefore need not be the order in memory, so the
    // grouping is worked out first and everything else follows it.
    std::vector<std::string> order;
    std::vector<std::vector<std::size_t>> groups;
    const std::vector<std::string> symbols = atoms.symbols();
    for (std::size_t atom = 0; atom < count; ++atom) {
        const auto found = std::find(order.begin(), order.end(), symbols[atom]);
        if (found == order.end()) {
            order.push_back(symbols[atom]);
            groups.push_back({atom});
        } else {
            groups[static_cast<std::size_t>(found - order.begin())].push_back(atom);
        }
    }

    std::string comment = options.comment;
    if (comment.empty()) {
        for (std::size_t k = 0; k < order.size(); ++k) {
            comment += (k ? " " : "");
            comment += order[k] + std::to_string(groups[k].size());
        }
    }

    std::string out = comment + "\n1.0\n";
    const Mat3& cell = atoms.cell().matrix();
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            out += "  ";
            append_double(out, cell.m[3 * row + column], options.precision);
        }
        out += '\n';
    }
    for (std::size_t k = 0; k < order.size(); ++k) {
        out += (k ? " " : "") + order[k];
    }
    out += '\n';
    for (std::size_t k = 0; k < order.size(); ++k) {
        out += (k ? " " : "") + std::to_string(groups[k].size());
    }
    out += '\n';

    // Which atoms are held still, so the flags can be written per atom.
    std::vector<bool> fixed(count, false);
    bool any_fixed = false;
    if (options.selective_dynamics) {
        for (const auto& constraint : atoms.constraints()) {
            for (const std::size_t index : constraint->indices()) {
                if (index < count) {
                    fixed[index] = true;
                    any_fixed = true;
                }
            }
        }
    }
    if (any_fixed) {
        out += "Selective dynamics\n";
    }
    out += options.direct ? "Direct\n" : "Cartesian\n";

    const std::vector<double> coordinates =
        options.direct ? atoms.scaled_positions() : atoms.positions();
    for (const auto& group : groups) {
        for (const std::size_t atom : group) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                out += "  ";
                append_double(out, coordinates[3 * atom + axis], options.precision);
            }
            if (any_fixed) {
                out += fixed[atom] ? "   F   F   F" : "   T   T   T";
            }
            out += '\n';
        }
    }
    return out;
}

void write_poscar(const std::string& path, const Atoms& atoms, PoscarOptions options) {
    std::ofstream stream(path);
    if (!stream) {
        throw ExtxyzError("vasp: cannot write '" + path + "'");
    }
    stream << poscar_to_string(atoms, std::move(options));
    if (!stream) {
        throw ExtxyzError("vasp: writing '" + path + "' failed");
    }
}

// ---------------------------------------------------------------------------
// XDATCAR
// ---------------------------------------------------------------------------

std::vector<Atoms> read_xdatcar(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw ExtxyzError("vasp: cannot open '" + path + "'");
    }

    std::vector<Atoms> frames;
    // The header appears once for a fixed cell and once per frame for a
    // variable one. Reading it whenever the next line is not a
    // "Direct configuration=" marker handles both without being told which.
    Mat3 cell;
    std::vector<std::string> symbols;
    bool have_header = false;

    std::string line;
    while (std::getline(stream, line)) {
        const auto fields = split(line);
        if (fields.empty()) {
            continue;
        }

        const bool is_marker = line.find("configuration=") != std::string::npos ||
                               (fields[0] == "Direct" && fields.size() <= 2);
        if (!is_marker) {
            // A header block: comment, scale, cell, species, counts.
            const auto scale_fields = split(require_line(stream, path, "scale factor"));
            if (scale_fields.empty()) {
                throw ExtxyzError("vasp: '" + path + "': no scale factor after the comment");
            }
            const double scale = to_double(scale_fields[0], path, "scale factor");
            cell = read_cell(stream, path, scale);
            const auto names = split(require_line(stream, path, "species"));
            if (!all_are_symbols(names)) {
                throw ExtxyzError("vasp: '" + path + "' does not name its species; an XDATCAR "
                                  "without a species line cannot be read on its own");
            }
            const auto counts = split(require_line(stream, path, "atom counts"));
            symbols = expand_species(names, counts, path);
            have_header = true;
            continue;
        }

        if (!have_header) {
            throw ExtxyzError("vasp: '" + path + "' starts with a frame marker but no header");
        }

        std::vector<double> scaled(3 * symbols.size(), 0.0);
        for (std::size_t atom = 0; atom < symbols.size(); ++atom) {
            const auto position = split(require_line(stream, path, "position"));
            if (position.size() < 3) {
                throw ExtxyzError("vasp: '" + path + "': frame " +
                                  std::to_string(frames.size() + 1) + " is short of positions");
            }
            for (std::size_t axis = 0; axis < 3; ++axis) {
                scaled[3 * atom + axis] = to_double(position[axis], path, "position");
            }
        }

        Atoms frame(symbols, std::vector<double>(3 * symbols.size(), 0.0));
        frame.set_cell(cell);
        frame.set_pbc(true);
        frame.set_scaled_positions(scaled);
        frames.push_back(std::move(frame));
    }

    if (frames.empty()) {
        throw ExtxyzError("vasp: '" + path + "' holds no frames");
    }
    return frames;
}

// ---------------------------------------------------------------------------
// OUTCAR
// ---------------------------------------------------------------------------

std::vector<Atoms> read_outcar(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw ExtxyzError("vasp: cannot open '" + path + "'");
    }

    std::vector<std::string> species;   // in POTCAR order
    std::vector<std::size_t> counts;
    std::vector<std::string> symbols;
    Mat3 cell;
    bool have_cell = false;

    std::vector<Atoms> frames;
    std::vector<double> positions;
    std::vector<double> forces;
    std::optional<std::array<double, 6>> stress;
    std::optional<double> energy;
    std::optional<double> free_energy;

    const auto flush = [&]() {
        if (positions.empty() || symbols.empty()) {
            return;
        }
        Atoms atoms(symbols, positions);
        if (have_cell) {
            atoms.set_cell(cell);
            atoms.set_pbc(true);
        }
        Results results;
        results.energy = energy;
        results.free_energy = free_energy ? free_energy : energy;
        if (forces.size() == positions.size()) {
            results.forces = forces;
        }
        results.stress = stress;
        if (results.energy || !results.forces.empty()) {
            atoms.set_calculator(std::make_shared<SinglePointCalculator>(atoms, results));
        }
        frames.push_back(std::move(atoms));
        positions.clear();
        forces.clear();
        stress.reset();
        energy.reset();
        free_energy.reset();
    };

    std::string line;
    while (std::getline(stream, line)) {
        // The species, one "VRHFIN =Cu:" line each, in POTCAR order.
        if (line.find("VRHFIN") != std::string::npos) {
            const std::size_t equals = line.find('=');
            if (equals != std::string::npos) {
                std::string symbol;
                for (std::size_t k = equals + 1; k < line.size(); ++k) {
                    if (std::isalpha(static_cast<unsigned char>(line[k]))) {
                        symbol += line[k];
                    } else if (!symbol.empty()) {
                        break;
                    }
                }
                if (try_number_from_symbol(symbol) > 0) {
                    species.push_back(symbol);
                }
            }
            continue;
        }

        if (line.find("ions per type") != std::string::npos) {
            const std::size_t equals = line.find('=');
            if (equals != std::string::npos) {
                counts.clear();
                for (const auto& field : split(line.substr(equals + 1))) {
                    counts.push_back(static_cast<std::size_t>(to_long(field, path, "ion count")));
                }
            }
            if (species.size() == counts.size()) {
                symbols.clear();
                for (std::size_t k = 0; k < species.size(); ++k) {
                    symbols.insert(symbols.end(), counts[k], species[k]);
                }
            }
            continue;
        }

        if (line.find("direct lattice vectors") != std::string::npos) {
            for (std::size_t row = 0; row < 3; ++row) {
                const auto fields = split(require_line(stream, path, "lattice vector"));
                if (fields.size() < 3) {
                    break;
                }
                for (std::size_t column = 0; column < 3; ++column) {
                    cell.m[3 * row + column] = to_double(fields[column], path, "lattice vector");
                }
                have_cell = true;
            }
            continue;
        }

        // "in kB" gives the stress in kilobar; VASP's sign is the opposite of
        // the one used here, and 1 kB = 0.1 GPa.
        if (line.find("in kB") != std::string::npos) {
            const auto fields = split(line);
            if (fields.size() >= 8) {
                std::array<double, 6> voigt{};
                const double kbar_to_ev_per_a3 = 0.1 * units::GPa;
                // XX YY ZZ XY YZ ZX in the file; xx yy zz yz xz xy here.
                const double xx = to_double(fields[2], path, "stress");
                const double yy = to_double(fields[3], path, "stress");
                const double zz = to_double(fields[4], path, "stress");
                const double xy = to_double(fields[5], path, "stress");
                const double yz = to_double(fields[6], path, "stress");
                const double zx = to_double(fields[7], path, "stress");
                voigt = {xx, yy, zz, yz, zx, xy};
                for (double& value : voigt) {
                    value *= -kbar_to_ev_per_a3;
                }
                stress = voigt;
            }
            continue;
        }

        if (line.find("TOTAL-FORCE") != std::string::npos) {
            require_line(stream, path, "force table rule");
            positions.clear();
            forces.clear();
            while (std::getline(stream, line)) {
                const auto fields = split(line);
                if (fields.size() < 6) {
                    break;
                }
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    positions.push_back(to_double(fields[axis], path, "position"));
                    forces.push_back(to_double(fields[axis + 3], path, "force"));
                }
            }
            continue;
        }

        if (line.find("free  energy   TOTEN") != std::string::npos) {
            const auto fields = split(line);
            if (fields.size() >= 5) {
                free_energy = to_double(fields[4], path, "free energy");
            }
            continue;
        }

        // "energy without entropy = X energy(sigma->0) = Y": the second is
        // the extrapolated one, which is what an energy should mean here.
        if (line.find("energy(sigma->0)") != std::string::npos) {
            const std::size_t marker = line.rfind('=');
            if (marker != std::string::npos) {
                const auto fields = split(line.substr(marker + 1));
                if (!fields.empty()) {
                    energy = to_double(fields[0], path, "energy");
                }
            }
            flush();
            continue;
        }
    }
    flush();

    if (frames.empty()) {
        throw ExtxyzError("vasp: '" + path + "' holds no completed ionic steps");
    }
    return frames;
}

}  // namespace fastatoms::io
