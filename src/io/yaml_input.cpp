// SPDX-License-Identifier: MIT
#include "fastatoms/io/yaml_input.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <string>

#include <yaml-cpp/yaml.h>

#include "fastatoms/core/elements.hpp"

namespace fastatoms::io {
namespace {

bool item_is_scalar(const Value& node) {
    return node.is_string() || node.is_number();
}

std::string lowercase(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

/// Read a list of atom indices, accepting `"4-11"` alongside plain integers.
///
/// A slab calculation holds a substrate of tens or hundreds of atoms, and
/// writing those out one per entry makes the input file unreadable. Both
/// `[0, 1, 2]` and `["0-63", 70]` are accepted, as is a bare `"0-63"`.
std::vector<std::int64_t> parse_index_list(const Value& node, const char* what) {
    std::vector<std::int64_t> out;

    const auto add_range = [&](const std::string& text) {
        const std::size_t dash = text.find('-', text.front() == '-' ? 1 : 0);
        if (dash == std::string::npos) {
            throw InputError(std::string(what) + ": '" + text +
                             "' is not an index or an 'first-last' range");
        }
        const std::string first = text.substr(0, dash);
        const std::string last = text.substr(dash + 1);
        try {
            const std::int64_t lo = std::stoll(first);
            const std::int64_t hi = std::stoll(last);
            if (lo < 0 || hi < lo) {
                throw InputError(std::string(what) + ": '" + text + "' is not a forward range");
            }
            for (std::int64_t k = lo; k <= hi; ++k) {
                out.push_back(k);
            }
        } catch (const std::invalid_argument&) {
            throw InputError(std::string(what) + ": '" + text + "' is not a range of integers");
        } catch (const std::out_of_range&) {
            throw InputError(std::string(what) + ": '" + text + "' is out of range");
        }
    };

    const auto add = [&](const Value& item) {
        if (item.is_string()) {
            add_range(item.as_string());
        } else if (item.is_number()) {
            const double value = item.as_double();
            if (value != std::floor(value) || value < 0.0) {
                throw InputError(std::string(what) + ": indices must be whole and not negative");
            }
            out.push_back(static_cast<std::int64_t>(value));
        } else {
            throw InputError(std::string(what) + ": expected integers or 'first-last' ranges");
        }
    };

    if (item_is_scalar(node)) {
        add(node);
    } else if (node.is_list()) {
        for (const Value& item : node.as_list()) {
            add(item);
        }
    } else {
        throw InputError(std::string(what) + ": expected a list of indices");
    }

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

/// Convert a plain YAML scalar to the narrowest type it fits.
///
/// Quoted scalars keep the "!" tag in yaml-cpp and are always left as strings,
/// so `symbol: "1"` stays a string while `charge: 1` becomes an integer.
Value scalar_to_value(const YAML::Node& node) {
    const std::string& text = node.Scalar();
    if (node.Tag() == "!") {
        return Value(text);
    }
    if (text.empty() || text == "~" || text == "null" || text == "Null" || text == "NULL") {
        return Value{};
    }

    const std::string lower = lowercase(text);
    if (lower == "true" || lower == "yes" || lower == "on") {
        return Value(true);
    }
    if (lower == "false" || lower == "no" || lower == "off") {
        return Value(false);
    }

    const char* begin = text.c_str();
    char* end = nullptr;
    const long long as_int = std::strtoll(begin, &end, 10);
    if (end != begin && *end == '\0') {
        return Value(static_cast<std::int64_t>(as_int));
    }
    end = nullptr;
    const double as_float = std::strtod(begin, &end);
    if (end != begin && *end == '\0') {
        return Value(as_float);
    }
    if (lower == ".inf" || lower == "+.inf") {
        return Value(HUGE_VAL);
    }
    if (lower == "-.inf") {
        return Value(-HUGE_VAL);
    }
    if (lower == ".nan") {
        return Value(std::nan(""));
    }
    return Value(text);
}

Value node_to_value(const YAML::Node& node) {
    if (!node || node.IsNull()) {
        return Value{};
    }
    if (node.IsScalar()) {
        return scalar_to_value(node);
    }
    if (node.IsSequence()) {
        Value::List items;
        items.reserve(node.size());
        for (const auto& child : node) {
            items.push_back(node_to_value(child));
        }
        return Value(std::move(items));
    }
    if (node.IsMap()) {
        Value::Map items;
        items.reserve(node.size());
        for (const auto& pair : node) {
            items.emplace_back(pair.first.Scalar(), node_to_value(pair.second));
        }
        return Value(std::move(items));
    }
    return Value{};
}

/// First present key among `names`, or the shared null value.
const Value& first_of(const Value& node, std::initializer_list<const char*> names) {
    for (const char* name : names) {
        if (const Value* found = node.find(name)) {
            return *found;
        }
    }
    return Value::null();
}

Cell parse_cell(const Value& node) {
    if (node.is_null()) {
        return Cell{};
    }

    if (node.is_number()) {
        return Cell::cubic(node.as_double());
    }

    if (node.is_map()) {
        const double a = first_of(node, {"a"}).as_double();
        const double b = first_of(node, {"b"}).as_double();
        const double c = first_of(node, {"c"}).as_double();
        const double alpha = first_of(node, {"alpha"}).value_or(90.0);
        const double beta = first_of(node, {"beta"}).value_or(90.0);
        const double gamma = first_of(node, {"gamma"}).value_or(90.0);
        return Cell::from_parameters(a, b, c, alpha, beta, gamma);
    }

    const std::vector<double> values = node.as_doubles();
    switch (values.size()) {
        case 1:
            return Cell::cubic(values[0]);
        case 3:
            return Cell::orthorhombic(values[0], values[1], values[2]);
        case 6:
            return Cell::from_parameters(values[0], values[1], values[2], values[3], values[4],
                                         values[5]);
        case 9:
            return Cell(Mat3(values[0], values[1], values[2], values[3], values[4], values[5],
                             values[6], values[7], values[8]),
                        {true, true, true});
        default:
            throw InputError("cell: expected a single length, 3 lengths, 6 lattice parameters, "
                             "or a 3x3 matrix, but got " +
                             std::to_string(values.size()) + " numbers");
    }
}

std::array<bool, 3> parse_pbc(const Value& node, const std::array<bool, 3>& fallback) {
    if (node.is_null()) {
        return fallback;
    }
    if (node.is_bool()) {
        const bool v = node.as_bool();
        return {v, v, v};
    }
    if (node.is_string()) {
        // "xyz", "xy", "z", "" - the letters that are periodic.
        const std::string text = lowercase(node.as_string());
        std::array<bool, 3> pbc{false, false, false};
        for (const char c : text) {
            switch (c) {
                case 'x': pbc[0] = true; break;
                case 'y': pbc[1] = true; break;
                case 'z': pbc[2] = true; break;
                case 'f': case 'n': case ' ': break;
                default:
                    throw InputError("pbc: unexpected character '" + std::string(1, c) +
                                     "' (expected some combination of x, y and z)");
            }
        }
        return pbc;
    }
    if (node.is_list() && node.size() == 3) {
        std::array<bool, 3> pbc{};
        for (std::size_t k = 0; k < 3; ++k) {
            const Value& item = node.at(k);
            pbc[k] = item.is_bool() ? item.as_bool() : item.as_int() != 0;
        }
        return pbc;
    }
    throw InputError("pbc: expected a boolean, a list of 3 booleans, or a string such as 'xy'");
}

/// Read a per-atom vector array (3 components per atom) in either the flat or
/// the nested layout.
std::vector<double> parse_vector_array(const Value& node, std::size_t natoms, const char* what) {
    if (node.is_null()) {
        return {};
    }
    std::vector<double> values = node.as_doubles();
    if (values.size() != 3 * natoms) {
        throw InputError(std::string(what) + ": expected " + std::to_string(3 * natoms) +
                         " numbers for " + std::to_string(natoms) + " atoms, got " +
                         std::to_string(values.size()));
    }
    return values;
}

std::vector<double> parse_scalar_array(const Value& node, std::size_t natoms, const char* what) {
    if (node.is_null()) {
        return {};
    }
    std::vector<double> values = node.as_doubles();
    if (values.size() != natoms) {
        throw InputError(std::string(what) + ": expected " + std::to_string(natoms) +
                         " numbers, got " + std::to_string(values.size()));
    }
    return values;
}

/// One entry of the `atoms:` list, in any of its accepted spellings.
struct AtomEntry {
    int number{0};
    Vec3 position{};
    bool has_mass{false};
    double mass{0.0};
    bool has_charge{false};
    double charge{0.0};
    bool has_tag{false};
    int tag{0};
    bool has_velocity{false};
    Vec3 velocity{};
    bool has_momentum{false};
    Vec3 momentum{};
};

int species_to_number(const Value& node) {
    if (node.is_string()) {
        return number_from_symbol(node.as_string());
    }
    if (node.is_int()) {
        const std::int64_t z = node.as_int();
        if (z < 0 || z > kMaxAtomicNumber) {
            throw InputError("atomic number " + std::to_string(z) + " is out of range");
        }
        return static_cast<int>(z);
    }
    throw InputError("expected a chemical symbol or an atomic number, found " +
                     std::string(node.type_name()));
}

AtomEntry parse_atom_entry(const Value& node, std::size_t index) {
    AtomEntry entry;
    const std::string where = "atoms[" + std::to_string(index) + "]";

    // Compact form: [Si, x, y, z]
    if (node.is_list()) {
        if (node.size() != 4) {
            throw InputError(where + ": the compact form must be [symbol, x, y, z]");
        }
        entry.number = species_to_number(node.at(0));
        entry.position = {node.at(1).as_double(), node.at(2).as_double(), node.at(3).as_double()};
        return entry;
    }

    if (!node.is_map()) {
        throw InputError(where + ": expected a mapping or a [symbol, x, y, z] list");
    }

    const Value& species = first_of(node, {"symbol", "element", "species", "number", "Z"});
    if (species.is_null()) {
        throw InputError(where + ": missing 'symbol' (or 'number')");
    }
    entry.number = species_to_number(species);

    const Value& position = first_of(node, {"position", "positions", "xyz", "r"});
    if (position.is_null()) {
        throw InputError(where + ": missing 'position'");
    }
    const std::vector<double> r = position.as_doubles();
    if (r.size() != 3) {
        throw InputError(where + ": 'position' must have 3 components, got " +
                         std::to_string(r.size()));
    }
    entry.position = {r[0], r[1], r[2]};

    if (const Value& mass = first_of(node, {"mass"}); !mass.is_null()) {
        entry.has_mass = true;
        entry.mass = mass.as_double();
    }
    if (const Value& charge = first_of(node, {"charge", "q"}); !charge.is_null()) {
        entry.has_charge = true;
        entry.charge = charge.as_double();
    }
    if (const Value& tag = first_of(node, {"tag"}); !tag.is_null()) {
        entry.has_tag = true;
        entry.tag = static_cast<int>(tag.as_int());
    }
    if (const Value& v = first_of(node, {"velocity"}); !v.is_null()) {
        const std::vector<double> values = v.as_doubles();
        if (values.size() != 3) {
            throw InputError(where + ": 'velocity' must have 3 components");
        }
        entry.has_velocity = true;
        entry.velocity = {values[0], values[1], values[2]};
    }
    if (const Value& p = first_of(node, {"momentum"}); !p.is_null()) {
        const std::vector<double> values = p.as_doubles();
        if (values.size() != 3) {
            throw InputError(where + ": 'momentum' must have 3 components");
        }
        entry.has_momentum = true;
        entry.momentum = {values[0], values[1], values[2]};
    }
    return entry;
}

/// Element and range problems from the core library arrive here as
/// std::invalid_argument / std::out_of_range. From the parser's point of view
/// those are all bad input, so they are reported as InputError just like
/// everything else in a malformed document.
template <typename Fn>
auto as_input_error(const char* section, Fn&& fn) -> decltype(fn()) {
    try {
        return fn();
    } catch (const InputError&) {
        throw;
    } catch (const std::invalid_argument& error) {
        throw InputError(std::string(section) + ": " + error.what());
    } catch (const std::out_of_range& error) {
        throw InputError(std::string(section) + ": " + error.what());
    }
}

Atoms atoms_from_value_impl(const Value& structure);
SimulationParameters simulation_from_value_impl(const Value& node);

}  // namespace

Atoms atoms_from_value(const Value& structure) {
    return as_input_error("structure", [&] { return atoms_from_value_impl(structure); });
}

SimulationParameters simulation_from_value(const Value& node) {
    return as_input_error("simulation", [&] { return simulation_from_value_impl(node); });
}

namespace {

Atoms atoms_from_value_impl(const Value& structure) {
    if (structure.is_null()) {
        throw InputError("structure: section is empty");
    }
    if (!structure.is_map()) {
        throw InputError("structure: expected a mapping, found " +
                         std::string(structure.type_name()));
    }

    Cell cell = parse_cell(first_of(structure, {"cell", "lattice", "unit_cell"}));
    // Default to periodic whenever a cell was given, matching the usual intent
    // of writing one down at all.
    const std::array<bool, 3> pbc_default =
        cell.is_zero() ? std::array<bool, 3>{false, false, false}
                       : std::array<bool, 3>{true, true, true};
    cell.set_pbc(parse_pbc(first_of(structure, {"pbc", "periodic", "boundary"}), pbc_default));

    const Value& coord_system =
        first_of(structure, {"coordinates", "coordinate_system", "positions_are"});
    bool scaled_input = false;
    if (!coord_system.is_null()) {
        const std::string mode = lowercase(coord_system.as_string());
        if (mode == "scaled" || mode == "fractional" || mode == "direct" || mode == "crystal" ||
            mode == "relative") {
            scaled_input = true;
        } else if (mode != "cartesian" && mode != "absolute" && mode != "angstrom") {
            throw InputError("coordinates: expected 'cartesian' or 'scaled', got '" + mode + "'");
        }
    }

    std::vector<int> numbers;
    std::vector<double> positions;
    std::vector<double> masses, charges, velocities, momenta;
    std::vector<int> tags;
    bool have_masses = false, have_charges = false, have_tags = false;
    bool have_velocities = false, have_momenta = false;

    const Value& atom_list = first_of(structure, {"atoms", "sites", "particles"});
    if (!atom_list.is_null()) {
        if (!atom_list.is_list()) {
            throw InputError("atoms: expected a list of atoms");
        }
        const std::size_t natoms = atom_list.size();
        numbers.reserve(natoms);
        positions.reserve(3 * natoms);
        masses.assign(natoms, 0.0);
        charges.assign(natoms, 0.0);
        tags.assign(natoms, 0);
        velocities.assign(3 * natoms, 0.0);
        momenta.assign(3 * natoms, 0.0);

        for (std::size_t i = 0; i < natoms; ++i) {
            const AtomEntry entry = parse_atom_entry(atom_list.at(i), i);
            numbers.push_back(entry.number);
            positions.push_back(entry.position.x);
            positions.push_back(entry.position.y);
            positions.push_back(entry.position.z);
            if (entry.has_mass) {
                have_masses = true;
                masses[i] = entry.mass;
            } else {
                masses[i] = atomic_mass(entry.number);
            }
            if (entry.has_charge) {
                have_charges = true;
                charges[i] = entry.charge;
            }
            if (entry.has_tag) {
                have_tags = true;
                tags[i] = entry.tag;
            }
            if (entry.has_velocity) {
                have_velocities = true;
                entry.velocity.store(velocities.data() + 3 * i);
            }
            if (entry.has_momentum) {
                have_momenta = true;
                entry.momentum.store(momenta.data() + 3 * i);
            }
        }
    } else {
        // Columnar form: symbols / numbers plus a positions array.
        const Value& symbols = first_of(structure, {"symbols", "elements", "species"});
        const Value& numbers_node = first_of(structure, {"numbers", "atomic_numbers"});
        if (symbols.is_null() && numbers_node.is_null()) {
            throw InputError(
                "structure: expected either an 'atoms' list or 'symbols'/'numbers' arrays");
        }
        if (!symbols.is_null()) {
            for (const std::string& s : symbols.as_strings()) {
                numbers.push_back(number_from_symbol(s));
            }
        } else {
            for (const Value& item : numbers_node.as_list()) {
                numbers.push_back(species_to_number(item));
            }
        }

        const Value& positions_node = first_of(structure, {"positions", "coords", "xyz"});
        if (positions_node.is_null()) {
            throw InputError("structure: missing 'positions'");
        }
        positions = parse_vector_array(positions_node, numbers.size(), "positions");

        masses = parse_scalar_array(first_of(structure, {"masses"}), numbers.size(), "masses");
        have_masses = !masses.empty();
        charges = parse_scalar_array(first_of(structure, {"charges"}), numbers.size(), "charges");
        have_charges = !charges.empty();
        velocities =
            parse_vector_array(first_of(structure, {"velocities"}), numbers.size(), "velocities");
        have_velocities = !velocities.empty();
        momenta = parse_vector_array(first_of(structure, {"momenta"}), numbers.size(), "momenta");
        have_momenta = !momenta.empty();

        if (const Value& tags_node = first_of(structure, {"tags"}); !tags_node.is_null()) {
            for (const double t : parse_scalar_array(tags_node, numbers.size(), "tags")) {
                tags.push_back(static_cast<int>(t));
            }
            have_tags = true;
        }
    }

    if (scaled_input && !cell.is_invertible()) {
        throw InputError("coordinates: fractional coordinates need a full 3x3 cell");
    }

    Atoms atoms(std::move(numbers), std::move(positions), cell);
    if (scaled_input) {
        const std::vector<double> fractional = atoms.positions();
        atoms.set_scaled_positions(fractional);
    }

    if (have_masses) {
        atoms.set_masses(masses);
    }
    if (have_charges) {
        atoms.set_charges(charges);
    }
    if (have_tags) {
        atoms.set_tags(tags);
    }
    if (have_momenta) {
        atoms.set_momenta(momenta);
    } else if (have_velocities) {
        atoms.set_velocities(velocities);
    }

    // Free-form extra arrays: `arrays: {forces: [[...], ...]}`.
    if (const Value& arrays = first_of(structure, {"arrays"}); !arrays.is_null()) {
        if (!arrays.is_map()) {
            throw InputError("arrays: expected a mapping of name to data");
        }
        for (const auto& [name, data] : arrays.as_map()) {
            const std::vector<double> values = data.as_doubles();
            if (atoms.empty() || values.size() % atoms.size() != 0) {
                throw InputError("arrays." + name + ": " + std::to_string(values.size()) +
                                 " values do not divide evenly among " +
                                 std::to_string(atoms.size()) + " atoms");
            }
            atoms.set_array(name, values, values.size() / atoms.size());
        }
    }

    if (const Value& wrap = first_of(structure, {"wrap"}); wrap.value_or(false)) {
        atoms.wrap();
    }
    return atoms;
}

SimulationParameters simulation_from_value_impl(const Value& node) {
    SimulationParameters params;
    params.raw = node;
    if (node.is_null()) {
        return params;
    }
    if (!node.is_map()) {
        throw InputError("simulation: expected a mapping, found " + std::string(node.type_name()));
    }

    params.task_name = first_of(node, {"task", "run", "mode"}).value_or("single_point");
    params.task = task_from_string(params.task_name);

    if (const Value& calc = first_of(node, {"calculator", "potential", "model"});
        !calc.is_null()) {
        if (calc.is_string()) {
            params.calculator.name = calc.as_string();
        } else if (calc.is_map()) {
            params.calculator.name = first_of(calc, {"name", "type"}).value_or("");
            params.calculator.parameters = first_of(calc, {"parameters", "params", "options"});
            if (params.calculator.parameters.is_null()) {
                // Treat any remaining keys as the parameter block.
                Value::Map rest;
                for (const auto& [key, value] : calc.as_map()) {
                    if (key != "name" && key != "type") {
                        rest.emplace_back(key, value);
                    }
                }
                if (!rest.empty()) {
                    params.calculator.parameters = Value(std::move(rest));
                }
            }
        } else {
            throw InputError("calculator: expected a name or a mapping");
        }
    }

    if (const Value& md = first_of(node, {"md", "dynamics", "molecular_dynamics"});
        !md.is_null()) {
        params.md.parameters = md;
        params.md.ensemble = lowercase(first_of(md, {"ensemble"}).value_or("nve"));
        params.md.timestep_fs = first_of(md, {"timestep", "dt", "time_step"}).value_or(1.0);
        params.md.steps = first_of(md, {"steps", "n_steps", "nsteps", "num_steps"}).value_or(
            static_cast<std::int64_t>(0));
        params.md.temperature_K =
            first_of(md, {"temperature", "temperature_K", "T"}).value_or(0.0);
        params.md.pressure_GPa = first_of(md, {"pressure", "pressure_GPa", "P"}).value_or(0.0);
        params.md.thermostat = first_of(md, {"thermostat"}).value_or("");
        params.md.friction = first_of(md, {"friction", "damping"}).value_or(0.01);
        params.md.seed = first_of(md, {"seed", "random_seed"}).value_or(static_cast<std::int64_t>(0));
        params.md.initialize_velocities =
            first_of(md, {"initialize_velocities", "init_velocities"}).value_or(false);

        if (params.md.timestep_fs <= 0.0) {
            throw InputError("md.timestep: must be positive");
        }
        if (params.md.steps < 0) {
            throw InputError("md.steps: must not be negative");
        }
    }

    if (const Value& opt = first_of(node, {"optimizer", "optimization", "relaxation"});
        !opt.is_null()) {
        params.optimizer.parameters = opt;
        params.optimizer.algorithm =
            lowercase(first_of(opt, {"algorithm", "method", "name"}).value_or("bfgs"));
        params.optimizer.fmax = first_of(opt, {"fmax", "force_tolerance"}).value_or(0.05);
        params.optimizer.max_steps =
            first_of(opt, {"max_steps", "steps", "maxiter"}).value_or(static_cast<std::int64_t>(200));
        params.optimizer.relax_cell =
            first_of(opt, {"relax_cell", "variable_cell", "cell"}).value_or(false);

        if (params.optimizer.fmax <= 0.0) {
            throw InputError("optimizer.fmax: must be positive");
        }
    }

    if (const Value& ph = first_of(node, {"phonons", "phonon", "vibrations"}); !ph.is_null()) {
        params.phonon.parameters = ph;

        if (const Value& sc = first_of(ph, {"supercell", "supercell_matrix", "dim"});
            !sc.is_null()) {
            std::vector<double> entries;
            if (sc.is_number()) {
                entries.assign(3, sc.as_double());
            } else {
                entries = sc.as_doubles();
            }
            if (entries.size() != 3 && entries.size() != 9) {
                throw InputError("phonons.supercell: expected 1, 3 or 9 integers");
            }
            for (const double n : entries) {
                if (n != std::floor(n)) {
                    throw InputError("phonons.supercell: entries must be whole numbers");
                }
            }
            params.phonon.supercell = std::move(entries);
        }

        params.phonon.displacement =
            first_of(ph, {"displacement", "distance", "delta"}).value_or(0.01);
        params.phonon.symmetry = first_of(ph, {"symmetry", "use_symmetry"}).value_or(true);
        params.phonon.symmetry_tolerance =
            first_of(ph, {"symmetry_tolerance", "symprec", "tolerance"}).value_or(1e-5);

        if (params.phonon.displacement <= 0.0) {
            throw InputError("phonons.displacement: must be positive");
        }
        if (params.phonon.symmetry_tolerance <= 0.0) {
            throw InputError("phonons.symmetry_tolerance: must be positive");
        }
    }

    if (const Value& band = first_of(node, {"neb", "band", "reaction_path"}); !band.is_null()) {
        params.neb.parameters = band;

        // An endpoint is either a path to a file or a structure written out
        // in place, so that a small system needs only one document.
        if (const Value& start = first_of(band, {"initial", "start", "reactant"});
            start.is_map()) {
            params.neb.initial_structure = start;
        } else {
            params.neb.initial = start.value_or("");
        }
        if (const Value& stop = first_of(band, {"final", "end", "product"}); stop.is_map()) {
            params.neb.final_structure = stop;
        } else {
            params.neb.final = stop.value_or("");
        }
        params.neb.band = first_of(band, {"band", "images_file", "path"}).value_or("");

        params.neb.images =
            first_of(band, {"images", "nimages"}).value_or(static_cast<std::int64_t>(7));
        params.neb.spring_constant =
            first_of(band, {"spring_constant", "k", "spring"}).value_or(0.1);
        params.neb.climbing = first_of(band, {"climb", "climbing", "cineb"}).value_or(false);
        params.neb.climb_threshold =
            first_of(band, {"climb_threshold", "climbing_threshold"}).value_or(0.0);
        params.neb.tangent = lowercase(first_of(band, {"tangent", "tangent_estimate"})
                                           .value_or("improved"));
        params.neb.variable_springs =
            first_of(band, {"variable_springs", "variable_k"}).value_or(false);
        params.neb.spring_softening =
            first_of(band, {"spring_softening", "k_min_fraction"}).value_or(0.5);
        params.neb.minimum_image =
            first_of(band, {"minimum_image", "wrap"}).value_or(true);

        if (const Value& fix = first_of(band, {"fix", "fixed", "fixed_atoms", "constrain"});
            !fix.is_null()) {
            params.neb.fixed_atoms = parse_index_list(fix, "neb.fix");
        }

        params.neb.automatic = first_of(band, {"auto", "autoneb", "adaptive"}).value_or(false);
        params.neb.max_images =
            first_of(band, {"max_images", "maximum_images"}).value_or(static_cast<std::int64_t>(11));
        params.neb.coarse_fmax =
            first_of(band, {"coarse_fmax", "growing_fmax"}).value_or(0.2);
        params.neb.insert = lowercase(first_of(band, {"insert", "insertion"}).value_or("energy"));
        params.neb.prune = first_of(band, {"prune", "remove"}).value_or(false);
        params.neb.prune_tolerance = first_of(band, {"prune_tolerance"}).value_or(0.01);
        params.neb.redistribute =
            first_of(band, {"redistribute", "respace"}).value_or(false);

        if (params.neb.band.empty() && params.neb.final.empty() &&
            params.neb.final_structure.is_null()) {
            throw InputError("neb: needs a 'final' endpoint, or a 'band' file to start from");
        }
        if (params.neb.images < 3) {
            throw InputError("neb.images: a band needs at least 3 images");
        }
        if (params.neb.spring_constant <= 0.0) {
            throw InputError("neb.spring_constant: must be positive");
        }
        if (params.neb.max_images < params.neb.images) {
            throw InputError("neb.max_images: must not be smaller than neb.images");
        }
    }

    if (const Value& out = first_of(node, {"output", "io"}); !out.is_null()) {
        params.output.trajectory = first_of(out, {"trajectory", "traj"}).value_or("");
        params.output.log = first_of(out, {"log", "logfile"}).value_or("");
        params.output.interval =
            first_of(out, {"interval", "every", "write_every"}).value_or(static_cast<std::int64_t>(1));
        if (params.output.interval < 1) {
            throw InputError("output.interval: must be at least 1");
        }
    }

    return params;
}

}  // namespace

Value parse_yaml_string(const std::string& text) {
    try {
        return node_to_value(YAML::Load(text));
    } catch (const YAML::Exception& error) {
        throw InputError("YAML parse error at line " + std::to_string(error.mark.line + 1) +
                         ", column " + std::to_string(error.mark.column + 1) + ": " + error.msg);
    }
}

Value parse_yaml_file(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw InputError("cannot open input file '" + path + "'");
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    try {
        return parse_yaml_string(buffer.str());
    } catch (const InputError& error) {
        throw InputError(path + ": " + error.what());
    }
}

InputDeck read_yaml_string(const std::string& text) {
    const Value document = parse_yaml_string(text);
    InputDeck deck;
    deck.document = document;

    const Value& structure = first_of(document, {"structure", "system", "atoms_object"});
    // A document that is itself a structure (no `structure:` wrapper) is also
    // accepted, which keeps small hand-written files short.
    deck.atoms = atoms_from_value(structure.is_null() ? document : structure);
    deck.simulation = simulation_from_value(first_of(document, {"simulation", "run", "calculation"}));
    return deck;
}

InputDeck read_yaml_file(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw InputError("cannot open input file '" + path + "'");
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    try {
        return read_yaml_string(buffer.str());
    } catch (const InputError& error) {
        throw InputError(path + ": " + error.what());
    }
}

std::string write_yaml_string(const Atoms& atoms, bool scaled) {
    Value structure;

    Value::List cell_rows;
    for (std::size_t r = 0; r < 3; ++r) {
        const Vec3 v = atoms.cell().matrix().row(r);
        cell_rows.push_back(Value(Value::List{Value(v.x), Value(v.y), Value(v.z)}));
    }
    structure.insert("cell", Value(std::move(cell_rows)));

    const auto& pbc = atoms.pbc();
    structure.insert("pbc",
                     Value(Value::List{Value(pbc[0]), Value(pbc[1]), Value(pbc[2])}));
    structure.insert("coordinates", Value(scaled ? "scaled" : "cartesian"));

    const std::vector<double> coords = scaled ? atoms.scaled_positions() : atoms.positions();
    const std::vector<double> momenta = atoms.momenta();
    const std::vector<double> charges = atoms.charges();
    const std::vector<int> tags = atoms.tags();

    Value::List entries;
    entries.reserve(atoms.size());
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        Value entry;
        entry.insert("symbol", Value(atoms.symbol(i)));
        entry.insert("position", Value(Value::List{Value(coords[3 * i]), Value(coords[3 * i + 1]),
                                                   Value(coords[3 * i + 2])}));
        if (atoms.has_masses()) {
            entry.insert("mass", Value(atoms.mass(i)));
        }
        if (atoms.has_charges()) {
            entry.insert("charge", Value(charges[i]));
        }
        if (atoms.has_tags()) {
            entry.insert("tag", Value(static_cast<std::int64_t>(tags[i])));
        }
        if (atoms.has_momenta()) {
            entry.insert("momentum", Value(Value::List{Value(momenta[3 * i]),
                                                       Value(momenta[3 * i + 1]),
                                                       Value(momenta[3 * i + 2])}));
        }
        entries.push_back(std::move(entry));
    }
    structure.insert("atoms", Value(std::move(entries)));

    const std::vector<std::string> extra = atoms.array_names();
    if (!extra.empty()) {
        Value arrays;
        for (const std::string& name : extra) {
            const Atoms::Array& array = atoms.array(name);
            Value::List rows;
            rows.reserve(atoms.size());
            if (array.ncols == 1) {
                // A scalar array reads better as one flat list.
                for (const double value : array.data) {
                    rows.push_back(Value(value));
                }
            } else {
                for (std::size_t i = 0; i < atoms.size(); ++i) {
                    Value::List row;
                    row.reserve(array.ncols);
                    for (std::size_t c = 0; c < array.ncols; ++c) {
                        row.push_back(Value(array.data[i * array.ncols + c]));
                    }
                    rows.push_back(Value(std::move(row)));
                }
            }
            arrays.insert(name, Value(std::move(rows)));
        }
        structure.insert("arrays", std::move(arrays));
    }

    Value document;
    document.insert("structure", std::move(structure));
    return document.to_yaml();
}

void write_yaml_file(const std::string& path, const Atoms& atoms, bool scaled) {
    std::ofstream stream(path);
    if (!stream) {
        throw InputError("cannot open '" + path + "' for writing");
    }
    stream << write_yaml_string(atoms, scaled);
    if (!stream) {
        throw InputError("failed while writing '" + path + "'");
    }
}

}  // namespace fastatoms::io
