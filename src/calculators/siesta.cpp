// SPDX-License-Identifier: MIT
#include "fastatoms/calculators/siesta.hpp"

#include <algorithm>
#include <cctype>
#include <optional>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>

#include "fastatoms/core/elements.hpp"
#include "fastatoms/core/units.hpp"
#include "fastatoms/util/text.hpp"

namespace fastatoms {
namespace {

const std::set<std::string>& reserved_keys() {
    static const std::set<std::string> keys{"directory",  "command",    "log_file", "timeout",
                                            "environment", "label",     "xc",       "kpts",
                                            "kpts_shift",  "pp_path",   "fractional", "fdf",
                                            "mesh_cutoff"};
    return keys;
}

/// XC.functional / XC.authors for a functional name.
std::pair<std::string, std::string> xc_setting(const std::string& xc) {
    static const std::map<std::string, std::pair<std::string, std::string>> table{
        {"lda", {"LDA", "PZ"}},        {"pz", {"LDA", "PZ"}},
        {"ca", {"LDA", "CA"}},         {"pw92", {"LDA", "PW92"}},
        {"pbe", {"GGA", "PBE"}},       {"pbesol", {"GGA", "PBEsol"}},
        {"revpbe", {"GGA", "revPBE"}}, {"rpbe", {"GGA", "RPBE"}},
        {"blyp", {"GGA", "LYP"}},      {"wc", {"GGA", "WC"}},
        {"vdw", {"VDW", "DRSLL"}}};
    const auto it = table.find(util::lowercase(xc));
    if (it == table.end()) {
        return {};
    }
    return it->second;
}

/// `dm_tolerance` -> `dm.tolerance`: fdf directives are dotted, and YAML keys
/// read better with underscores.
///
/// Only the separator changes; the letters are left exactly as written. fdf
/// label matching ignores case as well as dots, dashes and underscores, so
/// `dm.tolerance`, `DM.Tolerance` and `DM_Tolerance` are all the same
/// directive to SIESTA - and passing the user's spelling through unchanged is
/// more predictable than guessing at canonical capitalisation.
std::string fdf_key(const std::string& key) {
    std::string out(key);
    std::replace(out.begin(), out.end(), '_', '.');
    return out;
}

std::string fdf_scalar(const Value& value) {
    switch (value.type()) {
        case Value::Type::Bool:
            return value.as_bool() ? ".true." : ".false.";
        case Value::Type::Int:
            return std::to_string(value.as_int());
        case Value::Type::Float:
            return util::format_real(value.as_double());
        case Value::Type::String:
            return value.as_string();
        default:
            return value.to_string();
    }
}

/// Render one directive. Lists become %block sections, which is how fdf
/// expresses anything multi-valued.
std::string fdf_directive(const std::string& key, const Value& value) {
    const std::string name = fdf_key(key);
    if (!value.is_list()) {
        return name + "  " + fdf_scalar(value) + "\n";
    }

    std::ostringstream out;
    out << "%block " << name << "\n";
    for (const Value& row : value.as_list()) {
        out << "   ";
        if (row.is_list()) {
            for (const Value& item : row.as_list()) {
                out << fdf_scalar(item) << " ";
            }
        } else {
            out << fdf_scalar(row);
        }
        out << "\n";
    }
    out << "%endblock " << name << "\n";
    return out.str();
}

std::string format_coordinate(double v) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(9) << std::setw(16) << v;
    return out.str();
}

}  // namespace

SiestaCalculator::SiestaCalculator(Value parameters) : FileIOCalculator(std::move(parameters)) {}

std::vector<Property> SiestaCalculator::implemented_properties() const {
    return {Property::Energy, Property::FreeEnergy, Property::Forces, Property::Stress};
}

std::string SiestaCalculator::label() const { return parameter("label").value_or("siesta"); }

std::string SiestaCalculator::default_command() const {
    return "siesta < " + label() + ".fdf > " + label() + ".out";
}

std::vector<std::string> SiestaCalculator::command_environment_names() const {
    return {"FASTATOMS_SIESTA_COMMAND", "ASE_SIESTA_COMMAND", "SIESTA_COMMAND"};
}

std::string SiestaCalculator::fdf_string(const Atoms& atoms) const {
    if (!atoms.cell().is_invertible()) {
        fail("SIESTA needs a full 3x3 cell; give the structure a cell (for a molecule, "
             "call center(vacuum=...)) before running");
    }

    // Species are numbered in order of first appearance, and each atom carries
    // its species index, so no reordering is needed.
    std::vector<std::string> species;
    std::vector<std::size_t> species_of_atom(atoms.size());
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        const std::string symbol = atoms.symbol(i);
        const auto it = std::find(species.begin(), species.end(), symbol);
        if (it == species.end()) {
            species_of_atom[i] = species.size();
            species.push_back(symbol);
        } else {
            species_of_atom[i] = static_cast<std::size_t>(it - species.begin());
        }
    }

    std::ostringstream out;
    out << "# fdf deck written by FastAtoms\n\n";
    out << "SystemName      " << atoms.formula() << "\n";
    out << "SystemLabel     " << label() << "\n";
    out << "NumberOfAtoms   " << atoms.size() << "\n";
    out << "NumberOfSpecies " << species.size() << "\n\n";

    out << "%block ChemicalSpeciesLabel\n";
    for (std::size_t k = 0; k < species.size(); ++k) {
        out << "   " << (k + 1) << "  " << number_from_symbol(species[k]) << "  " << species[k]
            << "\n";
    }
    out << "%endblock ChemicalSpeciesLabel\n\n";

    out << "LatticeConstant 1.0 Ang\n";
    out << "%block LatticeVectors\n";
    for (std::size_t row = 0; row < 3; ++row) {
        const Vec3 v = atoms.cell().matrix().row(row);
        out << "   " << format_coordinate(v.x) << format_coordinate(v.y) << format_coordinate(v.z)
            << "\n";
    }
    out << "%endblock LatticeVectors\n\n";

    const bool fractional = parameter("fractional").value_or(true);
    out << "AtomicCoordinatesFormat " << (fractional ? "Fractional" : "Ang") << "\n";
    out << "%block AtomicCoordinatesAndAtomicSpecies\n";
    const std::vector<double> coords = fractional ? atoms.scaled_positions() : atoms.positions();
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        out << "   " << format_coordinate(coords[3 * i]) << format_coordinate(coords[3 * i + 1])
            << format_coordinate(coords[3 * i + 2]) << "   " << (species_of_atom[i] + 1) << "\n";
    }
    out << "%endblock AtomicCoordinatesAndAtomicSpecies\n\n";

    // --- k-points --------------------------------------------------------
    const Value& kpts = parameter("kpts");
    if (!kpts.is_null()) {
        std::array<long long, 3> grid{1, 1, 1};
        if (kpts.is_number()) {
            grid = {kpts.as_int(), kpts.as_int(), kpts.as_int()};
        } else if (kpts.is_list() && kpts.size() == 3) {
            for (std::size_t k = 0; k < 3; ++k) {
                grid[k] = kpts.at(k).as_int();
            }
        } else {
            fail("'kpts' must be a single integer or a list of three integers");
        }
        std::array<double, 3> shift{0.0, 0.0, 0.0};
        if (const Value& value = parameter("kpts_shift"); !value.is_null()) {
            const std::vector<double> values = value.as_doubles();
            if (values.size() != 3) {
                fail("'kpts_shift' must have three components");
            }
            std::copy(values.begin(), values.end(), shift.begin());
        }
        out << "%block kgrid_Monkhorst_Pack\n";
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t col = 0; col < 3; ++col) {
                out << "   " << (row == col ? grid[row] : 0);
            }
            out << "   " << util::format_real(shift[row]) << "\n";
        }
        out << "%endblock kgrid_Monkhorst_Pack\n\n";
    }

    // --- functional and the rest of the deck --------------------------------
    std::vector<std::pair<std::string, std::string>> directives;
    const auto emit = [&directives](const std::string& key, std::string text) {
        const auto it = std::find_if(directives.begin(), directives.end(),
                                     [&key](const auto& pair) { return pair.first == key; });
        if (it == directives.end()) {
            directives.emplace_back(key, std::move(text));
        } else {
            it->second = std::move(text);
        }
    };

    const std::string xc = parameter("xc").value_or("pbe");
    if (!xc.empty()) {
        const auto [functional, authors] = xc_setting(xc);
        if (functional.empty()) {
            fail("unknown functional '" + xc +
                 "'; use lda, pz, ca, pw92, pbe, pbesol, revpbe, rpbe, blyp, wc or vdw, or set "
                 "XC.functional directly");
        }
        emit("XC.functional", "XC.functional  " + functional + "\n");
        emit("XC.authors", "XC.authors     " + authors + "\n");
    }
    if (const Value& cutoff = parameter("mesh_cutoff"); !cutoff.is_null()) {
        emit("MeshCutoff", "MeshCutoff     " + util::format_real(cutoff.as_double()) + " Ry\n");
    }
    // Forces and stress land in the .FORCE_STRESS file.
    emit("WriteForces", "WriteForces    .true.\n");

    for (const auto& [key, value] : parameters().as_map()) {
        if (reserved_keys().count(key) != 0) {
            continue;
        }
        emit(fdf_key(key), fdf_directive(key, value));
    }
    if (const Value& block = parameter("fdf"); !block.is_null()) {
        if (!block.is_map()) {
            fail("the 'fdf' parameter must be a mapping of directives to values");
        }
        for (const auto& [key, value] : block.as_map()) {
            emit(fdf_key(key), fdf_directive(key, value));
        }
    }

    for (const auto& [key, text] : directives) {
        out << text;
    }
    return out.str();
}

void SiestaCalculator::copy_pseudopotentials(const std::filesystem::path& directory,
                                             const std::vector<std::string>& symbols) const {
    const std::string root =
        parameter("pp_path").value_or(std::string_view(util::environment_value("SIESTA_PP_PATH")));

    for (const std::string& symbol : symbols) {
        // A file already staged in the run directory is left alone, so users
        // can drop in hand-tuned pseudopotentials.
        bool present = false;
        for (const char* extension : {".psf", ".psml", ".vps"}) {
            if (std::filesystem::exists(directory / (symbol + extension))) {
                present = true;
                break;
            }
        }
        if (present) {
            continue;
        }
        if (root.empty()) {
            fail("no pseudopotential for " + symbol + ": set the 'pp_path' parameter or the "
                 "SIESTA_PP_PATH environment variable, or place " + symbol +
                 ".psf in '" + directory.string() + "'");
        }

        std::filesystem::path source;
        std::string tried;
        for (const char* extension : {".psf", ".psml", ".vps"}) {
            const std::filesystem::path candidate =
                std::filesystem::path(root) / (symbol + extension);
            tried += "\n    " + candidate.string();
            if (std::filesystem::is_regular_file(candidate)) {
                source = candidate;
                break;
            }
        }
        if (source.empty()) {
            fail("no pseudopotential found for " + symbol + "; tried:" + tried);
        }

        std::error_code error;
        std::filesystem::copy_file(source, directory / source.filename(),
                                   std::filesystem::copy_options::overwrite_existing, error);
        if (error) {
            fail("could not copy '" + source.string() + "' into the run directory: " +
                 error.message());
        }
    }
}

Results SiestaCalculator::parse_output_text(std::string_view text) {
    const std::vector<std::string> lines = util::split_lines(text);
    Results results;
    Value extra;

    // --- energies ------------------------------------------------------------
    // The spelling of the energy summary differs between SIESTA versions, so
    // several forms are tried in order of preference rather than relying on
    // one exact match. Each search takes the last occurrence in the file.
    const auto last_value_where = [&lines](auto&& predicate) -> std::optional<double> {
        for (std::size_t k = lines.size(); k > 0; --k) {
            const std::string lowered = util::lowercase(util::trim(lines[k - 1]));
            if (predicate(lowered)) {
                const std::vector<double> values = util::scan_numbers(lines[k - 1]);
                if (!values.empty()) {
                    return values.back();
                }
            }
        }
        return std::nullopt;
    };
    const auto is_siesta_line = [](const std::string& lowered, const char* key) {
        return lowered.rfind("siesta:", 0) == 0 && lowered.find(key) != std::string::npos;
    };

    results.energy = last_value_where(
        [&](const std::string& line) { return is_siesta_line(line, "etot"); });
    if (!results.energy.has_value()) {
        // The "Final energy (eV)" block ends with a "Total =" line.
        results.energy = last_value_where(
            [&](const std::string& line) { return is_siesta_line(line, "total ="); });
    }
    if (!results.energy.has_value()) {
        results.energy = last_value_where(
            [&](const std::string& line) { return is_siesta_line(line, "e_ks(ev)"); });
    }

    results.free_energy = last_value_where([&](const std::string& line) {
        return is_siesta_line(line, "freeeng") || is_siesta_line(line, "free =");
    });
    if (!results.free_energy.has_value()) {
        results.free_energy = results.energy;
    }
    if (!results.energy.has_value()) {
        results.energy = results.free_energy;
    }

    // --- forces (eV/Ang in the text output) ----------------------------------
    const std::ptrdiff_t force_header = util::find_last_line(lines, "Atomic forces");
    if (force_header >= 0) {
        for (std::size_t row = static_cast<std::size_t>(force_header) + 1; row < lines.size();
             ++row) {
            const std::vector<double> values = util::scan_numbers(lines[row]);
            if (values.size() < 4) {
                break;
            }
            results.forces.push_back(values[values.size() - 3]);
            results.forces.push_back(values[values.size() - 2]);
            results.forces.push_back(values[values.size() - 1]);
        }
    }

    // --- stress (eV/Ang**3 in the text output) --------------------------------
    const std::ptrdiff_t stress_header = util::find_last_line(lines, "Stress tensor");
    if (stress_header >= 0 && static_cast<std::size_t>(stress_header) + 3 < lines.size()) {
        Mat3 tensor;
        bool complete = true;
        for (std::size_t row = 0; row < 3; ++row) {
            const std::vector<double> values =
                util::scan_numbers(lines[static_cast<std::size_t>(stress_header) + 1 + row]);
            if (values.size() < 3) {
                complete = false;
                break;
            }
            for (std::size_t col = 0; col < 3; ++col) {
                tensor(row, col) = values[values.size() - 3 + col];
            }
        }
        if (complete) {
            results.stress = matrix_to_voigt(tensor);
        }
    }

    // --- extras ----------------------------------------------------------------
    const std::ptrdiff_t fermi = util::find_last_line(lines, "Fermi =");
    if (fermi >= 0) {
        const std::vector<double> values =
            util::scan_numbers(lines[static_cast<std::size_t>(fermi)]);
        if (!values.empty()) {
            extra.insert("fermi_energy", Value(values.back()));
        }
    }
    extra.insert("job_completed", Value(util::find_last_line(lines, "Job completed") >= 0));
    results.extra = std::move(extra);

    return results;
}

void SiestaCalculator::parse_force_stress(std::string_view text, Results& results) {
    const std::vector<std::string> lines = util::split_lines(text);
    if (lines.size() < 5) {
        return;
    }

    // Layout: total energy, then the 3x3 stress in Ry/bohr^3, then the atom
    // count, then one line per atom holding index, species and the force in
    // Ry/bohr.
    const double stress_scale = units::Rydberg / (units::Bohr * units::Bohr * units::Bohr);
    Mat3 tensor;
    for (std::size_t row = 0; row < 3; ++row) {
        const std::vector<double> values = util::scan_numbers(lines[1 + row]);
        if (values.size() < 3) {
            return;
        }
        for (std::size_t col = 0; col < 3; ++col) {
            tensor(row, col) = values[col] * stress_scale;
        }
    }
    results.stress = matrix_to_voigt(tensor);

    const std::vector<double> count_values = util::scan_numbers(lines[4]);
    if (count_values.empty()) {
        return;
    }
    const auto natoms = static_cast<std::size_t>(count_values.front());

    const double force_scale = units::Rydberg / units::Bohr;
    std::vector<double> forces;
    forces.reserve(3 * natoms);
    for (std::size_t row = 5; row < lines.size() && forces.size() < 3 * natoms; ++row) {
        const std::vector<double> values = util::scan_numbers(lines[row]);
        if (values.size() < 5) {
            continue;
        }
        forces.push_back(values[2] * force_scale);
        forces.push_back(values[3] * force_scale);
        forces.push_back(values[4] * force_scale);
    }
    if (forces.size() == 3 * natoms) {
        results.forces = std::move(forces);
    }
}

void SiestaCalculator::write_input(const std::filesystem::path& directory, const Atoms& atoms,
                                   const std::vector<Property>& properties) {
    (void)properties;  // SIESTA always reports forces and stress
    std::vector<std::string> species;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        const std::string symbol = atoms.symbol(i);
        if (std::find(species.begin(), species.end(), symbol) == species.end()) {
            species.push_back(symbol);
        }
    }

    util::write_text_file(directory / (label() + ".fdf"), fdf_string(atoms));
    copy_pseudopotentials(directory, species);
}

void SiestaCalculator::parse_output(const std::filesystem::path& directory) {
    const std::filesystem::path output = directory / (label() + ".out");
    if (!std::filesystem::is_regular_file(output)) {
        fail("no '" + label() + ".out' in '" + std::filesystem::absolute(directory).string() +
             "'; check that the run command redirects SIESTA's output there");
    }

    Results parsed = parse_output_text(util::read_text_file(output));

    // .FORCE_STRESS carries more digits than the text output, so prefer it.
    const std::filesystem::path force_stress = directory / (label() + ".FORCE_STRESS");
    if (std::filesystem::is_regular_file(force_stress)) {
        parse_force_stress(util::read_text_file(force_stress), parsed);
    }

    if (!parsed.energy.has_value()) {
        fail("no total energy in '" + std::filesystem::absolute(output).string() +
             "'; the run probably failed");
    }
    results_ = std::move(parsed);
}

}  // namespace fastatoms
