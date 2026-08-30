// SPDX-License-Identifier: MIT
#include "fastatoms/calculators/quantum_espresso.hpp"

#include <algorithm>
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
    static const std::set<std::string> keys{
        "directory",   "command",        "log_file",   "timeout",     "environment",
        "pseudopotentials", "pseudo_dir", "kpts",      "kpts_shift",  "input_file",
        "output_file", "crystal_coordinates",          "control",     "system",
        "electrons",   "ions",           "cell"};
    return keys;
}

/// pw.x keywords grouped by the namelist that owns them. Only the keywords a
/// driver realistically sets are listed; anything else falls through to
/// &SYSTEM, which is what ASE does too.
const std::map<std::string, std::string>& keyword_namelists() {
    static const std::map<std::string, std::string> table = [] {
        std::map<std::string, std::string> map;
        const auto add = [&map](const char* namelist, std::initializer_list<const char*> keys) {
            for (const char* key : keys) {
                map.emplace(key, namelist);
            }
        };
        add("control", {"calculation", "title", "verbosity", "restart_mode", "wf_collect",
                        "nstep", "iprint", "tstress", "tprnfor", "dt", "outdir", "wfcdir",
                        "prefix", "lkpoint_dir", "max_seconds", "etot_conv_thr",
                        "forc_conv_thr", "disk_io", "pseudo_dir", "tefield", "dipfield",
                        "lelfield", "nberrycyc", "lorbm", "lberry", "gdir", "nppstr"});
        add("system", {"ibrav", "nat", "ntyp", "nbnd", "tot_charge", "tot_magnetization",
                       "starting_magnetization", "ecutwfc", "ecutrho", "ecutfock", "nosym",
                       "noinv", "no_t_rev", "force_symmorphic", "occupations", "degauss",
                       "smearing", "nspin", "noncolin", "input_dft", "exx_fraction",
                       "screening_parameter", "exxdiv_treatment", "nqx1", "nqx2", "nqx3",
                       "lda_plus_u", "lda_plus_u_kind", "u_projection_type", "edir",
                       "emaxpos", "eopreg", "eamp", "lspinorb", "assume_isolated", "esm_bc",
                       "vdw_corr", "london", "london_s6", "london_rcut", "xdm", "space_group"});
        add("electrons", {"electron_maxstep", "scf_must_converge", "conv_thr", "adaptive_thr",
                          "conv_thr_init", "conv_thr_multi", "mixing_mode", "mixing_beta",
                          "mixing_ndim", "mixing_fixed_ns", "diagonalization",
                          "diago_thr_init", "diago_cg_maxiter", "diago_david_ndim",
                          "diago_full_acc", "startingpot", "startingwfc", "tqr"});
        add("ions", {"ion_dynamics", "ion_positions", "pot_extrapolation", "wfc_extrapolation",
                     "remove_rigid_rot", "ion_temperature", "tempw", "tolp", "delta_t",
                     "nraise", "refold_pos", "upscale", "bfgs_ndim", "trust_radius_max",
                     "trust_radius_min", "trust_radius_ini", "w_1", "w_2"});
        add("cell", {"cell_dynamics", "press", "wmass", "cell_factor", "press_conv_thr",
                     "cell_dofree"});
        return map;
    }();
    return table;
}

/// Render a value in Fortran namelist syntax.
std::string namelist_value(const Value& value) {
    switch (value.type()) {
        case Value::Type::Bool:
            return value.as_bool() ? ".true." : ".false.";
        case Value::Type::Int:
            return std::to_string(value.as_int());
        case Value::Type::Float:
            return util::format_real(value.as_double());
        case Value::Type::String:
            return "'" + value.as_string() + "'";
        default:
            return value.to_string();
    }
}

std::string format_coordinate(double v) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(10) << std::setw(18) << v;
    return out.str();
}

using Namelists = std::map<std::string, std::vector<std::pair<std::string, Value>>>;

void set_key(Namelists& namelists, const std::string& group, const std::string& key, Value value) {
    std::vector<std::pair<std::string, Value>>& entries = namelists[group];
    const auto it = std::find_if(entries.begin(), entries.end(),
                                 [&key](const auto& pair) { return pair.first == key; });
    if (it == entries.end()) {
        entries.emplace_back(key, std::move(value));
    } else {
        it->second = std::move(value);
    }
}

}  // namespace

QuantumEspressoCalculator::QuantumEspressoCalculator(Value parameters)
    : FileIOCalculator(std::move(parameters)) {}

std::vector<Property> QuantumEspressoCalculator::implemented_properties() const {
    return {Property::Energy, Property::FreeEnergy, Property::Forces, Property::Stress,
            Property::Magmom};
}

std::string QuantumEspressoCalculator::namelist_for(std::string_view keyword) {
    const auto& table = keyword_namelists();
    const auto it = table.find(util::lowercase(keyword));
    return it == table.end() ? std::string("system") : it->second;
}

std::string QuantumEspressoCalculator::input_file_name() const {
    return parameter("input_file").value_or("pw.in");
}

std::string QuantumEspressoCalculator::output_file_name() const {
    return parameter("output_file").value_or("pw.out");
}

std::string QuantumEspressoCalculator::default_command() const {
    return "pw.x -in " + input_file_name() + " > " + output_file_name();
}

std::vector<std::string> QuantumEspressoCalculator::command_environment_names() const {
    return {"FASTATOMS_ESPRESSO_COMMAND", "ASE_ESPRESSO_COMMAND", "ESPRESSO_COMMAND"};
}

std::string QuantumEspressoCalculator::input_string(
    const Atoms& atoms, const std::vector<Property>& properties) const {
    if (!atoms.cell().is_invertible()) {
        fail("pw.x needs a full 3x3 cell; give the structure a cell before running");
    }

    // --- species and pseudopotentials --------------------------------------
    std::vector<std::string> species;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        const std::string symbol = atoms.symbol(i);
        if (std::find(species.begin(), species.end(), symbol) == species.end()) {
            species.push_back(symbol);
        }
    }

    const Value& pseudos = parameter("pseudopotentials");
    if (pseudos.is_null()) {
        fail("no pseudopotentials given; set the 'pseudopotentials' parameter, e.g. "
             "{Si: Si.pbe-n-kjpaw_psl.1.0.0.UPF}");
    }
    if (!pseudos.is_map()) {
        fail("the 'pseudopotentials' parameter must be a mapping of element to UPF file name");
    }
    for (const std::string& symbol : species) {
        if (pseudos.find(symbol) == nullptr) {
            fail("no pseudopotential given for " + symbol +
                 "; add it to the 'pseudopotentials' mapping");
        }
    }

    // --- namelists ----------------------------------------------------------
    Namelists namelists;
    set_key(namelists, "control", "calculation", Value("scf"));
    set_key(namelists, "control", "outdir", Value("./out"));
    set_key(namelists, "control", "prefix", Value("pwscf"));

    const std::string pseudo_dir = parameter("pseudo_dir")
                                       .value_or(std::string_view(
                                           util::environment_value("ESPRESSO_PSEUDO")));
    if (!pseudo_dir.empty()) {
        set_key(namelists, "control", "pseudo_dir", Value(pseudo_dir));
    }

    const auto wants = [&properties](Property property) {
        return std::find(properties.begin(), properties.end(), property) != properties.end();
    };
    set_key(namelists, "control", "tprnfor", Value(wants(Property::Forces)));
    set_key(namelists, "control", "tstress", Value(wants(Property::Stress)));

    set_key(namelists, "system", "ibrav", Value(static_cast<std::int64_t>(0)));
    set_key(namelists, "system", "nat", Value(static_cast<std::int64_t>(atoms.size())));
    set_key(namelists, "system", "ntyp", Value(static_cast<std::int64_t>(species.size())));

    // Flat keys are routed to the namelist that owns them.
    for (const auto& [key, value] : parameters().as_map()) {
        if (reserved_keys().count(key) != 0) {
            continue;
        }
        set_key(namelists, namelist_for(key), util::lowercase(key), value);
    }
    // Explicit namelist blocks win.
    for (const char* group : {"control", "system", "electrons", "ions", "cell"}) {
        const Value& block = parameter(group);
        if (block.is_null()) {
            continue;
        }
        if (!block.is_map()) {
            fail(std::string("the '") + group + "' parameter must be a mapping");
        }
        for (const auto& [key, value] : block.as_map()) {
            set_key(namelists, group, util::lowercase(key), value);
        }
    }

    std::ostringstream out;
    for (const char* group : {"control", "system", "electrons", "ions", "cell"}) {
        const auto it = namelists.find(group);
        if (it == namelists.end() || it->second.empty()) {
            // &ELECTRONS is mandatory even when empty.
            if (std::string(group) == "electrons") {
                out << "&ELECTRONS\n/\n";
            }
            continue;
        }
        out << "&" << util::uppercase(group) << "\n";
        for (const auto& [key, value] : it->second) {
            out << "   " << std::left << std::setw(20) << key << " = " << namelist_value(value)
                << "\n";
        }
        out << "/\n";
    }

    // --- cards ---------------------------------------------------------------
    out << "\nATOMIC_SPECIES\n";
    for (const std::string& symbol : species) {
        out << " " << std::left << std::setw(4) << symbol << " "
            << util::format_real(atomic_mass(number_from_symbol(symbol)), 8) << " "
            << pseudos.at(symbol).as_string() << "\n";
    }

    out << "\nCELL_PARAMETERS angstrom\n";
    for (std::size_t row = 0; row < 3; ++row) {
        const Vec3 v = atoms.cell().matrix().row(row);
        out << " " << format_coordinate(v.x) << format_coordinate(v.y) << format_coordinate(v.z)
            << "\n";
    }

    const bool crystal = parameter("crystal_coordinates").value_or(true);
    out << "\nATOMIC_POSITIONS " << (crystal ? "crystal" : "angstrom") << "\n";
    const std::vector<double> coords = crystal ? atoms.scaled_positions() : atoms.positions();
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        out << " " << std::left << std::setw(4) << atoms.symbol(i) << " "
            << format_coordinate(coords[3 * i]) << format_coordinate(coords[3 * i + 1])
            << format_coordinate(coords[3 * i + 2]) << "\n";
    }

    // --- k-points --------------------------------------------------------------
    const Value& kpts = parameter("kpts");
    if (kpts.is_null() || (kpts.is_string() && util::lowercase(kpts.as_string()) == "gamma")) {
        out << "\nK_POINTS gamma\n";
    } else {
        std::array<long long, 3> grid{1, 1, 1};
        if (kpts.is_number()) {
            grid = {kpts.as_int(), kpts.as_int(), kpts.as_int()};
        } else if (kpts.is_list() && kpts.size() == 3) {
            for (std::size_t k = 0; k < 3; ++k) {
                grid[k] = kpts.at(k).as_int();
            }
        } else {
            fail("'kpts' must be 'gamma', a single integer, or a list of three integers");
        }
        std::array<long long, 3> shift{0, 0, 0};
        if (const Value& value = parameter("kpts_shift"); !value.is_null()) {
            if (!value.is_list() || value.size() != 3) {
                fail("'kpts_shift' must be a list of three values");
            }
            for (std::size_t k = 0; k < 3; ++k) {
                shift[k] = value.at(k).as_int() != 0 ? 1 : 0;
            }
        }
        out << "\nK_POINTS automatic\n " << grid[0] << " " << grid[1] << " " << grid[2] << " "
            << shift[0] << " " << shift[1] << " " << shift[2] << "\n";
    }

    return out.str();
}

Results QuantumEspressoCalculator::parse_output_text(std::string_view text) {
    const std::vector<std::string> lines = util::split_lines(text);
    Results results;
    Value extra;

    // --- energy --------------------------------------------------------------
    // The line that starts with '!' is the converged total (free) energy.
    for (std::size_t k = lines.size(); k > 0; --k) {
        const std::string_view line = util::trim(lines[k - 1]);
        if (!line.empty() && line.front() == '!' &&
            line.find("total energy") != std::string_view::npos) {
            const std::vector<double> values = util::scan_numbers(line);
            if (!values.empty()) {
                results.free_energy = values.back() * units::Rydberg;
                results.energy = results.free_energy;
            }
            break;
        }
    }

    const std::ptrdiff_t smearing = util::find_last_line(lines, "smearing contrib. (-TS)");
    if (smearing >= 0) {
        const std::vector<double> values =
            util::scan_numbers(lines[static_cast<std::size_t>(smearing)]);
        if (!values.empty()) {
            extra.insert("smearing_contribution", Value(values.back() * units::Rydberg));
        }
    }
    const std::ptrdiff_t internal = util::find_last_line(lines, "internal energy E=F+TS");
    if (internal >= 0) {
        const std::vector<double> values =
            util::scan_numbers(lines[static_cast<std::size_t>(internal)]);
        if (!values.empty()) {
            extra.insert("internal_energy", Value(values.back() * units::Rydberg));
        }
    }

    // --- forces --------------------------------------------------------------
    const std::ptrdiff_t force_header = util::find_last_line(lines, "Forces acting on atoms");
    if (force_header >= 0) {
        const double scale = units::Rydberg / units::Bohr;  // Ry/bohr -> eV/angstrom
        for (std::size_t row = static_cast<std::size_t>(force_header) + 1; row < lines.size();
             ++row) {
            const std::string& line = lines[row];
            if (line.find("force =") == std::string::npos) {
                // The block is a contiguous run of "atom N type M force = ..."
                // lines; stop once past it.
                if (!results.forces.empty()) {
                    break;
                }
                continue;
            }
            if (line.find("atom") == std::string::npos) {
                continue;  // the "Total force =" summary line
            }
            const std::vector<double> values = util::scan_numbers(line);
            if (values.size() < 5) {
                continue;
            }
            results.forces.push_back(values[values.size() - 3] * scale);
            results.forces.push_back(values[values.size() - 2] * scale);
            results.forces.push_back(values[values.size() - 1] * scale);
        }
    }

    // --- stress ---------------------------------------------------------------
    const std::ptrdiff_t stress_header = util::find_last_line(lines, "total   stress");
    if (stress_header >= 0 && static_cast<std::size_t>(stress_header) + 3 < lines.size()) {
        // Three rows, each printing the same tensor twice: Ry/bohr^3 first,
        // then kbar. Take the first three columns.
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
                tensor(row, col) = values[col];
            }
        }
        if (complete) {
            // Ry/bohr^3 -> eV/angstrom^3, and pw.x uses the opposite sign to ASE.
            const double scale =
                -units::Rydberg / (units::Bohr * units::Bohr * units::Bohr);
            for (std::size_t k = 0; k < 9; ++k) {
                tensor.m[k] *= scale;
            }
            results.stress = matrix_to_voigt(tensor);
        }
    }

    // --- extras ----------------------------------------------------------------
    const std::ptrdiff_t fermi = util::find_last_line(lines, "the Fermi energy is");
    if (fermi >= 0) {
        const std::vector<double> values =
            util::scan_numbers(lines[static_cast<std::size_t>(fermi)]);
        if (!values.empty()) {
            extra.insert("fermi_energy", Value(values.back()));  // already in eV
        }
    }
    const std::ptrdiff_t magnetization =
        util::find_last_line(lines, "total magnetization");
    if (magnetization >= 0) {
        const std::vector<double> values =
            util::scan_numbers(lines[static_cast<std::size_t>(magnetization)]);
        if (!values.empty()) {
            results.magmom = values.back();
        }
    }
    const std::ptrdiff_t converged = util::find_last_line(lines, "convergence has been achieved");
    extra.insert("converged", Value(converged >= 0));
    if (converged >= 0) {
        const std::vector<double> values =
            util::scan_numbers(lines[static_cast<std::size_t>(converged)]);
        if (!values.empty()) {
            extra.insert("scf_iterations", Value(static_cast<std::int64_t>(values.front())));
        }
    }
    extra.insert("job_done", Value(util::find_last_line(lines, "JOB DONE") >= 0));
    results.extra = std::move(extra);

    return results;
}

void QuantumEspressoCalculator::write_input(const std::filesystem::path& directory,
                                            const Atoms& atoms,
                                            const std::vector<Property>& properties) {
    util::write_text_file(directory / input_file_name(), input_string(atoms, properties));
}

void QuantumEspressoCalculator::parse_output(const std::filesystem::path& directory) {
    const std::filesystem::path output = directory / output_file_name();
    if (!std::filesystem::is_regular_file(output)) {
        fail("no '" + output_file_name() + "' in '" +
             std::filesystem::absolute(directory).string() +
             "'; check that the run command redirects pw.x output there");
    }

    const std::string text = util::read_text_file(output);
    Results parsed = parse_output_text(text);
    if (!parsed.energy.has_value()) {
        std::string message = "no total energy in '" +
                              std::filesystem::absolute(output).string() + "'";
        const std::ptrdiff_t error_line =
            util::find_last_line(util::split_lines(text), "%%%%%%%%%%%%");
        if (error_line >= 0) {
            message += "\n--- pw.x reported ---\n" + util::tail(text, 15);
        }
        fail(message);
    }
    results_ = std::move(parsed);
}

}  // namespace fastatoms
