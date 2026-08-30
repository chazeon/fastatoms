// SPDX-License-Identifier: MIT
#include "fastatoms/calculators/vasp.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>

#include "fastatoms/core/elements.hpp"
#include "fastatoms/core/units.hpp"
#include "fastatoms/util/text.hpp"

namespace fastatoms {
namespace {

/// Parameters the calculator interprets itself; everything else is an INCAR tag.
const std::set<std::string>& reserved_keys() {
    static const std::set<std::string> keys{
        // FileIOCalculator
        "directory", "command", "log_file", "timeout", "environment",
        // VaspCalculator
        "xc", "kpts", "kpts_shift", "gamma", "incar", "pp_path", "pp_flavour",
        "setups", "write_potcar", "direct"};
    return keys;
}

/// GGA tag for a functional name, or an empty string for LDA / unknown.
std::string gga_tag(const std::string& xc) {
    static const std::map<std::string, std::string> tags{
        {"pbe", "PE"},    {"pbesol", "PS"}, {"ps", "PS"},     {"rpbe", "RP"},
        {"revpbe", "RE"}, {"am05", "AM"},   {"b3lyp", "B3"},  {"pw91", "91"}};
    const auto it = tags.find(util::lowercase(xc));
    return it == tags.end() ? std::string{} : it->second;
}

/// Render a parameter value in INCAR syntax.
std::string incar_value(const Value& value) {
    switch (value.type()) {
        case Value::Type::Bool:
            return value.as_bool() ? ".TRUE." : ".FALSE.";
        case Value::Type::Int:
            return std::to_string(value.as_int());
        case Value::Type::Float:
            return util::format_real(value.as_double());
        case Value::Type::String:
            return value.as_string();
        case Value::Type::List: {
            std::string out;
            for (const Value& item : value.as_list()) {
                out += (out.empty() ? "" : " ") + incar_value(item);
            }
            return out;
        }
        default:
            return value.to_string();
    }
}

std::string format_coordinate(double v) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(16) << std::setw(22) << v;
    return out.str();
}

}  // namespace

VaspSpeciesLayout vasp_species_layout(const Atoms& atoms) {
    VaspSpeciesLayout layout;
    // Group by species while keeping the order in which species first appear,
    // which is what makes a POSCAR/POTCAR pair line up.
    std::vector<std::vector<std::size_t>> buckets;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        const std::string symbol = atoms.symbol(i);
        const auto it = std::find(layout.symbols.begin(), layout.symbols.end(), symbol);
        if (it == layout.symbols.end()) {
            layout.symbols.push_back(symbol);
            buckets.push_back(std::vector<std::size_t>{i});
        } else {
            buckets[static_cast<std::size_t>(it - layout.symbols.begin())].push_back(i);
        }
    }
    for (const std::vector<std::size_t>& bucket : buckets) {
        layout.counts.push_back(bucket.size());
        layout.order.insert(layout.order.end(), bucket.begin(), bucket.end());
    }
    return layout;
}

VaspCalculator::VaspCalculator(Value parameters) : FileIOCalculator(std::move(parameters)) {}

std::vector<Property> VaspCalculator::implemented_properties() const {
    return {Property::Energy, Property::FreeEnergy, Property::Forces, Property::Stress,
            Property::Magmom};
}

std::vector<std::string> VaspCalculator::command_environment_names() const {
    return {"FASTATOMS_VASP_COMMAND", "ASE_VASP_COMMAND", "VASP_COMMAND", "VASP_SCRIPT"};
}

std::string VaspCalculator::poscar_string(const Atoms& atoms,
                                          const VaspSpeciesLayout& layout) const {
    const bool direct = parameter("direct").value_or(true);
    if (direct && !atoms.cell().is_invertible()) {
        fail("fractional POSCAR coordinates need a full 3x3 cell; set direct: false");
    }

    std::ostringstream out;
    out << atoms.formula() << " (written by FastAtoms)\n";
    out << "   1.00000000000000\n";

    const Mat3& cell = atoms.cell().matrix();
    for (std::size_t row = 0; row < 3; ++row) {
        const Vec3 v = cell.row(row);
        out << "  " << format_coordinate(v.x) << format_coordinate(v.y) << format_coordinate(v.z)
            << "\n";
    }

    out << "  ";
    for (const std::string& symbol : layout.symbols) {
        out << " " << std::setw(4) << symbol;
    }
    out << "\n  ";
    for (const std::size_t count : layout.counts) {
        out << " " << std::setw(4) << count;
    }
    out << "\n";

    out << (direct ? "Direct\n" : "Cartesian\n");
    const std::vector<double> scaled = direct ? atoms.scaled_positions() : atoms.positions();
    for (const std::size_t index : layout.order) {
        out << "  " << format_coordinate(scaled[3 * index])
            << format_coordinate(scaled[3 * index + 1])
            << format_coordinate(scaled[3 * index + 2]) << "\n";
    }
    return out.str();
}

std::string VaspCalculator::incar_string(const std::vector<Property>& properties) const {
    // Ordered so the file is stable across runs and easy to diff.
    std::vector<std::pair<std::string, std::string>> tags;
    const auto set_tag = [&tags](const std::string& key, std::string value) {
        const auto it = std::find_if(tags.begin(), tags.end(),
                                     [&key](const auto& pair) { return pair.first == key; });
        if (it == tags.end()) {
            tags.emplace_back(key, std::move(value));
        } else {
            it->second = std::move(value);
        }
    };

    const std::string xc = parameter("xc").value_or(std::string_view{});
    if (!xc.empty()) {
        if (const std::string tag = gga_tag(xc); !tag.empty()) {
            set_tag("GGA", tag);
        } else if (util::lowercase(xc) != "lda") {
            fail("unknown functional '" + xc +
                 "'; use one of pbe, pbesol, rpbe, revpbe, am05, pw91, b3lyp, lda, or set the "
                 "GGA tag directly");
        }
    }

    // Stress is only reported for ISIF >= 2, which is also VASP's default.
    const bool wants_stress =
        std::find(properties.begin(), properties.end(), Property::Stress) != properties.end();
    if (wants_stress) {
        set_tag("ISIF", "2");
    }

    for (const auto& [key, value] : parameters().as_map()) {
        if (reserved_keys().count(key) != 0) {
            continue;
        }
        set_tag(util::uppercase(key), incar_value(value));
    }

    // The explicit `incar` block wins over everything else.
    if (const Value& block = parameter("incar"); !block.is_null()) {
        if (!block.is_map()) {
            fail("the 'incar' parameter must be a mapping of tags to values");
        }
        for (const auto& [key, value] : block.as_map()) {
            set_tag(util::uppercase(key), incar_value(value));
        }
    }

    std::ostringstream out;
    out << "! INCAR written by FastAtoms\n";
    for (const auto& [key, value] : tags) {
        out << key << " = " << value << "\n";
    }
    return out.str();
}

std::string VaspCalculator::kpoints_string() const {
    // KSPACING in INCAR replaces the KPOINTS file entirely.
    for (const auto& [key, value] : parameters().as_map()) {
        if (util::uppercase(key) == "KSPACING") {
            return {};
        }
    }
    if (const Value& block = parameter("incar"); block.is_map()) {
        for (const auto& [key, value] : block.as_map()) {
            if (util::uppercase(key) == "KSPACING") {
                return {};
            }
        }
    }

    std::array<long long, 3> grid{1, 1, 1};
    const Value& kpts = parameter("kpts");
    if (kpts.is_number()) {
        grid = {kpts.as_int(), kpts.as_int(), kpts.as_int()};
    } else if (kpts.is_list()) {
        if (kpts.size() != 3) {
            fail("'kpts' must be a single integer or a list of three integers");
        }
        for (std::size_t k = 0; k < 3; ++k) {
            grid[k] = kpts.at(k).as_int();
        }
    } else if (!kpts.is_null()) {
        fail("'kpts' must be a single integer or a list of three integers");
    }
    for (const long long n : grid) {
        if (n < 1) {
            fail("'kpts' entries must be at least 1");
        }
    }

    std::array<double, 3> shift{0.0, 0.0, 0.0};
    if (const Value& value = parameter("kpts_shift"); !value.is_null()) {
        const std::vector<double> values = value.as_doubles();
        if (values.size() != 3) {
            fail("'kpts_shift' must have three components");
        }
        std::copy(values.begin(), values.end(), shift.begin());
    }

    const bool gamma_centred = parameter("gamma").value_or(true);
    std::ostringstream out;
    out << "Automatic mesh written by FastAtoms\n";
    out << "0\n";
    out << (gamma_centred ? "Gamma\n" : "Monkhorst-Pack\n");
    out << " " << grid[0] << " " << grid[1] << " " << grid[2] << "\n";
    out << " " << util::format_real(shift[0]) << " " << util::format_real(shift[1]) << " "
        << util::format_real(shift[2]) << "\n";
    return out.str();
}

void VaspCalculator::write_potcar(const std::filesystem::path& directory,
                                  const std::vector<std::string>& symbols) const {
    if (!parameter("write_potcar").value_or(true)) {
        return;
    }

    const std::string root =
        parameter("pp_path").value_or(std::string_view(util::environment_value("VASP_PP_PATH")));
    if (root.empty()) {
        if (std::filesystem::exists(directory / "POTCAR")) {
            return;  // the user is managing POTCAR themselves
        }
        fail("no POTCAR library configured: set the 'pp_path' parameter or the VASP_PP_PATH "
             "environment variable, place a POTCAR in '" +
             directory.string() + "', or set write_potcar: false");
    }

    // The flavour subdirectory follows the functional unless it is pinned.
    std::string flavour = parameter("pp_flavour").value_or(std::string_view{});
    if (flavour.empty()) {
        const std::string xc = util::lowercase(parameter("xc").value_or(std::string_view("pbe")));
        flavour = xc == "lda" ? "potpaw_LDA" : "potpaw_PBE";
    }

    const Value& setups = parameter("setups");
    if (!setups.is_null() && !setups.is_map()) {
        fail("the 'setups' parameter must be a mapping of element to POTCAR name");
    }

    std::string combined;
    for (const std::string& symbol : symbols) {
        std::string folder = symbol;
        if (const Value* choice = setups.is_map() ? setups.find(symbol) : nullptr) {
            const std::string requested = choice->as_string();
            // "_sv" is shorthand for "<symbol>_sv", as in ASE.
            folder = requested.empty() ? symbol
                                       : (requested.front() == '_' ? symbol + requested : requested);
        }

        const std::vector<std::filesystem::path> candidates{
            std::filesystem::path(root) / flavour / folder / "POTCAR",
            std::filesystem::path(root) / folder / "POTCAR"};

        std::filesystem::path found;
        for (const std::filesystem::path& candidate : candidates) {
            if (std::filesystem::is_regular_file(candidate)) {
                found = candidate;
                break;
            }
        }

        if (found.empty()) {
            std::string tried;
            for (const std::filesystem::path& candidate : candidates) {
                tried += "\n    " + candidate.string();
                for (const char* suffix : {".gz", ".Z", ".bz2"}) {
                    if (std::filesystem::exists(candidate.string() + suffix)) {
                        tried += "  (found " + candidate.string() + suffix +
                                 " - FastAtoms reads plain POTCAR files, please decompress it)";
                    }
                }
            }
            fail("no POTCAR found for " + symbol + "; tried:" + tried);
        }
        combined += util::read_text_file(found);
        if (!combined.empty() && combined.back() != '\n') {
            combined += '\n';
        }
    }
    util::write_text_file(directory / "POTCAR", combined);
}

void VaspCalculator::write_input(const std::filesystem::path& directory, const Atoms& atoms,
                                 const std::vector<Property>& properties) {
    const VaspSpeciesLayout layout = vasp_species_layout(atoms);
    order_ = layout.order;

    util::write_text_file(directory / "POSCAR", poscar_string(atoms, layout));
    util::write_text_file(directory / "INCAR", incar_string(properties));

    const std::string kpoints = kpoints_string();
    if (kpoints.empty()) {
        // A stale KPOINTS would silently override KSPACING.
        std::error_code error;
        std::filesystem::remove(directory / "KPOINTS", error);
    } else {
        util::write_text_file(directory / "KPOINTS", kpoints);
    }

    write_potcar(directory, layout.symbols);
}

Results VaspCalculator::parse_outcar(std::string_view text) {
    const std::vector<std::string> lines = util::split_lines(text);
    Results results;

    // --- energies ----------------------------------------------------------
    for (std::size_t k = lines.size(); k > 0; --k) {
        const std::string& line = lines[k - 1];
        if (line.find("TOTEN") != std::string::npos && line.find("free") != std::string::npos) {
            const std::vector<double> values = util::scan_numbers(line);
            if (!values.empty()) {
                results.free_energy = values.back();
            }
            break;
        }
    }
    for (std::size_t k = lines.size(); k > 0; --k) {
        const std::string& line = lines[k - 1];
        const std::size_t at = line.find("energy(sigma->0)");
        if (at != std::string::npos) {
            const std::vector<double> values = util::scan_numbers(line.substr(at));
            if (!values.empty()) {
                results.energy = values.back();
            }
            break;
        }
    }
    // Older output only carries the free energy.
    if (!results.energy.has_value() && results.free_energy.has_value()) {
        results.energy = results.free_energy;
    }
    if (!results.free_energy.has_value() && results.energy.has_value()) {
        results.free_energy = results.energy;
    }

    // --- forces -------------------------------------------------------------
    const std::ptrdiff_t force_header = util::find_last_line(lines, "TOTAL-FORCE");
    if (force_header >= 0) {
        std::size_t row = static_cast<std::size_t>(force_header) + 1;
        // Skip the rule under the header.
        while (row < lines.size() && lines[row].find("---") != std::string::npos) {
            ++row;
        }
        for (; row < lines.size(); ++row) {
            const std::vector<double> values = util::scan_numbers(lines[row]);
            if (values.size() < 6) {
                break;  // the block ends at the next rule or blank line
            }
            results.forces.push_back(values[3]);
            results.forces.push_back(values[4]);
            results.forces.push_back(values[5]);
        }
    }

    // --- stress -------------------------------------------------------------
    const std::ptrdiff_t stress_line = util::find_last_line(lines, "in kB");
    if (stress_line >= 0) {
        const std::string& line = lines[static_cast<std::size_t>(stress_line)];
        const std::size_t at = line.find("in kB");
        const std::vector<double> values = util::scan_numbers(line.substr(at + 5));
        if (values.size() >= 6) {
            // OUTCAR prints XX YY ZZ XY YZ ZX in kB, with the opposite sign to
            // the ASE convention. Convert kB -> GPa -> eV/A^3 and reorder to
            // Voigt (xx, yy, zz, yz, xz, xy).
            const double scale = -0.1 * units::GPa;
            results.stress = std::array<double, 6>{values[0] * scale, values[1] * scale,
                                                   values[2] * scale, values[4] * scale,
                                                   values[5] * scale, values[3] * scale};
        }
    }

    // --- extras -------------------------------------------------------------
    Value extra;
    const std::ptrdiff_t fermi = util::find_last_line(lines, "E-fermi");
    if (fermi >= 0) {
        const std::vector<double> values =
            util::scan_numbers(lines[static_cast<std::size_t>(fermi)]);
        if (!values.empty()) {
            extra.insert("fermi_energy", Value(values.front()));
        }
    }
    const std::ptrdiff_t magnetization = util::find_last_line(lines, "magnetization");
    if (magnetization >= 0) {
        const std::string& line = lines[static_cast<std::size_t>(magnetization)];
        if (line.find("number of electron") != std::string::npos) {
            const std::vector<double> values = util::scan_numbers(line);
            if (values.size() >= 2) {
                results.magmom = values.back();
                extra.insert("nelect", Value(values[values.size() - 2]));
            }
        }
    }
    extra.insert("converged",
                 Value(util::find_last_line(lines, "reached required accuracy") >= 0));
    const std::ptrdiff_t iterations = util::find_last_line(lines, "Iteration");
    if (iterations >= 0) {
        const std::vector<double> values =
            util::scan_numbers(lines[static_cast<std::size_t>(iterations)]);
        if (values.size() >= 2) {
            extra.insert("ionic_steps", Value(static_cast<std::int64_t>(values[0])));
            extra.insert("electronic_steps", Value(static_cast<std::int64_t>(values[1])));
        }
    }
    results.extra = std::move(extra);

    return results;
}

void VaspCalculator::parse_output(const std::filesystem::path& directory) {
    const std::filesystem::path outcar = directory / "OUTCAR";
    if (!std::filesystem::is_regular_file(outcar)) {
        fail("no OUTCAR in '" + std::filesystem::absolute(directory).string() +
             "'; the run may not have started");
    }

    Results parsed = parse_outcar(util::read_text_file(outcar));
    if (!parsed.energy.has_value()) {
        fail("OUTCAR contains no total energy; the run probably failed - see '" +
             std::filesystem::absolute(outcar).string() + "'");
    }

    // OUTCAR lists atoms in POSCAR order, which is grouped by species.
    if (!order_.empty() && parsed.forces.size() == 3 * order_.size()) {
        std::vector<double> reordered(parsed.forces.size());
        for (std::size_t row = 0; row < order_.size(); ++row) {
            const std::size_t target = order_[row];
            reordered[3 * target] = parsed.forces[3 * row];
            reordered[3 * target + 1] = parsed.forces[3 * row + 1];
            reordered[3 * target + 2] = parsed.forces[3 * row + 2];
        }
        parsed.forces = std::move(reordered);
    }

    results_ = std::move(parsed);
}

}  // namespace fastatoms
