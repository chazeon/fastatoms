// SPDX-License-Identifier: MIT
#include "fastatoms/formats/lammps.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "fastatoms/core/elements.hpp"
#include "lammps_common.hpp"

namespace fastatoms::formats {
namespace {

using namespace lammps_common;

/// Largest type number the writer will expand into a Masses section.
constexpr std::int64_t kMaxWrittenType = 10000;

/// Column positions of one `Atoms` line for a given atom style.
struct AtomLayout {
    std::size_t base;   ///< columns without the optional image flags
    std::size_t type;   ///< column of the atom type
    int mol;            ///< column of the molecule id, or -1
    int charge;         ///< column of the charge, or -1
    std::size_t pos;    ///< column of x (y and z follow)
};

AtomLayout layout_for(const std::string& style) {
    if (style == "atomic") {
        return {5, 1, -1, -1, 2};
    }
    if (style == "charge") {
        return {6, 1, -1, 2, 3};
    }
    if (style == "molecular" || style == "bond" || style == "angle") {
        return {6, 2, 1, -1, 3};
    }
    if (style == "full") {
        return {7, 2, 1, 3, 4};
    }
    if (style == "sphere") {
        return {7, 1, -1, -1, 4};
    }
    throw FormatError("LAMMPS data: unsupported atom style '" + style + "'");
}

/// Style guessed from the field count of the first Atoms row, as ASE does
/// (5 or 8 atomic, 7 or 10 full), with 6 or 9 as charge. Any other count
/// reads as full, and the row check then reports it.
std::string guess_style(std::size_t fields) {
    if (fields == 5 || fields == 8) {
        return "atomic";
    }
    if (fields == 6 || fields == 9) {
        return "charge";
    }
    return "full";
}

/// LAMMPS section names. A line starting with a letter is only a section when
/// it is one of these; in Masses and Atom Type Labels the same letters are
/// a type label ("Sodium 22.99").
bool is_section_keyword(std::string_view body) {
    static const std::vector<std::string_view> names = {
        "Atoms",        "Velocities",   "Masses",        "Charges",   "Ellipsoids",
        "Lines",        "Triangles",    "Bodies",        "Bonds",     "Angles",
        "Dihedrals",    "Impropers",    "Atom Type Labels", "Bond Type Labels",
        "Angle Type Labels", "Dihedral Type Labels", "Improper Type Labels"};
    if (std::find(names.begin(), names.end(), body) != names.end()) {
        return true;
    }
    const auto words = fields(body);
    return !words.empty() && words.back() == "Coeffs";  // Pair Coeffs, PairIJ Coeffs, ...
}

struct AtomRecord {
    std::size_t line{0};
    std::int64_t id{0};
    std::int64_t type{0};
    std::string type_label{};  ///< non-empty when the type column is a label
    std::int64_t mol{0};
    double charge{0.0};
    Vec3 position{};
    std::array<std::int64_t, 3> image{0, 0, 0};
};

struct MassRecord {
    std::size_t line{0};
    double mass{0.0};
    std::string comment_label{};  ///< first word after '#', if any
};

struct TypeInfo {
    std::string label;
    int number;
};

enum class Section { Header, Atoms, Masses, Velocities, TypeLabels, Other };

/// Label and element of one LAMMPS atom type. Priority: explicit `types`, then
/// the Atom Type Labels section, then the Masses comment, then the mass.
///
/// A Masses comment is used only when its element has a matching mass, so
/// `12.011 # Ca` reads as carbon. A label that names no element (`Sodium`)
/// falls through to the mass match, and the label then becomes the matched
/// symbol.
TypeInfo resolve_type(std::int64_t type, std::size_t line, const LammpsTypeMap& types,
                      const std::map<std::int64_t, std::string>& type_labels,
                      const std::map<std::int64_t, MassRecord>& masses) {
    if (const auto it = types.find(static_cast<int>(type)); it != types.end()) {
        return {it->second, element_of_label(it->second, line)};
    }
    const auto mass_it = masses.find(type);
    const MassRecord* mass = mass_it == masses.end() ? nullptr : &mass_it->second;

    if (const auto it = type_labels.find(type); it != type_labels.end()) {
        const int z = try_number_from_label(it->second);
        if (z >= 0) {
            return {it->second, z};
        }
        const int zm = mass != nullptr ? element_from_mass(mass->mass) : -1;
        if (zm >= 0) {
            return {it->second, zm};
        }
        line_error(line, "atom type label '" + it->second +
                             "' names no element; give 'types' explicitly or a Masses entry");
    }

    if (mass != nullptr) {
        int zc = -1;
        if (!mass->comment_label.empty()) {
            zc = try_number_from_label(mass->comment_label);
            if (zc >= 0 && mass_agrees(zc, mass->mass)) {
                return {mass->comment_label, zc};
            }
        }
        const int zm = element_from_mass(mass->mass);
        if (zm >= 0) {
            return {std::string(symbol_from_number(zm)), zm};
        }
        if (zc >= 0) {
            // No element has this mass; the comment is the only evidence left.
            return {mass->comment_label, zc};
        }
        line_error(line, "no element has a mass near " + format_double(mass->mass) +
                             " for atom type " + std::to_string(type) +
                             "; give 'types' explicitly");
    }
    line_error(line, "no element for atom type " + std::to_string(type) +
                         " (no 'types' entry and no Masses section)");
}

}  // namespace

Atoms read_lammps_data(std::string_view text, const LammpsDataReadOptions& options) {
    text = strip_bom(text);
    if (text.empty()) {
        throw FormatError("empty LAMMPS data file");
    }
    const auto lines = split_lines(text);
    const std::string comment(trim(lines[0]));

    std::int64_t natoms = -1;
    std::int64_t ntypes = -1;
    std::optional<std::array<double, 3>> tilts;  // xy xz yz (restricted only)
    std::array<Vec3, 3> general_vectors{};
    std::array<bool, 3> have_vector{false, false, false};
    Vec3 origin{};
    std::array<double, 6> bounds{};
    std::array<bool, 3> have_bound{false, false, false};

    std::string style;
    bool style_known = false;
    AtomLayout layout{};
    Section section = Section::Header;
    std::vector<AtomRecord> atoms;
    std::map<std::int64_t, MassRecord> masses;
    std::vector<std::pair<std::string, MassRecord>> label_masses;
    std::map<std::int64_t, Vec3> velocities;
    std::map<std::int64_t, std::string> type_labels;
    std::map<std::string, std::int64_t> label_numbers;

    for (std::size_t li = 1; li < lines.size(); ++li) {
        const std::size_t lineno = li + 1;
        const std::string_view raw = lines[li];
        const std::string_view body = trim(without_comment(raw));
        if (body.empty()) {
            continue;
        }

        if (is_section_keyword(body)) {
            if (body == "Atoms") {
                section = Section::Atoms;
                std::string hint = options.atom_style;
                if (hint.empty()) {
                    const auto words = fields(comment_of(raw));
                    if (!words.empty()) {
                        hint = std::string(words.front());
                    }
                }
                style_known = !hint.empty();
                if (style_known) {
                    layout = layout_for(hint);
                    style = hint;
                }
            } else if (body == "Masses") {
                section = Section::Masses;
            } else if (body == "Velocities") {
                section = Section::Velocities;
            } else if (body == "Atom Type Labels") {
                section = Section::TypeLabels;
            } else {
                section = Section::Other;  // Bonds, Coeffs, ...: skipped by boundary
            }
            continue;
        }

        const auto f = fields(body);
        switch (section) {
        case Section::Header: {
            const std::size_t n = f.size();
            if (n == 2 && f[1] == "atoms") {
                natoms = parse_int(f[0], lineno);
                if (natoms < 0) {
                    line_error(lineno, "negative atom count");
                }
            } else if (n == 3 && f[1] == "atom" && f[2] == "types") {
                ntypes = parse_int(f[0], lineno);
                if (ntypes < 0) {
                    line_error(lineno, "negative atom type count");
                }
            } else if (n == 4 && f[2] == "xlo" && f[3] == "xhi") {
                bounds[0] = parse_double(f[0], lineno);
                bounds[1] = parse_double(f[1], lineno);
                have_bound[0] = true;
            } else if (n == 4 && f[2] == "ylo" && f[3] == "yhi") {
                bounds[2] = parse_double(f[0], lineno);
                bounds[3] = parse_double(f[1], lineno);
                have_bound[1] = true;
            } else if (n == 4 && f[2] == "zlo" && f[3] == "zhi") {
                bounds[4] = parse_double(f[0], lineno);
                bounds[5] = parse_double(f[1], lineno);
                have_bound[2] = true;
            } else if (n == 6 && f[3] == "xy" && f[4] == "xz" && f[5] == "yz") {
                tilts = std::array<double, 3>{parse_double(f[0], lineno),
                                              parse_double(f[1], lineno),
                                              parse_double(f[2], lineno)};
            } else if (n == 4 && (f[3] == "avec" || f[3] == "bvec" || f[3] == "cvec")) {
                const std::size_t k = f[3] == "avec" ? 0 : (f[3] == "bvec" ? 1 : 2);
                general_vectors[k] = Vec3(parse_double(f[0], lineno), parse_double(f[1], lineno),
                                          parse_double(f[2], lineno));
                have_vector[k] = true;
            } else if (n == 5 && f[3] == "abc" && f[4] == "origin") {
                origin = Vec3(parse_double(f[0], lineno), parse_double(f[1], lineno),
                              parse_double(f[2], lineno));
            }
            break;
        }
        case Section::TypeLabels: {
            if (f.size() < 2) {
                line_error(lineno, "an Atom Type Labels line needs a type and a label");
            }
            const std::int64_t t = parse_int(f[0], lineno);
            const std::string label(f[1]);
            if (type_labels.count(t) != 0 || label_numbers.count(label) != 0) {
                line_error(lineno, "duplicate atom type or label '" + label + "'");
            }
            type_labels[t] = label;
            label_numbers[label] = t;
            break;
        }
        case Section::Atoms: {
            if (!style_known) {
                style = guess_style(f.size());
                layout = layout_for(style);
                style_known = true;
            }
            const std::size_t n = f.size();
            if (n != layout.base && n != layout.base + 3) {
                line_error(lineno, "style '" + style + "' expects " +
                                       std::to_string(layout.base) + " or " +
                                       std::to_string(layout.base + 3) + " fields, got " +
                                       std::to_string(n));
            }
            AtomRecord record;
            record.line = lineno;
            record.id = parse_int(f[0], lineno);
            std::int64_t type = 0;
            if (try_parse_int(f[layout.type], type)) {
                record.type = type;
            } else {
                record.type_label = std::string(f[layout.type]);
            }
            if (layout.mol >= 0) {
                record.mol = parse_int(f[static_cast<std::size_t>(layout.mol)], lineno);
            }
            if (layout.charge >= 0) {
                record.charge = parse_double(f[static_cast<std::size_t>(layout.charge)], lineno);
            }
            record.position = Vec3(parse_double(f[layout.pos], lineno),
                                   parse_double(f[layout.pos + 1], lineno),
                                   parse_double(f[layout.pos + 2], lineno));
            if (n == layout.base + 3) {
                for (std::size_t k = 0; k < 3; ++k) {
                    record.image[k] = parse_int(f[layout.base + k], lineno);
                }
            }
            atoms.push_back(record);
            break;
        }
        case Section::Masses: {
            if (f.size() < 2) {
                line_error(lineno, "a Masses line needs a type and a mass");
            }
            MassRecord record;
            record.line = lineno;
            record.mass = parse_double(f[1], lineno);
            const auto words = fields(comment_of(raw));
            if (!words.empty()) {
                record.comment_label = std::string(words.front());
            }
            std::int64_t t = 0;
            if (try_parse_int(f[0], t)) {
                masses[t] = record;
            } else {
                label_masses.emplace_back(std::string(f[0]), record);
            }
            break;
        }
        case Section::Velocities: {
            if (f.size() < 4) {
                line_error(lineno, "a Velocities line needs an id and three components");
            }
            velocities[parse_int(f[0], lineno)] =
                Vec3(parse_double(f[1], lineno), parse_double(f[2], lineno),
                     parse_double(f[3], lineno));
            break;
        }
        case Section::Other:
            break;
        }
    }

    for (const auto& [label, record] : label_masses) {
        const auto it = label_numbers.find(label);
        if (it == label_numbers.end()) {
            line_error(record.line, "Masses names the unknown atom type label '" + label + "'");
        }
        masses[it->second] = record;
    }

    if (natoms < 0) {
        throw FormatError("LAMMPS data: no 'N atoms' header line");
    }
    if (static_cast<std::int64_t>(atoms.size()) != natoms) {
        throw FormatError("LAMMPS data: header says " + std::to_string(natoms) +
                          " atoms but the Atoms section has " + std::to_string(atoms.size()));
    }

    for (AtomRecord& r : atoms) {
        if (!r.type_label.empty()) {
            const auto it = label_numbers.find(r.type_label);
            if (it == label_numbers.end()) {
                line_error(r.line, "unknown atom type label '" + r.type_label + "'");
            }
            r.type = it->second;
        }
        if (r.type < 1) {
            line_error(r.line, "atom type " + std::to_string(r.type) + " must be 1 or more");
        }
        if (ntypes >= 0 && r.type > ntypes) {
            line_error(r.line, "atom type " + std::to_string(r.type) +
                                   " exceeds the 'atom types' header value " +
                                   std::to_string(ntypes));
        }
    }

    // Cell and origin.
    Mat3 matrix{};
    Vec3 shift{};
    if (have_vector[0] || have_vector[1] || have_vector[2]) {
        if (!(have_vector[0] && have_vector[1] && have_vector[2])) {
            throw FormatError("LAMMPS data: general triclinic box needs avec, bvec and cvec");
        }
        matrix = Mat3::from_rows(general_vectors[0], general_vectors[1], general_vectors[2]);
        shift = origin;
    } else {
        if (!(have_bound[0] && have_bound[1] && have_bound[2])) {
            throw FormatError("LAMMPS data: missing 'xlo xhi', 'ylo yhi' or 'zlo zhi' line");
        }
        const std::array<double, 3> tilt = tilts.value_or(std::array<double, 3>{0.0, 0.0, 0.0});
        const double lx = bounds[1] - bounds[0];
        const double ly = bounds[3] - bounds[2];
        const double lz = bounds[5] - bounds[4];
        matrix = restricted_cell(lx, ly, lz, tilt[0], tilt[1], tilt[2]).matrix();
        shift = Vec3(bounds[0], bounds[2], bounds[4]);
    }
    const Cell cell(matrix, {true, true, true});

    // Sorting by id keeps file order for ids that are already ascending.
    std::stable_sort(atoms.begin(), atoms.end(),
                     [](const AtomRecord& a, const AtomRecord& b) { return a.id < b.id; });
    for (std::size_t k = 1; k < atoms.size(); ++k) {
        if (atoms[k].id == atoms[k - 1].id) {
            line_error(atoms[k].line, "duplicate atom id " + std::to_string(atoms[k].id));
        }
    }

    // Type -> label/element, cached per type.
    std::map<std::int64_t, TypeInfo> type_info;
    const std::size_t n = atoms.size();
    std::vector<int> numbers(n);
    std::vector<double> positions(3 * n);
    std::vector<std::string> labels(n);
    std::vector<std::int64_t> ids(n), types(n), mols(n), images(3 * n);
    std::vector<double> charges(n);

    for (std::size_t i = 0; i < n; ++i) {
        const AtomRecord& r = atoms[i];
        auto it = type_info.find(r.type);
        if (it == type_info.end()) {
            it = type_info
                     .emplace(r.type, resolve_type(r.type, r.line, options.types, type_labels,
                                                   masses))
                     .first;
        }
        numbers[i] = it->second.number;
        labels[i] = it->second.label;

        Vec3 p = r.position - shift;
        if (options.unwrap) {
            for (std::size_t k = 0; k < 3; ++k) {
                p += static_cast<double>(r.image[k]) * matrix.row(k);
            }
        }
        p.store(positions.data() + 3 * i);

        ids[i] = r.id;
        types[i] = r.type;
        mols[i] = r.mol;
        charges[i] = r.charge;
        for (std::size_t k = 0; k < 3; ++k) {
            images[3 * i + k] = r.image[k];
        }
    }

    Atoms out(numbers, positions, cell);
    out.set_string_array("labels", labels);
    out.set_int_array("id", ids);
    out.set_int_array("type", types);
    out.set_int_array("mol", mols);
    out.set_int_array("image", images, 3);
    if (style == "charge" || style == "full") {
        out.set_charges(charges);
    }

    // Masses only when every type that occurs has one.
    if (!masses.empty()) {
        bool complete = true;
        for (const auto& entry : type_info) {
            complete = complete && masses.count(entry.first) != 0;
        }
        if (complete) {
            std::vector<double> per_atom(n);
            for (std::size_t i = 0; i < n; ++i) {
                per_atom[i] = masses.at(types[i]).mass;
            }
            out.set_masses(per_atom);
        }
    }

    if (!velocities.empty()) {
        std::vector<double> v(3 * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            if (const auto it = velocities.find(ids[i]); it != velocities.end()) {
                it->second.store(v.data() + 3 * i);
            }
        }
        std::size_t matched = 0;
        for (const std::int64_t id : ids) {
            matched += velocities.count(id);
        }
        if (matched != velocities.size()) {
            throw FormatError("LAMMPS data: Velocities section names an unknown atom id");
        }
        out.set_velocities(v);
    }

    Value::Map info;
    info.emplace_back("comment", Value(comment));
    out.set_info(Value(std::move(info)));
    return out;
}

std::string write_lammps_data(const Atoms& atoms, const LammpsDataWriteOptions& options) {
    const std::string& style = options.atom_style;
    if (style != "atomic" && style != "charge" && style != "full") {
        throw FormatError("LAMMPS data: cannot write atom style '" + style +
                          "' (atomic, charge and full are supported)");
    }
    if (!atoms.cell().is_invertible()) {
        throw FormatError("LAMMPS data: a non-degenerate cell is required");
    }

    // Rotate the cell into the restricted-triclinic frame: e1 along a, e2 in
    // the a-b plane, e3 along the normal of that plane on c's side. For a
    // left-handed cell this frame is a reflection, as in ASE.
    const Mat3 M = atoms.cell().matrix();
    const Vec3 a = M.row(0);
    const Vec3 b = M.row(1);
    const Vec3 c = M.row(2);
    const Vec3 e1 = a.normalized();
    const double lx = a.norm();
    const double xy = dot(b, e1);
    const Vec3 b_perp = b - xy * e1;
    const double ly = b_perp.norm();
    const Vec3 e2 = b_perp / ly;
    const double xz = dot(c, e1);
    const double yz = dot(c, e2);
    const Vec3 c_perp = c - xz * e1 - yz * e2;
    const double lz = c_perp.norm();
    const Vec3 e3 = c_perp / lz;
    const auto rotate = [&](const Vec3& v) {
        return Vec3(dot(v, e1), dot(v, e2), dot(v, e3));
    };

    const double tilt_tolerance = 1e-10 * std::max({lx, ly, lz});
    const bool skewed = options.force_skew || std::abs(xy) > tilt_tolerance ||
                        std::abs(xz) > tilt_tolerance || std::abs(yz) > tilt_tolerance;

    // Notes for the header. LAMMPS ignores '#' comments, so they are harmless.
    std::vector<std::string> notes;
    if (atoms.cell().signed_volume() < 0.0) {
        notes.emplace_back("note: left-handed cell written as its mirror image");
    }
    if (skewed && (std::abs(xy) > 0.5 * lx + tilt_tolerance ||
                   std::abs(xz) > 0.5 * lx + tilt_tolerance ||
                   std::abs(yz) > 0.5 * ly + tilt_tolerance)) {
        notes.emplace_back(
            "note: tilt factors exceed half the box lengths; LAMMPS needs 'box tilt large'");
    }

    // Species and the type of each atom (1-based). Without specorder, an int
    // "type" array gives the numbers (1..max); otherwise the labels are
    // numbered in order of first appearance.
    const std::size_t n = atoms.size();
    const std::vector<std::string> labels = atoms.labels();
    std::vector<std::string> species;
    std::vector<std::size_t> types(n, 0);
    bool use_file_types = false;
    if (options.specorder.empty() && atoms.has_int_array("type")) {
        const auto& file_types = atoms.int_array("type").data;
        use_file_types = file_types.size() == n;
        std::int64_t top = 0;
        for (const std::int64_t t : file_types) {
            if (t < 1 || t > kMaxWrittenType) {
                use_file_types = false;
                break;
            }
            top = std::max(top, t);
        }
        if (use_file_types) {
            species.assign(static_cast<std::size_t>(top), std::string{});
            std::vector<bool> seen(species.size(), false);
            for (std::size_t i = 0; i < n; ++i) {
                const auto k = static_cast<std::size_t>(file_types[i] - 1);
                types[i] = k + 1;
                if (!seen[k]) {
                    seen[k] = true;
                    species[k] = labels[i];
                }
            }
        }
    }
    if (!use_file_types) {
        species = options.specorder;
        if (species.empty()) {
            for (const auto& label : labels) {
                if (std::find(species.begin(), species.end(), label) == species.end()) {
                    species.push_back(label);
                }
            }
        }
        std::map<std::string, std::size_t> type_of;
        for (std::size_t k = species.size(); k-- > 0;) {
            type_of[species[k]] = k + 1;  // first occurrence wins
        }
        for (std::size_t i = 0; i < n; ++i) {
            auto it = type_of.find(labels[i]);
            if (it == type_of.end()) {
                // ASE users pass chemical symbols: Fe1 matches the entry Fe.
                it = type_of.find(std::string(symbol_from_number(atoms.number(i))));
            }
            if (it == type_of.end()) {
                throw FormatError("LAMMPS data: label '" + labels[i] + "' is not in specorder");
            }
            types[i] = it->second;
        }
    }

    std::string out;
    out.reserve(64 * n + 512);
    const auto line = [&out](const std::string& text) {
        out += text;
        out += '\n';
    };

    std::string title = "(written by FastAtoms)";
    if (const Value* c_info = atoms.info().is_map() ? atoms.info().find("comment") : nullptr;
        c_info != nullptr && c_info->is_string() &&
        c_info->as_string().find('\n') == std::string::npos) {
        title = c_info->as_string();
    }
    line(title);
    out += '\n';
    line(std::to_string(n) + " atoms");
    line(std::to_string(species.size()) + " atom types");
    out += '\n';
    line("0 " + format_double(lx) + " xlo xhi");
    line("0 " + format_double(ly) + " ylo yhi");
    line("0 " + format_double(lz) + " zlo zhi");
    if (skewed) {
        line(format_double(xy) + " " + format_double(xz) + " " + format_double(yz) +
             " xy xz yz");
    }
    for (const auto& note : notes) {
        line("# " + note);
    }

    if (options.masses) {
        // One mass per type: that of the first atom of the type, else the
        // weight of the element the label names. The comment is the label.
        std::vector<std::size_t> first(species.size(), n);
        for (std::size_t i = 0; i < n; ++i) {
            first[types[i] - 1] = std::min(first[types[i] - 1], i);
        }
        out += "\nMasses\n\n";
        for (std::size_t k = 0; k < species.size(); ++k) {
            if (first[k] < n) {
                line(std::to_string(k + 1) + " " + format_double(atoms.mass(first[k])) + " # " +
                     species[k]);
                continue;
            }
            if (species[k].empty()) {
                line(std::to_string(k + 1) + " 1.0 # unused type");
                continue;
            }
            const int z = try_number_from_label(species[k]);
            const double mass = z >= 0 ? atomic_mass(z) : 1.0;
            line(std::to_string(k + 1) + " " + format_double(mass) + " # " + species[k]);
        }
    }

    const std::vector<double> charges = atoms.charges();
    const bool has_mol = atoms.has_int_array("mol");
    out += "\nAtoms # " + style + "\n\n";
    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 r = rotate(atoms.position(i));
        std::string text = std::to_string(i + 1);
        if (style == "full") {
            text += " " + std::to_string(has_mol ? atoms.int_array("mol").data[i] : 0);
        }
        text += " " + std::to_string(types[i]);
        if (style != "atomic") {
            text += " " + format_double(charges[i]);
        }
        text += " " + format_double(r.x) + " " + format_double(r.y) + " " + format_double(r.z);
        line(text);
    }

    if (atoms.has_momenta()) {
        const std::vector<double> velocities = atoms.velocities();
        out += "\nVelocities\n\n";
        for (std::size_t i = 0; i < n; ++i) {
            const Vec3 v = rotate(Vec3(velocities[3 * i], velocities[3 * i + 1],
                                       velocities[3 * i + 2]));
            line(std::to_string(i + 1) + " " + format_double(v.x) + " " + format_double(v.y) +
                 " " + format_double(v.z));
        }
    }
    return out;
}

}  // namespace fastatoms::formats
