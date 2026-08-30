// SPDX-License-Identifier: MIT

#include "fastatoms/build/twod.hpp"

#include <algorithm>
#include <cmath>
#include <map>

namespace fastatoms::build {
namespace {

/// The hexagonal in-plane lattice used throughout this file:
/// `a1` along x, `a2` at 120 degrees to it. Same convention as `hcp`.
Cell hexagonal(double a, double height) {
    return Cell(Vec3{a, 0.0, 0.0}, Vec3{-a / 2, a * std::sqrt(3.0) / 2, 0.0},
                Vec3{0.0, 0.0, height}, {true, true, true});
}

void require_positive(double value, const char* what) {
    if (!(value > 0.0)) {
        throw BuildError(std::string(what) + " must be positive");
    }
}

/// Measured lattice constants and X-X thicknesses, in angstrom.
///
/// These are experimental monolayer values; a calculation will relax away from
/// them by a per cent or so depending on the functional, which is the point of
/// having them as a starting guess rather than an answer.
const std::map<std::string, std::pair<double, double>>& tmd_table() {
    static const std::map<std::string, std::pair<double, double>> table{
        {"MoS2", {3.160, 3.130}},  {"MoSe2", {3.288, 3.338}}, {"MoTe2", {3.519, 3.604}},
        {"WS2", {3.153, 3.143}},   {"WSe2", {3.282, 3.349}},  {"WTe2", {3.550, 3.620}},
        {"NbS2", {3.310, 3.150}},  {"NbSe2", {3.442, 3.350}}, {"TaS2", {3.315, 3.140}},
        {"TaSe2", {3.436, 3.340}}, {"TiS2", {3.407, 2.860}},  {"ZrS2", {3.662, 2.910}},
        {"HfS2", {3.635, 2.900}},  {"PtS2", {3.542, 2.600}},  {"PdS2", {3.540, 2.630}},
        {"SnS2", {3.648, 2.900}},
    };
    return table;
}

}  // namespace

std::string_view phase_name(Phase phase) { return phase == Phase::H ? "1H" : "1T"; }

Phase phase_from_string(std::string_view name) {
    std::string key(name);
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (key == "h" || key == "1h" || key == "2h" || key == "trigonal_prismatic") {
        return Phase::H;
    }
    if (key == "t" || key == "1t" || key == "octahedral") {
        return Phase::T;
    }
    throw BuildError("'" + std::string(name) + "' is not a phase; use 1H or 1T");
}

Atoms honeycomb(const std::vector<std::string>& symbols, double a, double vacuum,
                double buckling) {
    require_positive(a, "the lattice constant");
    if (symbols.size() != 1 && symbols.size() != 2) {
        throw BuildError("a honeycomb sheet needs one or two symbols, got " +
                         std::to_string(symbols.size()));
    }
    if (vacuum < 0.0) {
        throw BuildError("the vacuum gap must not be negative");
    }

    const double thickness = std::abs(buckling);
    const double height = thickness + vacuum;
    if (!(height > 0.0)) {
        throw BuildError("a sheet needs either a buckling or a vacuum gap to have a cell");
    }

    // Centred in the cell, so the vacuum is split evenly above and below and
    // the sheet is not sitting on a periodic boundary.
    const double z_low = (height - thickness) / 2;
    const double z_high = z_low + thickness;

    const std::vector<std::string> pair{symbols.front(), symbols.back()};
    const std::vector<double> scaled{
        1.0 / 3.0, 2.0 / 3.0, z_low / height,
        2.0 / 3.0, 1.0 / 3.0, z_high / height,
    };

    Atoms atoms(pair, std::vector<double>(6, 0.0), hexagonal(a, height));
    atoms.set_scaled_positions(scaled);
    atoms.set_pbc({true, true, true});
    return atoms;
}

Atoms graphene(double a, double vacuum, int nx, int ny) {
    if (nx < 1 || ny < 1) {
        throw BuildError("a sheet needs at least one cell in each direction");
    }
    const Atoms cell = honeycomb({"C"}, a, vacuum);
    return nx == 1 && ny == 1 ? cell : cell.repeat(nx, ny, 1);
}

Atoms graphene_rectangular(double a, double vacuum, int nx, int ny) {
    require_positive(a, "the lattice constant");
    if (nx < 1 || ny < 1) {
        throw BuildError("a sheet needs at least one cell in each direction");
    }
    if (vacuum <= 0.0) {
        throw BuildError("a flat sheet needs a positive vacuum gap");
    }

    // The four-atom orthorhombic cell: armchair along x, zigzag along y.
    const double bond = a / std::sqrt(3.0);
    const double x = a * std::sqrt(3.0);   // three bond lengths
    const double y = a;
    const std::vector<double> positions{
        0.0,             0.0,     vacuum / 2,
        bond,            0.0,     vacuum / 2,
        1.5 * bond,      y / 2,   vacuum / 2,
        2.5 * bond,      y / 2,   vacuum / 2,
    };
    Atoms atoms(std::vector<std::string>(4, "C"), positions,
                Cell(Mat3::diagonal(x, y, vacuum), {true, true, true}));
    atoms.set_pbc({true, true, true});
    return nx == 1 && ny == 1 ? atoms : atoms.repeat(nx, ny, 1);
}

std::pair<double, double> tmd_parameters(const std::string& metal,
                                         const std::string& chalcogen) {
    const std::string formula = metal + chalcogen + "2";
    const auto it = tmd_table().find(formula);
    if (it == tmd_table().end()) {
        throw BuildError(formula +
                         " is not in the table of measured dichalcogenides; give the "
                         "lattice constant and thickness explicitly");
    }
    return it->second;
}

std::vector<std::string> known_tmds() {
    std::vector<std::string> out;
    out.reserve(tmd_table().size());
    for (const auto& [formula, parameters] : tmd_table()) {
        out.push_back(formula);
    }
    return out;
}

Atoms tmd(const std::string& metal, const std::string& chalcogen, Phase phase, double a,
          double thickness, double vacuum) {
    if (a == 0.0 || thickness == 0.0) {
        const auto [table_a, table_thickness] = tmd_parameters(metal, chalcogen);
        if (a == 0.0) {
            a = table_a;
        }
        if (thickness == 0.0) {
            thickness = table_thickness;
        }
    }
    require_positive(a, "the lattice constant");
    require_positive(thickness, "the layer thickness");
    if (vacuum < 0.0) {
        throw BuildError("the vacuum gap must not be negative");
    }

    const double height = thickness + vacuum;
    const double z_metal = height / 2;
    const double z_low = z_metal - thickness / 2;
    const double z_high = z_metal + thickness / 2;

    // The metal sits on one hexagonal site. In 1H both chalcogens sit on the
    // second site, one directly above the other - the trigonal prism. In 1T
    // the lower one moves to the third site, and the metal is octahedrally
    // coordinated instead.
    const double mx = 1.0 / 3.0, my = 2.0 / 3.0;
    const double ux = 2.0 / 3.0, uy = 1.0 / 3.0;
    const double lx = phase == Phase::H ? ux : 0.0;
    const double ly = phase == Phase::H ? uy : 0.0;

    const std::vector<double> scaled{
        mx, my, z_metal / height,
        ux, uy, z_high / height,
        lx, ly, z_low / height,
    };
    Atoms atoms({metal, chalcogen, chalcogen}, std::vector<double>(9, 0.0),
                hexagonal(a, height));
    atoms.set_scaled_positions(scaled);
    atoms.set_pbc({true, true, true});
    return atoms;
}

void set_vacuum(Atoms& atoms, double vacuum) {
    if (atoms.empty()) {
        throw BuildError("an empty structure has nothing to pad");
    }
    if (vacuum < 0.0) {
        throw BuildError("the vacuum gap must not be negative");
    }

    double lowest = atoms.position(0).z;
    double highest = lowest;
    for (std::size_t i = 1; i < atoms.size(); ++i) {
        lowest = std::min(lowest, atoms.position(i).z);
        highest = std::max(highest, atoms.position(i).z);
    }
    const double thickness = highest - lowest;
    const double height = thickness + vacuum;

    // The cell's c axis is replaced outright rather than scaled: a sheet built
    // with one gap and re-padded to another should not have its in-plane
    // vectors touched, and any tilt in c is not something a vacuum gap can
    // meaningfully preserve.
    Mat3 matrix = atoms.cell().matrix();
    matrix(2, 0) = 0.0;
    matrix(2, 1) = 0.0;
    matrix(2, 2) = height;
    atoms.set_cell(Cell(matrix, atoms.pbc()), false);

    const double shift = (height - thickness) / 2 - lowest;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        Vec3 p = atoms.position(i);
        p.z += shift;
        atoms.set_position(i, p);
    }
}

}  // namespace fastatoms::build
