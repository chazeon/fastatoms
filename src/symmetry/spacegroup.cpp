// SPDX-License-Identifier: MIT

#include "fastatoms/symmetry/spacegroup.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>

namespace fastatoms::symmetry {
namespace {

double wrap01(double value) {
    value -= std::floor(value);
    return (value > 1.0 - 1e-9 || value < 1e-9) ? 0.0 : value;
}

/// The part of a translation that no choice of origin can remove.
///
/// Averaging `t` over the powers of `W` projects it onto the subspace `W`
/// leaves alone, and that component is what makes a screw a screw and a glide
/// a glide. The rest is the operation's location, which moves with the origin
/// and says nothing about which space group this is.
Vec3 intrinsic_translation(const Mat3& rotation, const Vec3& translation) {
    int order = 1;
    Mat3 power = rotation;
    while (order < 6) {
        bool identity = true;
        for (std::size_t k = 0; k < 9; ++k) {
            const double expected = (k % 4 == 0) ? 1.0 : 0.0;
            if (std::abs(power.m[k] - expected) > 1e-6) {
                identity = false;
            }
        }
        if (identity) {
            break;
        }
        power = power * rotation;
        ++order;
    }

    Vec3 sum = translation;
    Vec3 carried = translation;
    Mat3 accumulated = rotation;
    for (int k = 1; k < order; ++k) {
        carried = carried * rotation;
        sum = Vec3{sum.x + carried.x, sum.y + carried.y, sum.z + carried.z};
        accumulated = accumulated * rotation;
    }
    return Vec3{sum.x / order, sum.y / order, sum.z / order};
}

/// The direction a rotation or rotoinversion turns about, in the same basis as
/// the matrix, as the smallest integer vector.
Vec3 axis_direction(const Mat3& rotation, RotationType type) {
    Mat3 proper = rotation;
    const int order = rotation_order(type);
    if (order < 0) {
        for (std::size_t k = 0; k < 9; ++k) {
            proper.m[k] = -proper.m[k];   // -n turned into its proper partner
        }
    }
    const int turns = std::abs(order) == 1 ? 1 : std::abs(order);

    // Sum the powers: the result projects onto the invariant axis.
    Mat3 projector = Mat3::identity();
    Mat3 power = Mat3::identity();
    for (int k = 1; k < turns; ++k) {
        power = power * proper;
        for (std::size_t c = 0; c < 9; ++c) {
            projector.m[c] += power.m[c];
        }
    }
    // Rows, not columns. Fractional coordinates are row vectors here, so an
    // operation acts as `f -> f W` and its invariant direction is a *left*
    // eigenvector. For a cubic lattice the matrices are symmetric and the
    // distinction does not show; for a hexagonal one it puts the six-fold axis
    // in the wrong place and the symbol comes out for a different group.
    Vec3 best{0.0, 0.0, 0.0};
    for (std::size_t r = 0; r < 3; ++r) {
        const Vec3 row{projector(r, 0), projector(r, 1), projector(r, 2)};
        if (row.norm() > best.norm()) {
            best = row;
        }
    }
    return best;
}

/// The order of the cyclic group one element generates.
///
/// A rotoinversion of odd order generates twice as much as its order suggests:
/// -3 squares to a three-fold and cubes to the inversion, so it generates a
/// group of six. That is why the cubic symbol is `m-3m` and not `m3m` - the
/// slot names the generator of the axial subgroup, and along a body diagonal
/// with an inversion present that generator is -3.
int cyclic_order(int order) {
    const int magnitude = std::abs(order);
    if (order > 0) {
        return magnitude;
    }
    return magnitude % 2 == 0 ? magnitude : 2 * magnitude;
}

/// True when two directions are parallel, either way round.
bool parallel(const Vec3& a, const Vec3& b) {
    if (a.norm() < 1e-9 || b.norm() < 1e-9) {
        return false;
    }
    return cross(a, b).norm() < 1e-6 * a.norm() * b.norm();
}

/// The symmetry directions of each crystal system, in conventional
/// coordinates: primary, secondary, tertiary. A direction stands for the whole
/// family of equivalents, so only one representative of each is listed.
struct Directions {
    std::vector<Vec3> primary;
    std::vector<Vec3> secondary;
    std::vector<Vec3> tertiary;
};

Directions directions_for(CrystalSystem system) {
    switch (system) {
        case CrystalSystem::Triclinic:
            return {};
        case CrystalSystem::Monoclinic:
            return {{Vec3{0, 1, 0}}, {}, {}};
        case CrystalSystem::Orthorhombic:
            return {{Vec3{1, 0, 0}}, {Vec3{0, 1, 0}}, {Vec3{0, 0, 1}}};
        case CrystalSystem::Tetragonal:
            return {{Vec3{0, 0, 1}}, {Vec3{1, 0, 0}, Vec3{0, 1, 0}},
                    {Vec3{1, 1, 0}, Vec3{1, -1, 0}}};
        case CrystalSystem::Trigonal:
        case CrystalSystem::Hexagonal:
            return {{Vec3{0, 0, 1}},
                    {Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{-1, -1, 0}},
                    {Vec3{1, -1, 0}, Vec3{1, 2, 0}, Vec3{-2, -1, 0}}};
        case CrystalSystem::Cubic:
            return {{Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}},
                    {Vec3{1, 1, 1}, Vec3{1, -1, -1}, Vec3{-1, 1, -1}, Vec3{-1, -1, 1}},
                    {Vec3{1, 1, 0}, Vec3{1, -1, 0}, Vec3{0, 1, 1}, Vec3{0, 1, -1},
                     Vec3{1, 0, 1}, Vec3{-1, 0, 1}}};
    }
    return {};
}

/// What one direction carries.
struct DirectionSymbols {
    std::string axis;    ///< "4", "6_3", "-3", or empty
    std::string plane;   ///< "m", "c", "n", "d", "e", or empty
    int axis_order{0};   ///< for ordering; negative for rotoinversions
};

/// The screw index of a rotation of order `n` about `direction`.
///
/// The intrinsic translation of a screw is a fraction `s/n` of the lattice
/// repeat along the axis, and `s` is what the subscript in `4_1` or `6_3` is.
int screw_index(const Vec3& intrinsic, const Vec3& direction, int order) {
    const double along = dot(intrinsic, direction) / dot(direction, direction);
    const int index = static_cast<int>(std::lround(along * order));
    return ((index % order) + order) % order;
}

/// Which glide a mirror's intrinsic translation describes.
std::string glide_letter(const Vec3& intrinsic, const Vec3& normal) {
    const auto close = [](double value, double target) {
        return std::abs(value - std::round(value - target) - target) < 1e-4;
    };
    const double x = intrinsic.x - std::round(intrinsic.x);
    const double y = intrinsic.y - std::round(intrinsic.y);
    const double z = intrinsic.z - std::round(intrinsic.z);
    const Vec3 t{x, y, z};

    if (t.norm() < 1e-4) {
        return "m";
    }
    // A quarter translation in two directions at once is the diamond glide.
    const auto is_quarter = [&](double value) {
        return std::abs(std::abs(value) - 0.25) < 1e-4;
    };
    const int quarters = (is_quarter(x) ? 1 : 0) + (is_quarter(y) ? 1 : 0) +
                         (is_quarter(z) ? 1 : 0);
    if (quarters >= 2) {
        return "d";
    }
    const auto is_half = [&](double value) { return std::abs(std::abs(value) - 0.5) < 1e-4; };
    const int halves = (is_half(x) ? 1 : 0) + (is_half(y) ? 1 : 0) + (is_half(z) ? 1 : 0);
    if (halves == 1) {
        if (is_half(x)) return "a";
        if (is_half(y)) return "b";
        return "c";
    }
    if (halves >= 2) {
        return "n";
    }
    (void)close;
    (void)normal;
    return "g";   // some other glide, which the short symbol has no letter for
}

/// Which of two glide letters the symbol prefers when a plane carries both.
int glide_rank(const std::string& letter) {
    if (letter == "m") return 0;
    if (letter == "e") return 1;
    if (letter == "a" || letter == "b" || letter == "c") return 2;
    if (letter == "n") return 3;
    if (letter == "d") return 4;
    return 5;
}

/// Everything one symmetry direction carries, gathered from the operations.
DirectionSymbols analyse_direction(const std::vector<SymmetryOperation>& operations,
                                   const std::vector<Vec3>& family) {
    DirectionSymbols out;
    int best_order = 0;
    int best_screw = 0;
    std::vector<std::string> glides;

    for (const Vec3& direction : family) {
        for (const SymmetryOperation& operation : operations) {
            const RotationType type = classify_rotation(operation.rotation);
            if (type == RotationType::Identity || type == RotationType::Inversion) {
                continue;
            }
            const Vec3 axis = axis_direction(operation.rotation, type);
            if (!parallel(axis, direction)) {
                continue;
            }
            const Vec3 intrinsic = intrinsic_translation(operation.rotation,
                                                         operation.translation);

            if (type == RotationType::Mirror) {
                glides.push_back(glide_letter(intrinsic, direction));
                continue;
            }
            const int order = rotation_order(type);
            const int magnitude = std::abs(order);
            const int screw = order > 0 ? screw_index(intrinsic, direction, magnitude) : 0;

            // The slot names the generator of the axial subgroup, so the
            // element generating the largest cyclic group wins. Between two
            // that generate the same - a four-fold and a four-fold
            // rotoinversion - the proper one is written, and between two of
            // those the pure rotation beats the screw, which is what centring
            // often provides both of.
            const bool better =
                best_order == 0 || cyclic_order(order) > cyclic_order(best_order) ||
                (cyclic_order(order) == cyclic_order(best_order) && order > best_order) ||
                (order == best_order && screw < best_screw);
            if (better) {
                best_order = order;
                best_screw = screw;
            }
            (void)magnitude;
        }
    }

    if (best_order != 0) {
        out.axis_order = best_order;
        if (best_order < 0) {
            out.axis = "-" + std::to_string(-best_order);
        } else if (best_screw == 0) {
            out.axis = std::to_string(best_order);
        } else {
            out.axis = std::to_string(best_order) + "_" + std::to_string(best_screw);
        }
    }
    if (!glides.empty()) {
        // A plane carrying two different axial glides at once is the "e" of the
        // 1992 revision - Aem2 rather than Abm2.
        std::sort(glides.begin(), glides.end());
        glides.erase(std::unique(glides.begin(), glides.end()), glides.end());
        const bool double_glide =
            std::count_if(glides.begin(), glides.end(), [](const std::string& g) {
                return g == "a" || g == "b" || g == "c";
            }) >= 2;
        if (double_glide && std::find(glides.begin(), glides.end(), "m") == glides.end()) {
            out.plane = "e";
        } else {
            out.plane = *std::min_element(glides.begin(), glides.end(),
                                          [](const std::string& a, const std::string& b) {
                                              return glide_rank(a) < glide_rank(b);
                                          });
        }
    }
    return out;
}

/// The short-symbol entry for one direction: the plane when there is one,
/// otherwise the axis. The primary direction of the tetragonal, trigonal and
/// hexagonal systems is the exception, where both are written.
/// Whether an "axis over plane" entry is written at all.
///
/// Only for a proper rotation. A rotoinversion already contains its
/// perpendicular mirror where there is one - `-6` is `3/m` written another way
/// - so the symbol names the rotoinversion and stops: `P-6m2`, never
/// `P-6/mm2`.
bool combines(const DirectionSymbols& symbols) {
    return symbols.axis_order > 0 && !symbols.axis.empty() && !symbols.plane.empty();
}

std::string short_entry(const DirectionSymbols& symbols, bool keep_both) {
    if (symbols.axis.empty() && symbols.plane.empty()) {
        return "1";
    }
    if (keep_both && combines(symbols)) {
        return symbols.axis + "/" + symbols.plane;
    }
    if (symbols.axis_order < 0) {
        return symbols.axis;
    }
    return symbols.plane.empty() ? symbols.axis : symbols.plane;
}

std::string full_entry(const DirectionSymbols& symbols) {
    if (symbols.axis.empty() && symbols.plane.empty()) {
        return "1";
    }
    if (combines(symbols)) {
        return symbols.axis + "/" + symbols.plane;
    }
    if (symbols.axis_order < 0) {
        return symbols.axis;
    }
    return symbols.plane.empty() ? symbols.axis : symbols.plane;
}

/// The point group symbol implied by a space group symbol: drop the centring,
/// the screw subscripts, and turn every glide into a mirror.
std::string point_group_of_symbol(const std::string& symbol) {
    std::string out;
    for (std::size_t k = 1; k < symbol.size(); ++k) {   // skip the centring letter
        const char c = symbol[k];
        if (c == '_') {
            ++k;    // and the digit after it
            continue;
        }
        if (c == 'a' || c == 'b' || c == 'c' || c == 'd' || c == 'e' || c == 'n') {
            out += 'm';
        } else {
            out += c;
        }
    }
    return out;
}

}  // namespace

std::string WyckoffPosition::to_string() const {
    std::ostringstream out;
    out << multiplicity << letter << " (" << site_symmetry << ")";
    return out.str();
}

std::string SpaceGroup::to_string() const {
    std::ostringstream out;
    out << "SpaceGroup(" << hermann_mauguin;
    if (number > 0) {
        out << ", No. " << number;
    }
    out << ", " << point_group.symbol << ", " << operations.size() << " operations)";
    return out.str();
}

std::string SpaceGroup::summary() const {
    std::ostringstream out;
    out << "space group : " << hermann_mauguin;
    if (number > 0) {
        out << "  (No. " << number << ")";
    }
    out << "\nfull symbol : " << hermann_mauguin_full
        << "\nSchoenflies : " << schoenflies
        << "\npoint group : " << point_group.symbol << " (" << point_group.schoenflies << ")"
        << "\nsystem      : " << crystal_system_name(system) << ", "
        << centring_letter(centring) << "-centred"
        << "\noperations  : " << operations.size()
        << "\natoms       : " << primitive_atoms << " primitive, " << conventional_atoms
        << " conventional";
    if (!wyckoff.empty()) {
        out << "\nWyckoff     :";
        for (const WyckoffPosition& position : wyckoff) {
            out << " " << position.to_string();
        }
    }
    return out.str();
}

SpaceGroup analyse(const Atoms& atoms, double tolerance) {
    if (atoms.empty() || !atoms.cell().is_invertible()) {
        throw SymmetryError("determining a space group needs atoms and a cell that spans "
                            "three dimensions");
    }

    SpaceGroup group;
    group.tolerance = tolerance;

    const CellAnalysis cell = analyse_cell(atoms, tolerance);
    group.point_group = cell.point_group;
    group.system = cell.system;
    group.centring = cell.centring;
    group.primitive_atoms = cell.primitive_atoms;
    group.conventional_atoms = cell.conventional_atoms;

    // Everything below works on the primitive cell. A supercell admits fewer
    // operations than the crystal has - its lattice is a sublattice, and most
    // of the rotations do not map it onto itself - so analysing what was handed
    // in would answer a question about the box rather than about the crystal.
    group.primitive = cell.input_multiplicity == 1 ? atoms
                                                   : primitive_cell(atoms, tolerance);
    const SymmetryDataset dataset = find_symmetry(group.primitive, tolerance);
    group.operations = dataset.operations;

    // The symbol is read off the operations in the conventional basis, where
    // the symmetry directions are the axes. The change of basis for a rotation
    // is the usual conjugation.
    const Mat3 P = cell.conventional_cell * group.primitive.cell().inverse_matrix();
    const Mat3 P_inverse = P.inverse();
    std::vector<SymmetryOperation> conventional;
    conventional.reserve(dataset.operations.size());
    for (const SymmetryOperation& operation : dataset.operations) {
        SymmetryOperation moved = operation;
        moved.rotation = P * operation.rotation * P_inverse;
        moved.translation = operation.translation * P_inverse;
        moved.translation = Vec3{wrap01(moved.translation.x), wrap01(moved.translation.y),
                                 wrap01(moved.translation.z)};
        conventional.push_back(moved);
    }

    const Directions directions = directions_for(group.system);
    const DirectionSymbols primary = analyse_direction(conventional, directions.primary);
    const DirectionSymbols secondary = analyse_direction(conventional, directions.secondary);
    const DirectionSymbols tertiary = analyse_direction(conventional, directions.tertiary);

    const bool keep_both = group.system == CrystalSystem::Tetragonal ||
                           group.system == CrystalSystem::Trigonal ||
                           group.system == CrystalSystem::Hexagonal ||
                           group.system == CrystalSystem::Monoclinic;

    std::string symbol(1, centring_letter(group.centring));
    std::string full(1, centring_letter(group.centring));

    if (group.system == CrystalSystem::Triclinic) {
        symbol += group.point_group.centrosymmetric ? "-1" : "1";
        full = symbol;
    } else {
        std::string one = short_entry(primary, keep_both);
        std::string two = short_entry(secondary, false);
        std::string three = short_entry(tertiary, false);

        // Trailing "1"s carry information when only one of the two lower
        // directions is occupied - P3m1 and P31m are different groups - but
        // not when both are empty.
        if (two == "1" && three == "1") {
            two.clear();
            three.clear();
        }
        // A rhombohedral cell has no tertiary direction to speak of.
        if (group.centring == Centring::R) {
            three.clear();
        }
        // Cubic groups drop an empty tertiary as well: P23, Pm-3, Pa-3.
        if (group.system == CrystalSystem::Cubic && three == "1") {
            three.clear();
        }
        symbol += one + two + three;
        full += " " + full_entry(primary) + " " + full_entry(secondary) + " " +
                full_entry(tertiary);
    }

    group.hermann_mauguin = symbol;
    group.hermann_mauguin_full = full;
    group.number = space_group_number(symbol);

    // The Schoenflies superscript is the group's position among those sharing
    // its point group, which the table gives without needing a second one.
    if (group.number > 0) {
        int index = 0;
        for (const auto& [number, entry] : space_group_table()) {
            if (point_group_of_symbol(entry) == point_group_of_symbol(symbol)) {
                ++index;
                if (number == group.number) {
                    break;
                }
            }
        }
        group.schoenflies = group.point_group.schoenflies + "-" + std::to_string(index);
    } else {
        group.schoenflies = group.point_group.schoenflies;
    }

    // Wyckoff orbits: the atoms the operations tie together, with the site
    // symmetry of each read off the operations that leave a member alone.
    std::map<std::size_t, std::vector<std::size_t>> orbits;
    for (std::size_t i = 0; i < dataset.equivalent_atoms.size(); ++i) {
        orbits[dataset.equivalent_atoms[i]].push_back(i);
    }
    const int scale = group.primitive.empty()
                          ? 1
                          : static_cast<int>(cell.conventional_atoms / group.primitive.size());
    for (const auto& [representative, members] : orbits) {
        WyckoffPosition position;
        position.atoms = members;
        position.multiplicity = static_cast<int>(members.size()) * std::max(scale, 1);
        position.representative =
            group.primitive.position(representative) * group.primitive.cell().inverse_matrix();

        std::vector<Mat3> site;
        for (const SymmetryOperation& operation : phonon::site_symmetry(dataset, representative)) {
            site.push_back(operation.cartesian_rotation);
        }
        try {
            position.site_symmetry = identify_point_group(site).symbol;
        } catch (const SymmetryError&) {
            position.site_symmetry = "?";
        }
        group.wyckoff.push_back(std::move(position));
    }
    std::sort(group.wyckoff.begin(), group.wyckoff.end(),
              [](const WyckoffPosition& a, const WyckoffPosition& b) {
                  if (a.multiplicity != b.multiplicity) {
                      return a.multiplicity < b.multiplicity;
                  }
                  return a.atoms.front() < b.atoms.front();
              });
    char letter = 'a';
    for (WyckoffPosition& position : group.wyckoff) {
        position.letter = letter;
        letter = letter == 'z' ? 'z' : static_cast<char>(letter + 1);
    }
    return group;
}

}  // namespace fastatoms::symmetry
