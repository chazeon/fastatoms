// SPDX-License-Identifier: MIT

#include "fastatoms/symmetry/pointgroup.hpp"

#include <algorithm>
#include <cmath>
#include <map>

namespace fastatoms::symmetry {
namespace {

/// Determinant and trace of an integer rotation, rounded. A crystallographic
/// rotation in the lattice basis has integer entries, and in a Cartesian basis
/// it is orthogonal, so both quantities are integers either way.
std::pair<int, int> determinant_and_trace(const Mat3& rotation) {
    const double determinant = rotation.determinant();
    const double trace = rotation(0, 0) + rotation(1, 1) + rotation(2, 2);
    return {static_cast<int>(std::lround(determinant)), static_cast<int>(std::lround(trace))};
}

struct Entry {
    const char* symbol;
    const char* schoenflies;
    CrystalSystem system;
    /// Counts in the order of `RotationType`: 1, 2, 3, 4, 6, -1, m, -3, -4, -6.
    RotationCounts counts;
};

/// The thirty-two, by what they contain.
///
/// Each row is a fact about the group that can be checked by generating it -
/// and the tests do exactly that, as well as asserting that no two rows are
/// the same, which is what makes the identification well posed.
const std::vector<Entry>& table() {
    static const std::vector<Entry> entries{
        // triclinic
        {"1",     "C1",  CrystalSystem::Triclinic,    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"-1",    "Ci",  CrystalSystem::Triclinic,    {1, 0, 0, 0, 0, 1, 0, 0, 0, 0}},
        // monoclinic
        {"2",     "C2",  CrystalSystem::Monoclinic,   {1, 1, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"m",     "Cs",  CrystalSystem::Monoclinic,   {1, 0, 0, 0, 0, 0, 1, 0, 0, 0}},
        {"2/m",   "C2h", CrystalSystem::Monoclinic,   {1, 1, 0, 0, 0, 1, 1, 0, 0, 0}},
        // orthorhombic
        {"222",   "D2",  CrystalSystem::Orthorhombic, {1, 3, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"mm2",   "C2v", CrystalSystem::Orthorhombic, {1, 1, 0, 0, 0, 0, 2, 0, 0, 0}},
        {"mmm",   "D2h", CrystalSystem::Orthorhombic, {1, 3, 0, 0, 0, 1, 3, 0, 0, 0}},
        // tetragonal
        {"4",     "C4",  CrystalSystem::Tetragonal,   {1, 1, 0, 2, 0, 0, 0, 0, 0, 0}},
        {"-4",    "S4",  CrystalSystem::Tetragonal,   {1, 1, 0, 0, 0, 0, 0, 0, 2, 0}},
        {"4/m",   "C4h", CrystalSystem::Tetragonal,   {1, 1, 0, 2, 0, 1, 1, 0, 2, 0}},
        {"422",   "D4",  CrystalSystem::Tetragonal,   {1, 5, 0, 2, 0, 0, 0, 0, 0, 0}},
        {"4mm",   "C4v", CrystalSystem::Tetragonal,   {1, 1, 0, 2, 0, 0, 4, 0, 0, 0}},
        {"-42m",  "D2d", CrystalSystem::Tetragonal,   {1, 3, 0, 0, 0, 0, 2, 0, 2, 0}},
        {"4/mmm", "D4h", CrystalSystem::Tetragonal,   {1, 5, 0, 2, 0, 1, 5, 0, 2, 0}},
        // trigonal
        {"3",     "C3",  CrystalSystem::Trigonal,     {1, 0, 2, 0, 0, 0, 0, 0, 0, 0}},
        {"-3",    "C3i", CrystalSystem::Trigonal,     {1, 0, 2, 0, 0, 1, 0, 2, 0, 0}},
        {"32",    "D3",  CrystalSystem::Trigonal,     {1, 3, 2, 0, 0, 0, 0, 0, 0, 0}},
        {"3m",    "C3v", CrystalSystem::Trigonal,     {1, 0, 2, 0, 0, 0, 3, 0, 0, 0}},
        {"-3m",   "D3d", CrystalSystem::Trigonal,     {1, 3, 2, 0, 0, 1, 3, 2, 0, 0}},
        // hexagonal
        {"6",     "C6",  CrystalSystem::Hexagonal,    {1, 1, 2, 0, 2, 0, 0, 0, 0, 0}},
        {"-6",    "C3h", CrystalSystem::Hexagonal,    {1, 0, 2, 0, 0, 0, 1, 0, 0, 2}},
        {"6/m",   "C6h", CrystalSystem::Hexagonal,    {1, 1, 2, 0, 2, 1, 1, 2, 0, 2}},
        {"622",   "D6",  CrystalSystem::Hexagonal,    {1, 7, 2, 0, 2, 0, 0, 0, 0, 0}},
        {"6mm",   "C6v", CrystalSystem::Hexagonal,    {1, 1, 2, 0, 2, 0, 6, 0, 0, 0}},
        {"-6m2",  "D3h", CrystalSystem::Hexagonal,    {1, 3, 2, 0, 0, 0, 4, 0, 0, 2}},
        {"6/mmm", "D6h", CrystalSystem::Hexagonal,    {1, 7, 2, 0, 2, 1, 7, 2, 0, 2}},
        // cubic
        {"23",    "T",   CrystalSystem::Cubic,        {1, 3, 8, 0, 0, 0, 0, 0, 0, 0}},
        {"m-3",   "Th",  CrystalSystem::Cubic,        {1, 3, 8, 0, 0, 1, 3, 8, 0, 0}},
        {"432",   "O",   CrystalSystem::Cubic,        {1, 9, 8, 6, 0, 0, 0, 0, 0, 0}},
        {"-43m",  "Td",  CrystalSystem::Cubic,        {1, 3, 8, 0, 0, 0, 6, 0, 6, 0}},
        {"m-3m",  "Oh",  CrystalSystem::Cubic,        {1, 9, 8, 6, 0, 1, 9, 8, 6, 0}},
    };
    return entries;
}

PointGroup from_entry(const Entry& entry) {
    PointGroup group;
    group.symbol = entry.symbol;
    group.schoenflies = entry.schoenflies;
    group.system = entry.system;
    group.order = 0;
    for (const int count : entry.counts) {
        group.order += count;
    }
    group.centrosymmetric = entry.counts[static_cast<int>(RotationType::Inversion)] > 0;
    // Enantiomorphic when nothing improper is present: the four counts after
    // the proper rotations are all zero.
    group.enantiomorphic = true;
    for (int k = 5; k < 10; ++k) {
        if (entry.counts[k] != 0) {
            group.enantiomorphic = false;
        }
    }
    return group;
}

/// A key that identifies a matrix exactly, for deduplication.
std::array<long long, 9> matrix_key(const Mat3& m) {
    std::array<long long, 9> key{};
    for (std::size_t k = 0; k < 9; ++k) {
        key[k] = std::llround(m.m[k] * 1e6);
    }
    return key;
}

}  // namespace

std::string_view crystal_system_name(CrystalSystem system) {
    switch (system) {
        case CrystalSystem::Triclinic:    return "triclinic";
        case CrystalSystem::Monoclinic:   return "monoclinic";
        case CrystalSystem::Orthorhombic: return "orthorhombic";
        case CrystalSystem::Tetragonal:   return "tetragonal";
        case CrystalSystem::Trigonal:     return "trigonal";
        case CrystalSystem::Hexagonal:    return "hexagonal";
        case CrystalSystem::Cubic:        return "cubic";
    }
    return "triclinic";
}

RotationType classify_rotation(const Mat3& rotation) {
    const auto [determinant, trace] = determinant_and_trace(rotation);
    if (determinant == 1) {
        switch (trace) {
            case 3:  return RotationType::Identity;
            case -1: return RotationType::Two;
            case 0:  return RotationType::Three;
            case 1:  return RotationType::Four;
            case 2:  return RotationType::Six;
            default: break;
        }
    } else if (determinant == -1) {
        switch (trace) {
            case -3: return RotationType::Inversion;
            case 1:  return RotationType::Mirror;
            case 0:  return RotationType::BarThree;
            case -1: return RotationType::BarFour;
            case -2: return RotationType::BarSix;
            default: break;
        }
    }
    throw SymmetryError("a matrix with determinant " + std::to_string(determinant) +
                        " and trace " + std::to_string(trace) +
                        " is not a crystallographic rotation");
}

int rotation_order(RotationType type) {
    switch (type) {
        case RotationType::Identity:  return 1;
        case RotationType::Two:       return 2;
        case RotationType::Three:     return 3;
        case RotationType::Four:      return 4;
        case RotationType::Six:       return 6;
        case RotationType::Inversion: return -1;
        case RotationType::Mirror:    return -2;
        case RotationType::BarThree:  return -3;
        case RotationType::BarFour:   return -4;
        case RotationType::BarSix:    return -6;
    }
    return 1;
}

std::string_view rotation_type_name(RotationType type) {
    switch (type) {
        case RotationType::Identity:  return "1";
        case RotationType::Two:       return "2";
        case RotationType::Three:     return "3";
        case RotationType::Four:      return "4";
        case RotationType::Six:       return "6";
        case RotationType::Inversion: return "-1";
        case RotationType::Mirror:    return "m";
        case RotationType::BarThree:  return "-3";
        case RotationType::BarFour:   return "-4";
        case RotationType::BarSix:    return "-6";
    }
    return "1";
}

RotationCounts count_rotations(const std::vector<Mat3>& rotations) {
    RotationCounts counts{};
    std::vector<std::array<long long, 9>> seen;
    for (const Mat3& rotation : rotations) {
        const std::array<long long, 9> key = matrix_key(rotation);
        if (std::find(seen.begin(), seen.end(), key) != seen.end()) {
            continue;
        }
        seen.push_back(key);
        ++counts[static_cast<std::size_t>(classify_rotation(rotation))];
    }
    return counts;
}

PointGroup identify_point_group(const std::vector<Mat3>& rotations) {
    if (rotations.empty()) {
        throw SymmetryError("a point group needs at least the identity");
    }
    const RotationCounts counts = count_rotations(rotations);
    for (const Entry& entry : table()) {
        if (entry.counts == counts) {
            return from_entry(entry);
        }
    }

    int order = 0;
    for (const int count : counts) {
        order += count;
    }
    throw SymmetryError("these " + std::to_string(order) +
                        " rotations are not one of the thirty-two crystallographic point "
                        "groups; the tolerance may be admitting operations that are not "
                        "really there");
}

const std::vector<PointGroup>& crystallographic_point_groups() {
    static const std::vector<PointGroup> groups = [] {
        std::vector<PointGroup> out;
        out.reserve(table().size());
        for (const Entry& entry : table()) {
            out.push_back(from_entry(entry));
        }
        return out;
    }();
    return groups;
}

std::string PointGroup::to_string() const {
    return symbol + " (" + schoenflies + ", " + std::string(crystal_system_name(system)) +
           ", order " + std::to_string(order) + ")";
}

}  // namespace fastatoms::symmetry
