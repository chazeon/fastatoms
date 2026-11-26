// SPDX-License-Identifier: MIT
#pragma once

/// \file pointgroup.hpp
/// \brief Crystal systems, point groups and the error type of the symmetry
///        module. The identification itself is done by spglib.

#include <stdexcept>
#include <string>
#include <string_view>

namespace fastatoms::symmetry {

/// Raised when spglib cannot handle a structure, with its message attached.
class SymmetryError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class CrystalSystem { Triclinic, Monoclinic, Orthorhombic, Tetragonal, Trigonal, Hexagonal, Cubic };

[[nodiscard]] constexpr std::string_view crystal_system_name(CrystalSystem system) {
    switch (system) {
        case CrystalSystem::Triclinic: return "triclinic";
        case CrystalSystem::Monoclinic: return "monoclinic";
        case CrystalSystem::Orthorhombic: return "orthorhombic";
        case CrystalSystem::Tetragonal: return "tetragonal";
        case CrystalSystem::Trigonal: return "trigonal";
        case CrystalSystem::Hexagonal: return "hexagonal";
        case CrystalSystem::Cubic: return "cubic";
    }
    return "";
}

/// The crystal system of an International Tables space-group number (1..230).
[[nodiscard]] constexpr CrystalSystem crystal_system_of(int number) {
    if (number <= 2) return CrystalSystem::Triclinic;
    if (number <= 15) return CrystalSystem::Monoclinic;
    if (number <= 74) return CrystalSystem::Orthorhombic;
    if (number <= 142) return CrystalSystem::Tetragonal;
    if (number <= 167) return CrystalSystem::Trigonal;
    if (number <= 194) return CrystalSystem::Hexagonal;
    return CrystalSystem::Cubic;
}

/// One of the 32 crystallographic point groups.
struct PointGroup {
    std::string symbol{"1"};       ///< Hermann-Mauguin, in spglib's notation, e.g. "m-3m"
    std::string schoenflies{"C1"}; ///< e.g. "Oh"
    int order{1};                  ///< number of distinct rotations
    CrystalSystem system{CrystalSystem::Triclinic};
};

}  // namespace fastatoms::symmetry
