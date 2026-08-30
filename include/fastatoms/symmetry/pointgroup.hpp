// SPDX-License-Identifier: MIT
#pragma once

/// \file pointgroup.hpp
/// \brief The thirty-two crystallographic point groups, identified from a set
///        of rotation matrices.
///
/// A rotation in a crystal is one of ten kinds, fixed by its determinant and
/// trace: the identity, two-, three-, four- and six-fold rotations, and their
/// improper partners - inversion, a mirror, and the three-, four- and six-fold
/// rotoinversions. Counting how many of each a group contains identifies it
/// among the thirty-two, and does so without reference to any particular
/// basis, which is what makes it usable on a cell in an arbitrary setting.

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "fastatoms/core/vec3.hpp"

namespace fastatoms::symmetry {

/// Raised for a structure or a group this module cannot make sense of.
class SymmetryError : public std::runtime_error {
public:
    explicit SymmetryError(const std::string& message) : std::runtime_error(message) {}
};

/// The seven crystal systems.
enum class CrystalSystem {
    Triclinic,
    Monoclinic,
    Orthorhombic,
    Tetragonal,
    Trigonal,
    Hexagonal,
    Cubic,
};

[[nodiscard]] std::string_view crystal_system_name(CrystalSystem system);

/// What one rotation matrix is, geometrically.
///
/// The value is the conventional order with a sign: `1`, `2`, `3`, `4`, `6`
/// for proper rotations and `-1`, `-2` (a mirror), `-3`, `-4`, `-6` for
/// improper ones. Determinant and trace between them fix it: no two of the ten
/// share a pair.
enum class RotationType { Identity, Two, Three, Four, Six, Inversion, Mirror, BarThree,
                          BarFour, BarSix };

[[nodiscard]] RotationType classify_rotation(const Mat3& rotation);
[[nodiscard]] int rotation_order(RotationType type);
[[nodiscard]] std::string_view rotation_type_name(RotationType type);

/// How many of each kind a group contains, in the order of `RotationType`.
using RotationCounts = std::array<int, 10>;

[[nodiscard]] RotationCounts count_rotations(const std::vector<Mat3>& rotations);

/// One of the thirty-two crystallographic point groups.
struct PointGroup {
    /// The Hermann-Mauguin symbol, e.g. "4/mmm".
    std::string symbol;
    /// The Schoenflies symbol, e.g. "D4h".
    std::string schoenflies;
    int order{1};
    CrystalSystem system{CrystalSystem::Triclinic};
    bool centrosymmetric{false};
    /// True when the group contains no improper operation, so a structure with
    /// it can be chiral.
    bool enantiomorphic{false};

    [[nodiscard]] std::string to_string() const;
};

/// Identify a set of rotations as one of the thirty-two.
///
/// The rotations need not be in any particular basis, and duplicates are
/// ignored.
///
/// \throws SymmetryError if the counts match none of the thirty-two, which
///         means the input was not a crystallographic point group.
[[nodiscard]] PointGroup identify_point_group(const std::vector<Mat3>& rotations);

/// All thirty-two, for tests and for listing.
[[nodiscard]] const std::vector<PointGroup>& crystallographic_point_groups();

}  // namespace fastatoms::symmetry
