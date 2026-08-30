// SPDX-License-Identifier: MIT
#pragma once

/// \file nanotube.hpp
/// \brief Rolling a two-dimensional sheet into a tube.
///
/// The (n, m) construction, which is the same arithmetic whatever the sheet is
/// made of: a chiral vector `C = n a1 + m a2` becomes the circumference, the
/// shortest lattice vector perpendicular to it becomes the axis, and the
/// rectangle they span is wrapped onto a cylinder. A sheet with thickness -
/// a dichalcogenide, say - rolls into a tube with an inner and an outer wall,
/// because each atom keeps its height above the mid-plane as a radial offset.

#include <string>

#include "fastatoms/build/twod.hpp"
#include "fastatoms/core/atoms.hpp"

namespace fastatoms::build {

/// What the (n, m) indices imply for a sheet of lattice constant `a`, before
/// any atoms are placed.
struct TubeGeometry {
    double radius{0.0};           ///< angstrom
    double circumference{0.0};    ///< |C|
    double period{0.0};           ///< |T|, the length of one repeat along the axis
    double chiral_angle{0.0};     ///< degrees, 0 for zigzag and 30 for armchair
    int cells{0};                 ///< sheet cells in one repeat
    /// The translation vector's coefficients, `T = t1 a1 + t2 a2`.
    int t1{0};
    int t2{0};

    [[nodiscard]] std::string to_string() const;
};

/// The geometry of an (n, m) tube rolled from a sheet of lattice constant `a`.
/// \throws BuildError unless `n` and `m` are non-negative and not both zero.
[[nodiscard]] TubeGeometry tube_geometry(int n, int m, double a);

/// Roll any hexagonal monolayer into an (n, m) tube.
///
/// \param sheet   a monolayer with a hexagonal in-plane cell - what
///                `graphene`, `honeycomb` or `tmd` returns
/// \param length  repeats along the tube axis
/// \param vacuum  empty space between the tube and its periodic images across
///                the axis; the cell is periodic along z only
[[nodiscard]] Atoms roll(const Atoms& sheet, int n, int m, int length = 1,
                         double vacuum = 15.0);

/// A carbon nanotube.
[[nodiscard]] Atoms nanotube(int n, int m, int length = 1, double a = 2.46,
                             const std::string& symbol = "C", double vacuum = 15.0);

}  // namespace fastatoms::build
