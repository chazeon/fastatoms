// SPDX-License-Identifier: MIT
#pragma once

/// \file bandpath.hpp
/// \brief High-symmetry paths through the Brillouin zone.
///
/// A band structure is drawn along a path joining the special points of the
/// zone, and which points those are is fixed by the Bravais lattice. The
/// lattice is identified from the cell itself - via the symmetry module, by
/// asking what the point group and centring of a lattice with one atom in it
/// are - so a cell in any setting gets the right answer.
///
/// Coverage is stated rather than implied: the lattices below have their
/// special points tabulated, and anything else needs an explicit path. Nothing
/// here guesses.

#include <map>
#include <string>
#include <vector>

#include "fastatoms/core/cell.hpp"
#include "fastatoms/electronic/bands.hpp"

namespace fastatoms::electronic {

/// The fourteen Bravais lattices, in the naming Setyawan and Curtarolo use.
enum class BravaisLattice {
    CUB,   ///< simple cubic
    FCC,   ///< face-centred cubic
    BCC,   ///< body-centred cubic
    TET,   ///< simple tetragonal
    BCT,   ///< body-centred tetragonal
    ORC,   ///< simple orthorhombic
    ORCF,  ///< face-centred orthorhombic
    ORCI,  ///< body-centred orthorhombic
    ORCC,  ///< base-centred orthorhombic
    HEX,   ///< hexagonal
    RHL,   ///< rhombohedral
    MCL,   ///< simple monoclinic
    MCLC,  ///< base-centred monoclinic
    TRI,   ///< triclinic
};

[[nodiscard]] std::string_view bravais_name(BravaisLattice lattice);

/// Which lattice a cell is.
///
/// Works by handing the cell to the symmetry module with a single atom in it:
/// the point group of that structure is the lattice's own holohedry, and the
/// centring comes out of the same analysis.
[[nodiscard]] BravaisLattice identify_bravais(const Cell& cell, double tolerance = 1e-5);

/// The special points of a lattice, in fractional reciprocal coordinates.
///
/// \throws ElectronicError for a lattice whose points are not tabulated, which
///         says which one it is rather than returning something plausible.
[[nodiscard]] std::map<std::string, Vec3> special_points(const Cell& cell,
                                                         double tolerance = 1e-5);

/// The conventional path through those points, as labels separated by commas
/// where the path jumps: `"GXWKGLUW,LK,UX"` for face-centred cubic.
[[nodiscard]] std::string default_path(const Cell& cell, double tolerance = 1e-5);

/// Which lattices have their special points tabulated here.
[[nodiscard]] std::vector<BravaisLattice> tabulated_lattices();

/// A path through the zone, sampled.
struct BandPath {
    /// The k-points themselves, in fractional reciprocal coordinates.
    std::vector<Vec3> kpoints{};

    /// Cumulative distance along the path in inverse angstrom, which is what a
    /// band plot uses for its horizontal axis. Segments the path jumps across
    /// contribute nothing, so the axis has no gap where the path does.
    std::vector<double> distances{};

    /// The label of each k-point that is a special point, and empty elsewhere.
    std::vector<std::string> labels{};

    /// Just the special points: where they fall on the axis, and what they are
    /// called. This is what the tick marks of a plot are.
    std::vector<double> special_distances{};
    std::vector<std::string> special_labels{};

    /// The path string this came from.
    std::string path{};
    BravaisLattice lattice{BravaisLattice::TRI};

    [[nodiscard]] std::size_t size() const noexcept { return kpoints.size(); }
    [[nodiscard]] std::string to_string() const;
};

/// Sample a path through the Brillouin zone.
///
/// \param cell   the real-space cell; the reciprocal one is derived from it
/// \param path   labels to join, e.g. `"GXWKGLUW"`, with a comma where the
///               path should jump rather than interpolate. Empty asks for the
///               lattice's conventional path.
/// \param points roughly how many k-points in total; they are shared between
///               segments in proportion to length, so the sampling is even in
///               reciprocal space rather than even per segment
/// \throws ElectronicError for an unknown label, or a lattice with no
///         tabulated points and no explicit path
[[nodiscard]] BandPath bandpath(const Cell& cell, const std::string& path = "",
                                int points = 100, double tolerance = 1e-5);

/// Sample a path given explicitly, for a lattice with no table or a path of
/// your own.
///
/// \param corners the points to join, in fractional reciprocal coordinates
/// \param labels  one per corner; empty strings are allowed
[[nodiscard]] BandPath bandpath_from_points(const Cell& cell, const std::vector<Vec3>& corners,
                                            const std::vector<std::string>& labels,
                                            int points = 100);

}  // namespace fastatoms::electronic
