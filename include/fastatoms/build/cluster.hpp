// SPDX-License-Identifier: MIT
#pragma once

/// \file cluster.hpp
/// \brief Nanoparticles: the standard closed-shell shapes, and the Wulff
///        construction that produces the equilibrium one.
///
/// Clusters come back centred on the origin with a cell large enough to hold
/// them and periodicity off. Add a cell of your own if you want one; nothing
/// here assumes a particle is isolated except the boundary conditions it is
/// given.

#include <map>
#include <string>
#include <vector>

#include "fastatoms/build/bulk.hpp"
#include "fastatoms/core/atoms.hpp"

namespace fastatoms::build {

/// A Mackay icosahedron.
///
/// Twenty tetrahedra sharing a centre, which is not something an fcc lattice
/// can do without strain: the shells sit about five per cent closer together
/// than the atoms within a shell. That strain is why icosahedra win at small
/// sizes, where surface energy dominates, and lose at large ones.
///
/// `shells = 1` is the thirteen-atom cuboctahedral shell; the closed-shell
/// sizes are 13, 55, 147, 309, 561.
///
/// \param d nearest-neighbour distance in angstrom
/// \param vacuum empty space added around the particle
[[nodiscard]] Atoms icosahedron(const std::string& symbol, int shells, double d,
                                double vacuum = 10.0);

/// A cuboctahedron: the fcc particle with {111} and {100} facets.
///
/// The same closed-shell sizes as the icosahedron - 13, 55, 147 - because the
/// two have the same shell structure and differ only in how it is folded.
///
/// \param a the conventional cubic lattice constant, not the bond length
[[nodiscard]] Atoms cuboctahedron(const std::string& symbol, int shells, double a,
                                  double vacuum = 10.0);

/// A decahedron: five fcc tetrahedra twinned about a common five-fold axis.
///
/// Five tetrahedra placed around an axis leave a gap of 7.35 degrees, since
/// five times the 70.53-degree dihedral is not a full turn. A real particle
/// closes it by straining, and so does this: each wedge is stretched
/// azimuthally to exactly 72 degrees.
///
/// \param p atoms along the edge of the {100} side facets
/// \param q atomic planes along the five-fold axis
/// \param r depth of the Marks re-entrance; zero gives an Ino decahedron
[[nodiscard]] Atoms decahedron(const std::string& symbol, int p, int q, int r, double a,
                               double vacuum = 10.0);

/// One facet family and the energy it costs.
struct SurfaceEnergy {
    /// Miller indices of one representative; every symmetry-equivalent
    /// direction of the cubic point group is generated from it.
    int h{1};
    int k{0};
    int l{0};
    /// Surface energy, in any consistent unit - only the ratios matter to the
    /// shape, since the overall scale is fixed by the size you ask for.
    double energy{1.0};
};

/// The Wulff construction: the equilibrium shape of a crystal.
///
/// A facet's distance from the centre is proportional to its surface energy,
/// which is Wulff's theorem, and the particle is what survives cutting the
/// lattice with all of them at once. Cheap facets grow large; expensive ones
/// shrink and can vanish entirely, which is why a ratio above about 1.15
/// removes the {100} facets from an fcc metal and leaves an octahedron.
///
/// \param energies one entry per facet family; equivalents are generated
/// \param atom_count the particle is scaled until it holds about this many
///        atoms, then the closest achievable size is used
/// \throws BuildError if no energies are given, or none is positive
[[nodiscard]] Atoms wulff(const std::string& symbol, Lattice lattice, double a,
                          const std::vector<SurfaceEnergy>& energies, int atom_count,
                          double vacuum = 10.0);

/// Every symmetry-equivalent direction of `(h, k, l)` under cubic symmetry.
///
/// Exposed because it is the part of the Wulff construction most worth being
/// able to check: `{100}` must give six directions, `{111}` eight, `{110}`
/// twelve.
[[nodiscard]] std::vector<Vec3> equivalent_directions(int h, int k, int l);

}  // namespace fastatoms::build
