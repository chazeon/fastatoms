// SPDX-License-Identifier: MIT
#pragma once

/// \file surface.hpp
/// \brief Slabs cut from a bulk crystal along a Miller plane.
///
/// A surface calculation starts by cutting a crystal along a plane and putting
/// vacuum above it. Doing that by hand means finding two lattice vectors that
/// lie in the plane and a third that is not - arithmetic that is easy to get
/// subtly wrong, and wrong in a way that produces a slab which looks right and
/// has the wrong termination.

#include <array>
#include <string>

#include "fastatoms/build/bulk.hpp"
#include "fastatoms/core/atoms.hpp"

namespace fastatoms::build {

/// Cut a slab from a bulk crystal along the (h k l) plane.
///
///     auto slab = build::surface(build::bulk("Cu"), {2, 1, 1}, 9, 10.0);
///
/// \param crystal the bulk cell to cut - whatever `bulk()` returns
/// \param indices the Miller indices of the surface plane
/// \param layers  how many repeats of the surface cell along the normal
/// \param vacuum  vacuum added on each side, in angstrom; zero leaves the slab
///                periodic along the normal, which is what an interface wants
/// \param periodic keep periodicity along the normal even with vacuum
///
/// The result has `a1` along x and the surface normal along z, which is the
/// orientation every downstream tool assumes, and `pbc = (true, true, false)`
/// unless `periodic` says otherwise.
[[nodiscard]] Atoms surface(const Atoms& crystal, const std::array<int, 3>& indices, int layers,
                            double vacuum = 0.0, bool periodic = false);

/// The named low-index faces.
///
///     auto slab = build::fcc111("Pt", {4, 4, 5}, 0.0, 12.0);
///
/// `size` is (nx, ny, layers): repeats in the two surface directions, then the
/// number of atomic layers. `a` of zero looks the lattice constant up, and the
/// element's own structure has to match the face being asked for - `fcc111`
/// on iron is a question about a phase that iron is not in, so it raises.
///
/// `orthogonal` asks for a rectangular surface cell rather than the natural
/// hexagonal one, at the cost of twice the atoms.
/// @{
[[nodiscard]] Atoms fcc100(const std::string& symbol, const std::array<int, 3>& size,
                           double a = 0.0, double vacuum = 0.0, bool orthogonal = false);
[[nodiscard]] Atoms fcc110(const std::string& symbol, const std::array<int, 3>& size,
                           double a = 0.0, double vacuum = 0.0, bool orthogonal = false);
[[nodiscard]] Atoms fcc111(const std::string& symbol, const std::array<int, 3>& size,
                           double a = 0.0, double vacuum = 0.0, bool orthogonal = false);
[[nodiscard]] Atoms fcc211(const std::string& symbol, const std::array<int, 3>& size,
                           double a = 0.0, double vacuum = 0.0, bool orthogonal = false);
[[nodiscard]] Atoms bcc100(const std::string& symbol, const std::array<int, 3>& size,
                           double a = 0.0, double vacuum = 0.0, bool orthogonal = false);
[[nodiscard]] Atoms bcc110(const std::string& symbol, const std::array<int, 3>& size,
                           double a = 0.0, double vacuum = 0.0, bool orthogonal = false);
[[nodiscard]] Atoms bcc111(const std::string& symbol, const std::array<int, 3>& size,
                           double a = 0.0, double vacuum = 0.0, bool orthogonal = false);
[[nodiscard]] Atoms hcp0001(const std::string& symbol, const std::array<int, 3>& size,
                            double a = 0.0, double c = 0.0, double vacuum = 0.0,
                            bool orthogonal = false);
[[nodiscard]] Atoms diamond100(const std::string& symbol, const std::array<int, 3>& size,
                               double a = 0.0, double vacuum = 0.0);
[[nodiscard]] Atoms diamond111(const std::string& symbol, const std::array<int, 3>& size,
                               double a = 0.0, double vacuum = 0.0);
/// @}

/// Add vacuum along z, on top of whatever is already there.
///
/// Distinct from `set_vacuum`, which sets the total: this one adds, which is
/// what "give me another 5 angstrom of room" means.
void add_vacuum(Atoms& atoms, double vacuum);

/// Where an adsorbate goes on a slab.
enum class Site {
    Ontop,    ///< directly above a surface atom
    Bridge,   ///< between two neighbouring surface atoms
    Hollow,   ///< the three-fold hollow of a close-packed face
    FCC,      ///< the hollow with no atom directly beneath
    HCP,      ///< the hollow with an atom in the second layer beneath
};

[[nodiscard]] std::string_view site_name(Site site);
[[nodiscard]] Site site_from_string(std::string_view name);

/// Put an adsorbate on a slab.
///
///     build::add_adsorbate(slab, oxygen, 1.9, Site::FCC);
///
/// \param height how far above the topmost surface atom, in angstrom
/// \param site   which site; the slab has to know where its sites are, which
///               the named face builders record and `surface()` does not
/// \param offset extra displacement in surface-cell fractions, for putting a
///               second adsorbate in a different cell
/// \param mol_index which atom of the adsorbate sits at `height`; the rest
///               follow rigidly, so an OH binds through whichever atom is named
void add_adsorbate(Atoms& slab, const Atoms& adsorbate, double height, Site site,
                   const std::array<double, 2>& offset = {0.0, 0.0},
                   std::size_t mol_index = 0);

/// The same, at an explicit (x, y) in angstrom.
void add_adsorbate(Atoms& slab, const Atoms& adsorbate, double height,
                   const std::array<double, 2>& position, std::size_t mol_index = 0);

}  // namespace fastatoms::build
