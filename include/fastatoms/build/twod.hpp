// SPDX-License-Identifier: MIT
#pragma once

/// \file twod.hpp
/// \brief Two-dimensional materials: graphene and the transition-metal
///        dichalcogenides.
///
/// A monolayer is built periodic in all three directions with a vacuum gap
/// along z, because that is what a plane-wave code wants and what everything
/// downstream here assumes. The gap is the distance between one sheet and the
/// next image of it, not the height of the cell: ask for 15 A of vacuum and
/// you get 15 A of empty space, whatever the layer's own thickness.

#include <string>
#include <vector>

#include "fastatoms/build/bulk.hpp"
#include "fastatoms/core/atoms.hpp"

namespace fastatoms::build {

/// Which polymorph a dichalcogenide monolayer has.
///
/// The two differ only in where the lower chalcogen sits, and in almost
/// everything else: 1H MoS2 is a semiconductor, 1T is metallic.
enum class Phase {
    H,   ///< trigonal prismatic - the monolayer of the common 2H stacking
    T,   ///< octahedral
};

[[nodiscard]] std::string_view phase_name(Phase phase);
[[nodiscard]] Phase phase_from_string(std::string_view name);

/// A honeycomb monolayer of one or two species.
///
/// The general case behind graphene: two sites per cell on a hexagonal
/// lattice. Give one symbol for graphene or silicene, two for a binary sheet
/// such as hexagonal boron nitride.
///
/// \param a       the lattice constant, which is sqrt(3) times the bond length
/// \param vacuum  empty space between the sheet and its next periodic image
/// \param buckling out-of-plane offset between the two sites, for silicene and
///        germanene, which are not flat; zero for graphene and h-BN
[[nodiscard]] Atoms honeycomb(const std::vector<std::string>& symbols, double a,
                              double vacuum = 15.0, double buckling = 0.0);

/// A graphene sheet.
///
/// \param nx,ny repetitions of the two-atom cell
/// \param a     lattice constant; the default gives the usual 1.42 A bond
[[nodiscard]] Atoms graphene(double a = 2.46, double vacuum = 15.0, int nx = 1, int ny = 1);

/// A rectangular graphene cell, four atoms, with the armchair direction along
/// x and the zigzag along y.
///
/// The hexagonal cell is the smaller description; this is the one to repeat
/// when you want a rectangular ribbon or slab.
[[nodiscard]] Atoms graphene_rectangular(double a = 2.46, double vacuum = 15.0, int nx = 1,
                                         int ny = 1);

/// A transition-metal dichalcogenide monolayer, MX2.
///
/// \param metal      M, e.g. "Mo" or "W"
/// \param chalcogen  X, e.g. "S", "Se" or "Te"
/// \param a          in-plane lattice constant; zero looks the pair up in the
///                   table of measured values, and throws if it is not there
/// \param thickness  the vertical X-X separation; zero looks it up too
[[nodiscard]] Atoms tmd(const std::string& metal, const std::string& chalcogen,
                        Phase phase = Phase::H, double a = 0.0, double thickness = 0.0,
                        double vacuum = 15.0);

/// The measured lattice constant and X-X thickness of a known MX2, in
/// angstrom. \throws BuildError if the pair is not tabulated.
[[nodiscard]] std::pair<double, double> tmd_parameters(const std::string& metal,
                                                       const std::string& chalcogen);

/// The dichalcogenides this build has numbers for, as "MoS2" and so on.
[[nodiscard]] std::vector<std::string> known_tmds();

/// Set the vacuum gap of an existing slab or sheet along z, recentring it.
///
/// The gap is measured between the topmost atom and the bottom atom of the
/// next periodic image, so it is the number that actually decides whether the
/// layers can see each other.
void set_vacuum(Atoms& atoms, double vacuum);

}  // namespace fastatoms::build
