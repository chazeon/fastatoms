// SPDX-License-Identifier: MIT
#pragma once

/// \file water.hpp
/// \brief The rigid three-site water models, and the structures they describe.
///
/// SPC and SPC/E are the same shape - one Lennard-Jones centre on the oxygen,
/// three point charges, a fixed geometry - and differ only in their numbers.
/// Keeping them as data in one table rather than as two classes means a third
/// model is a row, not a file.

#include <array>
#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/forcefield/topology.hpp"

namespace fastatoms::ff {

/// The parameters of a rigid three-site water model.
///
/// Charges are in elementary charges, lengths in angstrom, energies in eV.
/// The published values are in kcal/mol and are converted on the way in, so
/// the table below can be read against the original papers.
struct WaterModel {
    std::string name{};

    double q_oxygen{0.0};       ///< e
    double q_hydrogen{0.0};     ///< e
    double epsilon_oxygen{0.0}; ///< eV, on the oxygen only
    double sigma_oxygen{0.0};   ///< angstrom

    double bond_length{0.0};    ///< O-H, angstrom
    double angle{0.0};          ///< H-O-H, degrees

    /// A constant added per molecule. SPC/E carries a self-energy polarisation
    /// correction of 5.22 kcal/mol, the work of polarising a gas-phase
    /// molecule up to the enhanced dipole the model uses; without it the
    /// computed vaporisation enthalpy is wrong by that amount. It shifts no
    /// force, only the energy.
    double self_energy{0.0};    ///< eV per molecule

    std::string reference{};

    /// The H-H distance implied by the bond length and angle. This is the
    /// third constraint that makes the molecule rigid.
    [[nodiscard]] double hydrogen_distance() const noexcept;

    /// The dipole moment of the isolated model geometry, in debye - the number
    /// these models are usually compared by (SPC 2.27 D, SPC/E 2.35 D).
    [[nodiscard]] double dipole_moment() const noexcept;

    [[nodiscard]] std::string to_string() const;
};

/// Look a model up by name: `spc`, `spce` (aliases `spc/e`, `spc_e`) or
/// `tip3p`. Matching ignores case, hyphens, underscores and slashes.
/// \throws ForcefieldError if the name is not known.
[[nodiscard]] const WaterModel& water_model(std::string_view name);

/// Every model name, in alphabetical order.
[[nodiscard]] std::vector<std::string> water_model_names();

/// One water molecule at its model geometry, oxygen first, in a cell of the
/// given size. Useful as a building block and as a test case.
[[nodiscard]] Atoms water_molecule(const WaterModel& model, double box = 20.0);

/// A cubic box of water molecules on a lattice, at a chosen density.
///
/// Molecules are placed on a face-centred cubic lattice - the densest simple
/// arrangement, so no two start unphysically close - with random orientations
/// from `seed`. This is a starting configuration to be equilibrated, not an
/// equilibrium structure.
///
/// \param molecules how many to place, rounded up to fill the lattice: 4, 32,
///                  108, 256, ... Filling it only partly would leave the
///                  molecules closer together than the density calls for.
/// \param density   g/cm^3, 0.997 by default (liquid water at 25 C)
[[nodiscard]] Atoms water_box(const WaterModel& model, int molecules, double density = 0.997,
                              std::uint64_t seed = 0);

/// Build the topology of a structure made of rigid water molecules.
///
/// Molecules are found by connectivity, so the atom order does not matter and
/// a box straddling the periodic boundary is handled. Every molecule gets its
/// charges, the oxygen its Lennard-Jones parameters, all three intramolecular
/// pairs are excluded, and three distance constraints are added.
///
/// \throws ForcefieldError if any molecule is not one oxygen and two hydrogens.
[[nodiscard]] Topology water_topology(const Atoms& atoms, const WaterModel& model);

}  // namespace fastatoms::ff
