// SPDX-License-Identifier: MIT
#pragma once

/// \file units.hpp
/// \brief Physical constants in the ASE unit system.
///
/// FastAtoms uses the same base units as ASE, so values can be passed between
/// the two without conversion:
///   - length: angstrom
///   - energy: electronvolt
///   - mass:   atomic mass unit (Dalton)
///   - time:   angstrom * sqrt(amu / eV)  ~= 10.1805 fs
///   - charge: elementary charge
///   - temperature: kelvin

namespace fastatoms::units {

/// Boltzmann constant, eV/K (CODATA 2018, exact by SI definition).
inline constexpr double kB = 8.617333262145e-5;

/// One femtosecond expressed in the internal time unit.
///
/// The internal time unit is angstrom * sqrt(amu / eV) = 1e-5 * sqrt(_e / _amu)
/// seconds ~= 10.1805 fs, so a femtosecond is a bit under a tenth of it. The
/// value matches ASE's `units.fs` to the last digit.
inline constexpr double fs = 9.822694788464063e-2;

/// One picosecond expressed in the internal time unit.
inline constexpr double ps = 1.0e3 * fs;

/// One second expressed in the internal time unit.
inline constexpr double second = 1.0e15 * fs;

/// Reduced Planck constant in eV * (internal time).
/// hbar is 0.6582119569... eV*fs; the factor converts that to internal time.
inline constexpr double hbar = 0.6582119569509066 * fs;

/// Bohr radius in angstrom.
inline constexpr double Bohr = 0.529177210903;

/// Hartree in electronvolt.
inline constexpr double Hartree = 27.211386245988;

/// Rydberg in electronvolt.
inline constexpr double Rydberg = 0.5 * Hartree;

/// kcal/mol in electronvolt.
inline constexpr double kcal_per_mol = 4.336410390059322e-2;

/// kJ/mol in electronvolt.
inline constexpr double kJ_per_mol = 1.0364269656262173e-2;

/// GPa in eV/angstrom^3.
inline constexpr double GPa = 1.0 / 160.21766208;

}  // namespace fastatoms::units
