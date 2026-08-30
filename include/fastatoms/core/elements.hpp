// SPDX-License-Identifier: MIT
#pragma once

/// \file elements.hpp
/// \brief Periodic-table reference data: symbols, standard atomic weights and
///        covalent radii.

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace fastatoms {

/// Highest supported atomic number (oganesson).
inline constexpr int kMaxAtomicNumber = 118;

/// Atomic number reserved for a dummy / ghost site.
inline constexpr int kDummyAtomicNumber = 0;

/// Chemical symbol for an atomic number, or "X" for Z = 0.
/// \throws std::out_of_range if `z` is outside [0, kMaxAtomicNumber].
[[nodiscard]] std::string_view symbol_from_number(int z);

/// Atomic number for a chemical symbol.
///
/// Matching is exact first, then falls back to canonical capitalisation, so
/// "si", "SI" and "Si" all resolve to 14. This tolerance exists because
/// hand-written YAML input is a primary entry point.
///
/// \throws std::invalid_argument if the symbol is not a known element.
[[nodiscard]] int number_from_symbol(std::string_view symbol);

/// Like `number_from_symbol` but returns -1 instead of throwing.
[[nodiscard]] int try_number_from_symbol(std::string_view symbol) noexcept;

/// Standard atomic weight in atomic mass units. Returns 0 for Z = 0.
[[nodiscard]] double atomic_mass(int z);

/// Cordero (2008) covalent radius in angstrom.
/// Returns 0.0 for elements with no tabulated value (Z = 0 and Z > 96).
[[nodiscard]] double covalent_radius(int z);

/// Per-atom cutoffs derived from covalent radii, as used for bond detection.
///
/// Each entry is `mult * covalent_radius(z)`; elements without a tabulated
/// radius fall back to `fallback`. Two atoms are considered bonded when their
/// distance is below the sum of their cutoffs.
[[nodiscard]] std::vector<double> natural_cutoffs(const std::vector<int>& numbers,
                                                  double mult = 1.0, double fallback = 0.2);

}  // namespace fastatoms
