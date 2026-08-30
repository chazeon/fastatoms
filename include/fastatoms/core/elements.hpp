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

/// What an element crystallises as at room temperature and pressure.
///
/// This is what lets `build::bulk("Cu")` work with nothing but a symbol: the
/// lattice and its constant are properties of the element, not of the call.
/// The values are the experimental ambient-condition ones, and they are a
/// starting point for a calculation rather than an answer - a relaxation with
/// any given functional will move them by a per cent or two.
struct ReferenceState {
    /// "fcc", "bcc", "hcp", "diamond", "sc", "tetragonal", "orthorhombic",
    /// "rhombohedral", "hexagonal", "cubic", "diatomic", or "" when the
    /// element has no tabulated ambient solid (the noble-gas-like cases that
    /// are liquid or gaseous, and the short-lived synthetics).
    std::string_view symmetry{};

    /// The conventional lattice constant in angstrom. Zero when unknown; for
    /// "diatomic" it is the bond length instead.
    double a{0.0};

    /// c/a for hexagonal, hcp and tetragonal; zero otherwise.
    double covera{0.0};

    /// The rhombohedral angle in degrees; zero otherwise.
    double alpha{0.0};

    /// b/a for orthorhombic; zero otherwise.
    double bovera{0.0};
};

/// The reference state for an atomic number.
///
/// Returns an entry with an empty `symmetry` when nothing is tabulated, which
/// is the caller's cue to ask for the lattice explicitly rather than guess.
[[nodiscard]] const ReferenceState& reference_state(int z);

/// Per-atom cutoffs derived from covalent radii, as used for bond detection.
///
/// Each entry is `mult * covalent_radius(z)`; elements without a tabulated
/// radius fall back to `fallback`. Two atoms are considered bonded when their
/// distance is below the sum of their cutoffs.
[[nodiscard]] std::vector<double> natural_cutoffs(const std::vector<int>& numbers,
                                                  double mult = 1.0, double fallback = 0.2);

}  // namespace fastatoms
