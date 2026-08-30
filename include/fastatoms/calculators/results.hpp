// SPDX-License-Identifier: MIT
#pragma once

/// \file results.hpp
/// \brief What a calculator returns, and the conventions those numbers follow.

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/core/vec3.hpp"
#include "fastatoms/io/parameters.hpp"

namespace fastatoms {

/// A quantity a calculator can be asked for.
enum class Property { Energy, FreeEnergy, Forces, Stress, Charges, Magmom };

[[nodiscard]] std::string_view property_name(Property property);
[[nodiscard]] Property property_from_string(std::string_view name);

/// Energy, forces and stress - the usual request.
[[nodiscard]] const std::vector<Property>& default_properties();

/// Raised when a calculation cannot be set up, run, or parsed.
class CalculatorError : public std::runtime_error {
public:
    explicit CalculatorError(const std::string& message) : std::runtime_error(message) {}
};

/// The outcome of one calculation.
///
/// Units follow ASE throughout:
///   - `energy`, `free_energy`: eV. `energy` is extrapolated to zero smearing
///     where the code reports it; `free_energy` is the quantity the forces are
///     consistent with.
///   - `forces`: eV/angstrom, 3N atom-major.
///   - `stress`: eV/angstrom^3 in Voigt order (xx, yy, zz, yz, xz, xy), with
///     the ASE sign convention sigma_ab = (1/V) dE/d(strain_ab). Positive
///     diagonal components mean the cell would lower its energy by shrinking.
///   - `charges`: elementary charges, one per atom.
///   - `magmom`: total magnetic moment in Bohr magnetons.
struct Results {
    std::optional<double> energy{};
    std::optional<double> free_energy{};
    std::vector<double> forces{};
    std::optional<std::array<double, 6>> stress{};
    std::vector<double> charges{};
    std::optional<double> magmom{};

    /// Code-specific extras: Fermi level, iteration counts, convergence flags.
    /// Reaches Python as a plain dict.
    Value extra{};

    void clear();

    [[nodiscard]] bool has(Property property) const;
};

/// Voigt (xx, yy, zz, yz, xz, xy) to a symmetric 3x3 matrix.
[[nodiscard]] Mat3 voigt_to_matrix(const std::array<double, 6>& voigt);

/// Symmetric 3x3 matrix to Voigt order. The off-diagonal pairs are averaged.
[[nodiscard]] std::array<double, 6> matrix_to_voigt(const Mat3& matrix);

/// Hydrostatic pressure in eV/angstrom^3: -(1/3) trace(sigma).
[[nodiscard]] double pressure_from_stress(const std::array<double, 6>& stress);

}  // namespace fastatoms
