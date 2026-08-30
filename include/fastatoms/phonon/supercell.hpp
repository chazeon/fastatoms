// SPDX-License-Identifier: MIT
#pragma once

/// \file supercell.hpp
/// \brief Building a supercell from a transformation matrix, and remembering
///        where every atom came from.

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"

namespace fastatoms::phonon {

/// Raised for a malformed supercell request or an inconsistent phonon setup.
class PhononError : public std::runtime_error {
public:
    explicit PhononError(const std::string& message) : std::runtime_error(message) {}
};

/// A supercell together with the bookkeeping a phonon calculation needs.
///
/// The supercell lattice is `cell_supercell = P * cell_unit` for an integer
/// matrix `P`, so the diagonal case `diag(n1, n2, n3)` is the familiar
/// repetition and a general `P` builds cells such as the conventional cubic
/// cell of an fcc primitive lattice.
///
/// Atoms are ordered so that the first `unit_cell_size` of them are the
/// original unit cell at the zero lattice translation. Force constants only
/// ever need rows for those.
struct SupercellMap {
    Atoms atoms{};
    Mat3 transformation{};          ///< P, integer valued
    std::size_t unit_cell_size{0};  ///< atoms per unit cell
    std::size_t cell_count{0};      ///< |det P|

    /// For each supercell atom, which unit-cell atom it is a copy of.
    std::vector<std::size_t> unit_cell_index{};

    /// For each supercell atom, its lattice translation in integer unit-cell
    /// coordinates. This is what enters the phase factor `exp(2 pi i q . n)`.
    std::vector<std::array<int, 3>> lattice_point{};

    /// The same translation in angstrom.
    std::vector<Vec3> lattice_translation{};

    [[nodiscard]] std::size_t size() const noexcept { return atoms.size(); }
    [[nodiscard]] std::string to_string() const;
};

/// Build a supercell from an integer transformation matrix.
///
/// \throws PhononError if `transformation` is not integer valued, or its
///         determinant is not a positive integer.
[[nodiscard]] SupercellMap make_supercell(const Atoms& unit_cell, const Mat3& transformation);

/// The common diagonal case.
[[nodiscard]] SupercellMap make_supercell(const Atoms& unit_cell, int n1, int n2, int n3);

}  // namespace fastatoms::phonon
