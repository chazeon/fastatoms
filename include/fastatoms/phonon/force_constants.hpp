// SPDX-License-Identifier: MIT
#pragma once

/// \file force_constants.hpp
/// \brief The harmonic force constant matrix.

#include <cstddef>
#include <string>
#include <vector>

#include "fastatoms/core/vec3.hpp"
#include "fastatoms/phonon/supercell.hpp"

namespace fastatoms::phonon {

/// The second derivatives of the energy, `Phi_ab(i,j) = d2E / du_ia du_jb`,
/// in eV/angstrom^2.
///
/// Only rows for the atoms of the first unit cell are stored. Translational
/// periodicity makes the rest redundant - `Phi(i+L, j+L) = Phi(i, j)` - and
/// those rows are all the dynamical matrix ever needs.
class ForceConstants {
public:
    ForceConstants() = default;
    ForceConstants(std::size_t rows, std::size_t columns)
        : rows_(rows), columns_(columns), blocks_(rows * columns) {}

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t columns() const noexcept { return columns_; }
    [[nodiscard]] bool empty() const noexcept { return blocks_.empty(); }

    [[nodiscard]] Mat3& block(std::size_t i, std::size_t j) {
        return blocks_[i * columns_ + j];
    }
    [[nodiscard]] const Mat3& block(std::size_t i, std::size_t j) const {
        return blocks_[i * columns_ + j];
    }

    /// Impose translational invariance: shifting the whole crystal costs no
    /// energy, so every row must sum to zero. Finite displacements satisfy
    /// that only to within their numerical error, and the residue would
    /// otherwise show up as acoustic modes with the wrong frequency at the
    /// zone centre.
    void apply_acoustic_sum_rule();

    /// Largest entry of `sum_j Phi(i,j)`, in eV/angstrom^2. A measure of how
    /// converged the calculation is; zero after the sum rule is applied.
    [[nodiscard]] double acoustic_sum_violation() const;

    /// Row-major `3*rows` by `3*columns`, for handing to numpy.
    [[nodiscard]] std::vector<double> to_array() const;

    [[nodiscard]] std::string to_string() const;

private:
    std::size_t rows_{0};
    std::size_t columns_{0};
    std::vector<Mat3> blocks_{};
};

}  // namespace fastatoms::phonon
