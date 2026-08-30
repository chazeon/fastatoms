// SPDX-License-Identifier: MIT
#pragma once

/// \file eigen.hpp
/// \brief A complex Hermitian eigensolver, written out by hand.
///
/// The dynamical matrix is complex Hermitian, and diagonalising it is the last
/// step of every phonon calculation. FastAtoms carries no linear algebra
/// dependency, so the solver lives here: a cyclic Jacobi method, which is a
/// few dozen lines, has no failure modes worth worrying about, and is
/// accurate to full precision on the matrix sizes a phonon calculation
/// produces (3N for a unit cell, so tens to a few hundred).

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

namespace fastatoms::phonon {

/// A dense square complex matrix, stored row-major.
class ComplexMatrix {
public:
    using value_type = std::complex<double>;

    ComplexMatrix() = default;
    explicit ComplexMatrix(std::size_t n) : n_(n), data_(n * n, value_type{}) {}

    /// The identity matrix.
    [[nodiscard]] static ComplexMatrix identity(std::size_t n);

    [[nodiscard]] std::size_t size() const noexcept { return n_; }
    [[nodiscard]] bool empty() const noexcept { return n_ == 0; }

    [[nodiscard]] value_type& operator()(std::size_t i, std::size_t j) {
        return data_[i * n_ + j];
    }
    [[nodiscard]] const value_type& operator()(std::size_t i, std::size_t j) const {
        return data_[i * n_ + j];
    }

    [[nodiscard]] const std::vector<value_type>& data() const noexcept { return data_; }
    [[nodiscard]] std::vector<value_type>& data() noexcept { return data_; }

    /// Replace the matrix by `(A + A^H) / 2`.
    ///
    /// For a dynamical matrix this is not cosmetic: Hermiticity is the
    /// permutation symmetry `Phi_ab(0, L) = Phi_ba(0, -L)^T` of the force
    /// constants, which finite displacements only satisfy to within their
    /// numerical error.
    void hermitize();

    /// Largest absolute deviation from Hermiticity, for diagnostics.
    [[nodiscard]] double hermiticity_error() const;

    [[nodiscard]] std::string to_string() const;

private:
    std::size_t n_{0};
    std::vector<value_type> data_{};
};

/// Eigenvalues in ascending order, with the matching eigenvectors as columns.
struct HermitianEigen {
    std::vector<double> values{};
    ComplexMatrix vectors{};
    /// Jacobi sweeps used; a well-behaved matrix needs fewer than ten.
    int sweeps{0};
    /// True when the off-diagonal norm reached the tolerance.
    bool converged{false};
};

/// Diagonalise a Hermitian matrix by the cyclic Jacobi method.
///
/// Each sweep annihilates every off-diagonal entry in turn with a unitary
/// rotation, chosen by writing the entry as `|z| e^{i phi}` and applying the
/// ordinary real Jacobi angle in the rotated phase. The off-diagonal norm
/// falls quadratically once the matrix is nearly diagonal, so a handful of
/// sweeps reaches machine precision.
///
/// The input is assumed Hermitian; only the upper triangle and the real part
/// of the diagonal are used, so a matrix that is Hermitian to within rounding
/// gives the same answer as one that is exactly so.
[[nodiscard]] HermitianEigen hermitian_eigen(const ComplexMatrix& matrix,
                                             double tolerance = 1e-12,
                                             int max_sweeps = 60);

}  // namespace fastatoms::phonon
