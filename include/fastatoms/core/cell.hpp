// SPDX-License-Identifier: MIT
#pragma once

/// \file cell.hpp
/// \brief Periodic simulation cell: lattice vectors, boundary conditions and
///        coordinate transformations.

#include <array>
#include <string>

#include "fastatoms/core/vec3.hpp"

namespace fastatoms {

/// A (possibly partially periodic, possibly triclinic) simulation cell.
///
/// The three lattice vectors are stored as the *rows* of a 3x3 matrix, so that
/// `cartesian = scaled * matrix`. The inverse matrix is cached on every
/// mutation because converting to scaled coordinates is on the hot path of
/// wrapping, neighbour searches and minimum-image distances.
///
/// Periodic boundary conditions live here rather than on `Atoms` so that there
/// is a single source of truth; `Atoms` forwards to this object.
class Cell {
public:
    Cell() = default;

    explicit Cell(const Mat3& matrix, const std::array<bool, 3>& pbc = {false, false, false})
        : matrix_(matrix), pbc_(pbc) {
        refresh();
    }

    Cell(const Vec3& a1, const Vec3& a2, const Vec3& a3,
         const std::array<bool, 3>& pbc = {true, true, true})
        : Cell(Mat3::from_rows(a1, a2, a3), pbc) {}

    /// Cubic cell of edge `a`, periodic in all directions.
    static Cell cubic(double a) {
        return Cell(Mat3::diagonal(a, a, a), {true, true, true});
    }

    /// Orthorhombic cell, periodic in all directions.
    static Cell orthorhombic(double a, double b, double c) {
        return Cell(Mat3::diagonal(a, b, c), {true, true, true});
    }

    /// Build a cell from the six crystallographic parameters. Angles are in
    /// degrees. Uses the standard convention: a along x, b in the xy plane.
    static Cell from_parameters(double a, double b, double c, double alpha, double beta,
                                double gamma,
                                const std::array<bool, 3>& pbc = {true, true, true});

    /// \name Lattice matrix
    /// @{
    [[nodiscard]] const Mat3& matrix() const noexcept { return matrix_; }
    [[nodiscard]] const Mat3& inverse_matrix() const noexcept { return inverse_; }

    void set_matrix(const Mat3& matrix) {
        matrix_ = matrix;
        refresh();
    }

    [[nodiscard]] Vec3 vector(std::size_t i) const noexcept { return matrix_.row(i); }
    void set_vector(std::size_t i, const Vec3& v) {
        matrix_.set_row(i, v);
        refresh();
    }
    /// @}

    /// \name Boundary conditions
    /// @{
    [[nodiscard]] const std::array<bool, 3>& pbc() const noexcept { return pbc_; }
    void set_pbc(const std::array<bool, 3>& pbc) noexcept { pbc_ = pbc; }
    void set_pbc(bool pbc) noexcept { pbc_ = {pbc, pbc, pbc}; }
    [[nodiscard]] bool any_pbc() const noexcept { return pbc_[0] || pbc_[1] || pbc_[2]; }
    [[nodiscard]] bool all_pbc() const noexcept { return pbc_[0] && pbc_[1] && pbc_[2]; }
    /// @}

    /// True if the cell matrix is exactly zero (an isolated molecule).
    [[nodiscard]] bool is_zero() const noexcept { return matrix_.is_zero(); }

    /// True if the *raw* lattice matrix is invertible, i.e. the three rows span
    /// three dimensions. A cell with a zero row (rank-deficient, periodic only
    /// in the directions it spans) reports false here even though it has
    /// scaled coordinates; see `has_scaled_coordinates()`.
    [[nodiscard]] bool is_invertible() const noexcept { return !singular_; }

    /// True if fractional coordinates are defined for this cell. That holds
    /// when the matrix is invertible, and also when it has one or two zero rows
    /// that can be completed with orthonormal vectors exactly as ASE's
    /// `complete_cell` does. A zero cell (all rows zero) and a cell with
    /// linearly dependent nonzero rows have no fractional coordinates.
    [[nodiscard]] bool has_scaled_coordinates() const noexcept { return scaled_ok_; }

    /// The lattice matrix with zero rows completed (see
    /// `has_scaled_coordinates()`). Equal to `matrix()` when the matrix is
    /// invertible.
    [[nodiscard]] const Mat3& completed_matrix() const noexcept { return completed_; }

    /// Signed cell volume; `volume()` returns its absolute value.
    [[nodiscard]] double signed_volume() const noexcept { return determinant_; }
    [[nodiscard]] double volume() const noexcept {
        return determinant_ < 0.0 ? -determinant_ : determinant_;
    }

    /// Lengths |a1|, |a2|, |a3|.
    [[nodiscard]] Vec3 lengths() const noexcept;

    /// Angles alpha, beta, gamma in degrees.
    [[nodiscard]] Vec3 angles() const noexcept;

    /// Perpendicular widths of the cell along each lattice direction.
    ///
    /// `heights()[k]` is the distance between the two periodic images of the
    /// plane spanned by the other two lattice vectors. This is the quantity
    /// that bounds a usable minimum-image cutoff (cutoff <= min(height) / 2).
    [[nodiscard]] Vec3 heights() const noexcept;

    /// Reciprocal lattice vectors as rows, *without* the 2*pi factor, i.e. the
    /// rows of the inverse-transpose. `dot(reciprocal().row(i), vector(j))` is
    /// the Kronecker delta.
    [[nodiscard]] Mat3 reciprocal() const noexcept { return inverse_.transposed(); }

    /// True if all lattice vectors are mutually orthogonal (within `tol`).
    [[nodiscard]] bool is_orthorhombic(double tol = 1e-10) const noexcept;

    /// \name Coordinate transformations
    /// @{

    /// Cartesian -> fractional, against `completed_matrix()`. Returns the input
    /// unchanged when there is no fractional basis (`!has_scaled_coordinates()`).
    [[nodiscard]] FASTATOMS_ALWAYS_INLINE Vec3 scaled(const Vec3& cartesian) const noexcept {
        return scaled_ok_ ? cartesian * completed_inverse_ : cartesian;
    }

    /// Fractional -> Cartesian, against `completed_matrix()`.
    [[nodiscard]] FASTATOMS_ALWAYS_INLINE Vec3 cartesian(const Vec3& scaled) const noexcept {
        return scaled * completed_;
    }

    /// Fold a fractional coordinate into [0, 1) along every periodic direction.
    ///
    /// `eps` nudges coordinates that sit exactly on the upper boundary back to
    /// zero, which keeps round-off from pushing atoms out of the cell.
    [[nodiscard]] FASTATOMS_ALWAYS_INLINE Vec3 wrap_scaled(Vec3 s, double eps = 1e-7) const noexcept {
        for (std::size_t k = 0; k < 3; ++k) {
            if (pbc_[k]) {
                double v = s[k] + eps;
                v -= std::floor(v);
                s[k] = v - eps;
            }
        }
        return s;
    }

    /// Fold a Cartesian position into the cell.
    [[nodiscard]] Vec3 wrap(const Vec3& position, double eps = 1e-7) const noexcept {
        if (!any_pbc() || !scaled_ok_) {
            return position;
        }
        return cartesian(wrap_scaled(scaled(position), eps));
    }
    /// @}

    /// Shortest periodic image of the displacement vector `dr`.
    ///
    /// Only the periodic directions are shifted: the periodic fractional
    /// coordinates are rounded to the nearest image and the surrounding images
    /// are then checked explicitly, which makes the result exact for any
    /// reduced triclinic cell. A non-periodic lattice direction (for example a
    /// zero row completed by `completed_matrix()`) is never shifted. Returns
    /// `dr` unchanged when there is no periodic direction or no fractional basis.
    [[nodiscard]] Vec3 minimum_image(const Vec3& dr) const noexcept;

    /// Minimum-image displacement together with the integer cell offset that
    /// produced it, so that `dr - cartesian(offset)` equals the result.
    ///
    /// \throws std::overflow_error if an offset component does not fit in `int`
    ///         (the displacement itself is then not representable as an offset).
    Vec3 minimum_image(const Vec3& dr, std::array<int, 3>& offset) const;

    /// Largest cutoff for which `minimum_image` is guaranteed unique, i.e.
    /// half the smallest perpendicular width over the periodic directions.
    /// Returns infinity when nothing is periodic and zero when there is no
    /// fractional basis.
    [[nodiscard]] double maximum_unique_cutoff() const noexcept;

    /// Human-readable one-line summary, used by the Python `repr`.
    [[nodiscard]] std::string to_string() const;

private:
    void refresh() noexcept;
    /// Minimum image of `dr`; `shift` receives the (integer-valued) lattice
    /// shift in each periodic direction, kept as doubles so that it cannot
    /// overflow while it is computed.
    Vec3 minimum_image_search(const Vec3& dr, std::array<double, 3>& shift) const noexcept;

    Mat3 matrix_{};
    Mat3 inverse_{};            ///< Inverse of `matrix_`; zero when singular.
    Mat3 completed_{};          ///< `matrix_` with zero rows completed.
    Mat3 completed_inverse_{};  ///< Inverse of `completed_`; zero when unavailable.
    std::array<bool, 3> pbc_{{false, false, false}};
    double determinant_{0.0};
    bool singular_{true};       ///< `matrix_` is singular.
    bool scaled_ok_{false};     ///< `completed_` is invertible.
};

}  // namespace fastatoms
