// SPDX-License-Identifier: MIT
#include "fastatoms/core/cell.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fastatoms {
namespace {

constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;

/// A cell whose |determinant| falls below this (in angstrom^3) is treated as
/// non-invertible; no physical cell is anywhere near that small.
constexpr double kSingularVolume = 1e-12;

/// Complete the zero rows of a lattice matrix the way ASE's
/// `ase.geometry.cell.complete_cell` does, so that fractional coordinates are
/// defined for cells that are periodic only in the directions they span.
///
/// - No zero row: returned unchanged.
/// - One zero row: the row becomes the unit normal of the other two (ASE).
/// - Two zero rows: the nonzero row is kept in place and the two missing rows
///   become an orthonormal basis of its complement, the first one flipped if
///   needed to make the determinant positive. ASE uses an SVD here, which can
///   move the nonzero row to another index; keeping it in place makes the
///   coordinates of the periodic direction refer to the same row as the input.
/// - All rows zero, or rows that do not complete to an invertible matrix:
///   returned unchanged, so `has_scaled_coordinates()` is false.
Mat3 complete_missing_rows(const Mat3& matrix) noexcept {
    std::array<bool, 3> missing{};
    std::size_t count = 0;
    for (std::size_t k = 0; k < 3; ++k) {
        const Vec3 row = matrix.row(k);
        missing[k] = row.x == 0.0 && row.y == 0.0 && row.z == 0.0;
        count += missing[k] ? 1 : 0;
    }
    if (count == 0 || count == 3) {
        return matrix;
    }

    Mat3 out = matrix;
    if (count == 1) {
        std::size_t i = 0;
        while (!missing[i]) {
            ++i;
        }
        const Vec3 normal = cross(matrix.row((i + 1) % 3), matrix.row((i + 2) % 3));
        const double n = normal.norm();
        if (n > 0.0) {
            out.set_row(i, normal / n);
        }
        return out;
    }

    // Two zero rows: exactly one nonzero row remains.
    std::size_t nonzero = 0;
    while (missing[nonzero]) {
        ++nonzero;
    }
    const Vec3 n = matrix.row(nonzero).normalized();
    // Start from the Cartesian axis least aligned with `n`, so that the first
    // completion vector is well conditioned.
    std::size_t axis = 0;
    for (std::size_t k = 1; k < 3; ++k) {
        if (std::abs(n[k]) < std::abs(n[axis])) {
            axis = k;
        }
    }
    Vec3 e{};
    e[axis] = 1.0;
    const Vec3 c2 = (e - dot(e, n) * n).normalized();
    const Vec3 c3 = cross(n, c2);

    std::size_t first = 0;
    while (!missing[first]) {
        ++first;
    }
    std::size_t second = first + 1;
    while (!missing[second]) {
        ++second;
    }
    out.set_row(first, c2);
    out.set_row(second, c3);
    if (out.determinant() < 0.0) {
        out.set_row(first, -c2);
    }
    return out;
}

/// Shortest decimal rendering of `value` through a printf `format`. Unlike a
/// fixed buffer, this never truncates: `snprintf` reports the length the output
/// needs, which can be far more than any buffer (`%.4f` of 1e40 has 46 digits).
std::string format_double(const char* format, double value) {
    const int n = std::snprintf(nullptr, 0, format, value);
    if (n < 0) {
        throw std::runtime_error("Cell::to_string: number formatting failed");
    }
    std::vector<char> buffer(static_cast<std::size_t>(n) + 1);
    std::snprintf(buffer.data(), buffer.size(), format, value);
    return std::string(buffer.data(), static_cast<std::size_t>(n));
}

}  // namespace

void Cell::refresh() noexcept {
    determinant_ = matrix_.determinant();
    singular_ = std::abs(determinant_) < kSingularVolume;
    inverse_ = singular_ ? Mat3{} : matrix_.inverse();

    // Zero rows only occur in singular matrices, so a regular matrix is its own
    // completion.
    completed_ = singular_ ? complete_missing_rows(matrix_) : matrix_;
    const double completed_det = completed_.determinant();
    scaled_ok_ = std::abs(completed_det) >= kSingularVolume;
    completed_inverse_ = scaled_ok_ ? completed_.inverse() : Mat3{};
}

Cell Cell::from_parameters(double a, double b, double c, double alpha, double beta, double gamma,
                          const std::array<bool, 3>& pbc) {
    if (a <= 0.0 || b <= 0.0 || c <= 0.0) {
        throw std::invalid_argument("Cell::from_parameters: lengths must be positive");
    }
    // Right angles are exact, as in ASE's cellpar_to_cell: cos(90 deg) in
    // floating point is 6e-17, which would otherwise leave noise in every
    // orthogonal cell and in everything written from it.
    const auto cos_deg = [](double angle) {
        return std::abs(std::abs(angle) - 90.0) < 1e-10 ? 0.0 : std::cos(angle * kDegToRad);
    };
    const double ca = cos_deg(alpha);
    const double cb = cos_deg(beta);
    const double cg = cos_deg(gamma);
    const double sg = std::abs(std::abs(gamma) - 90.0) < 1e-10 ? 1.0 : std::sin(gamma * kDegToRad);
    if (std::abs(sg) < 1e-14) {
        throw std::invalid_argument("Cell::from_parameters: gamma must not be 0 or 180 degrees");
    }

    const double cx = cb;
    double cy = (ca - cb * cg) / sg;
    if (std::abs(cy) < 1e-15) {
        cy = 0.0;
    }
    const double cz2 = 1.0 - cx * cx - cy * cy;
    if (cz2 <= 0.0) {
        throw std::invalid_argument(
            "Cell::from_parameters: the given angles do not describe a valid cell");
    }

    const Vec3 a1{a, 0.0, 0.0};
    const Vec3 a2{b * cg, b * sg, 0.0};
    const Vec3 a3{c * cx, c * cy, c * std::sqrt(cz2)};
    return Cell(Mat3::from_rows(a1, a2, a3), pbc);
}

Vec3 Cell::lengths() const noexcept {
    return {matrix_.row(0).norm(), matrix_.row(1).norm(), matrix_.row(2).norm()};
}

Vec3 Cell::angles() const noexcept {
    const Vec3 a = matrix_.row(0);
    const Vec3 b = matrix_.row(1);
    const Vec3 c = matrix_.row(2);
    const double la = a.norm();
    const double lb = b.norm();
    const double lc = c.norm();

    auto angle = [](const Vec3& u, const Vec3& v, double lu, double lv) -> double {
        if (lu == 0.0 || lv == 0.0) {
            return 90.0;
        }
        const double x = std::clamp(dot(u, v) / (lu * lv), -1.0, 1.0);
        return std::acos(x) * kRadToDeg;
    };

    return {angle(b, c, lb, lc), angle(a, c, la, lc), angle(a, b, la, lb)};
}

Vec3 Cell::heights() const noexcept {
    if (!scaled_ok_) {
        return {0.0, 0.0, 0.0};
    }
    // The reciprocal vector b_k satisfies a_k . b_k = 1 and is orthogonal to
    // the other two lattice vectors, so 1 / |b_k| is exactly the spacing
    // between the periodic images of the plane spanned by them. For a
    // completed cell the completion vectors are orthogonal to the periodic
    // ones, so the periodic heights are unchanged by the completion.
    Vec3 h;
    for (std::size_t k = 0; k < 3; ++k) {
        const double n = completed_inverse_.col(k).norm();
        h[k] = n > 0.0 ? 1.0 / n : 0.0;
    }
    return h;
}

bool Cell::is_orthorhombic(double tol) const noexcept {
    const Vec3 a = matrix_.row(0);
    const Vec3 b = matrix_.row(1);
    const Vec3 c = matrix_.row(2);
    return std::abs(dot(a, b)) <= tol && std::abs(dot(a, c)) <= tol && std::abs(dot(b, c)) <= tol;
}

Vec3 Cell::minimum_image(const Vec3& dr) const noexcept {
    std::array<double, 3> ignored{};
    return minimum_image_search(dr, ignored);
}

Vec3 Cell::minimum_image(const Vec3& dr, std::array<int, 3>& offset) const {
    std::array<double, 3> shift{};
    const Vec3 image = minimum_image_search(dr, shift);

    // Validate every component before writing anything. The bounds are exact
    // doubles and every shift is integer-valued, so the cast below is defined.
    constexpr double kLow = static_cast<double>(std::numeric_limits<int>::min());
    constexpr double kHigh = static_cast<double>(std::numeric_limits<int>::max());
    std::array<int, 3> out{0, 0, 0};
    for (std::size_t k = 0; k < 3; ++k) {
        if (!(shift[k] >= kLow && shift[k] <= kHigh)) {
            throw std::overflow_error(
                "Cell::minimum_image: the periodic image offset does not fit in int");
        }
        out[k] = static_cast<int>(shift[k]);
    }
    offset = out;
    return image;
}

Vec3 Cell::minimum_image_search(const Vec3& dr, std::array<double, 3>& shift) const noexcept {
    shift = {0.0, 0.0, 0.0};
    if (!scaled_ok_ || !any_pbc()) {
        return dr;
    }

    // Round the periodic fractional coordinates to the nearest image, then check
    // the surrounding images explicitly. Rounding alone is only correct for
    // near-orthogonal cells; the explicit search makes the result exact for any
    // reduced triclinic cell. Only periodic directions are searched, so a
    // non-periodic completion vector is never shifted.
    Vec3 s = scaled(dr);
    std::array<double, 3> base{0.0, 0.0, 0.0};
    for (std::size_t k = 0; k < 3; ++k) {
        if (pbc_[k]) {
            base[k] = -std::round(s[k]);
            s[k] += base[k];
        }
    }

    const int lo0 = pbc_[0] ? -1 : 0, hi0 = pbc_[0] ? 1 : 0;
    const int lo1 = pbc_[1] ? -1 : 0, hi1 = pbc_[1] ? 1 : 0;
    const int lo2 = pbc_[2] ? -1 : 0, hi2 = pbc_[2] ? 1 : 0;

    Vec3 best = cartesian(s);
    double best2 = best.norm2();
    std::array<double, 3> best_shift{0.0, 0.0, 0.0};

    for (int d0 = lo0; d0 <= hi0; ++d0) {
        for (int d1 = lo1; d1 <= hi1; ++d1) {
            for (int d2 = lo2; d2 <= hi2; ++d2) {
                if (d0 == 0 && d1 == 0 && d2 == 0) {
                    continue;
                }
                const Vec3 candidate = cartesian(Vec3{s.x + d0, s.y + d1, s.z + d2});
                const double n2 = candidate.norm2();
                if (n2 < best2) {
                    best2 = n2;
                    best = candidate;
                    best_shift = {static_cast<double>(d0), static_cast<double>(d1),
                                  static_cast<double>(d2)};
                }
            }
        }
    }

    for (std::size_t k = 0; k < 3; ++k) {
        shift[k] = base[k] + best_shift[k];
    }
    return best;
}

double Cell::maximum_unique_cutoff() const noexcept {
    if (!any_pbc()) {
        return std::numeric_limits<double>::infinity();
    }
    if (!scaled_ok_) {
        return 0.0;
    }
    const Vec3 h = heights();
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < 3; ++k) {
        if (pbc_[k]) {
            best = std::min(best, 0.5 * h[k]);
        }
    }
    return best;
}

std::string Cell::to_string() const {
    std::string out = "Cell([";
    for (std::size_t r = 0; r < 3; ++r) {
        const Vec3 v = matrix_.row(r);
        out += (r ? ", [" : "[") + format_double("%.6g", v.x) + ", " +
               format_double("%.6g", v.y) + ", " + format_double("%.6g", v.z) + "]";
    }
    out += std::string("], pbc=[") + (pbc_[0] ? "True" : "False") + ", " +
           (pbc_[1] ? "True" : "False") + ", " + (pbc_[2] ? "True" : "False") + "])";
    return out;
}

}  // namespace fastatoms
