// SPDX-License-Identifier: MIT
#include "fastatoms/core/cell.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace fastatoms {
namespace {

constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;

/// A cell whose |determinant| falls below this (in angstrom^3) is treated as
/// non-invertible; no physical cell is anywhere near that small.
constexpr double kSingularVolume = 1e-12;

}  // namespace

void Cell::refresh() noexcept {
    determinant_ = matrix_.determinant();
    singular_ = std::abs(determinant_) < kSingularVolume;
    inverse_ = singular_ ? Mat3{} : matrix_.inverse();
}

Cell Cell::from_parameters(double a, double b, double c, double alpha, double beta, double gamma,
                          const std::array<bool, 3>& pbc) {
    if (a <= 0.0 || b <= 0.0 || c <= 0.0) {
        throw std::invalid_argument("Cell::from_parameters: lengths must be positive");
    }
    const double ca = std::cos(alpha * kDegToRad);
    const double cb = std::cos(beta * kDegToRad);
    const double cg = std::cos(gamma * kDegToRad);
    const double sg = std::sin(gamma * kDegToRad);
    if (std::abs(sg) < 1e-14) {
        throw std::invalid_argument("Cell::from_parameters: gamma must not be 0 or 180 degrees");
    }

    const double cx = cb;
    const double cy = (ca - cb * cg) / sg;
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
    if (singular_) {
        return {0.0, 0.0, 0.0};
    }
    // The reciprocal vector b_k satisfies a_k . b_k = 1 and is orthogonal to
    // the other two lattice vectors, so 1 / |b_k| is exactly the spacing
    // between the periodic images of the plane spanned by them.
    Vec3 h;
    for (std::size_t k = 0; k < 3; ++k) {
        const double n = inverse_.col(k).norm();
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
    std::array<int, 3> ignored{};
    return minimum_image(dr, ignored);
}

Vec3 Cell::minimum_image(const Vec3& dr, std::array<int, 3>& offset) const noexcept {
    offset = {0, 0, 0};
    if (singular_ || !any_pbc()) {
        return dr;
    }

    // Round the fractional coordinates to the nearest image, then check the
    // surrounding images explicitly. Rounding alone is only correct for
    // near-orthogonal cells; the explicit search makes the result exact for
    // any reduced triclinic cell.
    Vec3 s = scaled(dr);
    std::array<int, 3> base{0, 0, 0};
    for (std::size_t k = 0; k < 3; ++k) {
        if (pbc_[k]) {
            base[k] = -static_cast<int>(std::nearbyint(s[k]));
            s[k] += base[k];
        }
    }

    const int lo0 = pbc_[0] ? -1 : 0, hi0 = pbc_[0] ? 1 : 0;
    const int lo1 = pbc_[1] ? -1 : 0, hi1 = pbc_[1] ? 1 : 0;
    const int lo2 = pbc_[2] ? -1 : 0, hi2 = pbc_[2] ? 1 : 0;

    Vec3 best = cartesian(s);
    double best2 = best.norm2();
    std::array<int, 3> best_shift{0, 0, 0};

    for (int d0 = lo0; d0 <= hi0; ++d0) {
        for (int d1 = lo1; d1 <= hi1; ++d1) {
            for (int d2 = lo2; d2 <= hi2; ++d2) {
                if (d0 == 0 && d1 == 0 && d2 == 0) {
                    continue;
                }
                const Vec3 candidate =
                    cartesian(Vec3{s.x + d0, s.y + d1, s.z + d2});
                const double n2 = candidate.norm2();
                if (n2 < best2) {
                    best2 = n2;
                    best = candidate;
                    best_shift = {d0, d1, d2};
                }
            }
        }
    }

    for (std::size_t k = 0; k < 3; ++k) {
        offset[k] = base[k] + best_shift[k];
    }
    return best;
}

double Cell::maximum_unique_cutoff() const noexcept {
    if (!any_pbc()) {
        return std::numeric_limits<double>::infinity();
    }
    if (singular_) {
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
    std::ostringstream out;
    out.precision(6);
    out << "Cell([";
    for (std::size_t r = 0; r < 3; ++r) {
        const Vec3 v = matrix_.row(r);
        out << (r ? ", [" : "[") << v.x << ", " << v.y << ", " << v.z << "]";
    }
    out << "], pbc=[" << (pbc_[0] ? "True" : "False") << ", " << (pbc_[1] ? "True" : "False")
        << ", " << (pbc_[2] ? "True" : "False") << "])";
    return out.str();
}

}  // namespace fastatoms
