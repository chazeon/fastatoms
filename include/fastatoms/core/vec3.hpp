// SPDX-License-Identifier: MIT
#pragma once

/// \file vec3.hpp
/// \brief Small fixed-size linear algebra primitives (3-vectors, 3x3 matrices).
///
/// These types are deliberately trivial, POD-like and header-only so that the
/// compiler can keep them in registers and vectorise the surrounding loops.

#include <array>
#include <cmath>
#include <cstddef>

namespace fastatoms {

#if defined(_MSC_VER)
#  define FASTATOMS_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#  define FASTATOMS_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#  define FASTATOMS_ALWAYS_INLINE inline
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define FASTATOMS_RESTRICT __restrict__
#elif defined(_MSC_VER)
#  define FASTATOMS_RESTRICT __restrict
#else
#  define FASTATOMS_RESTRICT
#endif

/// A Cartesian 3-vector.
struct Vec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    constexpr Vec3() = default;
    constexpr Vec3(double a, double b, double c) : x(a), y(b), z(c) {}

    /// Construct from a contiguous buffer of at least three doubles.
    static FASTATOMS_ALWAYS_INLINE Vec3 load(const double* p) { return {p[0], p[1], p[2]}; }

    /// Store into a contiguous buffer of at least three doubles.
    FASTATOMS_ALWAYS_INLINE void store(double* p) const {
        p[0] = x;
        p[1] = y;
        p[2] = z;
    }

    constexpr double& operator[](std::size_t i) noexcept {
        return i == 0 ? x : (i == 1 ? y : z);
    }
    constexpr const double& operator[](std::size_t i) const noexcept {
        return i == 0 ? x : (i == 1 ? y : z);
    }

    constexpr Vec3 operator-() const noexcept { return {-x, -y, -z}; }

    constexpr Vec3& operator+=(const Vec3& o) noexcept {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }
    constexpr Vec3& operator-=(const Vec3& o) noexcept {
        x -= o.x;
        y -= o.y;
        z -= o.z;
        return *this;
    }
    constexpr Vec3& operator*=(double s) noexcept {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }
    constexpr Vec3& operator/=(double s) noexcept { return *this *= (1.0 / s); }

    /// Squared Euclidean norm (cheap: no sqrt).
    [[nodiscard]] constexpr double norm2() const noexcept { return x * x + y * y + z * z; }
    [[nodiscard]] double norm() const noexcept { return std::sqrt(norm2()); }

    [[nodiscard]] Vec3 normalized() const noexcept {
        const double n = norm();
        return n > 0.0 ? Vec3{x / n, y / n, z / n} : Vec3{};
    }
};

constexpr Vec3 operator+(Vec3 a, const Vec3& b) noexcept { return a += b; }
constexpr Vec3 operator-(Vec3 a, const Vec3& b) noexcept { return a -= b; }
constexpr Vec3 operator*(Vec3 a, double s) noexcept { return a *= s; }
constexpr Vec3 operator*(double s, Vec3 a) noexcept { return a *= s; }
constexpr Vec3 operator/(Vec3 a, double s) noexcept { return a /= s; }

constexpr bool operator==(const Vec3& a, const Vec3& b) noexcept {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
constexpr bool operator!=(const Vec3& a, const Vec3& b) noexcept { return !(a == b); }

[[nodiscard]] constexpr double dot(const Vec3& a, const Vec3& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] constexpr Vec3 cross(const Vec3& a, const Vec3& b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

[[nodiscard]] inline double distance(const Vec3& a, const Vec3& b) noexcept {
    return (a - b).norm();
}

/// A row-major 3x3 matrix.
///
/// Throughout FastAtoms the *rows* of a cell matrix are the three lattice
/// vectors, matching the convention used by ASE, so that
/// `cartesian = scaled * matrix`.
struct Mat3 {
    /// Row-major storage: m[3 * row + col].
    std::array<double, 9> m{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

    constexpr Mat3() = default;

    constexpr Mat3(double m00, double m01, double m02, double m10, double m11, double m12,
                   double m20, double m21, double m22)
        : m{{m00, m01, m02, m10, m11, m12, m20, m21, m22}} {}

    /// Build a matrix from its three rows (i.e. from three lattice vectors).
    static constexpr Mat3 from_rows(const Vec3& a, const Vec3& b, const Vec3& c) {
        return {a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z};
    }

    static constexpr Mat3 identity() { return {1, 0, 0, 0, 1, 0, 0, 0, 1}; }

    static constexpr Mat3 diagonal(double a, double b, double c) {
        return {a, 0, 0, 0, b, 0, 0, 0, c};
    }

    constexpr double& operator()(std::size_t r, std::size_t c) noexcept { return m[3 * r + c]; }
    constexpr double operator()(std::size_t r, std::size_t c) const noexcept { return m[3 * r + c]; }

    [[nodiscard]] constexpr Vec3 row(std::size_t r) const noexcept {
        return {m[3 * r], m[3 * r + 1], m[3 * r + 2]};
    }
    [[nodiscard]] constexpr Vec3 col(std::size_t c) const noexcept {
        return {m[c], m[3 + c], m[6 + c]};
    }

    constexpr void set_row(std::size_t r, const Vec3& v) noexcept {
        m[3 * r] = v.x;
        m[3 * r + 1] = v.y;
        m[3 * r + 2] = v.z;
    }

    [[nodiscard]] constexpr double determinant() const noexcept {
        return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
               m[2] * (m[3] * m[7] - m[4] * m[6]);
    }

    [[nodiscard]] constexpr Mat3 transposed() const noexcept {
        return {m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]};
    }

    /// Matrix inverse. Returns the zero matrix for a singular matrix; callers
    /// that care should check `determinant()` first.
    [[nodiscard]] constexpr Mat3 inverse() const noexcept {
        const double det = determinant();
        if (det == 0.0) {
            return Mat3{};
        }
        const double inv_det = 1.0 / det;
        Mat3 r;
        r.m[0] = (m[4] * m[8] - m[5] * m[7]) * inv_det;
        r.m[1] = (m[2] * m[7] - m[1] * m[8]) * inv_det;
        r.m[2] = (m[1] * m[5] - m[2] * m[4]) * inv_det;
        r.m[3] = (m[5] * m[6] - m[3] * m[8]) * inv_det;
        r.m[4] = (m[0] * m[8] - m[2] * m[6]) * inv_det;
        r.m[5] = (m[2] * m[3] - m[0] * m[5]) * inv_det;
        r.m[6] = (m[3] * m[7] - m[4] * m[6]) * inv_det;
        r.m[7] = (m[1] * m[6] - m[0] * m[7]) * inv_det;
        r.m[8] = (m[0] * m[4] - m[1] * m[3]) * inv_det;
        return r;
    }

    [[nodiscard]] constexpr bool is_zero() const noexcept {
        for (double v : m) {
            if (v != 0.0) {
                return false;
            }
        }
        return true;
    }
};

/// Row-vector times matrix: `v * M`, i.e. sum_i v_i * M.row(i).
///
/// This is the operation that turns scaled coordinates into Cartesian ones
/// under the "rows are lattice vectors" convention.
[[nodiscard]] FASTATOMS_ALWAYS_INLINE constexpr Vec3 operator*(const Vec3& v,
                                                               const Mat3& M) noexcept {
    return {v.x * M.m[0] + v.y * M.m[3] + v.z * M.m[6],
            v.x * M.m[1] + v.y * M.m[4] + v.z * M.m[7],
            v.x * M.m[2] + v.y * M.m[5] + v.z * M.m[8]};
}

/// Matrix times column-vector: `M * v`.
[[nodiscard]] FASTATOMS_ALWAYS_INLINE constexpr Vec3 operator*(const Mat3& M,
                                                               const Vec3& v) noexcept {
    return {M.m[0] * v.x + M.m[1] * v.y + M.m[2] * v.z,
            M.m[3] * v.x + M.m[4] * v.y + M.m[5] * v.z,
            M.m[6] * v.x + M.m[7] * v.y + M.m[8] * v.z};
}

[[nodiscard]] constexpr Mat3 operator*(const Mat3& A, const Mat3& B) noexcept {
    Mat3 r;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            r.m[3 * i + j] =
                A.m[3 * i] * B.m[j] + A.m[3 * i + 1] * B.m[3 + j] + A.m[3 * i + 2] * B.m[6 + j];
        }
    }
    return r;
}

constexpr bool operator==(const Mat3& a, const Mat3& b) noexcept { return a.m == b.m; }
constexpr bool operator!=(const Mat3& a, const Mat3& b) noexcept { return !(a == b); }

/// Rotation matrix (acting on row vectors) for a rotation of `angle` radians
/// about `axis`, via Rodrigues' formula.
[[nodiscard]] inline Mat3 rotation_matrix(const Vec3& axis, double angle) noexcept {
    const Vec3 u = axis.normalized();
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double t = 1.0 - c;
    // Transposed relative to the usual column-vector form, because FastAtoms
    // multiplies row vectors from the left (v * R).
    return {t * u.x * u.x + c,        t * u.x * u.y + s * u.z,  t * u.x * u.z - s * u.y,
            t * u.x * u.y - s * u.z,  t * u.y * u.y + c,        t * u.y * u.z + s * u.x,
            t * u.x * u.z + s * u.y,  t * u.y * u.z - s * u.x,  t * u.z * u.z + c};
}

}  // namespace fastatoms
