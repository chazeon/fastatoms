// SPDX-License-Identifier: MIT
#pragma once

/// \file vector_math.hpp
/// \brief The linear algebra the optimizers need, written out by hand.
///
/// Deliberately dependency-free: everything here is a handful of loops over
/// `std::vector<double>`. Nothing in the optimization module reaches for an
/// external linear algebra or optimization library.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fastatoms::optimize {

/// Inner product of two equally sized vectors.
[[nodiscard]] inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("dot: vectors have different lengths");
    }
    double sum = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        sum += a[k] * b[k];
    }
    return sum;
}

[[nodiscard]] inline double norm(const std::vector<double>& a) { return std::sqrt(dot(a, a)); }

/// `y <- y + alpha * x`.
inline void axpy(double alpha, const std::vector<double>& x, std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("axpy: vectors have different lengths");
    }
    for (std::size_t k = 0; k < x.size(); ++k) {
        y[k] += alpha * x[k];
    }
}

/// `x <- alpha * x`.
inline void scale(double alpha, std::vector<double>& x) {
    for (double& value : x) {
        value *= alpha;
    }
}

[[nodiscard]] inline std::vector<double> scaled(double alpha, std::vector<double> x) {
    scale(alpha, x);
    return x;
}

/// `a - b`.
[[nodiscard]] inline std::vector<double> subtract(const std::vector<double>& a,
                                                  const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("subtract: vectors have different lengths");
    }
    std::vector<double> out(a.size());
    for (std::size_t k = 0; k < a.size(); ++k) {
        out[k] = a[k] - b[k];
    }
    return out;
}

/// Largest Euclidean norm among consecutive groups of three components.
///
/// This is how a force convergence threshold is defined: the largest force on
/// any single atom, not the largest single Cartesian component.
[[nodiscard]] inline double max_row_norm(const std::vector<double>& v) {
    if (v.size() % 3 != 0) {
        throw std::invalid_argument("max_row_norm: length must be a multiple of three");
    }
    double largest = 0.0;
    for (std::size_t row = 0; row < v.size() / 3; ++row) {
        const double x = v[3 * row];
        const double y = v[3 * row + 1];
        const double z = v[3 * row + 2];
        largest = std::max(largest, std::sqrt(x * x + y * y + z * z));
    }
    return largest;
}

/// As `max_row_norm`, but only over rows in `[begin, end)`.
[[nodiscard]] inline double max_row_norm(const std::vector<double>& v, std::size_t begin,
                                         std::size_t end) {
    double largest = 0.0;
    for (std::size_t row = begin; row < end && 3 * row + 2 < v.size(); ++row) {
        const double x = v[3 * row];
        const double y = v[3 * row + 1];
        const double z = v[3 * row + 2];
        largest = std::max(largest, std::sqrt(x * x + y * y + z * z));
    }
    return largest;
}

/// A dense symmetric matrix stored row-major, used for the BFGS inverse
/// Hessian. Only the operations BFGS actually needs are provided.
class SymmetricMatrix {
public:
    SymmetricMatrix() = default;

    /// `scale * I`.
    explicit SymmetricMatrix(std::size_t n, double diagonal = 1.0)
        : n_(n), data_(n * n, 0.0) {
        for (std::size_t k = 0; k < n; ++k) {
            data_[k * n + k] = diagonal;
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return n_; }
    [[nodiscard]] double& operator()(std::size_t i, std::size_t j) { return data_[i * n_ + j]; }
    [[nodiscard]] double operator()(std::size_t i, std::size_t j) const {
        return data_[i * n_ + j];
    }

    /// Reset to `scale * I`.
    void reset(std::size_t n, double diagonal) {
        n_ = n;
        data_.assign(n * n, 0.0);
        for (std::size_t k = 0; k < n; ++k) {
            data_[k * n + k] = diagonal;
        }
    }

    /// Matrix-vector product.
    [[nodiscard]] std::vector<double> apply(const std::vector<double>& v) const {
        if (v.size() != n_) {
            throw std::invalid_argument("SymmetricMatrix::apply: wrong vector length");
        }
        std::vector<double> out(n_, 0.0);
        for (std::size_t i = 0; i < n_; ++i) {
            const double* row = data_.data() + i * n_;
            double sum = 0.0;
            for (std::size_t j = 0; j < n_; ++j) {
                sum += row[j] * v[j];
            }
            out[i] = sum;
        }
        return out;
    }

    /// `M <- M + c1 * (a b^T + b a^T) + c2 * a a^T`, keeping symmetry exact.
    void add_symmetric_updates(double c1, const std::vector<double>& a,
                               const std::vector<double>& b, double c2) {
        for (std::size_t i = 0; i < n_; ++i) {
            double* row = data_.data() + i * n_;
            const double ai = a[i];
            const double bi = b[i];
            for (std::size_t j = 0; j < n_; ++j) {
                row[j] += c1 * (ai * b[j] + bi * a[j]) + c2 * ai * a[j];
            }
        }
    }

private:
    std::size_t n_{0};
    std::vector<double> data_{};
};

}  // namespace fastatoms::optimize
