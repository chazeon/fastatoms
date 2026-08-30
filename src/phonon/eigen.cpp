// SPDX-License-Identifier: MIT
#include "fastatoms/phonon/eigen.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace fastatoms::phonon {

ComplexMatrix ComplexMatrix::identity(std::size_t n) {
    ComplexMatrix out(n);
    for (std::size_t k = 0; k < n; ++k) {
        out(k, k) = 1.0;
    }
    return out;
}

void ComplexMatrix::hermitize() {
    for (std::size_t i = 0; i < n_; ++i) {
        data_[i * n_ + i] = std::complex<double>(data_[i * n_ + i].real(), 0.0);
        for (std::size_t j = i + 1; j < n_; ++j) {
            const value_type average =
                0.5 * (data_[i * n_ + j] + std::conj(data_[j * n_ + i]));
            data_[i * n_ + j] = average;
            data_[j * n_ + i] = std::conj(average);
        }
    }
}

double ComplexMatrix::hermiticity_error() const {
    double largest = 0.0;
    for (std::size_t i = 0; i < n_; ++i) {
        largest = std::max(largest, std::abs(data_[i * n_ + i].imag()));
        for (std::size_t j = i + 1; j < n_; ++j) {
            largest = std::max(
                largest, std::abs(data_[i * n_ + j] - std::conj(data_[j * n_ + i])));
        }
    }
    return largest;
}

std::string ComplexMatrix::to_string() const {
    std::ostringstream out;
    out << "ComplexMatrix(" << n_ << "x" << n_ << ")";
    return out.str();
}

HermitianEigen hermitian_eigen(const ComplexMatrix& matrix, double tolerance, int max_sweeps) {
    const std::size_t n = matrix.size();
    HermitianEigen result;
    result.values.assign(n, 0.0);
    result.vectors = ComplexMatrix::identity(n);
    if (n == 0) {
        result.converged = true;
        return result;
    }

    // Work on a copy, forced exactly Hermitian so the rotations stay unitary.
    ComplexMatrix a = matrix;
    a.hermitize();

    // Scale the convergence test by the size of the matrix, so the tolerance
    // means the same thing for a stiff crystal as for a soft one.
    double scale = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            scale = std::max(scale, std::abs(a(i, j)));
        }
    }
    if (scale == 0.0) {
        result.converged = true;
        return result;
    }
    const double threshold = tolerance * scale;

    for (int sweep = 1; sweep <= max_sweeps; ++sweep) {
        result.sweeps = sweep;

        double off_diagonal = 0.0;
        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                off_diagonal = std::max(off_diagonal, std::abs(a(p, q)));
            }
        }
        if (off_diagonal <= threshold) {
            result.converged = true;
            break;
        }

        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const std::complex<double> z = a(p, q);
                const double magnitude = std::abs(z);
                if (magnitude <= threshold * 1e-3) {
                    continue;
                }

                // Write the entry as |z| e^{i phi}; the phase comes out of the
                // rotation and what is left is the real Jacobi problem.
                const std::complex<double> phase = z / magnitude;

                const double app = a(p, p).real();
                const double aqq = a(q, q).real();

                // The rotation below is U(p,q) = -s e^{i phi}, which kills the
                // entry when t^2 + 2 tau t - 1 = 0 with tau as follows. Taking
                // the smaller root keeps the rotation angle under 45 degrees,
                // which is what makes the method numerically stable.
                const double tau = 0.5 * (app - aqq) / magnitude;
                const double t = (tau >= 0.0 ? 1.0 : -1.0) /
                                 (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = t * c;

                // The rotation, in the basis where columns p and q mix:
                //   U(p,p) = c            U(p,q) = -s * phase
                //   U(q,p) =  s * conj(phase)   U(q,q) = c
                for (std::size_t k = 0; k < n; ++k) {
                    if (k == p || k == q) {
                        continue;
                    }
                    const std::complex<double> akp = a(k, p);
                    const std::complex<double> akq = a(k, q);
                    const std::complex<double> new_kp = c * akp + s * std::conj(phase) * akq;
                    const std::complex<double> new_kq = -s * phase * akp + c * akq;
                    a(k, p) = new_kp;
                    a(p, k) = std::conj(new_kp);
                    a(k, q) = new_kq;
                    a(q, k) = std::conj(new_kq);
                }

                const double new_app = app + t * magnitude;
                const double new_aqq = aqq - t * magnitude;
                a(p, p) = new_app;
                a(q, q) = new_aqq;
                a(p, q) = 0.0;
                a(q, p) = 0.0;

                // Accumulate the same rotation into the eigenvectors.
                for (std::size_t k = 0; k < n; ++k) {
                    const std::complex<double> vkp = result.vectors(k, p);
                    const std::complex<double> vkq = result.vectors(k, q);
                    result.vectors(k, p) = c * vkp + s * std::conj(phase) * vkq;
                    result.vectors(k, q) = -s * phase * vkp + c * vkq;
                }
            }
        }
    }

    for (std::size_t k = 0; k < n; ++k) {
        result.values[k] = a(k, k).real();
    }

    // Sort ascending, carrying the eigenvectors along.
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&result](std::size_t i, std::size_t j) {
        return result.values[i] < result.values[j];
    });

    std::vector<double> sorted_values(n);
    ComplexMatrix sorted_vectors(n);
    for (std::size_t k = 0; k < n; ++k) {
        sorted_values[k] = result.values[order[k]];
        for (std::size_t row = 0; row < n; ++row) {
            sorted_vectors(row, k) = result.vectors(row, order[k]);
        }
    }
    result.values = std::move(sorted_values);
    result.vectors = std::move(sorted_vectors);
    return result;
}

}  // namespace fastatoms::phonon
