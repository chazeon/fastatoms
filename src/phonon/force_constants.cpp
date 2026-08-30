// SPDX-License-Identifier: MIT
#include "fastatoms/phonon/force_constants.hpp"

#include <cmath>
#include <sstream>

namespace fastatoms::phonon {

void ForceConstants::apply_acoustic_sum_rule() {
    for (std::size_t i = 0; i < rows_; ++i) {
        Mat3 sum{};
        for (std::size_t j = 0; j < columns_; ++j) {
            for (std::size_t k = 0; k < 9; ++k) {
                sum.m[k] += block(i, j).m[k];
            }
        }
        // The whole residue is charged to the self term, which is where the
        // convention puts it: Phi(i,i) = -sum_{j != i} Phi(i,j).
        for (std::size_t k = 0; k < 9; ++k) {
            block(i, i).m[k] -= sum.m[k];
        }
    }
}

double ForceConstants::acoustic_sum_violation() const {
    double largest = 0.0;
    for (std::size_t i = 0; i < rows_; ++i) {
        Mat3 sum{};
        for (std::size_t j = 0; j < columns_; ++j) {
            for (std::size_t k = 0; k < 9; ++k) {
                sum.m[k] += block(i, j).m[k];
            }
        }
        for (const double value : sum.m) {
            largest = std::max(largest, std::abs(value));
        }
    }
    return largest;
}

std::vector<double> ForceConstants::to_array() const {
    std::vector<double> out(9 * rows_ * columns_, 0.0);
    const std::size_t stride = 3 * columns_;
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < columns_; ++j) {
            const Mat3& value = block(i, j);
            for (std::size_t a = 0; a < 3; ++a) {
                for (std::size_t b = 0; b < 3; ++b) {
                    out[(3 * i + a) * stride + 3 * j + b] = value(a, b);
                }
            }
        }
    }
    return out;
}

std::string ForceConstants::to_string() const {
    std::ostringstream out;
    out << "ForceConstants(" << 3 * rows_ << "x" << 3 * columns_ << ")";
    return out.str();
}

}  // namespace fastatoms::phonon
