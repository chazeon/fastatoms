// SPDX-License-Identifier: MIT
#pragma once

/// \file constraint.hpp
/// \brief Geometric constraints, shared between the models that declare them
///        and the integrators that enforce them.
///
/// A force field knows that an O-H bond is rigid; the dynamics knows how to
/// keep it that way. The two need to agree on a vocabulary without depending
/// on each other, so the vocabulary lives here.

#include <cstddef>

namespace fastatoms {

/// One holonomic distance constraint, `|r_i - r_j| = length`.
///
/// Three of these per water molecule - the two O-H bonds and the H-H distance
/// that fixes the angle - make it rigid with no angular machinery at all. That
/// is the standard way to freeze a three-site model, and it is why SHAKE alone
/// is enough for SPC and SPC/E.
struct DistanceConstraint {
    std::size_t i{0};
    std::size_t j{0};
    double length{0.0};
};

}  // namespace fastatoms
