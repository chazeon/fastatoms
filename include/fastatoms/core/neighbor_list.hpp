// SPDX-License-Identifier: MIT
#pragma once

/// \file neighbor_list.hpp
/// \brief ASE-compatible neighbour pairs, computed with vesin.
///
/// Semantics match `ase.neighborlist.neighbor_list("ijSdD", atoms, cutoff)`:
///
///   D = r[j] - r[i] + S @ cell,  d = |D| < cutoff
///
/// periodic only along `atoms.pbc()`, cutoff may exceed the cell (several
/// images of the same j appear with different S), the pair (i, i, S=0) is
/// never returned, (i, i, S!=0) is. Pairs are sorted by (i, j, S) so the
/// output is reproducible across platforms.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "fastatoms/core/atoms.hpp"

namespace fastatoms {

struct NeighborOptions {
    double cutoff{0.0};
    bool full{true};    ///< both (i,j,S) and (j,i,-S); false keeps one of each
    bool sorted{true};  ///< sort by (i, j, S)
};

struct NeighborPairs {
    std::vector<std::int32_t> i;
    std::vector<std::int32_t> j;
    std::vector<std::int32_t> S;  ///< 3 per pair
    std::vector<double> d;
    std::vector<double> D;        ///< 3 per pair

    [[nodiscard]] std::size_t size() const noexcept { return i.size(); }
};

/// \throws std::invalid_argument for cutoff <= 0 or a singular cell along a
///         periodic axis.
[[nodiscard]] NeighborPairs neighbor_pairs(const Atoms& atoms, const NeighborOptions& options);

/// Same, on raw data: 3N positions, row-major cell (rows are lattice vectors).
[[nodiscard]] NeighborPairs neighbor_pairs(const double* positions, std::size_t n,
                                           const Mat3& cell, const std::array<bool, 3>& pbc,
                                           const NeighborOptions& options);

/// Neighbour count per atom from `pairs` (length n).
[[nodiscard]] std::vector<std::int32_t> coordination_numbers(const NeighborPairs& pairs,
                                                            std::size_t n);

}  // namespace fastatoms
