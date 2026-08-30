// SPDX-License-Identifier: MIT
#pragma once

/// \file neighbor_list.hpp
/// \brief Linear-scaling neighbour list based on cell-linked lists.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/core/vec3.hpp"

namespace fastatoms {

/// Parameters controlling neighbour-list construction.
struct NeighborListOptions {
    /// Global interaction cutoff in angstrom. Ignored when `per_atom_cutoffs`
    /// is non-empty.
    double cutoff{0.0};

    /// Optional per-atom radii. When given (length N), atoms `i` and `j` are
    /// neighbours if `r_ij < cutoff_i + cutoff_j`. This is what bond detection
    /// from covalent radii needs.
    std::vector<double> per_atom_cutoffs{};

    /// `true` stores both (i, j) and (j, i); `false` stores each pair once.
    bool full{true};

    /// Include the (i, i) self pair with zero shift.
    bool self_interaction{false};

    /// Also store the minimum-image displacement vector of every pair.
    bool store_vectors{true};

    /// Also store the pair distance.
    bool store_distances{true};
};

/// A pair list in compressed-sparse-row form.
///
/// Pairs are ordered by `i`, so `first[i] .. first[i + 1]` is the contiguous
/// block of neighbours of atom `i`. `shifts` holds the integer cell offset of
/// the periodic image of `j` that was used, such that
///
///     r_ij = r_j + shift * cell - r_i
///
/// which is what a force kernel needs to accumulate virial contributions.
struct NeighborList {
    std::vector<std::int32_t> i;       ///< npairs
    std::vector<std::int32_t> j;       ///< npairs
    std::vector<std::int32_t> shifts;  ///< 3 * npairs
    std::vector<double> vectors;       ///< 3 * npairs, empty unless requested
    std::vector<double> distances;     ///< npairs, empty unless requested
    std::vector<std::size_t> first;    ///< natoms + 1 CSR offsets

    double cutoff{0.0};
    bool full{true};

    [[nodiscard]] std::size_t size() const noexcept { return i.size(); }
    [[nodiscard]] bool empty() const noexcept { return i.empty(); }
    [[nodiscard]] std::size_t natoms() const noexcept {
        return first.empty() ? 0 : first.size() - 1;
    }

    /// Number of neighbours of atom `a` (CSR row length).
    [[nodiscard]] std::size_t count(std::size_t a) const noexcept {
        return first[a + 1] - first[a];
    }

    /// Neighbour indices of atom `a` as a (pointer, length) pair.
    [[nodiscard]] const std::int32_t* neighbors_of(std::size_t a,
                                                   std::size_t& n) const noexcept {
        n = count(a);
        return j.data() + first[a];
    }

    /// Coordination number of every atom.
    [[nodiscard]] std::vector<std::int32_t> coordination_numbers() const;
};

/// Build a neighbour list for `atoms`.
///
/// Cost is O(N) for a fixed cutoff and roughly uniform density: atoms are
/// binned into cells of at least the cutoff size and only the 27 surrounding
/// bins are searched (more when the cell is thinner than the cutoff, so that
/// slabs and long thin cells stay correct).
///
/// Works for any combination of periodic and free directions, and does not
/// require the cutoff to be smaller than half the cell - multiple periodic
/// images of the same pair are returned with different `shifts`.
///
/// \throws std::invalid_argument for a non-positive cutoff, a per-atom cutoff
///         array of the wrong length, or periodicity with a singular cell.
[[nodiscard]] NeighborList build_neighbor_list(const Atoms& atoms,
                                               const NeighborListOptions& options);

/// Convenience overload for the common "one global cutoff, full list" case.
[[nodiscard]] NeighborList build_neighbor_list(const Atoms& atoms, double cutoff,
                                               bool full = true);

}  // namespace fastatoms
