// SPDX-License-Identifier: MIT
#include "fastatoms/core/neighbor_list.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace fastatoms {
namespace {

/// Integer division rounding towards negative infinity.
FASTATOMS_ALWAYS_INLINE int floor_div(int a, int b) noexcept {
    int q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) {
        --q;
    }
    return q;
}

/// Everything the pair loop needs about the binning geometry.
struct BinGrid {
    Mat3 basis{Mat3::identity()};    ///< rows are the binning lattice vectors
    Mat3 inverse{Mat3::identity()};
    std::array<bool, 3> pbc{{false, false, false}};
    std::array<int, 3> nbins{{1, 1, 1}};
    std::array<int, 3> search{{1, 1, 1}};
    Vec3 origin{};                   ///< scaled-coordinate origin of bin 0
    Vec3 inv_width{};                ///< bins per unit of scaled coordinate
};

/// Keep the bin array from exploding for very sparse or very large cells.
/// Shrinking a dimension only makes bins wider, which stays correct.
void limit_total_bins(std::array<int, 3>& nbins, std::size_t natoms) {
    const std::size_t cap = std::max<std::size_t>(64, 8 * natoms);
    auto total = [&nbins]() -> std::size_t {
        return static_cast<std::size_t>(nbins[0]) * static_cast<std::size_t>(nbins[1]) *
               static_cast<std::size_t>(nbins[2]);
    };
    while (total() > cap) {
        const auto largest =
            static_cast<std::size_t>(std::max_element(nbins.begin(), nbins.end()) - nbins.begin());
        if (nbins[largest] <= 1) {
            break;
        }
        nbins[largest] = std::max(1, nbins[largest] / 2);
    }
}

}  // namespace

std::vector<std::int32_t> NeighborList::coordination_numbers() const {
    std::vector<std::int32_t> counts(natoms(), 0);
    for (std::size_t a = 0; a < counts.size(); ++a) {
        counts[a] = static_cast<std::int32_t>(count(a));
    }
    if (!full) {
        // A half list stores each pair once, so the partner needs counting too.
        for (const std::int32_t partner : j) {
            ++counts[static_cast<std::size_t>(partner)];
        }
    }
    return counts;
}

NeighborList build_neighbor_list(const Atoms& atoms, double cutoff, bool full) {
    NeighborListOptions options;
    options.cutoff = cutoff;
    options.full = full;
    return build_neighbor_list(atoms, options);
}

NeighborList build_neighbor_list(const Atoms& atoms, const NeighborListOptions& options) {
    const std::size_t natoms = atoms.size();
    const Cell& cell = atoms.cell();

    const bool per_atom = !options.per_atom_cutoffs.empty();
    if (per_atom && options.per_atom_cutoffs.size() != natoms) {
        throw std::invalid_argument("build_neighbor_list: per_atom_cutoffs has " +
                                    std::to_string(options.per_atom_cutoffs.size()) +
                                    " entries but there are " + std::to_string(natoms) + " atoms");
    }

    // The search radius must bound the largest accepted pair distance.
    double rc = options.cutoff;
    if (per_atom) {
        const double rmax =
            *std::max_element(options.per_atom_cutoffs.begin(), options.per_atom_cutoffs.end());
        rc = 2.0 * rmax;
    }
    if (!(rc > 0.0) || !std::isfinite(rc)) {
        throw std::invalid_argument("build_neighbor_list: the cutoff must be positive and finite");
    }

    NeighborList result;
    result.cutoff = per_atom ? rc : options.cutoff;
    result.full = options.full;
    result.first.assign(natoms + 1, 0);
    if (natoms == 0) {
        return result;
    }

    // ---- 1. Choose the binning basis -------------------------------------
    //
    // With any periodicity we bin in the cell's own lattice space, so that
    // wrapping and image offsets are exact for triclinic cells. Without it we
    // bin in Cartesian space over the bounding box of the atoms.
    BinGrid grid;
    grid.pbc = cell.pbc();
    const bool periodic = cell.any_pbc();
    if (periodic) {
        if (!cell.is_invertible()) {
            throw std::invalid_argument(
                "build_neighbor_list: periodic boundary conditions require an invertible cell");
        }
        grid.basis = cell.matrix();
        grid.inverse = cell.inverse_matrix();
    } else {
        grid.pbc = {false, false, false};
    }

    // ---- 2. Scaled coordinates, wrapped along periodic directions --------
    //
    // `wrap_offset` records how far each atom was folded, so that the shifts
    // we report refer to the caller's original (possibly unwrapped) positions:
    //     r_ij = r_j + shift * cell - r_i
    std::vector<double> scaled(3 * natoms);
    std::vector<double> wrapped(3 * natoms);
    std::vector<std::array<int, 3>> wrap_offset(natoms, {0, 0, 0});
    for (std::size_t i = 0; i < natoms; ++i) {
        Vec3 s = atoms.position(i) * grid.inverse;
        for (std::size_t k = 0; k < 3; ++k) {
            if (grid.pbc[k]) {
                const double w = std::floor(s[k]);
                wrap_offset[i][k] = static_cast<int>(w);
                s[k] -= w;
                // Guard against s[k] landing exactly on 1.0 through rounding.
                if (s[k] >= 1.0) {
                    s[k] = 0.0;
                    wrap_offset[i][k] += 1;
                }
            }
        }
        s.store(scaled.data() + 3 * i);
        (s * grid.basis).store(wrapped.data() + 3 * i);
    }

    // ---- 3. Bin layout ----------------------------------------------------
    const Vec3 heights = periodic ? cell.heights() : Vec3{1.0, 1.0, 1.0};
    Vec3 span{1.0, 1.0, 1.0};
    for (std::size_t k = 0; k < 3; ++k) {
        double lo = 0.0;
        double hi = 1.0;
        if (!grid.pbc[k]) {
            lo = std::numeric_limits<double>::max();
            hi = std::numeric_limits<double>::lowest();
            for (std::size_t i = 0; i < natoms; ++i) {
                lo = std::min(lo, scaled[3 * i + k]);
                hi = std::max(hi, scaled[3 * i + k]);
            }
        }
        grid.origin[k] = lo;
        // A degenerate span (all atoms in a plane) still needs one usable bin.
        span[k] = std::max(hi - lo, 1e-12);
        const double extent = span[k] * (heights[k] > 0.0 ? heights[k] : 1.0);
        grid.nbins[k] = std::max(1, static_cast<int>(std::floor(extent / rc)));
    }
    limit_total_bins(grid.nbins, natoms);

    for (std::size_t k = 0; k < 3; ++k) {
        grid.inv_width[k] = static_cast<double>(grid.nbins[k]) / span[k];
        // |dr| >= |ds_k| * height_k, so a pair within rc spans at most this
        // many bins along direction k. Values above 1 only occur when the cell
        // is thinner than the cutoff.
        const double h = heights[k] > 0.0 ? heights[k] : 1.0;
        double reach = rc * static_cast<double>(grid.nbins[k]) / (h * span[k]);

        if (!grid.pbc[k]) {
            // Bins outside the grid are skipped, so searching further than the
            // grid is wasted work. The clamp also keeps a degenerate direction
            // finite: a flat or collinear cluster has a span of essentially
            // zero, which would otherwise ask for an astronomical range.
            reach = std::min(reach, static_cast<double>(grid.nbins[k]));
        } else if (!(reach < 1e6)) {
            // Only reachable by asking for a cutoff vastly larger than the
            // periodic cell, which is a mistake worth naming.
            throw std::invalid_argument(
                "build_neighbor_list: a cutoff of " + std::to_string(rc) +
                " would need more than a million periodic images along lattice direction " +
                std::to_string(k) + "; check the cutoff and the cell");
        }
        grid.search[k] = std::max(1, static_cast<int>(std::ceil(reach)));
    }

    const int nb0 = grid.nbins[0];
    const int nb1 = grid.nbins[1];
    const int nb2 = grid.nbins[2];
    const std::size_t total_bins =
        static_cast<std::size_t>(nb0) * static_cast<std::size_t>(nb1) *
        static_cast<std::size_t>(nb2);

    // ---- 4. Counting sort of the atoms into bins -------------------------
    std::vector<int> bin_of_atom(natoms);
    std::vector<std::size_t> bin_start(total_bins + 1, 0);
    for (std::size_t i = 0; i < natoms; ++i) {
        int b[3];
        for (std::size_t k = 0; k < 3; ++k) {
            const double t = (scaled[3 * i + k] - grid.origin[k]) * grid.inv_width[k];
            b[k] = std::clamp(static_cast<int>(std::floor(t)), 0, grid.nbins[k] - 1);
        }
        const int index = (b[0] * nb1 + b[1]) * nb2 + b[2];
        bin_of_atom[i] = index;
        ++bin_start[static_cast<std::size_t>(index) + 1];
    }
    std::partial_sum(bin_start.begin(), bin_start.end(), bin_start.begin());

    std::vector<std::int32_t> atoms_in_bin(natoms);
    {
        std::vector<std::size_t> fill(bin_start.begin(), bin_start.end() - 1);
        for (std::size_t i = 0; i < natoms; ++i) {
            atoms_in_bin[fill[static_cast<std::size_t>(bin_of_atom[i])]++] =
                static_cast<std::int32_t>(i);
        }
    }

    // ---- 5. Pair search ---------------------------------------------------
    //
    // Atoms are split into contiguous chunks, one per thread, so concatenating
    // the per-thread buffers in thread order yields a list already sorted by
    // `i`. That gives a valid CSR layout with no locks and no final sort.
    struct Buffer {
        std::vector<std::int32_t> i, j, shifts;
        std::vector<double> vectors, distances;
    };

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = std::max(1, omp_get_max_threads());
    nthreads = std::min<int>(nthreads, static_cast<int>(natoms));
#endif
    std::vector<Buffer> buffers(static_cast<std::size_t>(nthreads));

    const double* FASTATOMS_RESTRICT pos = wrapped.data();
    const double* FASTATOMS_RESTRICT radii =
        per_atom ? options.per_atom_cutoffs.data() : nullptr;
    const double rc2 = rc * rc;
    const bool store_vectors = options.store_vectors;
    const bool store_distances = options.store_distances;
    const bool want_full = options.full;
    const bool self_interaction = options.self_interaction;

#ifdef _OPENMP
#  pragma omp parallel num_threads(nthreads)
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const std::size_t chunk = (natoms + static_cast<std::size_t>(nthreads) - 1) /
                                  static_cast<std::size_t>(nthreads);
        const std::size_t begin = std::min(natoms, static_cast<std::size_t>(tid) * chunk);
        const std::size_t end = std::min(natoms, begin + chunk);
        Buffer& buf = buffers[static_cast<std::size_t>(tid)];
        buf.i.reserve((end - begin) * 16);
        buf.j.reserve((end - begin) * 16);
        buf.shifts.reserve((end - begin) * 48);

        for (std::size_t i = begin; i < end; ++i) {
            const Vec3 ri = Vec3::load(pos + 3 * i);
            const double rc_i = per_atom ? radii[i] : 0.0;
            const int base = bin_of_atom[i];
            const int b2 = base % nb2;
            const int b1 = (base / nb2) % nb1;
            const int b0 = base / (nb2 * nb1);

            for (int d0 = -grid.search[0]; d0 <= grid.search[0]; ++d0) {
                int c0 = b0 + d0;
                int image0 = 0;
                if (grid.pbc[0]) {
                    image0 = floor_div(c0, nb0);
                    c0 -= image0 * nb0;
                } else if (c0 < 0 || c0 >= nb0) {
                    continue;
                }
                for (int d1 = -grid.search[1]; d1 <= grid.search[1]; ++d1) {
                    int c1 = b1 + d1;
                    int image1 = 0;
                    if (grid.pbc[1]) {
                        image1 = floor_div(c1, nb1);
                        c1 -= image1 * nb1;
                    } else if (c1 < 0 || c1 >= nb1) {
                        continue;
                    }
                    for (int d2 = -grid.search[2]; d2 <= grid.search[2]; ++d2) {
                        int c2 = b2 + d2;
                        int image2 = 0;
                        if (grid.pbc[2]) {
                            image2 = floor_div(c2, nb2);
                            c2 -= image2 * nb2;
                        } else if (c2 < 0 || c2 >= nb2) {
                            continue;
                        }

                        const Vec3 offset = grid.basis.row(0) * image0 +
                                            grid.basis.row(1) * image1 +
                                            grid.basis.row(2) * image2;
                        const std::size_t bin =
                            static_cast<std::size_t>((c0 * nb1 + c1) * nb2 + c2);

                        for (std::size_t slot = bin_start[bin]; slot < bin_start[bin + 1]; ++slot) {
                            const std::int32_t jj = atoms_in_bin[slot];
                            const std::size_t j = static_cast<std::size_t>(jj);

                            // Shift relative to the caller's unwrapped positions.
                            const std::int32_t sx = static_cast<std::int32_t>(
                                image0 - wrap_offset[j][0] + wrap_offset[i][0]);
                            const std::int32_t sy = static_cast<std::int32_t>(
                                image1 - wrap_offset[j][1] + wrap_offset[i][1]);
                            const std::int32_t sz = static_cast<std::int32_t>(
                                image2 - wrap_offset[j][2] + wrap_offset[i][2]);

                            const bool same_site = (j == i) && sx == 0 && sy == 0 && sz == 0;
                            if (same_site && !self_interaction) {
                                continue;
                            }
                            if (!want_full && !same_site) {
                                // Keep exactly one of (i, j, S) and (j, i, -S).
                                const bool keep = (sx > 0) || (sx == 0 && sy > 0) ||
                                                  (sx == 0 && sy == 0 && sz > 0) ||
                                                  (sx == 0 && sy == 0 && sz == 0 && j > i);
                                if (!keep) {
                                    continue;
                                }
                            }

                            const Vec3 dr = Vec3::load(pos + 3 * j) + offset - ri;
                            const double d2r = dr.norm2();
                            double limit2 = rc2;
                            if (per_atom) {
                                const double sum = rc_i + radii[j];
                                limit2 = sum * sum;
                            }
                            if (d2r >= limit2) {
                                continue;
                            }

                            buf.i.push_back(static_cast<std::int32_t>(i));
                            buf.j.push_back(jj);
                            buf.shifts.push_back(sx);
                            buf.shifts.push_back(sy);
                            buf.shifts.push_back(sz);
                            if (store_vectors) {
                                buf.vectors.push_back(dr.x);
                                buf.vectors.push_back(dr.y);
                                buf.vectors.push_back(dr.z);
                            }
                            if (store_distances) {
                                buf.distances.push_back(std::sqrt(d2r));
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- 6. Concatenate and build the CSR index --------------------------
    std::size_t npairs = 0;
    for (const Buffer& buf : buffers) {
        npairs += buf.i.size();
    }

    result.i.reserve(npairs);
    result.j.reserve(npairs);
    result.shifts.reserve(3 * npairs);
    if (store_vectors) {
        result.vectors.reserve(3 * npairs);
    }
    if (store_distances) {
        result.distances.reserve(npairs);
    }
    for (const Buffer& buf : buffers) {
        result.i.insert(result.i.end(), buf.i.begin(), buf.i.end());
        result.j.insert(result.j.end(), buf.j.begin(), buf.j.end());
        result.shifts.insert(result.shifts.end(), buf.shifts.begin(), buf.shifts.end());
        result.vectors.insert(result.vectors.end(), buf.vectors.begin(), buf.vectors.end());
        result.distances.insert(result.distances.end(), buf.distances.begin(), buf.distances.end());
    }

    for (const std::int32_t owner : result.i) {
        ++result.first[static_cast<std::size_t>(owner) + 1];
    }
    std::partial_sum(result.first.begin(), result.first.end(), result.first.begin());

    return result;
}

}  // namespace fastatoms
