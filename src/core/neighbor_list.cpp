// SPDX-License-Identifier: MIT
#include "fastatoms/core/neighbor_list.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "vesin.h"

namespace fastatoms {
namespace {

/// vesin refuses cutoffs at or below this value.
constexpr double kMinCutoff = 1e-6;

/// Cutoffs below this are searched in a scaled copy of the geometry, so that
/// the vesin cutoff stays well above kMinCutoff. The pairs are still tested
/// against the original cutoff, and D and d are recomputed from the original
/// coordinates, so the scaling only affects which candidates vesin returns.
constexpr double kScaleBelow = 1e-3;

/// Relative tolerance for declaring a periodic part of the cell singular.
constexpr double kSingularTol = 1e-12;

/// Owns a vesin list so that it is freed on every exit path.
struct VesinList {
    VesinNeighborList list;
    ~VesinList() { vesin_free(&list); }
};

/// Periodic rows must span the dimensions they claim: one non-zero vector,
/// two non-parallel vectors, or three independent vectors.
void check_periodic_cell(const Mat3& cell, const std::array<bool, 3>& pbc) {
    std::vector<Vec3> rows;
    for (std::size_t k = 0; k < 3; ++k) {
        if (pbc[k]) {
            rows.push_back(cell.row(k));
        }
    }

    bool ok = true;
    switch (rows.size()) {
        case 0:
            break;
        case 1:
            ok = rows[0].norm() > kSingularTol;
            break;
        case 2: {
            const double scale = rows[0].norm() * rows[1].norm();
            ok = scale > 0.0 && cross(rows[0], rows[1]).norm() > kSingularTol * scale;
            break;
        }
        default: {
            const double scale = rows[0].norm() * rows[1].norm() * rows[2].norm();
            ok = scale > 0.0 && std::abs(dot(rows[0], cross(rows[1], rows[2]))) > kSingularTol * scale;
            break;
        }
    }
    if (!ok) {
        throw std::invalid_argument("neighbor_pairs: the cell is singular along a periodic axis");
    }
}

/// The box handed to vesin. Periodic rows are the cell's own vectors. Each
/// non-periodic row is replaced by a vector orthogonal to the periodic ones,
/// with length (extent + cutoff) of the bounding box along that axis, so that
/// any cell, including a zero one for a molecule, gives a well-formed box.
/// Shifts along non-periodic axes are always zero, so these rows never enter
/// the result.
[[nodiscard]] Mat3 vesin_box(const double* positions, std::size_t n, const Mat3& cell,
                             const std::array<bool, 3>& pbc, double cutoff) {
    std::array<double, 3> lo{positions[0], positions[1], positions[2]};
    std::array<double, 3> hi = lo;
    for (std::size_t i = 1; i < n; ++i) {
        for (std::size_t k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], positions[3 * i + k]);
            hi[k] = std::max(hi[k], positions[3 * i + k]);
        }
    }
    std::array<Vec3, 3> out{cell.row(0), cell.row(1), cell.row(2)};

    std::vector<std::size_t> periodic;
    std::vector<std::size_t> free_axes;
    for (std::size_t k = 0; k < 3; ++k) {
        (pbc[k] ? periodic : free_axes).push_back(k);
    }
    if (free_axes.empty()) {
        return cell;
    }

    std::array<Vec3, 3> replacement{};
    if (periodic.empty()) {
        // No periodicity: any orthogonal frame will do.
        replacement = {Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}};
    } else if (periodic.size() == 1) {
        // One periodic row: two free directions orthogonal to it.
        const Vec3 a = cell.row(periodic[0]).normalized();
        // Start from the Cartesian axis least aligned with a.
        std::size_t e = 0;
        for (std::size_t k = 1; k < 3; ++k) {
            if (std::abs(a[k]) < std::abs(a[e])) {
                e = k;
            }
        }
        Vec3 axis{};
        axis[e] = 1.0;
        const Vec3 u = cross(a, axis).normalized();
        const Vec3 v = cross(a, u);
        replacement = {u, v, Vec3{}};
    } else {
        // Two periodic rows: the free one is their normal.
        const Vec3 normal = cross(cell.row(periodic[0]), cell.row(periodic[1])).normalized();
        replacement = {normal, normal, normal};
    }

    for (std::size_t f = 0; f < free_axes.size(); ++f) {
        const std::size_t k = free_axes[f];
        const double length = (hi[k] - lo[k]) + cutoff;
        out[k] = replacement[f] * length;
    }
    return Mat3::from_rows(out[0], out[1], out[2]);
}

/// vesin is asked for a slightly larger cutoff than the requested one. The
/// margin covers the rounding difference between vesin's distances and the
/// ones recomputed in finalize_pairs, so no pair with d < cutoff is lost. The
/// rounding error of a displacement grows with the coordinate magnitude, not
/// with the cutoff, so the margin also includes a few ulps of max_abs.
[[nodiscard]] double search_cutoff(double cutoff, double max_abs) {
    return cutoff * (1.0 + 1e-10) + 1e-12 +
           max_abs * 4.0 * std::numeric_limits<double>::epsilon();
}

/// D = r_j - r_i + S @ cell and d = |D|, evaluated the way ASE's
/// primitive_neighbor_list does it: S @ cell is a BLAS product, which on the
/// reference platform accumulates each component as an FMA chain over the cell
/// rows 0, 1, 2 (verified bit-for-bit against numpy); that shift is added to
/// r_j - r_i, and d = sqrt of the plain sum of squares in x, y, z order.
/// The squares and the sum are separate statements so that the compiler cannot
/// fuse them into FMAs.
struct Image {
    std::array<double, 3> D;
    double d;
};

[[nodiscard]] Image ase_displacement(const double* positions, const Mat3& cell, std::int32_t i,
                                     std::int32_t j, const std::array<std::int32_t, 3>& S) {
    const double* r_i = positions + 3 * static_cast<std::size_t>(i);
    const double* r_j = positions + 3 * static_cast<std::size_t>(j);

    std::array<double, 3> shift{0.0, 0.0, 0.0};
    for (std::size_t c = 0; c < 3; ++c) {
        const double s = static_cast<double>(S[c]);
        for (std::size_t k = 0; k < 3; ++k) {
            shift[k] = std::fma(s, cell(c, k), shift[k]);
        }
    }

    Image image{};
    for (std::size_t k = 0; k < 3; ++k) {
        const double diff = r_j[k] - r_i[k];
        image.D[k] = diff + shift[k];
    }
    const double xx = image.D[0] * image.D[0];
    const double yy = image.D[1] * image.D[1];
    const double zz = image.D[2] * image.D[2];
    image.d = std::sqrt((xx + yy) + zz);
    return image;
}

/// One pair of the output, before it is written into the struct-of-arrays.
struct Candidate {
    std::int32_t i;
    std::int32_t j;
    std::array<std::int32_t, 3> S;
    Image image;
};

[[nodiscard]] bool key_less(const Candidate& a, const Candidate& b) {
    return std::tie(a.i, a.j, a.S[0], a.S[1], a.S[2]) < std::tie(b.i, b.j, b.S[0], b.S[1], b.S[2]);
}

/// Final step for vesin's raw list: recompute D and d for each pair the way
/// ASE does, keep the pairs with d < cutoff (ASE's comparison), and sort by
/// (i, j, S) if requested.
[[nodiscard]] NeighborPairs finalize_pairs(const double* positions, const Mat3& cell,
                                           const std::array<bool, 3>& pbc, double cutoff,
                                           bool sorted, const VesinNeighborList& list) {
    std::vector<Candidate> cand;
    cand.reserve(list.length);
    for (std::size_t k = 0; k < list.length; ++k) {
        Candidate c{};
        c.i = static_cast<std::int32_t>(list.pairs[k][0]);
        c.j = static_cast<std::int32_t>(list.pairs[k][1]);
        for (std::size_t a = 0; a < 3; ++a) {
            c.S[a] = pbc[a] ? static_cast<std::int32_t>(list.shifts[k][a]) : 0;
        }
        c.image = ase_displacement(positions, cell, c.i, c.j, c.S);
        if (c.image.d < cutoff) {
            cand.push_back(c);
        }
    }
    if (sorted) {
        std::sort(cand.begin(), cand.end(), key_less);
    }

    NeighborPairs out;
    const std::size_t m = cand.size();
    out.i.resize(m);
    out.j.resize(m);
    out.S.resize(3 * m);
    out.d.resize(m);
    out.D.resize(3 * m);
    for (std::size_t q = 0; q < m; ++q) {
        const Candidate& c = cand[q];
        out.i[q] = c.i;
        out.j[q] = c.j;
        for (std::size_t a = 0; a < 3; ++a) {
            out.S[3 * q + a] = c.S[a];
            out.D[3 * q + a] = c.image.D[a];
        }
        out.d[q] = c.image.d;
    }
    return out;
}

}  // namespace


NeighborPairs neighbor_pairs(const double* positions, std::size_t n, const Mat3& cell,
                             const std::array<bool, 3>& pbc, const NeighborOptions& options) {
    if (!std::isfinite(options.cutoff) || options.cutoff <= 0.0) {
        throw std::invalid_argument("neighbor_pairs: cutoff must be finite and positive");
    }
    check_periodic_cell(cell, pbc);

    NeighborPairs out;
    if (n == 0) {
        return out;
    }
    if (positions == nullptr) {
        throw std::invalid_argument("neighbor_pairs: positions is a null pointer");
    }

    // Search geometry: the original, or a copy scaled so that the cutoff is
    // above vesin's minimum. Scaling every coordinate and the cell by the same
    // factor leaves the integer shifts unchanged.
    double max_abs = 0.0;
    for (std::size_t k = 0; k < 3 * n; ++k) {
        max_abs = std::max(max_abs, std::abs(positions[k]));
    }
    const double scale = options.cutoff < kScaleBelow ? kScaleBelow / options.cutoff : 1.0;
    const double search = search_cutoff(options.cutoff, max_abs) * scale;
    if (!std::isfinite(scale) || !std::isfinite(search) || search <= kMinCutoff) {
        throw std::invalid_argument("neighbor_pairs: cutoff is too small to search");
    }

    std::vector<double> scaled_positions;
    const double* search_positions = positions;
    Mat3 search_cell = cell;
    if (scale != 1.0) {
        scaled_positions.resize(3 * n);
        for (std::size_t k = 0; k < 3 * n; ++k) {
            scaled_positions[k] = positions[k] * scale;
        }
        search_positions = scaled_positions.data();
        search_cell = cell * scale;
    }

    const Mat3 box = vesin_box(search_positions, n, search_cell, pbc, search);
    double box_c[3][3];
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            box_c[r][c] = box(r, c);
        }
    }
    const bool periodic_c[3] = {pbc[0], pbc[1], pbc[2]};

    VesinOptions vopts{};
    vopts.cutoff = search;
    vopts.full = options.full;
    vopts.sorted = false;  // finalize_pairs sorts by (i, j, S)
    vopts.algorithm = VesinAutoAlgorithm;
    vopts.skin = 0.0;
#if defined(__EMSCRIPTEN__)
    vopts.n_threads = 1;  // no pthreads in the default WASM build
#else
    vopts.n_threads = 0;  // vesin picks the thread count
#endif
    vopts.return_shifts = true;
    vopts.return_distances = false;
    vopts.return_vectors = false;

    const auto* points = reinterpret_cast<const double (*)[3]>(search_positions);
    VesinList result;
    const char* error = nullptr;
    const int status = vesin_neighbors(points, n, box_c, periodic_c, VesinDevice{VesinCPU, 0},
                                       vopts, &result.list, &error);
    if (status != EXIT_SUCCESS) {
        throw std::runtime_error(std::string("neighbor_pairs: vesin failed: ") +
                                 (error != nullptr ? error : "unknown error"));
    }
    return finalize_pairs(positions, cell, pbc, options.cutoff, options.sorted, result.list);
}

NeighborPairs neighbor_pairs(const Atoms& atoms, const NeighborOptions& options) {
    return neighbor_pairs(atoms.positions_data(), atoms.size(), atoms.cell().matrix(), atoms.pbc(),
                          options);
}

std::vector<std::int32_t> coordination_numbers(const NeighborPairs& pairs, std::size_t n) {
    // Counts the pairs whose first index is each atom. This is the coordination
    // for a full list; a half list stores each bond once, so use full=true.
    std::vector<std::int32_t> counts(n, 0);
    for (const std::int32_t i : pairs.i) {
        if (i < 0 || static_cast<std::size_t>(i) >= n) {
            throw std::out_of_range("coordination_numbers: pair index outside [0, n)");
        }
        ++counts[static_cast<std::size_t>(i)];
    }
    return counts;
}

}  // namespace fastatoms
