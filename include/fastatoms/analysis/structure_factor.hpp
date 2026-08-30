// SPDX-License-Identifier: MIT
#pragma once

/// \file structure_factor.hpp
/// \brief The static structure factor, S(q).
///
/// What a diffraction experiment measures, and the Fourier partner of g(r).
/// Three routes are offered because they are good at different things and
/// disagreeing with each other is diagnostic:
///
///   - the Debye scattering equation, exact for a finite cluster and the right
///     tool for a nanoparticle;
///   - the transform of g(r), which is what you use when g(r) is what you have;
///   - the sum over the reciprocal lattice, which is exact for a periodic cell
///     and is the one to trust for bulk MD.
///
/// The first two need a truncation, and a window to keep the truncation from
/// ringing; the third needs neither, but only knows the q values the cell
/// allows.

#include <string>
#include <vector>

#include "fastatoms/analysis/rdf.hpp"
#include "fastatoms/core/atoms.hpp"

namespace fastatoms::analysis {

/// A computed S(q).
struct StructureFactor {
    /// Wavevector magnitudes, in inverse angstrom.
    std::vector<double> q{};
    /// The structure factor itself, which tends to one at large q.
    std::vector<double> s{};
    /// How many reciprocal lattice points fell in each bin. Only the
    /// reciprocal-lattice method fills this; a bin with few points is noisy.
    std::vector<double> weight{};

    int frames{0};
    /// "debye", "rdf" or "reciprocal".
    std::string method{};

    [[nodiscard]] std::size_t size() const noexcept { return q.size(); }
    [[nodiscard]] std::string to_string() const;
};

/// Settings shared by the transform-based methods.
struct StructureFactorOptions {
    double qmin{0.5};
    double qmax{15.0};
    int points{300};

    /// Apply the Lorch window, `sin(pi r / rmax) / (pi r / rmax)`.
    ///
    /// Truncating the integral at a finite rmax puts ripples into S(q) that
    /// look like structure and are not. The window removes them at the cost of
    /// broadening real peaks, which is the trade every diffraction study makes.
    bool lorch{true};

    /// Largest pair separation to include, for the Debye sum. Zero means half
    /// the smallest cell width for a periodic frame, or every pair for a
    /// cluster.
    double rmax{0.0};
};

/// S(q) by the Debye scattering equation.
///
/// `S(q) = 1 + (1/N) sum over i != j of sin(q r_ij) / (q r_ij)`.
///
/// Quadratic in the number of atoms, and exact for a finite cluster. For a
/// periodic cell the sum is cut off at `rmax`, which makes it approximate; the
/// reciprocal-lattice method below has no such problem.
[[nodiscard]] StructureFactor structure_factor_debye(const std::vector<Atoms>& frames,
                                                     StructureFactorOptions options = {});

/// S(q) from a g(r) that has already been computed.
///
/// `S(q) = 1 + 4 pi rho integral r^2 (g(r) - 1) sin(qr)/(qr) w(r) dr`.
[[nodiscard]] StructureFactor structure_factor_from_rdf(const RadialDistribution& rdf,
                                                        StructureFactorOptions options = {});

/// S(q) by summing over the reciprocal lattice.
///
/// `S(q) = < |sum_j exp(i q . r_j)|^2 > / N`, evaluated at the wavevectors the
/// periodic cell actually supports and binned by magnitude. Exact for a
/// periodic system - no cutoff, no window, no truncation ripple - but it only
/// knows about the discrete q the cell allows, so a small cell gives a coarse
/// curve.
[[nodiscard]] StructureFactor structure_factor_reciprocal(const std::vector<Atoms>& frames,
                                                          double qmax = 15.0, int bins = 150);

}  // namespace fastatoms::analysis
