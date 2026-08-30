// SPDX-License-Identifier: MIT
#pragma once

/// \file bands.hpp
/// \brief Eigenvalues on a set of k-points, and the gaps they imply.

#include <stdexcept>
#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"

namespace fastatoms::electronic {

/// Raised for a band structure or an integration that cannot be carried out.
class ElectronicError : public std::runtime_error {
public:
    explicit ElectronicError(const std::string& message) : std::runtime_error(message) {}
};

/// Where a band gap is and how big.
struct BandGap {
    /// The fundamental gap: the smallest separation between any occupied and
    /// any empty state, wherever in the zone they are. Zero for a metal.
    double gap{0.0};
    /// The smallest separation at a single k-point. Never less than `gap`, and
    /// the one that matters for optical absorption.
    double direct_gap{0.0};

    /// True when a band crosses the Fermi level, in which case both gaps are
    /// zero and the rest of this is not meaningful.
    bool metallic{false};
    /// True when the fundamental gap is found at one k-point, so the two gaps
    /// coincide.
    bool direct{false};

    double valence_maximum{0.0};
    double conduction_minimum{0.0};

    std::size_t valence_kpoint{0};
    std::size_t conduction_kpoint{0};
    std::size_t direct_kpoint{0};
    std::size_t valence_band{0};
    std::size_t conduction_band{0};

    [[nodiscard]] std::string to_string() const;
};

/// Eigenvalues on a set of k-points.
///
/// Energies are stored flat in `[spin][kpoint][band]` order, which is how every
/// electronic structure code writes them and what keeps the band loop
/// contiguous.
class BandStructure {
public:
    BandStructure() = default;

    /// \param kpoints  fractional coordinates in the reciprocal cell
    /// \param energies `spins * kpoints * bands` values in eV
    /// \param bands    how many bands per k-point
    /// \param spins    1 for a non-spin-polarised calculation, 2 otherwise
    /// \param weights  k-point weights; empty means all equal
    /// \throws ElectronicError if the sizes do not agree
    BandStructure(std::vector<Vec3> kpoints, std::vector<double> energies, std::size_t bands,
                  std::size_t spins = 1, double fermi_level = 0.0,
                  std::vector<double> weights = {});

    [[nodiscard]] std::size_t nkpoints() const noexcept { return kpoints_.size(); }
    [[nodiscard]] std::size_t nbands() const noexcept { return bands_; }
    [[nodiscard]] std::size_t nspins() const noexcept { return spins_; }
    [[nodiscard]] bool empty() const noexcept { return kpoints_.empty() || bands_ == 0; }

    [[nodiscard]] const std::vector<Vec3>& kpoints() const noexcept { return kpoints_; }
    [[nodiscard]] const std::vector<double>& energies() const noexcept { return energies_; }
    [[nodiscard]] const std::vector<double>& weights() const noexcept { return weights_; }

    /// One eigenvalue. \throws ElectronicError if any index is out of range.
    [[nodiscard]] double energy(std::size_t spin, std::size_t kpoint, std::size_t band) const;

    /// The whole band `band` of spin `spin`, one value per k-point - what a
    /// band-structure plot draws as a single line.
    [[nodiscard]] std::vector<double> band(std::size_t spin, std::size_t band) const;

    [[nodiscard]] double fermi_level() const noexcept { return fermi_level_; }
    void set_fermi_level(double value) noexcept { fermi_level_ = value; }

    [[nodiscard]] double minimum() const;
    [[nodiscard]] double maximum() const;

    /// Where the gap is and how big. States at or below the Fermi level count
    /// as occupied.
    [[nodiscard]] BandGap band_gap() const;

    /// True when a band crosses the Fermi level.
    [[nodiscard]] bool is_metal(double tolerance = 1e-6) const;

    /// A copy with every energy measured from the Fermi level, which is what a
    /// plot wants.
    [[nodiscard]] BandStructure shifted() const;

    /// Read a band structure out of a calculator's extra results.
    ///
    /// Codes that report eigenvalues put them there under `eigenvalues`, with
    /// `kpoints` and `fermi_level` alongside. None of the calculators shipped
    /// here produce them yet, so this is the hook for one that does rather
    /// than something with a caller today.
    ///
    /// \throws ElectronicError if the calculator reported no eigenvalues.
    [[nodiscard]] static BandStructure from_calculator(Atoms& atoms);

    [[nodiscard]] std::string to_string() const;

private:
    std::vector<Vec3> kpoints_{};
    std::vector<double> energies_{};
    std::vector<double> weights_{};
    std::size_t bands_{0};
    std::size_t spins_{1};
    double fermi_level_{0.0};
};

}  // namespace fastatoms::electronic
