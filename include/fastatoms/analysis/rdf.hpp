// SPDX-License-Identifier: MIT
#pragma once

/// \file rdf.hpp
/// \brief The radial distribution function, g(r).
///
/// How much more likely you are to find an atom a distance r from another than
/// you would be in an ideal gas of the same density. It is the first thing to
/// look at in a liquid, the thing that says whether a glass is really a glass,
/// and the quantity a structure factor is the Fourier transform of.
///
/// Pairs come from the neighbour list, so the cost is linear in the number of
/// atoms rather than quadratic, and the minimum-image convention and periodic
/// boundaries are handled there rather than here.

#include <stdexcept>
#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"

namespace fastatoms::analysis {

/// Raised for a trajectory or a request that cannot be analysed.
class AnalysisError : public std::runtime_error {
public:
    explicit AnalysisError(const std::string& message) : std::runtime_error(message) {}
};

/// What to compute.
struct RDFOptions {
    /// Largest separation to bin, in angstrom.
    double rmax{10.0};
    /// Number of bins between zero and `rmax`.
    int bins{200};

    /// Which species the first and second atom of a pair may be, by chemical
    /// symbol. Empty means any, so the default is the total g(r).
    ///
    /// `{"O"}, {"O"}` gives the oxygen-oxygen correlation, `{"O"}, {"H"}` the
    /// oxygen-hydrogen one.
    std::vector<std::string> first{};
    std::vector<std::string> second{};
};

/// A computed g(r), and the pieces it was built from.
struct RadialDistribution {
    /// Bin centres, in angstrom.
    std::vector<double> r{};
    /// The pair correlation itself, which tends to one at large r.
    std::vector<double> g{};
    /// Raw ordered-pair counts per bin, summed over every frame. Useful when
    /// you want to normalise differently, or to combine runs.
    std::vector<double> counts{};
    /// The running coordination number: how many second-species atoms lie
    /// within r of a first-species one, on average. Its plateaux are the
    /// coordination shells.
    std::vector<double> coordination{};

    double rmax{0.0};
    double bin_width{0.0};
    int frames{0};

    /// Atoms of each selection, per frame, and the number density of the
    /// second selection used for the normalisation.
    double first_atoms{0.0};
    double second_atoms{0.0};
    double density{0.0};
    double volume{0.0};

    /// The largest radius at which g(r) is free of periodic self-overlap:
    /// half the smallest perpendicular width of the cell. Beyond it a shell no
    /// longer fits inside the box and the result is biased, which is worth
    /// knowing rather than being protected from.
    double reliable_rmax{0.0};

    /// e.g. "O-H", or "all-all".
    std::string label{};

    [[nodiscard]] std::size_t size() const noexcept { return r.size(); }
    [[nodiscard]] std::string to_string() const;
};

/// Accumulate g(r) one frame at a time.
///
/// A trajectory of ten thousand frames need not be held in memory: feed them
/// in as they are read, and ask for the result at the end.
class RadialDistributionAccumulator {
public:
    explicit RadialDistributionAccumulator(RDFOptions options = {});

    /// Add one frame. Every frame must have the same species content, since
    /// the normalisation is an average over frames.
    /// \throws AnalysisError if the frame has no cell, or does not match the
    ///         frames already added.
    void add(const Atoms& frame);

    [[nodiscard]] int frames() const noexcept { return frames_; }
    [[nodiscard]] RadialDistribution result() const;

private:
    RDFOptions options_;
    std::vector<double> counts_;
    int frames_{0};
    double first_atoms_{0.0};
    double second_atoms_{0.0};
    double volume_{0.0};
    double reliable_rmax_{0.0};
    std::size_t natoms_{0};
};

/// g(r) over a whole trajectory.
[[nodiscard]] RadialDistribution radial_distribution(const std::vector<Atoms>& frames,
                                                     RDFOptions options = {});

/// g(r) of a single structure.
[[nodiscard]] RadialDistribution radial_distribution(const Atoms& atoms,
                                                     RDFOptions options = {});

}  // namespace fastatoms::analysis
