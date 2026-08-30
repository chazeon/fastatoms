// SPDX-License-Identifier: MIT

#include "fastatoms/analysis/rdf.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "fastatoms/core/neighbor_list.hpp"

namespace fastatoms::analysis {
namespace {

constexpr double kPi = 3.14159265358979323846;

/// Whether an atom belongs to a selection. An empty selection means all.
bool selected(const Atoms& atoms, std::size_t index, const std::vector<std::string>& species) {
    if (species.empty()) {
        return true;
    }
    const std::string symbol = atoms.symbol(index);
    return std::find(species.begin(), species.end(), symbol) != species.end();
}

std::string selection_label(const std::vector<std::string>& species) {
    if (species.empty()) {
        return "all";
    }
    std::string out;
    for (const std::string& symbol : species) {
        if (!out.empty()) {
            out += ",";
        }
        out += symbol;
    }
    return out;
}

}  // namespace

std::string RadialDistribution::to_string() const {
    std::ostringstream out;
    out << "RadialDistribution(" << label << ", " << size() << " bins to " << rmax << " A, "
        << frames << " frame" << (frames == 1 ? "" : "s") << ")";
    return out.str();
}

RadialDistributionAccumulator::RadialDistributionAccumulator(RDFOptions options)
    : options_(std::move(options)) {
    if (!(options_.rmax > 0.0)) {
        throw AnalysisError("rmax must be positive");
    }
    if (options_.bins < 1) {
        throw AnalysisError("a histogram needs at least one bin");
    }
    counts_.assign(static_cast<std::size_t>(options_.bins), 0.0);
}

void RadialDistributionAccumulator::add(const Atoms& frame) {
    if (frame.empty()) {
        throw AnalysisError("cannot bin an empty frame");
    }
    if (!frame.cell().is_invertible()) {
        throw AnalysisError("g(r) needs a cell: without one there is no density to "
                            "normalise against");
    }
    if (frames_ > 0 && frame.size() != natoms_) {
        throw AnalysisError("frame " + std::to_string(frames_) + " has " +
                            std::to_string(frame.size()) + " atoms where the first had " +
                            std::to_string(natoms_) +
                            "; g(r) averages over frames and needs them to match");
    }
    natoms_ = frame.size();

    // Which atoms are in which selection, worked out once per frame rather
    // than once per pair.
    std::vector<char> in_first(frame.size());
    std::vector<char> in_second(frame.size());
    double first_count = 0.0;
    double second_count = 0.0;
    for (std::size_t k = 0; k < frame.size(); ++k) {
        in_first[k] = selected(frame, k, options_.first) ? 1 : 0;
        in_second[k] = selected(frame, k, options_.second) ? 1 : 0;
        first_count += in_first[k];
        second_count += in_second[k];
    }

    NeighborListOptions neighbours;
    neighbours.cutoff = options_.rmax;
    // Each unordered pair once; both orderings are counted below, which costs
    // the same and halves the list.
    neighbours.full = false;
    neighbours.store_vectors = false;
    neighbours.store_distances = true;
    const NeighborList list = build_neighbor_list(frame, neighbours);

    const double scale = static_cast<double>(options_.bins) / options_.rmax;
    for (std::size_t p = 0; p < list.size(); ++p) {
        const double distance = list.distances[p];
        if (!(distance < options_.rmax) || distance <= 0.0) {
            continue;
        }
        const auto bin = static_cast<std::size_t>(distance * scale);
        if (bin >= counts_.size()) {
            continue;
        }
        const auto a = static_cast<std::size_t>(list.i[p]);
        const auto b = static_cast<std::size_t>(list.j[p]);
        // Ordered pairs: (a in first, b in second) and the other way round.
        counts_[bin] += (in_first[a] && in_second[b]) ? 1.0 : 0.0;
        counts_[bin] += (in_first[b] && in_second[a]) ? 1.0 : 0.0;
    }

    first_atoms_ += first_count;
    second_atoms_ += second_count;
    volume_ += frame.volume();
    const Vec3 heights = frame.cell().heights();
    const double half = 0.5 * std::min({heights.x, heights.y, heights.z});
    reliable_rmax_ = frames_ == 0 ? half : std::min(reliable_rmax_, half);
    ++frames_;
}

RadialDistribution RadialDistributionAccumulator::result() const {
    if (frames_ == 0) {
        throw AnalysisError("no frames have been added");
    }
    RadialDistribution out;
    out.rmax = options_.rmax;
    out.frames = frames_;
    out.first_atoms = first_atoms_ / frames_;
    out.second_atoms = second_atoms_ / frames_;
    out.volume = volume_ / frames_;
    out.reliable_rmax = reliable_rmax_;
    out.label = selection_label(options_.first) + "-" + selection_label(options_.second);

    const double width = options_.rmax / options_.bins;
    out.bin_width = width;
    out.density = out.volume > 0.0 ? out.second_atoms / out.volume : 0.0;

    // The ideal-gas count that g(r) is measured against. Using N_B - 1 rather
    // than N_B when the two selections are the same is what makes g(r) come out
    // at exactly one for an ideal gas of finitely many atoms, rather than one
    // minus 1/N.
    const bool same = options_.first == options_.second;
    const double partners = same ? out.second_atoms - 1.0 : out.second_atoms;
    const double pair_density = out.volume > 0.0 ? out.first_atoms * partners / out.volume : 0.0;

    double running = 0.0;
    for (std::size_t k = 0; k < counts_.size(); ++k) {
        const double inner = k * width;
        const double outer = inner + width;
        out.r.push_back(0.5 * (inner + outer));
        out.counts.push_back(counts_[k]);

        const double shell = (4.0 / 3.0) * kPi * (outer * outer * outer - inner * inner * inner);
        const double ideal = pair_density * shell * frames_;
        out.g.push_back(ideal > 0.0 ? counts_[k] / ideal : 0.0);

        // The coordination number is the raw count per first-selection atom,
        // which needs no ideal-gas reference at all.
        running += out.first_atoms > 0.0
                       ? counts_[k] / (out.first_atoms * frames_)
                       : 0.0;
        out.coordination.push_back(running);
    }
    return out;
}

RadialDistribution radial_distribution(const std::vector<Atoms>& frames, RDFOptions options) {
    if (frames.empty()) {
        throw AnalysisError("a trajectory needs at least one frame");
    }
    RadialDistributionAccumulator accumulator(std::move(options));
    for (const Atoms& frame : frames) {
        accumulator.add(frame);
    }
    return accumulator.result();
}

RadialDistribution radial_distribution(const Atoms& atoms, RDFOptions options) {
    RadialDistributionAccumulator accumulator(std::move(options));
    accumulator.add(atoms);
    return accumulator.result();
}

}  // namespace fastatoms::analysis
