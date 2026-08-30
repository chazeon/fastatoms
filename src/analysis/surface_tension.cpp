// SPDX-License-Identifier: MIT

#include "fastatoms/analysis/surface_tension.hpp"

#include <cmath>
#include <sstream>

#include "fastatoms/md/types.hpp"

namespace fastatoms::analysis {
namespace {

void check_axis(int normal) {
    if (normal < 0 || normal > 2) {
        throw AnalysisError("the interface normal must be 0, 1 or 2, not " +
                            std::to_string(normal));
    }
}

}  // namespace

std::string SurfaceTension::to_string() const {
    std::ostringstream out;
    out << "SurfaceTension(" << gamma_mN_per_m << " +/- " << standard_error_mN_per_m
        << " mN/m, normal=" << "xyz"[normal] << ", " << frames << " frame"
        << (frames == 1 ? "" : "s") << ")";
    return out.str();
}

Mat3 pressure_tensor(Atoms& atoms) {
    // `total_stress` is the potential stress plus the kinetic one, in the ASE
    // convention where sigma = (1/V) dE/d(strain). Pressure is its negative,
    // which is what makes a compressed solid read positive.
    Mat3 pressure = md::total_stress(atoms);
    for (std::size_t k = 0; k < 9; ++k) {
        pressure.m[k] = -pressure.m[k];
    }
    return pressure;
}

SurfaceTension surface_tension_from_pressures(const std::vector<Mat3>& pressures,
                                              const std::vector<double>& lengths, int normal) {
    check_axis(normal);
    if (pressures.empty()) {
        throw AnalysisError("a trajectory needs at least one frame");
    }
    if (pressures.size() != lengths.size()) {
        throw AnalysisError("there are " + std::to_string(pressures.size()) +
                            " pressure tensors but " + std::to_string(lengths.size()) +
                            " cell lengths");
    }

    const std::size_t n = static_cast<std::size_t>(normal);
    const std::size_t t1 = (n + 1) % 3;
    const std::size_t t2 = (n + 2) % 3;

    SurfaceTension out;
    out.normal = normal;
    out.frames = static_cast<int>(pressures.size());
    out.per_frame.reserve(pressures.size());

    Mat3 mean{};
    for (std::size_t f = 0; f < pressures.size(); ++f) {
        const Mat3& p = pressures[f];
        // The normal component less the average of the two in the plane, times
        // half the box length: the half is the two interfaces a periodic slab
        // necessarily has, not a normalisation of the average.
        const double difference =
            p(n, n) - 0.5 * (p(t1, t1) + p(t2, t2));
        out.per_frame.push_back(0.5 * lengths[f] * difference);
        for (std::size_t k = 0; k < 9; ++k) {
            mean.m[k] += p.m[k] / static_cast<double>(pressures.size());
        }
    }
    out.mean_pressure = mean;

    double sum = 0.0;
    for (const double value : out.per_frame) {
        sum += value;
    }
    out.gamma = sum / out.per_frame.size();

    if (out.per_frame.size() > 1) {
        double variance = 0.0;
        for (const double value : out.per_frame) {
            const double deviation = value - out.gamma;
            variance += deviation * deviation;
        }
        variance /= static_cast<double>(out.per_frame.size() - 1);
        out.standard_error = std::sqrt(variance / static_cast<double>(out.per_frame.size()));
    }

    out.gamma_mN_per_m = out.gamma * kMilliNewtonPerMetre;
    out.standard_error_mN_per_m = out.standard_error * kMilliNewtonPerMetre;
    return out;
}

SurfaceTension surface_tension(std::vector<Atoms>& frames, int normal) {
    check_axis(normal);
    if (frames.empty()) {
        throw AnalysisError("a trajectory needs at least one frame");
    }

    std::vector<Mat3> pressures;
    std::vector<double> lengths;
    pressures.reserve(frames.size());
    lengths.reserve(frames.size());
    for (Atoms& frame : frames) {
        if (!frame.cell().is_invertible()) {
            throw AnalysisError("surface tension needs a cell to measure the slab across");
        }
        pressures.push_back(pressure_tensor(frame));
        const Vec3 lengths_of = frame.cell().lengths();
        lengths.push_back(normal == 0 ? lengths_of.x : (normal == 1 ? lengths_of.y
                                                                   : lengths_of.z));
    }
    return surface_tension_from_pressures(pressures, lengths, normal);
}

}  // namespace fastatoms::analysis
