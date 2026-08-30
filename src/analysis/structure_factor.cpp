// SPDX-License-Identifier: MIT

#include "fastatoms/analysis/structure_factor.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "fastatoms/core/neighbor_list.hpp"

namespace fastatoms::analysis {
namespace {

constexpr double kPi = 3.14159265358979323846;

/// `sin(x)/x`, continued to one at the origin where the ratio is not defined
/// but the limit is.
double sinc(double x) {
    return std::abs(x) < 1e-8 ? 1.0 - x * x / 6.0 : std::sin(x) / x;
}

/// The Lorch window, which tapers the integrand to zero at the cutoff so the
/// truncation does not ring.
double lorch(double r, double rmax) {
    if (rmax <= 0.0) {
        return 1.0;
    }
    return sinc(kPi * r / rmax);
}

std::vector<double> q_grid(const StructureFactorOptions& options) {
    if (options.points < 1) {
        throw AnalysisError("S(q) needs at least one q point");
    }
    if (!(options.qmax > options.qmin) || options.qmin < 0.0) {
        throw AnalysisError("the q range must run upwards from a non-negative qmin");
    }
    std::vector<double> q;
    q.reserve(static_cast<std::size_t>(options.points));
    const double step = options.points == 1
                            ? 0.0
                            : (options.qmax - options.qmin) / (options.points - 1);
    for (int k = 0; k < options.points; ++k) {
        q.push_back(options.qmin + k * step);
    }
    return q;
}

}  // namespace

std::string StructureFactor::to_string() const {
    std::ostringstream out;
    out << "StructureFactor(" << method << ", " << size() << " points, " << frames << " frame"
        << (frames == 1 ? "" : "s") << ")";
    return out.str();
}

StructureFactor structure_factor_debye(const std::vector<Atoms>& frames,
                                       StructureFactorOptions options) {
    if (frames.empty()) {
        throw AnalysisError("a trajectory needs at least one frame");
    }
    StructureFactor out;
    out.method = "debye";
    out.frames = static_cast<int>(frames.size());
    out.q = q_grid(options);
    out.s.assign(out.q.size(), 0.0);

    double total_atoms = 0.0;
    for (const Atoms& frame : frames) {
        if (frame.empty()) {
            throw AnalysisError("cannot transform an empty frame");
        }
        const bool periodic = frame.pbc()[0] || frame.pbc()[1] || frame.pbc()[2];

        // How far to go. For a periodic frame the pairs beyond half the cell
        // width are images of nearer ones, so the sum is cut there unless the
        // caller asked otherwise.
        double rmax = options.rmax;
        if (rmax <= 0.0) {
            if (periodic && frame.cell().is_invertible()) {
                const Vec3 heights = frame.cell().heights();
                rmax = 0.5 * std::min({heights.x, heights.y, heights.z});
            } else {
                // A cluster: every pair, so the diameter is enough.
                for (std::size_t i = 0; i < frame.size(); ++i) {
                    for (std::size_t j = i + 1; j < frame.size(); ++j) {
                        rmax = std::max(rmax, (frame.position(i) - frame.position(j)).norm());
                    }
                }
                rmax *= 1.001;
            }
        }
        if (!(rmax > 0.0)) {
            throw AnalysisError("no pairs to sum: the frame has no extent");
        }

        // Distances once, then every q against them.
        std::vector<double> distances;
        if (periodic && frame.cell().is_invertible()) {
            NeighborListOptions neighbours;
            neighbours.cutoff = rmax;
            neighbours.full = false;
            neighbours.store_vectors = false;
            neighbours.store_distances = true;
            const NeighborList list = build_neighbor_list(frame, neighbours);
            distances = list.distances;
        } else {
            for (std::size_t i = 0; i < frame.size(); ++i) {
                for (std::size_t j = i + 1; j < frame.size(); ++j) {
                    const double d = (frame.position(i) - frame.position(j)).norm();
                    if (d < rmax) {
                        distances.push_back(d);
                    }
                }
            }
        }

        const double n = static_cast<double>(frame.size());

        // For a periodic frame the sum had to be cut off, and what was cut off
        // is not nothing: a uniform sea of atoms out to infinity. Its
        // contribution has to come back off, or the answer carries the
        // ideal-gas background instead of the deviation from it - which shows
        // up as S(q) sitting well above one at small q for a structureless
        // liquid. A finite cluster in vacuum has no such background, and gets
        // the Debye equation as it is usually written.
        const bool subtract_background = periodic && frame.cell().is_invertible();
        const double density = subtract_background ? n / frame.volume() : 0.0;

        for (std::size_t k = 0; k < out.q.size(); ++k) {
            const double q = out.q[k];
            double sum = 0.0;
            for (const double r : distances) {
                const double window = options.lorch ? lorch(r, rmax) : 1.0;
                sum += sinc(q * r) * window;
            }
            // Twice, because each unordered pair stands for two ordered ones.
            double value = 2.0 * sum / n;

            if (subtract_background) {
                // 4 pi rho integral of r^2 sinc(qr) w(r) dr out to the cutoff,
                // on a fine grid: the window makes an analytic form awkward and
                // the integral is cheap.
                constexpr int kSteps = 2000;
                const double step = rmax / kSteps;
                double background = 0.0;
                for (int step_index = 0; step_index < kSteps; ++step_index) {
                    const double r = (step_index + 0.5) * step;
                    const double window = options.lorch ? lorch(r, rmax) : 1.0;
                    background += r * r * sinc(q * r) * window * step;
                }
                value -= 4.0 * kPi * density * background;
            }
            out.s[k] += value;
        }
        total_atoms += n;
    }

    for (double& value : out.s) {
        value = 1.0 + value / frames.size();
    }
    (void)total_atoms;
    return out;
}

StructureFactor structure_factor_from_rdf(const RadialDistribution& rdf,
                                          StructureFactorOptions options) {
    if (rdf.size() == 0) {
        throw AnalysisError("that g(r) is empty");
    }
    StructureFactor out;
    out.method = "rdf";
    out.frames = rdf.frames;
    out.q = q_grid(options);
    out.s.reserve(out.q.size());

    const double rmax = options.rmax > 0.0 ? options.rmax : rdf.rmax;
    for (const double q : out.q) {
        double integral = 0.0;
        for (std::size_t k = 0; k < rdf.size(); ++k) {
            const double r = rdf.r[k];
            if (r > rmax) {
                break;
            }
            const double window = options.lorch ? lorch(r, rmax) : 1.0;
            integral += r * r * (rdf.g[k] - 1.0) * sinc(q * r) * window * rdf.bin_width;
        }
        out.s.push_back(1.0 + 4.0 * kPi * rdf.density * integral);
    }
    return out;
}

StructureFactor structure_factor_reciprocal(const std::vector<Atoms>& frames, double qmax,
                                            int bins) {
    if (frames.empty()) {
        throw AnalysisError("a trajectory needs at least one frame");
    }
    if (!(qmax > 0.0)) {
        throw AnalysisError("qmax must be positive");
    }
    if (bins < 1) {
        throw AnalysisError("a histogram needs at least one bin");
    }

    StructureFactor out;
    out.method = "reciprocal";
    out.frames = static_cast<int>(frames.size());
    std::vector<double> sums(static_cast<std::size_t>(bins), 0.0);
    std::vector<double> weights(static_cast<std::size_t>(bins), 0.0);

    for (const Atoms& frame : frames) {
        if (!frame.cell().is_invertible()) {
            throw AnalysisError("summing over the reciprocal lattice needs a cell");
        }
        // Rows of the reciprocal matrix, times 2 pi: the wavevectors a
        // periodic cell supports are integer combinations of these, and no
        // others - which is why this method needs no window.
        const Mat3 reciprocal = frame.cell().reciprocal();
        const Vec3 b1{reciprocal(0, 0), reciprocal(0, 1), reciprocal(0, 2)};
        const Vec3 b2{reciprocal(1, 0), reciprocal(1, 1), reciprocal(1, 2)};
        const Vec3 b3{reciprocal(2, 0), reciprocal(2, 1), reciprocal(2, 2)};

        const int n1 = static_cast<int>(std::ceil(qmax / (2 * kPi * b1.norm()))) + 1;
        const int n2 = static_cast<int>(std::ceil(qmax / (2 * kPi * b2.norm()))) + 1;
        const int n3 = static_cast<int>(std::ceil(qmax / (2 * kPi * b3.norm()))) + 1;

        const double scale = bins / qmax;
        const double natoms = static_cast<double>(frame.size());

        for (int h = -n1; h <= n1; ++h) {
            for (int k = -n2; k <= n2; ++k) {
                for (int l = -n3; l <= n3; ++l) {
                    if (h == 0 && k == 0 && l == 0) {
                        continue;   // the forward beam carries no structure
                    }
                    const Vec3 q{2 * kPi * (h * b1.x + k * b2.x + l * b3.x),
                                 2 * kPi * (h * b1.y + k * b2.y + l * b3.y),
                                 2 * kPi * (h * b1.z + k * b2.z + l * b3.z)};
                    const double magnitude = q.norm();
                    if (magnitude >= qmax) {
                        continue;
                    }
                    double real = 0.0;
                    double imaginary = 0.0;
                    for (std::size_t a = 0; a < frame.size(); ++a) {
                        const double phase = dot(q, frame.position(a));
                        real += std::cos(phase);
                        imaginary += std::sin(phase);
                    }
                    const auto bin = static_cast<std::size_t>(magnitude * scale);
                    if (bin < sums.size()) {
                        sums[bin] += (real * real + imaginary * imaginary) / natoms;
                        weights[bin] += 1.0;
                    }
                }
            }
        }
    }

    const double width = qmax / bins;
    for (std::size_t k = 0; k < sums.size(); ++k) {
        out.q.push_back((k + 0.5) * width);
        out.s.push_back(weights[k] > 0.0 ? sums[k] / weights[k] : 0.0);
        out.weight.push_back(weights[k]);
    }
    return out;
}

}  // namespace fastatoms::analysis
