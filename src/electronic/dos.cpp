// SPDX-License-Identifier: MIT

#include "fastatoms/electronic/dos.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace fastatoms::electronic {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSqrtPi = 1.7724538509055159;

/// The physicists' Hermite polynomial `H_n(x)`, by the standard recurrence.
double hermite(int n, double x) {
    double previous = 1.0;   // H_0
    if (n == 0) {
        return previous;
    }
    double current = 2.0 * x;   // H_1
    for (int k = 2; k <= n; ++k) {
        const double next = 2.0 * x * current - 2.0 * (k - 1) * previous;
        previous = current;
        current = next;
    }
    return current;
}

double factorial(int n) {
    double out = 1.0;
    for (int k = 2; k <= n; ++k) {
        out *= k;
    }
    return out;
}

int default_spin_degeneracy(std::size_t spins) { return spins == 1 ? 2 : 1; }

/// The energy window, taken from the data when the caller did not give one.
std::pair<double, double> window(const BandStructure& bands, const DOSOptions& options) {
    if (options.emax > options.emin) {
        return {options.emin, options.emax};
    }
    const double margin = std::max(5.0 * options.width, 0.5);
    return {bands.minimum() - margin, bands.maximum() + margin};
}

}  // namespace

std::string_view smearing_name(Smearing smearing) {
    switch (smearing) {
        case Smearing::Gaussian:         return "gaussian";
        case Smearing::MethfesselPaxton: return "methfessel-paxton";
        case Smearing::FermiDirac:       return "fermi-dirac";
    }
    return "gaussian";
}

Smearing smearing_from_string(std::string_view name) {
    std::string key(name);
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    key.erase(std::remove_if(key.begin(), key.end(),
                             [](unsigned char c) { return c == '-' || c == '_' || c == ' '; }),
              key.end());
    if (key == "gaussian" || key == "gauss") return Smearing::Gaussian;
    if (key == "methfesselpaxton" || key == "mp") return Smearing::MethfesselPaxton;
    if (key == "fermidirac" || key == "fd" || key == "fermi") return Smearing::FermiDirac;
    throw ElectronicError("'" + std::string(name) +
                          "' is not a smearing; use gaussian, methfessel-paxton or fermi-dirac");
}

double smearing_delta(double x, Smearing smearing, int order) {
    switch (smearing) {
        case Smearing::Gaussian:
            return std::exp(-x * x) / kSqrtPi;
        case Smearing::FermiDirac: {
            // sech^2(x/2) / 4, written so that a large |x| does not overflow.
            const double e = std::exp(-std::abs(x));
            const double denominator = (1.0 + e) * (1.0 + e);
            return e / denominator;
        }
        case Smearing::MethfesselPaxton: {
            if (order < 0) {
                throw ElectronicError("the Methfessel-Paxton order must not be negative");
            }
            // D_N(x) = sum over n of A_n H_2n(x) exp(-x^2), with
            // A_n = (-1)^n / (n! 4^n sqrt(pi)). Order zero is a plain Gaussian.
            const double gaussian = std::exp(-x * x);
            double sum = 0.0;
            for (int n = 0; n <= order; ++n) {
                const double a = ((n % 2 == 0) ? 1.0 : -1.0) /
                                 (factorial(n) * std::pow(4.0, n) * kSqrtPi);
                sum += a * hermite(2 * n, x);
            }
            return sum * gaussian;
        }
    }
    return 0.0;
}

double smearing_occupation(double x, Smearing smearing, int order) {
    switch (smearing) {
        case Smearing::Gaussian:
            return 0.5 * std::erfc(x);
        case Smearing::FermiDirac:
            return 1.0 / (1.0 + std::exp(x));
        case Smearing::MethfesselPaxton: {
            if (order < 0) {
                throw ElectronicError("the Methfessel-Paxton order must not be negative");
            }
            // S_N(x) = erfc(x)/2 + sum over n>=1 of A_n H_{2n-1}(x) exp(-x^2).
            double sum = 0.5 * std::erfc(x);
            const double gaussian = std::exp(-x * x);
            for (int n = 1; n <= order; ++n) {
                const double a = ((n % 2 == 0) ? 1.0 : -1.0) /
                                 (factorial(n) * std::pow(4.0, n) * kSqrtPi);
                sum += a * hermite(2 * n - 1, x) * gaussian;
            }
            return sum;
        }
    }
    return 0.0;
}

double DensityOfStates::at_fermi() const {
    if (energy.empty()) {
        return 0.0;
    }
    if (fermi_level <= energy.front()) {
        return dos.front();
    }
    if (fermi_level >= energy.back()) {
        return dos.back();
    }
    const auto upper = std::lower_bound(energy.begin(), energy.end(), fermi_level);
    const std::size_t k = static_cast<std::size_t>(upper - energy.begin());
    if (k == 0) {
        return dos.front();
    }
    const double span = energy[k] - energy[k - 1];
    const double t = span > 0.0 ? (fermi_level - energy[k - 1]) / span : 0.0;
    return dos[k - 1] + t * (dos[k] - dos[k - 1]);
}

std::string DensityOfStates::to_string() const {
    std::ostringstream out;
    out << "DensityOfStates(" << method << ", " << size() << " points";
    if (spins == 2) {
        out << ", spin-resolved";
    }
    out << ")";
    return out.str();
}

std::vector<Vec3> monkhorst_pack(int n1, int n2, int n3, bool gamma_centred) {
    if (n1 < 1 || n2 < 1 || n3 < 1) {
        throw ElectronicError("a k-point grid needs at least one point in each direction");
    }
    std::vector<Vec3> out;
    out.reserve(static_cast<std::size_t>(n1) * n2 * n3);
    const auto coordinate = [&](int index, int n) {
        return gamma_centred ? static_cast<double>(index) / n
                             : (2.0 * index - n + 1) / (2.0 * n);
    };
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            for (int k = 0; k < n3; ++k) {
                out.push_back(Vec3{coordinate(i, n1), coordinate(j, n2), coordinate(k, n3)});
            }
        }
    }
    return out;
}

DensityOfStates density_of_states(const BandStructure& bands, DOSOptions options) {
    if (bands.empty()) {
        throw ElectronicError("an empty band structure has no density of states");
    }
    if (options.points < 2) {
        throw ElectronicError("a density of states needs at least two energy points");
    }
    if (!(options.width > 0.0)) {
        throw ElectronicError("the smearing width must be positive");
    }

    const int degeneracy = options.spin_degeneracy > 0
                               ? options.spin_degeneracy
                               : default_spin_degeneracy(bands.nspins());

    DensityOfStates out;
    out.method = std::string(smearing_name(options.smearing));
    out.fermi_level = bands.fermi_level();
    out.spins = bands.nspins();

    const auto [emin, emax] = window(bands, options);
    const double step = (emax - emin) / (options.points - 1);
    out.energy.reserve(static_cast<std::size_t>(options.points));
    for (int k = 0; k < options.points; ++k) {
        out.energy.push_back(emin + k * step);
    }
    out.dos.assign(out.energy.size(), 0.0);
    if (bands.nspins() == 2) {
        out.spin_dos.assign(2 * out.energy.size(), 0.0);
    }

    for (std::size_t spin = 0; spin < bands.nspins(); ++spin) {
        for (std::size_t k = 0; k < bands.nkpoints(); ++k) {
            const double weight = bands.weights()[k] * degeneracy;
            for (std::size_t band = 0; band < bands.nbands(); ++band) {
                const double eigenvalue = bands.energy(spin, k, band);
                for (std::size_t point = 0; point < out.energy.size(); ++point) {
                    const double x = (out.energy[point] - eigenvalue) / options.width;
                    if (std::abs(x) > 20.0) {
                        continue;   // the bump has died long before here
                    }
                    const double value =
                        weight * smearing_delta(x, options.smearing, options.order) /
                        options.width;
                    out.dos[point] += value;
                    if (!out.spin_dos.empty()) {
                        out.spin_dos[spin * out.energy.size() + point] += value;
                    }
                }
            }
        }
    }

    out.integrated.assign(out.energy.size(), 0.0);
    double running = 0.0;
    for (std::size_t point = 0; point < out.energy.size(); ++point) {
        if (point > 0) {
            running += 0.5 * (out.dos[point] + out.dos[point - 1]) * step;
        }
        out.integrated[point] = running;
    }
    return out;
}

namespace {

/// The eight corners of a microcell, as offsets in bits: bit 2 is the first
/// axis, bit 1 the second, bit 0 the third.
std::array<int, 3> corner_offset(int corner) {
    return {(corner >> 2) & 1, (corner >> 1) & 1, corner & 1};
}

/// The six tetrahedra of a cube sharing the main diagonal from 0 to 7.
const std::array<std::array<int, 4>, 6>& tetrahedra() {
    static const std::array<std::array<int, 4>, 6> list{{
        {0, 1, 3, 7}, {0, 1, 5, 7}, {0, 2, 3, 7},
        {0, 2, 6, 7}, {0, 4, 5, 7}, {0, 4, 6, 7},
    }};
    return list;
}

/// Blochl's expression for the density of states of one tetrahedron whose
/// corner energies are `e`, sorted ascending, carrying weight `w`.
double tetrahedron_dos(double energy, const std::array<double, 4>& e, double w) {
    const double e1 = e[0], e2 = e[1], e3 = e[2], e4 = e[3];
    if (energy <= e1 || energy >= e4) {
        return 0.0;
    }
    // The three regimes are the three ways the constant-energy plane can cut
    // a tetrahedron: off one corner, across the middle, off the opposite
    // corner. Degenerate corners make denominators vanish, and there the
    // contribution is genuinely zero rather than infinite.
    constexpr double tiny = 1e-12;
    if (energy < e2) {
        const double d = (e2 - e1) * (e3 - e1) * (e4 - e1);
        return d > tiny ? 3.0 * w * (energy - e1) * (energy - e1) / d : 0.0;
    }
    if (energy < e3) {
        const double d = (e3 - e1) * (e4 - e1);
        if (!(d > tiny)) {
            return 0.0;
        }
        const double inner = (e3 - e2) * (e4 - e2);
        const double t = energy - e2;
        double value = 3.0 * (e2 - e1) + 6.0 * t;
        if (inner > tiny) {
            value -= 3.0 * (e3 - e1 + e4 - e2) * t * t / inner;
        }
        return w * value / d;
    }
    const double d = (e4 - e1) * (e4 - e2) * (e4 - e3);
    return d > tiny ? 3.0 * w * (e4 - energy) * (e4 - energy) / d : 0.0;
}

}  // namespace

DensityOfStates density_of_states_tetrahedron(const BandStructure& bands,
                                              std::array<int, 3> grid, DOSOptions options,
                                              Mat3 reciprocal) {
    if (bands.empty()) {
        throw ElectronicError("an empty band structure has no density of states");
    }
    if (grid[0] < 1 || grid[1] < 1 || grid[2] < 1) {
        throw ElectronicError("a k-point grid needs at least one point in each direction");
    }
    const std::size_t expected =
        static_cast<std::size_t>(grid[0]) * grid[1] * grid[2];
    if (bands.nkpoints() != expected) {
        throw ElectronicError(
            "the tetrahedron method needs the bands on the regular grid " +
            std::to_string(grid[0]) + "x" + std::to_string(grid[1]) + "x" +
            std::to_string(grid[2]) + ", which is " + std::to_string(expected) +
            " k-points, but there are " + std::to_string(bands.nkpoints()));
    }
    if (options.points < 2) {
        throw ElectronicError("a density of states needs at least two energy points");
    }

    const int degeneracy = options.spin_degeneracy > 0
                               ? options.spin_degeneracy
                               : default_spin_degeneracy(bands.nspins());

    DensityOfStates out;
    out.method = "tetrahedron";
    out.fermi_level = bands.fermi_level();
    out.spins = bands.nspins();

    DOSOptions windowed = options;
    windowed.width = std::max(options.width, 0.0);
    const auto [emin, emax] = window(bands, windowed);
    const double step = (emax - emin) / (options.points - 1);
    out.energy.reserve(static_cast<std::size_t>(options.points));
    for (int k = 0; k < options.points; ++k) {
        out.energy.push_back(emin + k * step);
    }
    out.dos.assign(out.energy.size(), 0.0);
    if (bands.nspins() == 2) {
        out.spin_dos.assign(2 * out.energy.size(), 0.0);
    }

    // Which body diagonal to cut along. Blochl's recipe is the shortest one,
    // because a long thin tetrahedron interpolates badly; on a cubic grid all
    // four are equal and the choice does not matter.
    int diagonal = 0;
    double shortest = -1.0;
    for (int candidate = 0; candidate < 4; ++candidate) {
        const std::array<int, 3> a = corner_offset(candidate);
        const std::array<int, 3> b = corner_offset(7 - candidate);
        const Vec3 step_vector{
            static_cast<double>(b[0] - a[0]) / grid[0],
            static_cast<double>(b[1] - a[1]) / grid[1],
            static_cast<double>(b[2] - a[2]) / grid[2],
        };
        const double length = (step_vector * reciprocal).norm();
        if (shortest < 0.0 || length < shortest) {
            shortest = length;
            diagonal = candidate;
        }
    }

    // Every microcell contributes six tetrahedra, and the whole grid of them
    // fills the zone exactly once.
    const double weight = 1.0 / (6.0 * static_cast<double>(expected)) * degeneracy;
    const auto index_of = [&](int i, int j, int k) {
        const int a = ((i % grid[0]) + grid[0]) % grid[0];
        const int b = ((j % grid[1]) + grid[1]) % grid[1];
        const int c = ((k % grid[2]) + grid[2]) % grid[2];
        return static_cast<std::size_t>((a * grid[1] + b) * grid[2] + c);
    };

    for (std::size_t spin = 0; spin < bands.nspins(); ++spin) {
        for (int i = 0; i < grid[0]; ++i) {
            for (int j = 0; j < grid[1]; ++j) {
                for (int k = 0; k < grid[2]; ++k) {
                    std::array<std::size_t, 8> corners{};
                    for (int corner = 0; corner < 8; ++corner) {
                        // XOR by the chosen diagonal relabels the cube so that
                        // the diagonal always runs from 0 to 7, which lets one
                        // fixed list of six tetrahedra serve all four choices.
                        const std::array<int, 3> offset = corner_offset(corner ^ diagonal);
                        corners[static_cast<std::size_t>(corner)] =
                            index_of(i + offset[0], j + offset[1], k + offset[2]);
                    }
                    for (std::size_t band = 0; band < bands.nbands(); ++band) {
                        for (const std::array<int, 4>& tetrahedron : tetrahedra()) {
                            std::array<double, 4> e{};
                            for (int c = 0; c < 4; ++c) {
                                e[static_cast<std::size_t>(c)] = bands.energy(
                                    spin, corners[static_cast<std::size_t>(tetrahedron[c])],
                                    band);
                            }
                            std::sort(e.begin(), e.end());
                            for (std::size_t point = 0; point < out.energy.size(); ++point) {
                                const double value =
                                    tetrahedron_dos(out.energy[point], e, weight);
                                if (value == 0.0) {
                                    continue;
                                }
                                out.dos[point] += value;
                                if (!out.spin_dos.empty()) {
                                    out.spin_dos[spin * out.energy.size() + point] += value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    out.integrated.assign(out.energy.size(), 0.0);
    double running = 0.0;
    for (std::size_t point = 0; point < out.energy.size(); ++point) {
        if (point > 0) {
            running += 0.5 * (out.dos[point] + out.dos[point - 1]) * step;
        }
        out.integrated[point] = running;
    }
    (void)kPi;
    return out;
}

}  // namespace fastatoms::electronic
