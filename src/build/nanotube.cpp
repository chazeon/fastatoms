// SPDX-License-Identifier: MIT

#include "fastatoms/build/nanotube.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <tuple>

namespace fastatoms::build {
namespace {

constexpr double kPi = 3.14159265358979323846;

/// Wrap a fractional coordinate into [0, 1), treating anything within `eps` of
/// 1 as 0. Without the tolerance an atom exactly on the far edge of the
/// rectangle is kept as well as its image at the near edge, and the tube comes
/// out with a seam of doubled atoms.
double wrap_fraction(double value, double eps = 1e-9) {
    value -= std::floor(value);
    if (value > 1.0 - eps || value < eps) {
        return 0.0;
    }
    return value;
}

}  // namespace

std::string TubeGeometry::to_string() const {
    std::ostringstream out;
    out << "TubeGeometry(radius=" << radius << " A, period=" << period
        << " A, chiral angle=" << chiral_angle << " deg, " << cells << " cells)";
    return out.str();
}

TubeGeometry tube_geometry(int n, int m, double a) {
    if (n < 0 || m < 0) {
        throw BuildError("the chiral indices must not be negative");
    }
    if (n == 0 && m == 0) {
        throw BuildError("(0, 0) is not a tube");
    }
    if (!(a > 0.0)) {
        throw BuildError("the lattice constant must be positive");
    }

    TubeGeometry geometry;
    const int squared = n * n + n * m + m * m;
    geometry.circumference = a * std::sqrt(static_cast<double>(squared));
    geometry.radius = geometry.circumference / (2.0 * kPi);

    // T is the shortest lattice vector perpendicular to C. Dividing by d_R is
    // what makes it the shortest rather than merely a perpendicular one.
    const int d_r = std::gcd(2 * n + m, 2 * m + n);
    geometry.t1 = (2 * m + n) / d_r;
    geometry.t2 = -(2 * n + m) / d_r;
    geometry.period = std::sqrt(3.0) * geometry.circumference / d_r;
    geometry.cells = 2 * squared / d_r;

    // Zero for zigzag (m = 0), thirty degrees for armchair (n = m).
    geometry.chiral_angle = std::atan2(std::sqrt(3.0) * m, 2.0 * n + m) * 180.0 / kPi;
    return geometry;
}

Atoms roll(const Atoms& sheet, int n, int m, int length, double vacuum) {
    if (sheet.empty()) {
        throw BuildError("there is nothing to roll");
    }
    if (length < 1) {
        throw BuildError("a tube needs at least one repeat along its axis");
    }
    if (vacuum < 0.0) {
        throw BuildError("the vacuum gap must not be negative");
    }

    const Mat3 matrix = sheet.cell().matrix();
    const Vec3 a1{matrix(0, 0), matrix(0, 1), matrix(0, 2)};
    const Vec3 a2{matrix(1, 0), matrix(1, 1), matrix(1, 2)};
    if (std::abs(a1.z) > 1e-9 || std::abs(a2.z) > 1e-9) {
        throw BuildError("the sheet's in-plane vectors must lie in the xy plane");
    }
    const double a = a1.norm();
    const double cosine = (a1.x * a2.x + a1.y * a2.y) / (a * a);
    if (std::abs(std::abs(cosine) - 0.5) > 1e-6 || std::abs(a2.norm() - a) > 1e-6 * a) {
        throw BuildError("rolling needs a hexagonal sheet with equal-length lattice vectors");
    }

    // The chiral indices are defined against a basis whose vectors meet at
    // sixty degrees. The sheets here are built on the hundred-and-twenty
    // degree description of the same lattice, so the second vector is swapped
    // for `a1 + a2` - another lattice vector, so nothing about the lattice
    // changes, but (5, 5) now means the 6.78 A armchair tube everyone else
    // means by it rather than a different tube of the same name.
    const Vec3 b1 = a1;
    const Vec3 b2 = cosine < 0.0 ? Vec3{a1.x + a2.x, a1.y + a2.y, 0.0} : a2;

    const TubeGeometry geometry = tube_geometry(n, m, a);
    const int t1 = geometry.t1;
    const int t2 = geometry.t2;

    // C and T in Cartesian coordinates. They are perpendicular by construction,
    // which is what lets the fractional coordinates below be simple dot
    // products rather than a matrix solve.
    const Vec3 c_vector{n * b1.x + m * b2.x, n * b1.y + m * b2.y, 0.0};
    const Vec3 t_vector{t1 * b1.x + t2 * b2.x, t1 * b1.y + t2 * b2.y, 0.0};
    const double c_squared = c_vector.x * c_vector.x + c_vector.y * c_vector.y;
    const double t_squared = t_vector.x * t_vector.x + t_vector.y * t_vector.y;

    // The mid-plane the sheet is bent about: heights are measured from it and
    // become radial offsets, so a layer with thickness keeps it.
    double lowest = sheet.position(0).z;
    double highest = lowest;
    for (std::size_t i = 1; i < sheet.size(); ++i) {
        lowest = std::min(lowest, sheet.position(i).z);
        highest = std::max(highest, sheet.position(i).z);
    }
    const double mid = (lowest + highest) / 2;

    // Enumerate enough lattice translations to cover the rectangle C x T, then
    // keep whatever lands inside it.
    const int reach = std::abs(n) + std::abs(m) + std::abs(geometry.t1) +
                      std::abs(geometry.t2) + 2;

    std::vector<std::string> symbols;
    std::vector<double> positions;
    // Keyed by basis atom as well as position: in the 1H dichalcogenides the
    // two chalcogens sit at the same place in plane, one above the other, and
    // deduplicating on position alone would silently drop one of them.
    std::vector<std::tuple<std::size_t, double, double>> seen;

    for (int i = -reach; i <= reach; ++i) {
        for (int j = -reach; j <= reach; ++j) {
            for (std::size_t k = 0; k < sheet.size(); ++k) {
                const Vec3 p = sheet.position(k);
                const double x = p.x + i * b1.x + j * b2.x;
                const double y = p.y + i * b1.y + j * b2.y;

                const double u = wrap_fraction((x * c_vector.x + y * c_vector.y) / c_squared);
                const double v = wrap_fraction((x * t_vector.x + y * t_vector.y) / t_squared);
                if (u >= 1.0 || v >= 1.0) {
                    continue;
                }
                // A lattice point can be reached by more than one (i, j) once
                // it has been wrapped, so the rectangle is filled by hand.
                const bool duplicate =
                    std::any_of(seen.begin(), seen.end(), [&](const auto& other) {
                        return std::get<0>(other) == k &&
                               std::abs(std::get<1>(other) - u) < 1e-7 &&
                               std::abs(std::get<2>(other) - v) < 1e-7;
                    });
                if (duplicate) {
                    continue;
                }
                seen.emplace_back(k, u, v);

                for (int repeat = 0; repeat < length; ++repeat) {
                    // u runs around the circumference, v along the axis, and
                    // the atom's own height becomes a radial offset.
                    const double angle = 2.0 * kPi * u;
                    const double radius = geometry.radius + (p.z - mid);
                    symbols.push_back(sheet.symbol(k));
                    positions.push_back(radius * std::cos(angle));
                    positions.push_back(radius * std::sin(angle));
                    positions.push_back((v + repeat) * geometry.period);
                }
            }
        }
    }

    if (symbols.size() != static_cast<std::size_t>(geometry.cells * length) * sheet.size()) {
        throw BuildError("rolling produced " + std::to_string(symbols.size()) +
                         " atoms where the (n, m) indices call for " +
                         std::to_string(geometry.cells * length * static_cast<int>(sheet.size())));
    }

    const double width = 2.0 * (geometry.radius + (highest - lowest) / 2) + vacuum;
    Atoms tube(symbols, positions,
               Cell(Mat3::diagonal(width, width, geometry.period * length),
                    {false, false, true}));
    tube.set_pbc({false, false, true});
    tube.center();
    return tube;
}

Atoms nanotube(int n, int m, int length, double a, const std::string& symbol, double vacuum) {
    // A bare honeycomb with no vacuum of its own: the tube supplies its own
    // cell, and the sheet is only ever a template here.
    const Atoms sheet = honeycomb({symbol}, a, 1.0);
    return roll(sheet, n, m, length, vacuum);
}

}  // namespace fastatoms::build
