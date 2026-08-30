// SPDX-License-Identifier: MIT

#include "fastatoms/build/cluster.hpp"

#include <algorithm>
#include <cmath>
#include <set>

namespace fastatoms::build {
namespace {

constexpr double kPi = 3.14159265358979323846;

/// Collected positions, deduplicated on a grid so that points arriving from
/// different faces or wedges coincide exactly rather than nearly.
class PointSet {
public:
    explicit PointSet(double tolerance) : tolerance_(tolerance) {}

    bool add(const Vec3& p) {
        const Key key{quantise(p.x), quantise(p.y), quantise(p.z)};
        if (!keys_.insert(key).second) {
            return false;
        }
        points_.push_back(p);
        return true;
    }

    [[nodiscard]] const std::vector<Vec3>& points() const noexcept { return points_; }

private:
    using Key = std::array<long long, 3>;

    [[nodiscard]] long long quantise(double value) const {
        return std::llround(value / tolerance_);
    }

    double tolerance_;
    std::set<Key> keys_;
    std::vector<Vec3> points_;
};

/// Assemble a finite particle: centred, with a box around it and no
/// periodicity.
Atoms finish(const std::string& symbol, const std::vector<Vec3>& points, double vacuum) {
    if (points.empty()) {
        throw BuildError("that construction produced no atoms; ask for a larger one");
    }
    std::vector<double> positions;
    positions.reserve(3 * points.size());
    for (const Vec3& p : points) {
        positions.push_back(p.x);
        positions.push_back(p.y);
        positions.push_back(p.z);
    }
    Atoms atoms(std::vector<std::string>(points.size(), symbol), positions);

    Vec3 low = points.front();
    Vec3 high = points.front();
    for (const Vec3& p : points) {
        low = Vec3{std::min(low.x, p.x), std::min(low.y, p.y), std::min(low.z, p.z)};
        high = Vec3{std::max(high.x, p.x), std::max(high.y, p.y), std::max(high.z, p.z)};
    }
    atoms.set_cell(Cell(Mat3::diagonal(high.x - low.x + vacuum, high.y - low.y + vacuum,
                                       high.z - low.z + vacuum),
                        {false, false, false}),
                   false);
    atoms.set_pbc({false, false, false});
    atoms.center();
    return atoms;
}

/// The twelve vertices of a unit icosahedron, and the twenty triangles joining
/// adjacent triples of them.
struct Icosahedron {
    std::vector<Vec3> vertices;
    std::vector<std::array<std::size_t, 3>> faces;
    double edge{0.0};
};

const Icosahedron& unit_icosahedron() {
    static const Icosahedron shape = [] {
        Icosahedron out;
        const double phi = (1.0 + std::sqrt(5.0)) / 2;
        const double norm = std::sqrt(1.0 + phi * phi);
        for (const double s1 : {-1.0, 1.0}) {
            for (const double s2 : {-1.0, 1.0}) {
                out.vertices.push_back(Vec3{0.0, s1 / norm, s2 * phi / norm});
                out.vertices.push_back(Vec3{s1 / norm, s2 * phi / norm, 0.0});
                out.vertices.push_back(Vec3{s1 * phi / norm, 0.0, s2 / norm});
            }
        }

        // The edge length is the shortest distance between two vertices; the
        // faces are the triples that are all that distance apart. Finding them
        // this way rather than writing out a table means the two can never
        // disagree.
        out.edge = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < out.vertices.size(); ++i) {
            for (std::size_t j = i + 1; j < out.vertices.size(); ++j) {
                out.edge = std::min(out.edge, (out.vertices[i] - out.vertices[j]).norm());
            }
        }
        const auto adjacent = [&](std::size_t i, std::size_t j) {
            return std::abs((out.vertices[i] - out.vertices[j]).norm() - out.edge) < 1e-9;
        };
        for (std::size_t i = 0; i < out.vertices.size(); ++i) {
            for (std::size_t j = i + 1; j < out.vertices.size(); ++j) {
                for (std::size_t k = j + 1; k < out.vertices.size(); ++k) {
                    if (adjacent(i, j) && adjacent(j, k) && adjacent(i, k)) {
                        out.faces.push_back({i, j, k});
                    }
                }
            }
        }
        return out;
    }();
    return shape;
}

/// fcc lattice points inside a bounding radius, in Cartesian coordinates.
std::vector<Vec3> fcc_points(double a, double radius) {
    std::vector<Vec3> out;
    const int reach = static_cast<int>(std::ceil(2.0 * radius / a)) + 1;
    for (int i = -reach; i <= reach; ++i) {
        for (int j = -reach; j <= reach; ++j) {
            for (int k = -reach; k <= reach; ++k) {
                if (((i + j + k) % 2 + 2) % 2 != 0) {
                    continue;   // fcc: only the even sites of the half-integer grid
                }
                const Vec3 p{i * a / 2, j * a / 2, k * a / 2};
                if (p.norm() <= radius) {
                    out.push_back(p);
                }
            }
        }
    }
    return out;
}

}  // namespace

Atoms icosahedron(const std::string& symbol, int shells, double d, double vacuum) {
    if (shells < 1) {
        throw BuildError("an icosahedron needs at least one shell");
    }
    if (!(d > 0.0)) {
        throw BuildError("the nearest-neighbour distance must be positive");
    }

    const Icosahedron& shape = unit_icosahedron();
    // Scale so that neighbouring points along a face edge are `d` apart. The
    // radial spacing then comes out at `d / 1.0515`, which is the icosahedron's
    // internal strain and not an error.
    const double scale = d / shape.edge;

    PointSet points(d * 1e-4);
    points.add(Vec3{0.0, 0.0, 0.0});
    for (int shell = 1; shell < shells + 1; ++shell) {
        for (const auto& face : shape.faces) {
            const Vec3& va = shape.vertices[face[0]];
            const Vec3& vb = shape.vertices[face[1]];
            const Vec3& vc = shape.vertices[face[2]];
            // Every lattice point of the triangular net on this face of this
            // shell. Interior points of a face belong to it alone; edges and
            // vertices are reached from several faces and dedupe away.
            for (int i = 0; i <= shell; ++i) {
                for (int j = 0; j + i <= shell; ++j) {
                    const int rest = shell - i - j;
                    const Vec3 p{
                        (rest * va.x + i * vb.x + j * vc.x) * scale,
                        (rest * va.y + i * vb.y + j * vc.y) * scale,
                        (rest * va.z + i * vb.z + j * vc.z) * scale,
                    };
                    points.add(p);
                }
            }
        }
    }
    return finish(symbol, points.points(), vacuum);
}

Atoms cuboctahedron(const std::string& symbol, int shells, double a, double vacuum) {
    if (shells < 1) {
        throw BuildError("a cuboctahedron needs at least one shell");
    }
    if (!(a > 0.0)) {
        throw BuildError("the lattice constant must be positive");
    }

    // Six {100} planes at `shells * a / 2` and eight {111} planes at the
    // distance that makes the two facet types meet edge to edge, which is what
    // a cuboctahedron is.
    const double cube = shells * a / 2;
    const double octahedron = shells * a;

    PointSet points(a * 1e-4);
    for (const Vec3& p : fcc_points(a, octahedron * 1.1)) {
        if (std::max({std::abs(p.x), std::abs(p.y), std::abs(p.z)}) <= cube + 1e-9 &&
            std::abs(p.x) + std::abs(p.y) + std::abs(p.z) <= octahedron + 1e-9) {
            points.add(p);
        }
    }
    return finish(symbol, points.points(), vacuum);
}

Atoms decahedron(const std::string& symbol, int p, int q, int r, double a, double vacuum) {
    if (p < 1 || q < 1) {
        throw BuildError("a decahedron needs p and q of at least one");
    }
    if (r < 0) {
        throw BuildError("the Marks re-entrance depth must not be negative");
    }
    if (!(a > 0.0)) {
        throw BuildError("the lattice constant must be positive");
    }

    const double d = a / std::sqrt(2.0);   // the fcc nearest-neighbour distance

    // One wedge, in a frame where the five-fold axis is z. The axis is a <110>
    // of the constituent fcc crystal, and the two bounding facets are the
    // {111} planes that contain it.
    const Vec3 e3{1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0), 0.0};
    const Vec3 e1{1.0 / std::sqrt(2.0), -1.0 / std::sqrt(2.0), 0.0};
    const Vec3 e2{0.0, 0.0, -1.0};

    const Vec3 n1{1.0 / std::sqrt(3.0), -1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
    const Vec3 n2{-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};

    // The gap five tetrahedra leave, closed by stretching each one.
    const double wedge = std::acos(-1.0 / 3.0);          // 109.47 degrees between normals
    const double half_opening = (kPi - wedge) / 2;       // 35.26 degrees
    const double strain = (kPi / 5) / half_opening;      // to exactly 36 degrees

    const double axis_half = (q - 1) * d / 2;
    const double side = p * d;
    const double reach = std::sqrt(axis_half * axis_half + side * side) + 2 * d;

    // Collect one wedge first and replicate it afterwards, rather than
    // deduplicating the finished particle. Two atoms arriving at a twin
    // boundary from neighbouring wedges land a whisker apart, and a tolerance
    // wide enough to merge them is wide enough to merge one pair and not the
    // next - which quietly destroys the five-fold symmetry the whole shape is
    // about. Replicating a deduplicated wedge cannot do that.
    struct WedgePoint {
        double radius;
        double angle;
        double height;
    };
    std::vector<WedgePoint> wedge_points;
    PointSet wedge_seen(d * 1e-3);

    for (const Vec3& lattice : fcc_points(a, reach)) {
        const double u = lattice.x * e1.x + lattice.y * e1.y + lattice.z * e1.z;
        const double v = lattice.x * e2.x + lattice.y * e2.y + lattice.z * e2.z;
        const double w = lattice.x * e3.x + lattice.y * e3.y + lattice.z * e3.z;

        if (std::abs(w) > axis_half + 1e-9) {
            continue;
        }
        const double radius = std::sqrt(u * u + v * v);
        if (radius > side + 1e-9) {
            continue;
        }
        // Half-open in angle, so an atom sitting exactly on a twin boundary
        // belongs to one wedge and is placed once.
        const double angle = radius < 1e-9 ? 0.0 : std::atan2(u, v);
        if (radius > 1e-9 &&
            (angle < -half_opening - 1e-9 || angle >= half_opening - 1e-9)) {
            continue;
        }
        // The Marks re-entrance: a notch cut into each twin boundary, which is
        // what makes a Marks decahedron out of an Ino one.
        if (r > 0 && radius > side - r * d + 1e-9 && std::abs(w) < r * d) {
            continue;
        }
        if (!wedge_seen.add(Vec3{u, v, w})) {
            continue;
        }
        wedge_points.push_back({radius, angle * strain, w});
    }

    PointSet points(d * 1e-4);
    for (const WedgePoint& point : wedge_points) {
        if (point.radius < 1e-9) {
            points.add(Vec3{0.0, 0.0, point.height});   // on the axis: one atom, not five
            continue;
        }
        for (int copy = 0; copy < 5; ++copy) {
            const double turned = point.angle + copy * 2 * kPi / 5;
            points.add(Vec3{point.radius * std::sin(turned), point.radius * std::cos(turned),
                            point.height});
        }
    }
    return finish(symbol, points.points(), vacuum);
}

std::vector<Vec3> equivalent_directions(int h, int k, int l) {
    if (h == 0 && k == 0 && l == 0) {
        throw BuildError("(0, 0, 0) is not a direction");
    }
    // The cubic point group: every permutation of the indices, every sign.
    std::set<std::array<int, 3>> unique;
    const std::array<int, 3> base{h, k, l};
    std::array<int, 3> order{0, 1, 2};
    std::sort(order.begin(), order.end());
    do {
        for (const int s0 : {-1, 1}) {
            for (const int s1 : {-1, 1}) {
                for (const int s2 : {-1, 1}) {
                    unique.insert({s0 * base[order[0]], s1 * base[order[1]],
                                   s2 * base[order[2]]});
                }
            }
        }
    } while (std::next_permutation(order.begin(), order.end()));

    std::vector<Vec3> out;
    out.reserve(unique.size());
    for (const auto& v : unique) {
        const double norm = std::sqrt(static_cast<double>(v[0] * v[0] + v[1] * v[1] +
                                                          v[2] * v[2]));
        out.push_back(Vec3{v[0] / norm, v[1] / norm, v[2] / norm});
    }
    return out;
}

Atoms wulff(const std::string& symbol, Lattice lattice, double a,
            const std::vector<SurfaceEnergy>& energies, int atom_count, double vacuum) {
    if (energies.empty()) {
        throw BuildError("the Wulff construction needs at least one surface energy");
    }
    if (atom_count < 1) {
        throw BuildError("ask for at least one atom");
    }
    if (!(a > 0.0)) {
        throw BuildError("the lattice constant must be positive");
    }
    double smallest = std::numeric_limits<double>::infinity();
    for (const SurfaceEnergy& facet : energies) {
        if (!(facet.energy > 0.0)) {
            throw BuildError("surface energies must be positive");
        }
        smallest = std::min(smallest, facet.energy);
    }

    // Every plane, as an outward normal and a distance proportional to its
    // surface energy. Wulff's theorem is exactly that proportionality; the
    // constant of it is the particle's size, found by bisection below.
    struct Plane {
        Vec3 normal;
        double distance;
    };
    std::vector<Plane> planes;
    for (const SurfaceEnergy& facet : energies) {
        for (const Vec3& direction : equivalent_directions(facet.h, facet.k, facet.l)) {
            planes.push_back({direction, facet.energy / smallest});
        }
    }

    // One conventional cell's worth of sites, tiled far enough to cover the
    // largest particle the search below will try.
    const Atoms cell = bulk({symbol}, lattice, a, 0.0, true);
    const auto grow = [&](double scale) {
        PointSet points(a * 1e-4);
        const int reach = static_cast<int>(std::ceil(scale)) + 2;
        for (int i = -reach; i <= reach; ++i) {
            for (int j = -reach; j <= reach; ++j) {
                for (int k = -reach; k <= reach; ++k) {
                    for (std::size_t s = 0; s < cell.size(); ++s) {
                        const Vec3 site = cell.position(s);
                        const Vec3 p{site.x + i * a, site.y + j * a, site.z + k * a};
                        const bool inside =
                            std::all_of(planes.begin(), planes.end(), [&](const Plane& plane) {
                                const double along = p.x * plane.normal.x + p.y * plane.normal.y +
                                                     p.z * plane.normal.z;
                                return along <= plane.distance * scale * a + 1e-9;
                            });
                        if (inside) {
                            points.add(p);
                        }
                    }
                }
            }
        }
        return points.points();
    };

    // Bisect on the scale until the count is as close to what was asked for as
    // the lattice allows. The count jumps as whole facets of atoms appear, so
    // landing exactly on a target is not generally possible and the nearest
    // achievable size is used instead.
    double low = 0.1;
    double high = 1.0;
    while (grow(high).size() < static_cast<std::size_t>(atom_count) && high < 200.0) {
        high *= 1.6;
    }
    // Track the closest count seen anywhere in the search, not merely the
    // smallest one at or above the target. The achievable sizes are whatever
    // whole shells of atoms the lattice offers - near 500 an fcc Wulff shape
    // may jump from 459 straight to 711 - and 459 is the better answer to
    // "about 500" even though it is under it.
    std::vector<Vec3> best;
    std::size_t best_distance = std::numeric_limits<std::size_t>::max();
    const auto consider = [&](const std::vector<Vec3>& candidate) {
        const std::size_t count = candidate.size();
        const std::size_t target = static_cast<std::size_t>(atom_count);
        const std::size_t distance = count > target ? count - target : target - count;
        if (count > 0 && distance < best_distance) {
            best_distance = distance;
            best = candidate;
        }
    };

    consider(grow(high));
    for (int step = 0; step < 40; ++step) {
        const double middle = (low + high) / 2;
        const std::vector<Vec3> candidate = grow(middle);
        consider(candidate);
        if (candidate.size() >= static_cast<std::size_t>(atom_count)) {
            high = middle;
        } else {
            low = middle;
        }
    }
    return finish(symbol, best, vacuum);
}

}  // namespace fastatoms::build
