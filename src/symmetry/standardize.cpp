// SPDX-License-Identifier: MIT

#include "fastatoms/symmetry/standardize.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

#include "fastatoms/phonon/symmetry.hpp"

namespace fastatoms::symmetry {
namespace {

/// Wrap a fractional coordinate into [0, 1), snapping what is within `eps` of
/// either end to zero.
double wrap(double value, double eps = 1e-8) {
    value -= std::floor(value);
    if (value > 1.0 - eps || value < eps) {
        return 0.0;
    }
    return value;
}

Vec3 wrap(const Vec3& v, double eps = 1e-8) {
    return Vec3{wrap(v.x, eps), wrap(v.y, eps), wrap(v.z, eps)};
}

/// The Hermite normal form of an integer matrix given as rows.
///
/// Reduces the rows to an upper-triangular basis of the same lattice, which is
/// what turns "here are some translations that happen to be in the lattice"
/// into "here are three vectors that generate it".
Mat3 hermite_normal_form(std::vector<std::array<long long, 3>> rows) {
    std::array<std::array<long long, 3>, 3> basis{};
    std::size_t placed = 0;

    for (std::size_t column = 0; column < 3 && placed < 3; ++column) {
        // Repeatedly reduce the rows against each other on this column until
        // one non-zero entry is left - the integer analogue of pivoting.
        while (true) {
            std::size_t pivot = rows.size();
            for (std::size_t k = 0; k < rows.size(); ++k) {
                if (rows[k][column] != 0 &&
                    (pivot == rows.size() ||
                     std::abs(rows[k][column]) < std::abs(rows[pivot][column]))) {
                    pivot = k;
                }
            }
            if (pivot == rows.size()) {
                break;
            }
            bool reduced = false;
            for (std::size_t k = 0; k < rows.size(); ++k) {
                if (k == pivot || rows[k][column] == 0) {
                    continue;
                }
                const long long factor = rows[k][column] / rows[pivot][column];
                for (std::size_t c = 0; c < 3; ++c) {
                    rows[k][c] -= factor * rows[pivot][c];
                }
                if (rows[k][column] != 0) {
                    reduced = true;
                }
            }
            if (!reduced) {
                basis[placed++] = rows[pivot];
                rows.erase(rows.begin() + static_cast<std::ptrdiff_t>(pivot));
                break;
            }
        }
    }
    if (placed != 3) {
        throw SymmetryError("the translations do not span three dimensions");
    }

    Mat3 out;
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            out(r, c) = static_cast<double>(basis[r][c]);
        }
    }
    return out;
}

/// The translation lattice of a structure, as a basis in fractional
/// coordinates of the input cell.
///
/// The pure translations found by the symmetry search form a finite group
/// under addition modulo one, so with `n` of them every one is a multiple of
/// `1/n` - which is what makes the integer reduction above possible.
Mat3 translation_basis(const phonon::SymmetryDataset& dataset) {
    std::vector<Vec3> pure{Vec3{0.0, 0.0, 0.0}};
    for (const phonon::SymmetryOperation& operation : dataset.operations) {
        if (operation.is_pure_translation()) {
            const Vec3 t = wrap(operation.translation);
            if (t.norm() > 1e-8) {
                pure.push_back(t);
            }
        }
    }
    const long long n = static_cast<long long>(pure.size());

    std::vector<std::array<long long, 3>> rows;
    for (const Vec3& t : pure) {
        rows.push_back({std::llround(t.x * static_cast<double>(n)),
                        std::llround(t.y * static_cast<double>(n)),
                        std::llround(t.z * static_cast<double>(n))});
    }
    rows.push_back({n, 0, 0});
    rows.push_back({0, n, 0});
    rows.push_back({0, 0, n});

    Mat3 basis = hermite_normal_form(std::move(rows));
    for (std::size_t k = 0; k < 9; ++k) {
        basis.m[k] /= static_cast<double>(n);
    }
    return basis;
}

/// Every lattice point of `cell` within `radius` of the origin, in Cartesian
/// coordinates, excluding the origin itself.
std::vector<Vec3> lattice_points(const Mat3& cell, double radius) {
    const Cell wrapper(cell, {true, true, true});
    const Vec3 widths = wrapper.heights();
    const int nx = static_cast<int>(std::ceil(radius / widths.x)) + 1;
    const int ny = static_cast<int>(std::ceil(radius / widths.y)) + 1;
    const int nz = static_cast<int>(std::ceil(radius / widths.z)) + 1;

    std::vector<Vec3> out;
    for (int i = -nx; i <= nx; ++i) {
        for (int j = -ny; j <= ny; ++j) {
            for (int k = -nz; k <= nz; ++k) {
                if (i == 0 && j == 0 && k == 0) {
                    continue;
                }
                const Vec3 p = Vec3{static_cast<double>(i), static_cast<double>(j),
                                    static_cast<double>(k)} * cell;
                if (p.norm() <= radius) {
                    out.push_back(p);
                }
            }
        }
    }
    std::sort(out.begin(), out.end(),
              [](const Vec3& a, const Vec3& b) { return a.norm() < b.norm(); });
    return out;
}

/// The shortest lattice vector parallel to `direction`, or a zero vector when
/// none is found within `radius`.
Vec3 shortest_along(const std::vector<Vec3>& points, const Vec3& direction, double tolerance) {
    const double length = direction.norm();
    if (!(length > 0.0)) {
        return Vec3{0.0, 0.0, 0.0};
    }
    const Vec3 unit{direction.x / length, direction.y / length, direction.z / length};
    for (const Vec3& p : points) {
        const double along = p.x * unit.x + p.y * unit.y + p.z * unit.z;
        const Vec3 perpendicular{p.x - along * unit.x, p.y - along * unit.y,
                                 p.z - along * unit.z};
        if (perpendicular.norm() < tolerance && along > 0.0) {
            return p;   // the points are sorted by length, so this is the shortest
        }
    }
    return Vec3{0.0, 0.0, 0.0};
}

/// The rotation axis of a proper rotation, in Cartesian coordinates.
///
/// The eigenvector with eigenvalue one. Found by summing the powers of the
/// rotation, which projects onto exactly that subspace.
Vec3 rotation_axis(const Mat3& rotation, int order) {
    // A rotoinversion is minus its proper partner, and it is the partner whose
    // axis this is. Without the negation the loop below runs zero times for a
    // negative order, the projector stays the identity, and the "axis" is
    // whichever coordinate direction happens to come first - which puts the
    // conventional c axis somewhere it has no business being.
    Mat3 proper = rotation;
    if (order < 0) {
        for (std::size_t k = 0; k < 9; ++k) {
            proper.m[k] = -proper.m[k];
        }
    }
    const int turns = std::abs(order);

    Mat3 projector = Mat3::identity();
    Mat3 power = Mat3::identity();
    for (int k = 1; k < turns; ++k) {
        power = power * proper;
        for (std::size_t c = 0; c < 9; ++c) {
            projector.m[c] += power.m[c];
        }
    }
    // Any non-zero column of the projector spans the axis.
    Vec3 best{0.0, 0.0, 0.0};
    for (std::size_t c = 0; c < 3; ++c) {
        const Vec3 column{projector(0, c), projector(1, c), projector(2, c)};
        if (column.norm() > best.norm()) {
            best = column;
        }
    }
    const double length = best.norm();
    return length > 1e-9 ? Vec3{best.x / length, best.y / length, best.z / length} : best;
}

}  // namespace

char centring_letter(Centring centring) {
    switch (centring) {
        case Centring::P: return 'P';
        case Centring::A: return 'A';
        case Centring::B: return 'B';
        case Centring::C: return 'C';
        case Centring::I: return 'I';
        case Centring::F: return 'F';
        case Centring::R: return 'R';
    }
    return 'P';
}

int centring_multiplicity(Centring centring) {
    switch (centring) {
        case Centring::P: return 1;
        case Centring::A:
        case Centring::B:
        case Centring::C:
        case Centring::I: return 2;
        case Centring::R: return 3;
        case Centring::F: return 4;
    }
    return 1;
}

std::vector<Vec3> centring_vectors(Centring centring) {
    switch (centring) {
        case Centring::P: return {};
        case Centring::A: return {Vec3{0.0, 0.5, 0.5}};
        case Centring::B: return {Vec3{0.5, 0.0, 0.5}};
        case Centring::C: return {Vec3{0.5, 0.5, 0.0}};
        case Centring::I: return {Vec3{0.5, 0.5, 0.5}};
        case Centring::F: return {Vec3{0.0, 0.5, 0.5}, Vec3{0.5, 0.0, 0.5}, Vec3{0.5, 0.5, 0.0}};
        case Centring::R: return {Vec3{2.0 / 3, 1.0 / 3, 1.0 / 3},
                                  Vec3{1.0 / 3, 2.0 / 3, 2.0 / 3}};
    }
    return {};
}


namespace {

/// Rebuild a structure on a new lattice, keeping one copy of each distinct
/// site. `basis` holds the new vectors in fractional coordinates of the old.
Atoms rebuild(const Atoms& atoms, const Mat3& basis, double tolerance) {
    const Mat3 new_matrix = basis * atoms.cell().matrix();
    const Cell new_cell(new_matrix, {true, true, true});
    const Mat3 inverse = new_cell.inverse_matrix();

    // The new cell may be smaller than the old one or larger, so every atom is
    // offered together with its images under the old lattice, and whatever
    // lands inside is kept. That covers reducing a supercell and building a
    // centred conventional cell out of a primitive one with the same code.
    const double ratio = std::abs(new_matrix.determinant() / atoms.cell().volume());
    const int reach = static_cast<int>(std::ceil(std::cbrt(std::max(ratio, 1.0)))) + 1;

    std::vector<std::string> symbols;
    std::vector<double> scaled;
    std::vector<Vec3> kept;

    const auto already_there = [&](const Vec3& fractional) {
        return std::any_of(kept.begin(), kept.end(), [&](const Vec3& other) {
            Vec3 difference{fractional.x - other.x, fractional.y - other.y,
                            fractional.z - other.z};
            difference = Vec3{difference.x - std::round(difference.x),
                              difference.y - std::round(difference.y),
                              difference.z - std::round(difference.z)};
            // Compared in angstrom, not in fractional coordinates, so the
            // tolerance means the same thing whatever the cell's shape.
            return (difference * new_matrix).norm() < tolerance;
        });
    };

    for (std::size_t i = 0; i < atoms.size(); ++i) {
        for (int ix = -reach; ix <= reach; ++ix) {
            for (int iy = -reach; iy <= reach; ++iy) {
                for (int iz = -reach; iz <= reach; ++iz) {
                    const Vec3 shift = Vec3{static_cast<double>(ix), static_cast<double>(iy),
                                            static_cast<double>(iz)} * atoms.cell().matrix();
                    const Vec3 position{atoms.position(i).x + shift.x,
                                        atoms.position(i).y + shift.y,
                                        atoms.position(i).z + shift.z};
                    const Vec3 fractional = wrap(position * inverse, 1e-6);
                    if (already_there(fractional)) {
                        continue;
                    }
                    kept.push_back(fractional);
                    symbols.push_back(atoms.symbol(i));
                    scaled.push_back(fractional.x);
                    scaled.push_back(fractional.y);
                    scaled.push_back(fractional.z);
                }
            }
        }
    }

    Atoms out(symbols, std::vector<double>(3 * symbols.size(), 0.0), new_cell);
    out.set_scaled_positions(scaled);
    out.set_pbc({true, true, true});
    return out;
}

/// Mutually perpendicular axes of the given rotation order, longest-lived
/// first. Used to pick out the conventional directions.
std::vector<Vec3> axes_of_order(const std::vector<Mat3>& cartesian, int order) {
    std::vector<Vec3> axes;
    for (const Mat3& rotation : cartesian) {
        const RotationType type = classify_rotation(rotation);
        if (rotation_order(type) != order) {
            continue;
        }
        const Vec3 axis = rotation_axis(rotation, order);
        if (axis.norm() < 0.5) {
            continue;
        }
        const bool known = std::any_of(axes.begin(), axes.end(), [&](const Vec3& other) {
            const double dot = axis.x * other.x + axis.y * other.y + axis.z * other.z;
            return std::abs(std::abs(dot) - 1.0) < 1e-6;
        });
        if (!known) {
            axes.push_back(axis);
        }
    }
    return axes;
}

/// The axis a mirror is perpendicular to.
std::vector<Vec3> mirror_normals(const std::vector<Mat3>& cartesian) {
    std::vector<Vec3> normals;
    for (const Mat3& rotation : cartesian) {
        if (classify_rotation(rotation) != RotationType::Mirror) {
            continue;
        }
        // A mirror is minus a two-fold about its own normal.
        Mat3 negated = rotation;
        for (std::size_t k = 0; k < 9; ++k) {
            negated.m[k] = -negated.m[k];
        }
        const Vec3 axis = rotation_axis(negated, 2);
        const bool known = std::any_of(normals.begin(), normals.end(), [&](const Vec3& other) {
            const double dot = axis.x * other.x + axis.y * other.y + axis.z * other.z;
            return std::abs(std::abs(dot) - 1.0) < 1e-6;
        });
        if (!known && axis.norm() > 0.5) {
            normals.push_back(axis);
        }
    }
    return normals;
}

}  // namespace

Atoms niggli_reduce(const Atoms& atoms, double tolerance) {
    if (!atoms.cell().is_invertible()) {
        throw SymmetryError("Niggli reduction needs a cell that spans three dimensions");
    }
    // Krivy and Gruber's algorithm, working on the metric and carrying the
    // basis change alongside so the atoms can be brought over at the end.
    // Every step below is guarded to fire only when it changes something: an
    // unguarded sign-fixing step applies the identity forever once the angles
    // are already the right sign, which looks exactly like convergence and is
    // not.
    Mat3 cell = atoms.cell().matrix();
    Mat3 transform = Mat3::identity();

    const double scale = std::cbrt(std::abs(cell.determinant()));
    const double eps = 1e-9 * scale * scale;

    const auto row = [&](std::size_t k) { return Vec3{cell(k, 0), cell(k, 1), cell(k, 2)}; };
    const auto apply = [&](const Mat3& operation) {
        cell = operation * cell;
        transform = operation * transform;
    };
    const auto swap_rows = [&](std::size_t i, std::size_t j) {
        Mat3 operation{};
        // A swap alone flips the handedness, so one of the two rows is negated
        // to keep the determinant positive.
        const std::size_t other = 3 - i - j;
        operation(i, j) = 1.0;
        operation(j, i) = 1.0;
        operation(other, other) = -1.0;
        apply(operation);
    };

    bool changed = true;
    for (int iteration = 0; iteration < 1000 && changed; ++iteration) {
        changed = false;
        const Vec3 a = row(0), b = row(1), c = row(2);
        const double A = dot(a, a), B = dot(b, b), C = dot(c, c);
        const double xi = 2 * dot(b, c), eta = 2 * dot(a, c), zeta = 2 * dot(a, b);

        // 1, 2: sort by length.
        if (A > B + eps) {
            swap_rows(0, 1);
            changed = true;
            continue;
        }
        if (B > C + eps) {
            swap_rows(1, 2);
            changed = true;
            continue;
        }

        // 3, 4: make the three angle terms all positive, or all non-positive.
        // Which of the two depends on the sign of their product, and both are
        // reached by flipping the signs of whole basis vectors.
        const int negatives = (xi < -eps ? 1 : 0) + (eta < -eps ? 1 : 0) + (zeta < -eps ? 1 : 0);
        const int zeros = (std::abs(xi) < eps ? 1 : 0) + (std::abs(eta) < eps ? 1 : 0) +
                          (std::abs(zeta) < eps ? 1 : 0);
        const bool want_positive = zeros == 0 && negatives % 2 == 0;
        if (zeros == 0) {
            const bool wrong = want_positive ? negatives > 0 : negatives < 3;
            if (wrong) {
                // Flipping two vectors changes the sign of two of the three
                // terms and keeps the handedness, which is enough to reach
                // either target.
                const double sx = want_positive ? (eta < 0 && zeta < 0 ? -1.0 : 1.0)
                                                : (eta > 0 && zeta > 0 ? -1.0 : 1.0);
                const double sy = want_positive ? (xi < 0 && zeta < 0 ? -1.0 : 1.0)
                                                : (xi > 0 && zeta > 0 ? -1.0 : 1.0);
                const double sz = want_positive ? (xi < 0 && eta < 0 ? -1.0 : 1.0)
                                                : (xi > 0 && eta > 0 ? -1.0 : 1.0);
                const Mat3 flip = Mat3::diagonal(sx, sy, sz);
                if (std::abs(flip.determinant() - 1.0) < 1e-9 &&
                    std::abs(sx * sy * sz - 1.0) < 1e-9 && (sx < 0 || sy < 0 || sz < 0)) {
                    apply(flip);
                    changed = true;
                    continue;
                }
            }
        }

        // 5, 6, 7: shorten each vector against the earlier ones.
        const auto shorten = [&](std::size_t target, std::size_t source, double numerator,
                                 double denominator) {
            const double factor = std::round(numerator / denominator);
            if (std::abs(factor) < 0.5) {
                return false;
            }
            Mat3 operation = Mat3::identity();
            operation(target, source) = -factor;
            apply(operation);
            return true;
        };
        if (std::abs(xi) > B + eps && shorten(2, 1, xi, 2 * B)) { changed = true; continue; }
        if (std::abs(eta) > A + eps && shorten(2, 0, eta, 2 * A)) { changed = true; continue; }
        if (std::abs(zeta) > A + eps && shorten(1, 0, zeta, 2 * A)) { changed = true; continue; }

        // 8: the remaining case, where c can be shortened against a + b.
        if (xi + eta + zeta + A + B < -eps) {
            Mat3 operation = Mat3::identity();
            operation(2, 0) = 1.0;
            operation(2, 1) = 1.0;
            apply(operation);
            changed = true;
            continue;
        }
    }

    if (transform.determinant() < 0) {
        apply(Mat3::diagonal(-1.0, -1.0, -1.0));
    }
    return rebuild(atoms, transform, tolerance);
}

namespace {

/// The conventional lattice vectors, in Cartesian coordinates, for a crystal
/// whose translation lattice is `lattice` and whose point group contains
/// `cartesian` rotations.
///
/// The conventional cell is the one whose axes lie along the crystal system's
/// symmetry directions. Finding it is therefore a matter of asking the point
/// group where those directions are, and then asking the lattice for its
/// shortest vector along each - which is what makes a face-centred cubic
/// primitive cell come back as the four-atom cube.
Mat3 conventional_axes(const Mat3& lattice, const std::vector<Mat3>& cartesian,
                       CrystalSystem system, double tolerance) {
    const double scale = std::cbrt(std::abs(lattice.determinant()));
    const std::vector<Vec3> points = lattice_points(lattice, 6.0 * scale);
    const double straight = tolerance + 1e-6 * scale;

    const auto build = [](const Vec3& a, const Vec3& b, const Vec3& c) {
        Mat3 out;
        out(0, 0) = a.x; out(0, 1) = a.y; out(0, 2) = a.z;
        out(1, 0) = b.x; out(1, 1) = b.y; out(1, 2) = b.z;
        out(2, 0) = c.x; out(2, 1) = c.y; out(2, 2) = c.z;
        return out;
    };
    const auto shortest = [&](const Vec3& direction) {
        const Vec3 found = shortest_along(points, direction, straight);
        if (found.norm() < 1e-9) {
            throw SymmetryError("no lattice vector lies along a required symmetry direction; "
                                "the tolerance may be too loose for this cell");
        }
        return found;
    };
    /// The shortest lattice vector perpendicular to `axis`.
    const auto shortest_perpendicular = [&](const Vec3& axis) {
        for (const Vec3& p : points) {
            if (std::abs(dot(p, axis)) < straight) {
                return p;
            }
        }
        throw SymmetryError("no lattice vector lies perpendicular to the principal axis");
    };

    switch (system) {
        case CrystalSystem::Cubic: {
            // The three four-folds, or the three two-folds when there are no
            // four-folds (point groups 23 and m-3).
            std::vector<Vec3> axes = axes_of_order(cartesian, 4);
            if (axes.size() < 3) {
                axes = axes_of_order(cartesian, 2);
            }
            if (axes.size() < 3) {
                throw SymmetryError("a cubic point group must have three principal axes");
            }
            return build(shortest(axes[0]), shortest(axes[1]), shortest(axes[2]));
        }
        case CrystalSystem::Tetragonal: {
            std::vector<Vec3> principal = axes_of_order(cartesian, 4);
            if (principal.empty()) {
                principal = axes_of_order(cartesian, -4);
            }
            if (principal.empty()) {
                throw SymmetryError("a tetragonal point group must have a four-fold axis");
            }
            const Vec3 c = shortest(principal.front());
            const Vec3 a = shortest_perpendicular(principal.front());
            // b is a turned by ninety degrees about c, which the four-fold
            // guarantees is a lattice vector.
            const Vec3 unit_c{c.x / c.norm(), c.y / c.norm(), c.z / c.norm()};
            const Vec3 b = cross(unit_c, a);
            return build(a, b, c);
        }
        case CrystalSystem::Hexagonal:
        case CrystalSystem::Trigonal: {
            std::vector<Vec3> principal = axes_of_order(cartesian, 6);
            if (principal.empty()) principal = axes_of_order(cartesian, -6);
            if (principal.empty()) principal = axes_of_order(cartesian, 3);
            if (principal.empty()) principal = axes_of_order(cartesian, -3);
            if (principal.empty()) {
                throw SymmetryError("a hexagonal or trigonal point group must have a "
                                    "three- or six-fold axis");
            }
            const Vec3 c = shortest(principal.front());
            const Vec3 a = shortest_perpendicular(principal.front());
            // b is a turned by a hundred and twenty degrees about c: the
            // hexagonal convention, with gamma = 120 rather than 60.
            const Vec3 unit_c{c.x / c.norm(), c.y / c.norm(), c.z / c.norm()};
            const double angle = 2.0 * 3.14159265358979323846 / 3;
            const Vec3 perpendicular = cross(unit_c, a);
            const Vec3 b{a.x * std::cos(angle) + perpendicular.x * std::sin(angle),
                         a.y * std::cos(angle) + perpendicular.y * std::sin(angle),
                         a.z * std::cos(angle) + perpendicular.z * std::sin(angle)};
            return build(a, b, c);
        }
        case CrystalSystem::Orthorhombic: {
            std::vector<Vec3> axes = axes_of_order(cartesian, 2);
            if (axes.size() < 3) {
                // Point group mm2 has one two-fold; the other two directions
                // are the normals of its mirrors.
                for (const Vec3& normal : mirror_normals(cartesian)) {
                    const bool known = std::any_of(axes.begin(), axes.end(),
                                                   [&](const Vec3& other) {
                                                       return std::abs(std::abs(dot(normal, other)) -
                                                                       1.0) < 1e-6;
                                                   });
                    if (!known) {
                        axes.push_back(normal);
                    }
                }
            }
            if (axes.size() < 3) {
                throw SymmetryError("an orthorhombic point group must have three axes");
            }
            // Order them so that the two-fold, when there is only one, is c -
            // the convention for mm2.
            return build(shortest(axes[0]), shortest(axes[1]), shortest(axes[2]));
        }
        case CrystalSystem::Monoclinic: {
            std::vector<Vec3> unique = axes_of_order(cartesian, 2);
            if (unique.empty()) {
                unique = mirror_normals(cartesian);
            }
            if (unique.empty()) {
                throw SymmetryError("a monoclinic point group must have a unique axis");
            }
            const Vec3 b = shortest(unique.front());
            // a and c are the two shortest independent lattice vectors in the
            // plane perpendicular to b.
            std::vector<Vec3> plane;
            for (const Vec3& p : points) {
                if (std::abs(dot(p, unique.front())) > straight) {
                    continue;
                }
                const bool parallel = std::any_of(plane.begin(), plane.end(),
                                                  [&](const Vec3& other) {
                                                      return cross(p, other).norm() < straight;
                                                  });
                if (!parallel) {
                    plane.push_back(p);
                }
                if (plane.size() == 2) {
                    break;
                }
            }
            if (plane.size() < 2) {
                throw SymmetryError("the plane perpendicular to the unique axis is not spanned");
            }
            return build(plane[0], b, plane[1]);
        }
        case CrystalSystem::Triclinic:
            break;
    }
    return lattice;   // triclinic: nothing to align, the Niggli cell is used
}

}  // namespace

CellAnalysis analyse_cell(const Atoms& atoms, double tolerance) {
    if (atoms.empty() || !atoms.cell().is_invertible()) {
        throw SymmetryError("analysing a cell needs atoms and a cell that spans three "
                            "dimensions");
    }
    const phonon::SymmetryDataset given = phonon::find_symmetry(atoms, tolerance);

    CellAnalysis analysis;
    analysis.tolerance = tolerance;

    const Mat3 primitive_basis = translation_basis(given);
    analysis.to_primitive = primitive_basis;
    analysis.input_multiplicity =
        static_cast<int>(std::lround(1.0 / std::abs(primitive_basis.determinant())));

    const Atoms primitive = rebuild(atoms, primitive_basis, tolerance);
    analysis.primitive_atoms = primitive.size();

    // The point group has to be read off the primitive cell, not the one that
    // was handed in. A search can only find operations that map the given cell
    // onto itself, and a 2x1x1 supercell of a cubic crystal does not admit the
    // cubic rotations - it would come back monoclinic, which is true of the
    // box and false of the crystal.
    const phonon::SymmetryDataset dataset =
        analysis.input_multiplicity == 1 ? given : phonon::find_symmetry(primitive, tolerance);
    analysis.point_group = identify_point_group(dataset.point_group);
    analysis.system = analysis.point_group.system;

    std::vector<Mat3> cartesian;
    cartesian.reserve(dataset.operations.size());
    for (const phonon::SymmetryOperation& operation : dataset.operations) {
        cartesian.push_back(operation.cartesian_rotation);
    }

    Mat3 conventional = analysis.system == CrystalSystem::Triclinic
                            ? niggli_reduce(primitive, tolerance).cell().matrix()
                            : conventional_axes(primitive.cell().matrix(), cartesian,
                                                analysis.system, tolerance);
    if (conventional.determinant() < 0) {
        for (std::size_t k = 6; k < 9; ++k) {
            conventional.m[k] = -conventional.m[k];
        }
    }
    analysis.to_conventional = conventional * atoms.cell().inverse_matrix();
    analysis.conventional_cell = conventional;

    const int multiplicity = static_cast<int>(
        std::lround(std::abs(conventional.determinant()) / std::abs(primitive.cell().volume())));
    analysis.conventional_atoms = primitive.size() * static_cast<std::size_t>(multiplicity);

    // Which centring gives that multiplicity, decided by checking the actual
    // vectors rather than by the count alone: I and C both double the cell.
    const Mat3 conventional_inverse = Cell(conventional, {true, true, true}).inverse_matrix();
    const auto is_lattice_vector = [&](const Vec3& fractional) {
        const Vec3 cartesian_vector = fractional * conventional;
        const Vec3 in_primitive = cartesian_vector * primitive.cell().inverse_matrix();
        const Vec3 rounded{in_primitive.x - std::round(in_primitive.x),
                           in_primitive.y - std::round(in_primitive.y),
                           in_primitive.z - std::round(in_primitive.z)};
        return (rounded * primitive.cell().matrix()).norm() < tolerance;
    };
    (void)conventional_inverse;

    analysis.centring = Centring::P;
    for (const Centring candidate : {Centring::F, Centring::R, Centring::I, Centring::C,
                                     Centring::A, Centring::B}) {
        if (centring_multiplicity(candidate) != multiplicity) {
            continue;
        }
        const std::vector<Vec3> vectors = centring_vectors(candidate);
        if (std::all_of(vectors.begin(), vectors.end(), is_lattice_vector)) {
            analysis.centring = candidate;
            break;
        }
    }
    return analysis;
}

Atoms primitive_cell(const Atoms& atoms, double tolerance) {
    if (atoms.empty() || !atoms.cell().is_invertible()) {
        throw SymmetryError("reducing a cell needs atoms and a cell that spans three dimensions");
    }
    const phonon::SymmetryDataset dataset = phonon::find_symmetry(atoms, tolerance);
    return rebuild(atoms, translation_basis(dataset), tolerance);
}

Atoms conventional_cell(const Atoms& atoms, double tolerance) {
    const CellAnalysis analysis = analyse_cell(atoms, tolerance);
    return rebuild(atoms, analysis.to_conventional, tolerance);
}

std::string CellAnalysis::to_string() const {
    std::ostringstream out;
    out << "CellAnalysis(" << point_group.symbol << ", " << crystal_system_name(system) << ", "
        << centring_letter(centring) << "-centred, " << primitive_atoms
        << " atoms primitive, " << conventional_atoms << " conventional)";
    return out.str();
}

}  // namespace fastatoms::symmetry
