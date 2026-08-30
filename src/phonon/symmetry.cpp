// SPDX-License-Identifier: MIT
#include "fastatoms/phonon/symmetry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <sstream>

namespace fastatoms::phonon {
namespace {

double wrap(double value) {
    double folded = value - std::floor(value);
    if (folded > 1.0 - 1e-9) {
        folded = 0.0;
    }
    return folded;
}

Vec3 wrap(const Vec3& v) { return {wrap(v.x), wrap(v.y), wrap(v.z)}; }

/// A coarse grid over fractional space, so that "which atom is at this
/// position" is a constant-time question rather than a scan.
class PositionIndex {
public:
    PositionIndex(const std::vector<Vec3>& scaled, const std::vector<int>& species,
                  const Cell& cell, double tolerance)
        : scaled_(scaled), species_(species), cell_(cell), tolerance_(tolerance) {
        // Bins must be comfortably wider than the tolerance, or a match could
        // sit more than one bin away.
        bins_ = std::max(1, static_cast<int>(std::cbrt(static_cast<double>(scaled.size()))) * 2);
        const Vec3 heights = cell.heights();
        for (std::size_t k = 0; k < 3; ++k) {
            const double width = heights[k] / static_cast<double>(bins_);
            if (width < 10.0 * tolerance) {
                bins_ = std::max(1, static_cast<int>(heights[k] / (10.0 * tolerance)));
            }
        }
        buckets_.resize(static_cast<std::size_t>(bins_) * bins_ * bins_);
        for (std::size_t i = 0; i < scaled.size(); ++i) {
            buckets_[bucket_of(scaled[i])].push_back(i);
        }
    }

    /// The atom of the given species at `position`, or -1.
    [[nodiscard]] std::ptrdiff_t find(const Vec3& position, int species) const {
        const Vec3 folded = wrap(position);
        const int bx = coordinate(folded.x);
        const int by = coordinate(folded.y);
        const int bz = coordinate(folded.z);

        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    const std::size_t bucket =
                        index_of(fold_bin(bx + dx), fold_bin(by + dy), fold_bin(bz + dz));
                    for (const std::size_t candidate : buckets_[bucket]) {
                        if (species_[candidate] != species) {
                            continue;
                        }
                        const Vec3 difference = folded - scaled_[candidate];
                        const Vec3 cartesian = cell_.minimum_image(cell_.cartesian(difference));
                        if (cartesian.norm() <= tolerance_) {
                            return static_cast<std::ptrdiff_t>(candidate);
                        }
                    }
                }
            }
        }
        return -1;
    }

private:
    [[nodiscard]] int coordinate(double value) const {
        return std::clamp(static_cast<int>(value * bins_), 0, bins_ - 1);
    }
    [[nodiscard]] int fold_bin(int value) const { return ((value % bins_) + bins_) % bins_; }
    [[nodiscard]] std::size_t index_of(int x, int y, int z) const {
        return static_cast<std::size_t>((x * bins_ + y) * bins_ + z);
    }
    [[nodiscard]] std::size_t bucket_of(const Vec3& v) const {
        return index_of(coordinate(v.x), coordinate(v.y), coordinate(v.z));
    }

    const std::vector<Vec3>& scaled_;
    const std::vector<int>& species_;
    const Cell& cell_;
    double tolerance_;
    int bins_{1};
    std::vector<std::vector<std::size_t>> buckets_{};
};

}  // namespace

bool SymmetryOperation::is_pure_translation(double tolerance) const {
    const Mat3 identity = Mat3::identity();
    for (std::size_t k = 0; k < 9; ++k) {
        if (std::abs(rotation.m[k] - identity.m[k]) > tolerance) {
            return false;
        }
    }
    return true;
}

std::string SymmetryOperation::to_string() const {
    std::ostringstream out;
    out.precision(4);
    out << "SymmetryOperation(rotation=[";
    for (std::size_t r = 0; r < 3; ++r) {
        out << (r ? ", [" : "[") << static_cast<int>(std::nearbyint(rotation(r, 0))) << " "
            << static_cast<int>(std::nearbyint(rotation(r, 1))) << " "
            << static_cast<int>(std::nearbyint(rotation(r, 2))) << "]";
    }
    out << "], translation=[" << translation.x << ", " << translation.y << ", "
        << translation.z << "])";
    return out.str();
}

std::string SymmetryDataset::summary() const {
    std::ostringstream out;
    out << operations.size() << " operations, point group of order " << point_group.size()
        << ", " << independent_atoms.size() << " symmetrically independent atom"
        << (independent_atoms.size() == 1 ? "" : "s") << " of "
        << equivalent_atoms.size();
    return out.str();
}

std::vector<Mat3> lattice_point_group(const Cell& cell, double tolerance) {
    if (!cell.is_invertible()) {
        throw PhononError("the lattice point group needs a cell that spans three dimensions");
    }

    const Mat3 matrix = cell.matrix();
    const Mat3 metric = matrix * matrix.transposed();
    const Mat3 inverse = cell.inverse_matrix();

    // A rotation maps each lattice vector onto a lattice vector of the same
    // length, so the integer coefficients of the image are bounded by the
    // vector's length times the reciprocal vector lengths. That bound is
    // exact, which is what lets this handle any cell rather than only reduced
    // ones.
    const Vec3 lengths = cell.lengths();
    const double longest = std::max({lengths.x, lengths.y, lengths.z});
    std::array<int, 3> bound{};
    for (std::size_t k = 0; k < 3; ++k) {
        const double reciprocal = inverse.col(k).norm();
        bound[k] = std::max(1, static_cast<int>(std::ceil(longest * reciprocal + 1e-9)));
        // A cell this skewed is pathological; refuse rather than hang.
        if (bound[k] > 12) {
            throw PhononError(
                "the cell is too skewed for a symmetry search; reduce it first "
                "(the search bound came out as " + std::to_string(bound[k]) + ")");
        }
    }

    // Candidate rows for each lattice vector: integer triples of the right
    // length.
    const double scale = std::max(1.0, metric(0, 0) + metric(1, 1) + metric(2, 2));
    const double metric_tolerance = tolerance * scale;

    std::array<std::vector<Vec3>, 3> candidates;
    for (std::size_t i = 0; i < 3; ++i) {
        for (int a = -bound[0]; a <= bound[0]; ++a) {
            for (int b = -bound[1]; b <= bound[1]; ++b) {
                for (int c = -bound[2]; c <= bound[2]; ++c) {
                    const Vec3 w{static_cast<double>(a), static_cast<double>(b),
                                 static_cast<double>(c)};
                    // |w M|^2 = w G w^T must match |a_i|^2.
                    const double length2 = dot(w * metric, w);
                    if (std::abs(length2 - metric(i, i)) <= metric_tolerance) {
                        candidates[i].push_back(w);
                    }
                }
            }
        }
    }

    std::vector<Mat3> group;
    for (const Vec3& w0 : candidates[0]) {
        const Vec3 g0 = w0 * metric;
        for (const Vec3& w1 : candidates[1]) {
            if (std::abs(dot(g0, w1) - metric(0, 1)) > metric_tolerance) {
                continue;
            }
            const Vec3 g1 = w1 * metric;
            for (const Vec3& w2 : candidates[2]) {
                if (std::abs(dot(g0, w2) - metric(0, 2)) > metric_tolerance ||
                    std::abs(dot(g1, w2) - metric(1, 2)) > metric_tolerance) {
                    continue;
                }
                group.push_back(Mat3::from_rows(w0, w1, w2));
            }
        }
    }
    return group;
}

SymmetryDataset find_symmetry(const Atoms& atoms, double tolerance) {
    if (atoms.empty()) {
        throw PhononError("cannot find the symmetry of an empty structure");
    }
    if (!atoms.cell().is_invertible()) {
        throw PhononError("finding the symmetry needs a cell that spans three dimensions");
    }

    SymmetryDataset dataset;
    dataset.tolerance = tolerance;

    const Cell& cell = atoms.cell();
    const Mat3 matrix = cell.matrix();
    const Mat3 inverse = cell.inverse_matrix();
    const std::size_t count = atoms.size();

    std::vector<Vec3> scaled(count);
    std::vector<int> species(count);
    for (std::size_t i = 0; i < count; ++i) {
        scaled[i] = wrap(cell.scaled(atoms.position(i)));
        species[i] = atoms.number(i);
    }
    const PositionIndex index(scaled, species, cell, tolerance);

    // Translations are generated from the rarest species: fewer candidates,
    // and every one of them has to work for the whole structure anyway.
    std::map<int, std::vector<std::size_t>> by_species;
    for (std::size_t i = 0; i < count; ++i) {
        by_species[species[i]].push_back(i);
    }
    const std::vector<std::size_t>* rarest = nullptr;
    for (const auto& [element, members] : by_species) {
        if (rarest == nullptr || members.size() < rarest->size()) {
            rarest = &members;
        }
    }
    const std::size_t reference = rarest->front();

    for (const Mat3& rotation : lattice_point_group(cell, tolerance)) {
        // Rotate every atom once; only the translation changes per candidate.
        std::vector<Vec3> rotated(count);
        for (std::size_t i = 0; i < count; ++i) {
            rotated[i] = scaled[i] * rotation;
        }

        for (const std::size_t target : *rarest) {
            const Vec3 translation = wrap(scaled[target] - rotated[reference]);

            std::vector<std::size_t> map(count);
            bool matches = true;
            for (std::size_t i = 0; i < count && matches; ++i) {
                const std::ptrdiff_t image = index.find(rotated[i] + translation, species[i]);
                if (image < 0) {
                    matches = false;
                } else {
                    map[i] = static_cast<std::size_t>(image);
                }
            }
            if (!matches) {
                continue;
            }

            SymmetryOperation operation;
            operation.rotation = rotation;
            operation.translation = translation;
            operation.cartesian_rotation = inverse * rotation * matrix;
            operation.atom_map = std::move(map);
            dataset.operations.push_back(std::move(operation));
        }
    }

    if (dataset.operations.empty()) {
        throw PhononError(
            "no symmetry operations found, not even the identity; the structure and its cell "
            "are inconsistent at the requested tolerance");
    }

    // The distinct rotations are the crystal's point group.
    for (const SymmetryOperation& operation : dataset.operations) {
        const bool seen = std::any_of(
            dataset.point_group.begin(), dataset.point_group.end(),
            [&operation](const Mat3& known) { return known == operation.rotation; });
        if (!seen) {
            dataset.point_group.push_back(operation.rotation);
        }
    }

    // Orbits: two atoms are equivalent when some operation carries one to the
    // other. Union-find over the atom maps, labelled by the lowest index.
    std::vector<std::size_t> parent(count);
    std::iota(parent.begin(), parent.end(), std::size_t{0});
    const auto root = [&parent](std::size_t i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    for (const SymmetryOperation& operation : dataset.operations) {
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t a = root(i);
            const std::size_t b = root(operation.atom_map[i]);
            if (a != b) {
                parent[std::max(a, b)] = std::min(a, b);
            }
        }
    }

    dataset.equivalent_atoms.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        dataset.equivalent_atoms[i] = root(i);
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (dataset.equivalent_atoms[i] == i) {
            dataset.independent_atoms.push_back(i);
        }
    }
    return dataset;
}

std::vector<SymmetryOperation> site_symmetry(const SymmetryDataset& symmetry,
                                             std::size_t atom) {
    std::vector<SymmetryOperation> site;
    for (const SymmetryOperation& operation : symmetry.operations) {
        if (atom < operation.atom_map.size() && operation.atom_map[atom] == atom) {
            site.push_back(operation);
        }
    }
    return site;
}

const SymmetryOperation* mapping_operation(const SymmetryDataset& symmetry, std::size_t from,
                                           std::size_t to) {
    for (const SymmetryOperation& operation : symmetry.operations) {
        if (from < operation.atom_map.size() && operation.atom_map[from] == to) {
            return &operation;
        }
    }
    return nullptr;
}

}  // namespace fastatoms::phonon
