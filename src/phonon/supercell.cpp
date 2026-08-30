// SPDX-License-Identifier: MIT
#include "fastatoms/phonon/supercell.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>

namespace fastatoms::phonon {
namespace {

/// Nearest integer, with a check that the value really was one.
int require_integer(double value, const char* what) {
    const double rounded = std::nearbyint(value);
    if (std::abs(value - rounded) > 1e-8) {
        std::ostringstream out;
        out << what << " must be integer valued, got " << value;
        throw PhononError(out.str());
    }
    return static_cast<int>(rounded);
}

/// Fold a fractional coordinate into [0, 1), nudging values that sit on the
/// upper boundary through rounding.
double wrap(double value) {
    double folded = value - std::floor(value);
    if (folded > 1.0 - 1e-9) {
        folded = 0.0;
    }
    return folded;
}

}  // namespace

std::string SupercellMap::to_string() const {
    std::ostringstream out;
    out << "SupercellMap(" << atoms.size() << " atoms, " << cell_count << " cells of "
        << unit_cell_size << ")";
    return out.str();
}

SupercellMap make_supercell(const Atoms& unit_cell, const Mat3& transformation) {
    if (unit_cell.empty()) {
        throw PhononError("cannot build a supercell of an empty structure");
    }
    if (!unit_cell.cell().is_invertible()) {
        throw PhononError("a supercell needs a unit cell that spans three dimensions");
    }

    // The transformation has to be an integer matrix: the supercell lattice is
    // a sublattice of the unit-cell lattice, or the mapping back is undefined.
    std::array<std::array<int, 3>, 3> p{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            p[i][j] = require_integer(transformation(i, j), "the supercell matrix");
        }
    }

    const double determinant = transformation.determinant();
    const int cells = require_integer(determinant, "the supercell determinant");
    if (cells <= 0) {
        throw PhononError("the supercell matrix must have a positive determinant, got " +
                          std::to_string(cells));
    }

    SupercellMap map;
    map.transformation = transformation;
    map.unit_cell_size = unit_cell.size();
    map.cell_count = static_cast<std::size_t>(cells);

    const Mat3 unit_matrix = unit_cell.cell().matrix();
    const Mat3 super_matrix = transformation * unit_matrix;
    const Mat3 inverse_transformation = transformation.inverse();

    // Enumerate the lattice points of the unit cell that fall inside the
    // supercell. A translation n (integer, unit-cell basis) sits at fractional
    // coordinates n * P^-1 in the supercell, so the ones in [0,1)^3 are the
    // distinct cells. Searching the bounding box of P covers all of them.
    int bound = 1;
    for (std::size_t i = 0; i < 3; ++i) {
        int row_sum = 0;
        for (std::size_t j = 0; j < 3; ++j) {
            row_sum += std::abs(p[j][i]);
        }
        bound = std::max(bound, row_sum);
    }

    // Keyed by the wrapped fractional coordinate, rounded, so duplicates from
    // different integer triples collapse.
    std::map<std::array<long long, 3>, std::array<int, 3>> found;
    for (int i = -bound; i <= bound; ++i) {
        for (int j = -bound; j <= bound; ++j) {
            for (int k = -bound; k <= bound; ++k) {
                const Vec3 n{static_cast<double>(i), static_cast<double>(j),
                             static_cast<double>(k)};
                const Vec3 s = n * inverse_transformation;
                const Vec3 folded{wrap(s.x), wrap(s.y), wrap(s.z)};
                const std::array<long long, 3> key{
                    static_cast<long long>(std::llround(folded.x * 1e7)),
                    static_cast<long long>(std::llround(folded.y * 1e7)),
                    static_cast<long long>(std::llround(folded.z * 1e7))};
                // Several integer triples land on the same cell; keep the
                // one closest to the origin, so that in particular the cell at
                // the origin is represented by (0, 0, 0).
                const std::array<int, 3> triple{i, j, k};
                const auto [entry, inserted] = found.emplace(key, triple);
                if (!inserted) {
                    const auto& current = entry->second;
                    const int existing = current[0] * current[0] + current[1] * current[1] +
                                         current[2] * current[2];
                    const int candidate = i * i + j * j + k * k;
                    if (candidate < existing) {
                        entry->second = triple;
                    }
                }
            }
        }
    }

    if (found.size() != map.cell_count) {
        throw PhononError("found " + std::to_string(found.size()) + " lattice points but the "
                          "supercell matrix has determinant " + std::to_string(cells) +
                          "; the matrix is probably not a valid sublattice");
    }

    // Order the cells with the zero translation first, so that the leading
    // block of atoms is the original unit cell.
    std::vector<std::array<int, 3>> translations;
    translations.reserve(map.cell_count);
    translations.push_back({0, 0, 0});
    for (const auto& [key, triple] : found) {
        if (triple[0] != 0 || triple[1] != 0 || triple[2] != 0) {
            translations.push_back(triple);
        }
    }
    if (translations.size() != map.cell_count) {
        throw PhononError("the zero translation was not among the supercell lattice points");
    }

    const std::size_t total = map.cell_count * unit_cell.size();
    std::vector<int> numbers;
    std::vector<double> positions;
    numbers.reserve(total);
    positions.reserve(3 * total);
    map.unit_cell_index.reserve(total);
    map.lattice_point.reserve(total);
    map.lattice_translation.reserve(total);

    for (const std::array<int, 3>& n : translations) {
        const Vec3 shift = unit_matrix.row(0) * n[0] + unit_matrix.row(1) * n[1] +
                           unit_matrix.row(2) * n[2];
        for (std::size_t atom = 0; atom < unit_cell.size(); ++atom) {
            const Vec3 position = unit_cell.position(atom) + shift;
            numbers.push_back(unit_cell.number(atom));
            positions.push_back(position.x);
            positions.push_back(position.y);
            positions.push_back(position.z);
            map.unit_cell_index.push_back(atom);
            map.lattice_point.push_back(n);
            map.lattice_translation.push_back(shift);
        }
    }

    Cell cell(super_matrix, unit_cell.cell().pbc());
    map.atoms = Atoms(std::move(numbers), std::move(positions), cell);

    // Masses follow the atoms; anything else would be lost anyway.
    if (unit_cell.has_masses()) {
        std::vector<double> masses;
        masses.reserve(total);
        const std::vector<double> unit_masses = unit_cell.masses();
        for (std::size_t k = 0; k < total; ++k) {
            masses.push_back(unit_masses[map.unit_cell_index[k]]);
        }
        map.atoms.set_masses(masses);
    }
    return map;
}

SupercellMap make_supercell(const Atoms& unit_cell, int n1, int n2, int n3) {
    if (n1 < 1 || n2 < 1 || n3 < 1) {
        throw PhononError("supercell repetitions must all be at least 1");
    }
    return make_supercell(unit_cell, Mat3::diagonal(n1, n2, n3));
}

}  // namespace fastatoms::phonon
