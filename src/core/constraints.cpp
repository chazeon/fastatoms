// SPDX-License-Identifier: MIT

#include "fastatoms/core/constraints.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>

#include "fastatoms/core/atoms.hpp"

namespace fastatoms {
namespace {

/// Sorted, duplicate-free indices. Applying a constraint twice to the same
/// atom is harmless for a projection, but it would double the degrees of
/// freedom it claims to remove, and that silently corrupts the temperature.
std::vector<std::size_t> tidy(std::vector<std::size_t> indices) {
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    return indices;
}

Vec3 unit(const Vec3& v, const char* what) {
    const double length = v.norm();
    if (!(length > 0.0)) {
        throw ConstraintError(std::string(what) + " must have a direction, and this one has zero "
                                                 "length");
    }
    return Vec3{v.x / length, v.y / length, v.z / length};
}

/// Keep only the component of each named row along `axis`.
void project_onto(const std::vector<std::size_t>& indices, const Vec3& axis,
                  std::vector<double>& rows) {
    for (const std::size_t i : indices) {
        const std::size_t o = 3 * i;
        if (o + 2 >= rows.size()) {
            continue;
        }
        const double along = rows[o] * axis.x + rows[o + 1] * axis.y + rows[o + 2] * axis.z;
        rows[o] = along * axis.x;
        rows[o + 1] = along * axis.y;
        rows[o + 2] = along * axis.z;
    }
}

/// Remove the component of each named row along `axis`.
void project_out(const std::vector<std::size_t>& indices, const Vec3& axis,
                 std::vector<double>& rows) {
    for (const std::size_t i : indices) {
        const std::size_t o = 3 * i;
        if (o + 2 >= rows.size()) {
            continue;
        }
        const double along = rows[o] * axis.x + rows[o + 1] * axis.y + rows[o + 2] * axis.z;
        rows[o] -= along * axis.x;
        rows[o + 1] -= along * axis.y;
        rows[o + 2] -= along * axis.z;
    }
}

/// The displacement from `reference`, projected, then put back. Constraining
/// the displacement rather than the position is what makes a line or a plane
/// pass through wherever the atom already is.
template <typename Project>
void adjust_displacement(const std::vector<std::size_t>& indices,
                         const std::vector<double>& reference, std::vector<double>& positions,
                         Project project) {
    if (reference.size() != positions.size()) {
        return;
    }
    std::vector<double> delta(positions.size());
    for (std::size_t c = 0; c < positions.size(); ++c) {
        delta[c] = positions[c] - reference[c];
    }
    project(delta);
    for (const std::size_t i : indices) {
        const std::size_t o = 3 * i;
        if (o + 2 >= positions.size()) {
            continue;
        }
        positions[o] = reference[o] + delta[o];
        positions[o + 1] = reference[o + 1] + delta[o + 1];
        positions[o + 2] = reference[o + 2] + delta[o + 2];
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Constraint
// ---------------------------------------------------------------------------

std::size_t Constraint::extent() const noexcept {
    const std::vector<std::size_t>& list = indices();
    return list.empty() ? 0 : *std::max_element(list.begin(), list.end()) + 1;
}

void Constraint::require_fits(std::size_t natoms) const {
    if (extent() > natoms) {
        throw ConstraintError(name() + " names atom " + std::to_string(extent() - 1) +
                              ", but the structure has " + std::to_string(natoms) + " atoms");
    }
}

std::string Constraint::to_string() const {
    return name() + "(" + std::to_string(indices().size()) + " atoms)";
}

// ---------------------------------------------------------------------------
// FixAtoms
// ---------------------------------------------------------------------------

FixAtoms::FixAtoms(std::vector<std::size_t> indices) : indices_(tidy(std::move(indices))) {}

void FixAtoms::adjust_forces(const Atoms& atoms, std::vector<double>& forces) const {
    require_fits(atoms.size());
    for (const std::size_t i : indices_) {
        const std::size_t o = 3 * i;
        if (o + 2 < forces.size()) {
            forces[o] = 0.0;
            forces[o + 1] = 0.0;
            forces[o + 2] = 0.0;
        }
    }
}

void FixAtoms::adjust_positions(const Atoms& atoms, const std::vector<double>& reference,
                                std::vector<double>& positions) const {
    require_fits(atoms.size());
    if (reference.size() != positions.size()) {
        return;
    }
    for (const std::size_t i : indices_) {
        const std::size_t o = 3 * i;
        if (o + 2 < positions.size()) {
            positions[o] = reference[o];
            positions[o + 1] = reference[o + 1];
            positions[o + 2] = reference[o + 2];
        }
    }
}

void FixAtoms::adjust_momenta(const Atoms& atoms, std::vector<double>& momenta) const {
    adjust_forces(atoms, momenta);   // the same projection, to zero
}

int FixAtoms::removed_degrees_of_freedom(const Atoms&) const {
    return 3 * static_cast<int>(indices_.size());
}

std::shared_ptr<Constraint> FixAtoms::clone() const {
    return std::make_shared<FixAtoms>(indices_);
}

std::string FixAtoms::to_string() const {
    return "FixAtoms(" + std::to_string(indices_.size()) + " atoms)";
}

// ---------------------------------------------------------------------------
// FixLine
// ---------------------------------------------------------------------------

FixLine::FixLine(std::vector<std::size_t> indices, const Vec3& direction)
    : indices_(tidy(std::move(indices))), direction_(unit(direction, "FixLine's direction")) {}

void FixLine::adjust_forces(const Atoms& atoms, std::vector<double>& forces) const {
    require_fits(atoms.size());
    project_onto(indices_, direction_, forces);
}

void FixLine::adjust_positions(const Atoms& atoms, const std::vector<double>& reference,
                               std::vector<double>& positions) const {
    require_fits(atoms.size());
    adjust_displacement(indices_, reference, positions, [&](std::vector<double>& delta) {
        project_onto(indices_, direction_, delta);
    });
}

void FixLine::adjust_momenta(const Atoms& atoms, std::vector<double>& momenta) const {
    require_fits(atoms.size());
    project_onto(indices_, direction_, momenta);
}

int FixLine::removed_degrees_of_freedom(const Atoms&) const {
    return 2 * static_cast<int>(indices_.size());
}

std::shared_ptr<Constraint> FixLine::clone() const {
    return std::make_shared<FixLine>(indices_, direction_);
}

std::string FixLine::to_string() const {
    std::ostringstream out;
    out << "FixLine(" << indices_.size() << " atoms, along [" << direction_.x << ", "
        << direction_.y << ", " << direction_.z << "])";
    return out.str();
}

// ---------------------------------------------------------------------------
// FixPlane
// ---------------------------------------------------------------------------

FixPlane::FixPlane(std::vector<std::size_t> indices, const Vec3& normal)
    : indices_(tidy(std::move(indices))), normal_(unit(normal, "FixPlane's normal")) {}

void FixPlane::adjust_forces(const Atoms& atoms, std::vector<double>& forces) const {
    require_fits(atoms.size());
    project_out(indices_, normal_, forces);
}

void FixPlane::adjust_positions(const Atoms& atoms, const std::vector<double>& reference,
                                std::vector<double>& positions) const {
    require_fits(atoms.size());
    adjust_displacement(indices_, reference, positions, [&](std::vector<double>& delta) {
        project_out(indices_, normal_, delta);
    });
}

void FixPlane::adjust_momenta(const Atoms& atoms, std::vector<double>& momenta) const {
    require_fits(atoms.size());
    project_out(indices_, normal_, momenta);
}

int FixPlane::removed_degrees_of_freedom(const Atoms&) const {
    return static_cast<int>(indices_.size());
}

std::shared_ptr<Constraint> FixPlane::clone() const {
    return std::make_shared<FixPlane>(indices_, normal_);
}

std::string FixPlane::to_string() const {
    std::ostringstream out;
    out << "FixPlane(" << indices_.size() << " atoms, normal [" << normal_.x << ", " << normal_.y
        << ", " << normal_.z << "])";
    return out.str();
}

// ---------------------------------------------------------------------------
// Whole lists
// ---------------------------------------------------------------------------

void adjust_forces(const ConstraintList& constraints, const Atoms& atoms,
                   std::vector<double>& forces) {
    for (const std::shared_ptr<Constraint>& constraint : constraints) {
        if (constraint) {
            constraint->adjust_forces(atoms, forces);
        }
    }
}

void adjust_positions(const ConstraintList& constraints, const Atoms& atoms,
                      const std::vector<double>& reference, std::vector<double>& positions) {
    for (const std::shared_ptr<Constraint>& constraint : constraints) {
        if (constraint) {
            constraint->adjust_positions(atoms, reference, positions);
        }
    }
}

void adjust_momenta(const ConstraintList& constraints, const Atoms& atoms,
                    std::vector<double>& momenta) {
    for (const std::shared_ptr<Constraint>& constraint : constraints) {
        if (constraint) {
            constraint->adjust_momenta(atoms, momenta);
        }
    }
}

int removed_degrees_of_freedom(const ConstraintList& constraints, const Atoms& atoms) {
    if (constraints.empty()) {
        return 0;
    }
    // Summing what each constraint claims would double-count an atom named by
    // two of them - FixPlane and FixLine on the same atom together leave one
    // direction, not zero, and a naive sum would report four removed of three.
    // So the answer is measured instead: push the three unit directions of
    // every atom through the very projection the code enforces, and count how
    // many independent directions survive. It cannot disagree with what is
    // actually enforced, because it is what is actually enforced.
    const std::size_t n = atoms.size();
    std::array<std::vector<double>, 3> probe;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        probe[axis].assign(3 * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            probe[axis][3 * i + axis] = 1.0;
        }
        adjust_momenta(constraints, atoms, probe[axis]);
    }

    int removed = 0;
    for (std::size_t i = 0; i < n; ++i) {
        // Rank of the 3x3 block by Gram-Schmidt. The rows are projections of
        // an orthonormal basis, so their norms are bounded by one and a fixed
        // tolerance is meaningful.
        std::array<Vec3, 3> rows{
            Vec3{probe[0][3 * i], probe[0][3 * i + 1], probe[0][3 * i + 2]},
            Vec3{probe[1][3 * i], probe[1][3 * i + 1], probe[1][3 * i + 2]},
            Vec3{probe[2][3 * i], probe[2][3 * i + 1], probe[2][3 * i + 2]},
        };
        std::vector<Vec3> basis;
        for (Vec3 row : rows) {
            for (const Vec3& b : basis) {
                const double along = row.x * b.x + row.y * b.y + row.z * b.z;
                row = Vec3{row.x - along * b.x, row.y - along * b.y, row.z - along * b.z};
            }
            const double length = row.norm();
            if (length > 1e-9) {
                basis.push_back(Vec3{row.x / length, row.y / length, row.z / length});
            }
        }
        removed += 3 - static_cast<int>(basis.size());
    }
    return removed;
}

// ---------------------------------------------------------------------------
// The Atoms side
//
// Defined here rather than in atoms.cpp so that the core container keeps its
// dependency on the constraint hierarchy to the forward declaration in the
// header, the same arrangement the calculator link uses.
// ---------------------------------------------------------------------------

void Atoms::set_constraints(ConstraintList constraints) {
    for (const std::shared_ptr<Constraint>& constraint : constraints) {
        if (constraint) {
            constraint->require_fits(size());
        }
    }
    constraints_ = std::move(constraints);
}

void Atoms::add_constraint(std::shared_ptr<Constraint> constraint) {
    if (!constraint) {
        return;
    }
    constraint->require_fits(size());
    constraints_.push_back(std::move(constraint));
}

void Atoms::adjust_forces(std::vector<double>& forces) const {
    fastatoms::adjust_forces(constraints_, *this, forces);
}

void Atoms::adjust_positions(const std::vector<double>& reference,
                             std::vector<double>& positions) const {
    fastatoms::adjust_positions(constraints_, *this, reference, positions);
}

void Atoms::adjust_momenta(std::vector<double>& momenta) const {
    fastatoms::adjust_momenta(constraints_, *this, momenta);
}

int Atoms::constrained_degrees_of_freedom() const {
    return fastatoms::removed_degrees_of_freedom(constraints_, *this);
}

void Atoms::move_to(const std::vector<double>& positions) {
    if (constraints_.empty()) {
        set_positions(positions);
        return;
    }
    std::vector<double> allowed = positions;
    adjust_positions(this->positions(), allowed);
    set_positions(allowed);
}

}  // namespace fastatoms
