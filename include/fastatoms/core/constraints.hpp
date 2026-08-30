// SPDX-License-Identifier: MIT
#pragma once

/// \file constraints.hpp
/// \brief Constraints that restrict how atoms may move.
///
/// A constraint here is a *linear* restriction on the motion of individual
/// atoms: a frozen substrate, an atom allowed to slide along one direction, an
/// adsorbate held to a plane. Each one answers the same three questions - what
/// may a force do, what may a displacement do, what may a velocity do - and
/// every part of the code that moves atoms asks them.
///
/// This is a different thing from `DistanceConstraint` and `md::Constraints`
/// in \ref constraint.hpp, which are holonomic bond-length constraints solved
/// iteratively by SHAKE and RATTLE. Those couple atoms to each other and need
/// a solver; these are projections, exact in one step, and the two compose:
/// rigid water molecules on a frozen slab is both at once.

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "fastatoms/core/vec3.hpp"

namespace fastatoms {

class Atoms;

/// Raised for a constraint that cannot be built or does not fit a structure.
class ConstraintError : public std::runtime_error {
public:
    explicit ConstraintError(const std::string& message) : std::runtime_error(message) {}
};

/// A restriction on how atoms may move.
///
/// The three `adjust_*` methods are the whole interface, and they must agree
/// with each other: whatever subspace of motion a constraint forbids, it has
/// to forbid in the force, in the displacement and in the velocity alike.
/// Removing it from only the force is the classic mistake - the atom then
/// stays put in a relaxation and drifts in dynamics, because it kept whatever
/// velocity it started with.
class Constraint {
public:
    virtual ~Constraint() = default;

    Constraint(const Constraint&) = delete;
    Constraint& operator=(const Constraint&) = delete;

    /// Short identifier, e.g. "FixAtoms".
    [[nodiscard]] virtual std::string name() const = 0;

    /// Project the forbidden components out of `forces`, 3N atom-major.
    virtual void adjust_forces(const Atoms& atoms, std::vector<double>& forces) const = 0;

    /// Project a proposed set of positions back onto the allowed subspace.
    ///
    /// \param reference where the atoms were before the move; the constraint
    ///        acts on the displacement from it, which is what makes a plane or
    ///        a line an offset one through the atom's own starting point
    ///        rather than through the origin.
    virtual void adjust_positions(const Atoms& atoms, const std::vector<double>& reference,
                                  std::vector<double>& positions) const = 0;

    /// Project the forbidden components out of momenta or velocities. The two
    /// differ by a mass per row, which no constraint here is sensitive to.
    virtual void adjust_momenta(const Atoms& atoms, std::vector<double>& momenta) const = 0;

    /// How many degrees of freedom this removes from `atoms`.
    ///
    /// The temperature of a constrained system is `2K / (dof kB)` with this
    /// already subtracted; getting it wrong shows up as a thermostat holding
    /// the system at the wrong temperature, not as anything more obvious.
    [[nodiscard]] virtual int removed_degrees_of_freedom(const Atoms& atoms) const = 0;

    /// The atoms this constraint acts on.
    [[nodiscard]] virtual const std::vector<std::size_t>& indices() const noexcept = 0;

    /// One past the largest index referred to, for checking against a size.
    [[nodiscard]] std::size_t extent() const noexcept;

    /// A copy, so that copying an `Atoms` does not share mutable state.
    [[nodiscard]] virtual std::shared_ptr<Constraint> clone() const = 0;

    [[nodiscard]] virtual std::string to_string() const;

    /// Throw unless every index this constraint names exists in `natoms`.
    ///
    /// Checked when a constraint is attached, and again on every use: a
    /// structure can be shrunk after the fact, and a constraint that then
    /// silently addressed nothing would be worse than one that complains.
    void require_fits(std::size_t natoms) const;

protected:
    Constraint() = default;
};

/// Atoms that do not move at all.
///
/// The substrate of a surface calculation, the far end of a cluster, the bulk
/// region around a defect. Three degrees of freedom go per atom.
class FixAtoms final : public Constraint {
public:
    explicit FixAtoms(std::vector<std::size_t> indices);

    [[nodiscard]] std::string name() const override { return "FixAtoms"; }
    void adjust_forces(const Atoms& atoms, std::vector<double>& forces) const override;
    void adjust_positions(const Atoms& atoms, const std::vector<double>& reference,
                          std::vector<double>& positions) const override;
    void adjust_momenta(const Atoms& atoms, std::vector<double>& momenta) const override;
    [[nodiscard]] int removed_degrees_of_freedom(const Atoms& atoms) const override;
    [[nodiscard]] const std::vector<std::size_t>& indices() const noexcept override {
        return indices_;
    }
    [[nodiscard]] std::shared_ptr<Constraint> clone() const override;
    [[nodiscard]] std::string to_string() const override;

private:
    std::vector<std::size_t> indices_;
};

/// Atoms confined to a line: motion is allowed *along* `direction` only.
///
/// An atom pulled through a channel, or a reaction coordinate held to one
/// axis. Two degrees of freedom go per atom, leaving one.
class FixLine final : public Constraint {
public:
    /// \param direction the line's direction; normalised on construction.
    /// \throws ConstraintError if `direction` has no length.
    FixLine(std::vector<std::size_t> indices, const Vec3& direction);

    [[nodiscard]] std::string name() const override { return "FixLine"; }
    void adjust_forces(const Atoms& atoms, std::vector<double>& forces) const override;
    void adjust_positions(const Atoms& atoms, const std::vector<double>& reference,
                          std::vector<double>& positions) const override;
    void adjust_momenta(const Atoms& atoms, std::vector<double>& momenta) const override;
    [[nodiscard]] int removed_degrees_of_freedom(const Atoms& atoms) const override;
    [[nodiscard]] const std::vector<std::size_t>& indices() const noexcept override {
        return indices_;
    }
    [[nodiscard]] const Vec3& direction() const noexcept { return direction_; }
    [[nodiscard]] std::shared_ptr<Constraint> clone() const override;
    [[nodiscard]] std::string to_string() const override;

private:
    std::vector<std::size_t> indices_;
    Vec3 direction_;
};

/// Atoms confined to a plane: motion perpendicular to `normal` only.
///
/// An adsorbate kept at a fixed height, or a slab layer allowed to relax
/// in-plane but not to buckle. One degree of freedom goes per atom, leaving
/// two.
class FixPlane final : public Constraint {
public:
    /// \param normal the plane's normal; normalised on construction.
    /// \throws ConstraintError if `normal` has no length.
    FixPlane(std::vector<std::size_t> indices, const Vec3& normal);

    [[nodiscard]] std::string name() const override { return "FixPlane"; }
    void adjust_forces(const Atoms& atoms, std::vector<double>& forces) const override;
    void adjust_positions(const Atoms& atoms, const std::vector<double>& reference,
                          std::vector<double>& positions) const override;
    void adjust_momenta(const Atoms& atoms, std::vector<double>& momenta) const override;
    [[nodiscard]] int removed_degrees_of_freedom(const Atoms& atoms) const override;
    [[nodiscard]] const std::vector<std::size_t>& indices() const noexcept override {
        return indices_;
    }
    [[nodiscard]] const Vec3& normal() const noexcept { return normal_; }
    [[nodiscard]] std::shared_ptr<Constraint> clone() const override;
    [[nodiscard]] std::string to_string() const override;

private:
    std::vector<std::size_t> indices_;
    Vec3 normal_;
};

/// A list of constraints, as carried by an `Atoms`.
using ConstraintList = std::vector<std::shared_ptr<Constraint>>;

/// \name Applying a whole list
/// Free functions so that code holding a list without an `Atoms` - the band in
/// \ref neb.hpp, for one - can apply them the same way.
/// @{
void adjust_forces(const ConstraintList& constraints, const Atoms& atoms,
                   std::vector<double>& forces);
void adjust_positions(const ConstraintList& constraints, const Atoms& atoms,
                      const std::vector<double>& reference, std::vector<double>& positions);
void adjust_momenta(const ConstraintList& constraints, const Atoms& atoms,
                    std::vector<double>& momenta);
[[nodiscard]] int removed_degrees_of_freedom(const ConstraintList& constraints,
                                             const Atoms& atoms);

/// How many of the three uniform translations the constraints still allow.
///
/// Three when nothing is constrained. Removing the net momentum of a system
/// takes away one degree of freedom per translation that is still a symmetry,
/// and a constraint removes exactly those that move an atom it holds: a frozen
/// atom leaves none, a plane leaves the two translations lying in it, a line
/// leaves the one along it. Subtracting three regardless is how a slab with a
/// frozen substrate ends up appearing to have no degrees of freedom at all.
[[nodiscard]] int free_translations(const ConstraintList& constraints, const Atoms& atoms);
/// @}

}  // namespace fastatoms
