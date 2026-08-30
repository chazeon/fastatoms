// SPDX-License-Identifier: MIT
#pragma once

/// \file atoms.hpp
/// \brief The central `Atoms` container.

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "fastatoms/core/cell.hpp"
#include "fastatoms/core/constraints.hpp"
#include "fastatoms/io/parameters.hpp"
#include "fastatoms/core/vec3.hpp"

namespace fastatoms {

class Calculator;

/// A collection of atoms in an optionally periodic cell.
///
/// Layout: every per-atom quantity lives in its own contiguous `std::vector`,
/// with vector quantities stored atom-major (`positions[3*i + k]`). That keeps
/// the data numpy-compatible with zero copies while remaining friendly to
/// auto-vectorisation, and it lets a whole array be handed to BLAS or a
/// force kernel as a raw pointer.
///
/// Optional arrays (masses, momenta, tags, charges and user-defined arrays)
/// are empty until first written, so a plain structure carries no dead weight.
class Atoms {
public:
    /// One extra per-atom array of doubles, with `ncols` components per atom.
    struct Array {
        std::vector<double> data;
        std::size_t ncols{1};
    };

    Atoms() = default;

    /// Construct from atomic numbers and a flat 3N array of Cartesian positions.
    /// \throws std::invalid_argument on a length mismatch.
    Atoms(std::vector<int> numbers, std::vector<double> positions, Cell cell = Cell{});

    /// Construct from chemical symbols.
    Atoms(const std::vector<std::string>& symbols, std::vector<double> positions,
          Cell cell = Cell{});

    /// \name Size
    /// @{
    [[nodiscard]] std::size_t size() const noexcept { return numbers_.size(); }
    [[nodiscard]] bool empty() const noexcept { return numbers_.empty(); }
    void reserve(std::size_t n);
    void clear();
    /// @}

    /// \name Positions
    /// @{
    [[nodiscard]] const std::vector<double>& positions() const noexcept { return positions_; }
    [[nodiscard]] std::vector<double>& positions() noexcept { return positions_; }
    [[nodiscard]] const double* positions_data() const noexcept { return positions_.data(); }
    [[nodiscard]] double* positions_data() noexcept { return positions_.data(); }

    /// \throws std::invalid_argument unless `positions.size() == 3 * size()`.
    void set_positions(const std::vector<double>& positions);

    [[nodiscard]] FASTATOMS_ALWAYS_INLINE Vec3 position(std::size_t i) const noexcept {
        return Vec3::load(positions_.data() + 3 * i);
    }
    FASTATOMS_ALWAYS_INLINE void set_position(std::size_t i, const Vec3& r) noexcept {
        r.store(positions_.data() + 3 * i);
    }
    /// @}

    /// \name Species
    /// @{
    [[nodiscard]] const std::vector<int>& numbers() const noexcept { return numbers_; }
    [[nodiscard]] const int* numbers_data() const noexcept { return numbers_.data(); }
    [[nodiscard]] int* numbers_data() noexcept { return numbers_.data(); }
    void set_numbers(const std::vector<int>& numbers);
    [[nodiscard]] int number(std::size_t i) const noexcept { return numbers_[i]; }
    void set_number(std::size_t i, int z);

    [[nodiscard]] std::vector<std::string> symbols() const;
    [[nodiscard]] std::string symbol(std::size_t i) const;
    void set_symbols(const std::vector<std::string>& symbols);

    /// Compact formula, e.g. "H2O" (Hill order: C, H, then alphabetical).
    [[nodiscard]] std::string formula() const;
    /// @}

    /// \name Cell and boundary conditions
    /// @{
    [[nodiscard]] const Cell& cell() const noexcept { return cell_; }
    [[nodiscard]] Cell& cell() noexcept { return cell_; }

    /// Replace the cell. With `scale_atoms = true` the fractional coordinates
    /// are preserved, which is what a cell relaxation step needs.
    void set_cell(const Mat3& matrix, bool scale_atoms = false);
    void set_cell(const Cell& cell, bool scale_atoms = false);

    [[nodiscard]] const std::array<bool, 3>& pbc() const noexcept { return cell_.pbc(); }
    void set_pbc(const std::array<bool, 3>& pbc) noexcept { cell_.set_pbc(pbc); }
    void set_pbc(bool pbc) noexcept { cell_.set_pbc(pbc); }

    [[nodiscard]] double volume() const noexcept { return cell_.volume(); }
    /// @}

    /// \name Fractional coordinates
    /// @{
    [[nodiscard]] std::vector<double> scaled_positions(bool wrapped = false) const;
    void set_scaled_positions(const std::vector<double>& scaled);

    /// Fold every atom into the cell along the periodic directions.
    void wrap(double eps = 1e-7);
    /// @}

    /// \name Optional per-atom arrays
    /// @{
    [[nodiscard]] bool has_masses() const noexcept { return !masses_.empty(); }
    /// Masses in amu; falls back to standard atomic weights when unset.
    [[nodiscard]] std::vector<double> masses() const;
    void set_masses(const std::vector<double>& masses);
    [[nodiscard]] double mass(std::size_t i) const;

    [[nodiscard]] bool has_momenta() const noexcept { return !momenta_.empty(); }
    /// Momenta in amu*angstrom/(internal time unit); zeros when unset.
    [[nodiscard]] std::vector<double> momenta() const;
    void set_momenta(const std::vector<double>& momenta);

    /// Velocities are momenta divided by mass.
    [[nodiscard]] std::vector<double> velocities() const;
    void set_velocities(const std::vector<double>& velocities);

    [[nodiscard]] bool has_tags() const noexcept { return !tags_.empty(); }
    [[nodiscard]] std::vector<int> tags() const;
    void set_tags(const std::vector<int>& tags);

    [[nodiscard]] bool has_charges() const noexcept { return !charges_.empty(); }
    [[nodiscard]] std::vector<double> charges() const;
    void set_charges(const std::vector<double>& charges);

    /// Register an arbitrary per-atom array of `ncols` doubles per atom.
    /// \throws std::invalid_argument on a length mismatch, or if `name`
    ///         collides with a built-in array.
    void set_array(const std::string& name, const std::vector<double>& data, std::size_t ncols);
    [[nodiscard]] bool has_array(const std::string& name) const;
    /// \throws std::out_of_range if the array does not exist.
    [[nodiscard]] const Array& array(const std::string& name) const;
    void remove_array(const std::string& name);
    [[nodiscard]] std::vector<std::string> array_names() const;
    /// @}

    /// \name Frame information
    ///
    /// Per-structure scalars, as opposed to the per-atom arrays above: the
    /// energy of a trajectory frame, its timestep, a label. Extended XYZ
    /// carries exactly this on its comment line, and without somewhere to put
    /// it a reader would have to drop `energy=` on the floor.
    ///
    /// The name and the shape follow ASE's `atoms.info`, so a script that
    /// knows one knows the other.
    /// @{
    [[nodiscard]] const Value& info() const noexcept { return info_; }
    [[nodiscard]] Value& info() noexcept { return info_; }
    void set_info(Value info);
    /// @}

    /// \name Geometry
    /// @{
    /// Displacement `r_j - r_i`, minimum-image convention when `mic` is set.
    [[nodiscard]] Vec3 distance_vector(std::size_t i, std::size_t j, bool mic = false) const;
    [[nodiscard]] double distance(std::size_t i, std::size_t j, bool mic = false) const;

    /// Distances from atom `i` to every atom, including itself (0.0).
    [[nodiscard]] std::vector<double> distances_from(std::size_t i, bool mic = false) const;

    /// Full N x N distance matrix, row-major. O(N^2) - use a NeighborList for
    /// anything large.
    [[nodiscard]] std::vector<double> all_distances(bool mic = false) const;

    /// Mass-weighted centre of mass.
    [[nodiscard]] Vec3 center_of_mass() const;

    /// Unweighted geometric centre.
    [[nodiscard]] Vec3 center_of_geometry() const;

    /// Symmetric 3x3 moment-of-inertia tensor about the centre of mass.
    [[nodiscard]] Mat3 moments_of_inertia_tensor() const;

    void translate(const Vec3& shift);
    /// Per-atom translation; `shifts` must be 3N long.
    void translate(const std::vector<double>& shifts);

    /// Rotate about `axis` through `center` by `angle` radians.
    void rotate(double angle, const Vec3& axis, const Vec3& center);
    /// Rotate about the centre of mass.
    void rotate(double angle, const Vec3& axis);

    /// Centre the atoms in the cell.
    ///
    /// When `vacuum >= 0` the cell is additionally resized along the selected
    /// axes so that there is `vacuum` angstrom of empty space on each side.
    /// `axes` selects which Cartesian directions are affected.
    void center(double vacuum = -1.0, const std::array<bool, 3>& axes = {true, true, true});
    /// @}

    /// \name Dynamics helpers
    /// @{
    /// Kinetic energy in eV.
    [[nodiscard]] double kinetic_energy() const;

    /// Instantaneous temperature in K. `dof` defaults to 3N.
    [[nodiscard]] double temperature(int dof = -1) const;

    /// Total linear momentum.
    [[nodiscard]] Vec3 total_momentum() const;

    /// Remove net translation (and optionally net rotation) from the momenta.
    void zero_center_of_mass_momentum();
    /// @}

    /// \name Composition
    /// @{
    void append(int number, const Vec3& position);
    void append(const std::string& symbol, const Vec3& position);

    /// Concatenate `other` onto this object. The cell of `other` is ignored.
    void extend(const Atoms& other);

    /// A new `Atoms` holding only `indices`, in the given order.
    [[nodiscard]] Atoms subset(const std::vector<std::size_t>& indices) const;

    /// Delete atoms; `indices` may be unsorted and may contain duplicates.
    void remove(std::vector<std::size_t> indices);

    /// Build an (nx, ny, nz) supercell. Optional arrays are replicated.
    [[nodiscard]] Atoms repeat(int nx, int ny, int nz) const;
    /// @}

    /// \name Calculator
    ///
    /// Attaching a calculator lets the structure evaluate its own energy,
    /// forces and stress. The calculator caches results and only re-runs the
    /// underlying engine when the structure has actually changed.
    ///
    /// `subset`, `repeat` and the Python `copy()` deliberately do not carry the
    /// calculator over, matching ASE; a plain C++ copy shares it.
    /// @{
    void set_calculator(std::shared_ptr<Calculator> calculator);
    [[nodiscard]] const std::shared_ptr<Calculator>& calculator() const noexcept {
        return calculator_;
    }
    [[nodiscard]] bool has_calculator() const noexcept { return calculator_ != nullptr; }

    /// Potential energy in eV.
    /// \throws CalculatorError if no calculator is attached.
    [[nodiscard]] double get_potential_energy(bool force_consistent = false);

    /// Forces in eV/angstrom, 3N atom-major. Also stored as the per-atom array
    /// "forces".
    [[nodiscard]] std::vector<double> get_forces();

    /// Stress in eV/angstrom^3, Voigt order (xx, yy, zz, yz, xz, xy).
    [[nodiscard]] std::array<double, 6> get_stress();
    /// @}

    /// \name Constraints
    ///
    /// Constraints restrict how the atoms may move, and every part of the
    /// code that moves them respects the list attached here: `get_forces()`
    /// returns forces with the forbidden components already projected out,
    /// the optimizers project each proposed step, the integrators project the
    /// velocities too, and the nudged elastic band projects again after its
    /// own tangent projection.
    ///
    /// A constraint is immutable once built, so copies of an `Atoms` share
    /// theirs. Nothing can modify a constraint through one structure and
    /// surprise another; to change one, attach a different one.
    /// @{
    /// \throws ConstraintError if a constraint names an atom that is not here.
    void set_constraints(ConstraintList constraints);
    void add_constraint(std::shared_ptr<Constraint> constraint);
    void clear_constraints() noexcept { constraints_.clear(); }
    [[nodiscard]] const ConstraintList& constraints() const noexcept { return constraints_; }
    [[nodiscard]] bool has_constraints() const noexcept { return !constraints_.empty(); }

    /// Project the forbidden components out of `forces` in place. Called for
    /// you by `get_forces()`; call it yourself only on forces obtained some
    /// other way.
    void adjust_forces(std::vector<double>& forces) const;

    /// Project a proposed set of positions onto what the constraints allow,
    /// treating `reference` as where the atoms moved from.
    void adjust_positions(const std::vector<double>& reference,
                          std::vector<double>& positions) const;

    /// As above for momenta or velocities.
    void adjust_momenta(std::vector<double>& momenta) const;

    /// Degrees of freedom the constraints remove, counted without
    /// double-counting an atom that two of them name.
    [[nodiscard]] int constrained_degrees_of_freedom() const;

    /// Set the positions, projected onto what the constraints allow, using the
    /// current positions as the reference. This is what an optimizer or an
    /// integrator wants; `set_positions` itself stays unconstrained, because
    /// building a structure is not a move.
    void move_to(const std::vector<double>& positions);
    /// @}

    /// One-line summary used by the Python `repr`.
    [[nodiscard]] std::string to_string() const;

private:
    /// Throws if `i` is out of range. Only used on non-hot paths.
    void check_index(std::size_t i) const;
    /// Resize every optional array that is currently active.
    void resize_optional(std::size_t n);

    std::vector<int> numbers_;
    std::vector<double> positions_;  ///< 3N, atom-major

    std::vector<double> masses_;   ///< empty or N
    std::vector<double> momenta_;  ///< empty or 3N
    std::vector<double> charges_;  ///< empty or N
    Value info_{};                 ///< per-frame scalars, e.g. energy
    std::vector<int> tags_;        ///< empty or N

    std::map<std::string, Array> extra_;

    Cell cell_{};
    std::shared_ptr<Calculator> calculator_{};
    ConstraintList constraints_{};
};

}  // namespace fastatoms
