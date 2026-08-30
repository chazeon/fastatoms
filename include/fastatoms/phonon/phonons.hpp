// SPDX-License-Identifier: MIT
#pragma once

/// \file phonons.hpp
/// \brief Harmonic phonons by finite displacements.

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "fastatoms/calculators/calculator.hpp"
#include "fastatoms/core/atoms.hpp"
#include "fastatoms/io/parameters.hpp"
#include "fastatoms/phonon/eigen.hpp"
#include "fastatoms/phonon/force_constants.hpp"
#include "fastatoms/phonon/supercell.hpp"
#include "fastatoms/phonon/symmetry.hpp"

namespace fastatoms::phonon {

/// One displaced configuration.
struct Displacement {
    std::size_t atom{0};   ///< index into the supercell
    Vec3 vector{};         ///< the displacement itself, in angstrom
};

/// Frequencies and polarisation vectors at one q-point.
struct PhononModes {
    /// Frequencies in THz, ascending. A negative value is an imaginary
    /// frequency, `-sqrt(-omega^2)`, which is how a dynamical instability
    /// announces itself.
    std::vector<double> frequencies{};

    /// Eigenvalues of the dynamical matrix, `omega^2`, in eV/(angstrom^2 amu).
    std::vector<double> squared_frequencies{};

    /// Polarisation vectors as columns, mass-weighted and orthonormal.
    ComplexMatrix eigenvectors{};
};

/// Harmonic phonons from finite displacements.
///
/// The calculation runs in four steps, each of which is separately reachable:
///
///   1. build a supercell from an integer transformation matrix,
///   2. find the space group and use it to decide which atoms, and which
///      directions, actually have to be displaced,
///   3. displace them, collect forces, and solve for the force constants,
///   4. Fourier transform to the dynamical matrix at a q-point and diagonalise.
///
/// The symmetry reduction is the step that decides whether a calculation is
/// affordable: for silicon in a 2x2x2 supercell it turns 48 displacements into
/// two. What makes it safe is that the reduced and unreduced routes are
/// checked against each other in the test suite - `symmetry: false` runs every
/// displacement and must give the same force constants.
///
/// Parameters:
///   - `supercell`   - three integers, or nine for a full transformation
///                     matrix. Default `[2, 2, 2]`.
///   - `displacement`- the finite displacement in angstrom. Default 0.01.
///   - `symmetry`    - use the space group to reduce the work. Default true.
///   - `central`     - displace by both `+d` and `-d` and difference them,
///                     which cancels the leading anharmonic error. Default
///                     true, at twice the cost.
///   - `acoustic_sum_rule` - impose translational invariance. Default true.
///   - `symmetry_tolerance` - distance in angstrom. Default 1e-5.
class Phonons {
public:
    explicit Phonons(Atoms unit_cell, Value parameters = {});

    /// \name Setup
    /// @{
    [[nodiscard]] const Atoms& unit_cell() const noexcept { return unit_cell_; }
    [[nodiscard]] const SupercellMap& supercell() const noexcept { return supercell_; }
    [[nodiscard]] const SymmetryDataset& symmetry() const noexcept { return symmetry_; }
    [[nodiscard]] Mat3 supercell_matrix() const;
    [[nodiscard]] double displacement_distance() const;
    [[nodiscard]] double symmetry_tolerance() const;
    [[nodiscard]] bool use_symmetry() const;
    [[nodiscard]] bool central_differences() const;
    [[nodiscard]] bool acoustic_sum_rule() const;
    /// @}

    /// \name Displacements
    /// @{
    /// The configurations that have to be evaluated, in order.
    [[nodiscard]] const std::vector<Displacement>& displacements() const noexcept {
        return displacements_;
    }

    /// The supercells to run, one per displacement. Use these to farm the
    /// calculations out yourself and feed the forces back with `set_forces`.
    [[nodiscard]] std::vector<Atoms> displaced_supercells() const;

    /// How many force evaluations the reduced set needs, and how many it would
    /// have needed without symmetry. The ratio is the point of the exercise.
    [[nodiscard]] std::size_t calculation_count() const noexcept {
        return displacements_.size();
    }
    [[nodiscard]] std::size_t calculation_count_without_symmetry() const;
    /// @}

    /// \name Running
    /// @{
    /// Displace, evaluate, and solve for the force constants.
    ///
    /// Uses the calculator attached to the unit cell unless one is given.
    /// \throws PhononError if there is no calculator.
    void run(std::shared_ptr<Calculator> calculator = nullptr);

    /// Supply forces computed elsewhere, one 3N array per displacement in the
    /// order `displacements()` gives, plus the equilibrium forces first when
    /// `central` is off.
    void set_forces(const std::vector<std::vector<double>>& forces,
                    const std::vector<double>& equilibrium_forces = {});

    [[nodiscard]] bool has_force_constants() const noexcept {
        return !force_constants_.empty();
    }
    [[nodiscard]] const ForceConstants& force_constants() const;

    /// Forces on the undisplaced supercell, in eV/angstrom. Large values mean
    /// the structure was not relaxed and the harmonic expansion is being taken
    /// about the wrong point.
    [[nodiscard]] const std::vector<double>& equilibrium_forces() const noexcept {
        return equilibrium_forces_;
    }
    [[nodiscard]] double residual_force() const;
    /// @}

    /// \name Results
    /// @{
    /// The dynamical matrix at `q`, given in fractional coordinates of the
    /// unit cell's reciprocal lattice.
    [[nodiscard]] ComplexMatrix dynamical_matrix(const Vec3& q) const;

    /// Frequencies in THz at `q`, ascending.
    [[nodiscard]] std::vector<double> frequencies(const Vec3& q) const;

    /// Frequencies and polarisation vectors at `q`.
    [[nodiscard]] PhononModes modes(const Vec3& q) const;

    /// Frequencies along a list of q-points, one row per point.
    [[nodiscard]] std::vector<std::vector<double>> band_structure(
        const std::vector<Vec3>& path) const;

    /// Frequencies on a uniform Gamma-centred mesh, flattened. The natural
    /// input for a density of states.
    [[nodiscard]] std::vector<double> mesh_frequencies(int n1, int n2, int n3) const;
    /// @}

    /// \name Unit conversions from THz
    /// @{
    static constexpr double kTHzToWavenumber = 33.35641;   ///< cm^-1
    static constexpr double kTHzToMilliElectronVolt = 4.135667696;
    /// @}

    [[nodiscard]] std::string to_string() const;

private:
    /// Which atom is displaced along which directions, and where the
    /// corresponding entries live in `displacements_`.
    struct DisplacementGroup {
        std::size_t atom{0};
        Vec3 direction{};
        std::size_t plus{0};
        std::ptrdiff_t minus{-1};
    };

    void build_displacements();
    void solve_force_constants(const std::vector<std::vector<double>>& forces);
    void build_shortest_vectors();
    [[nodiscard]] const Value& parameter(std::string_view key) const;
    [[noreturn]] void fail(const std::string& message) const;

    Atoms unit_cell_;
    Value parameters_;
    SupercellMap supercell_;
    SymmetryDataset symmetry_;

    std::vector<Displacement> displacements_{};
    std::vector<DisplacementGroup> groups_{};

    ForceConstants force_constants_{};
    std::vector<double> equilibrium_forces_{};

    /// For each (unit-cell atom, supercell atom) pair, the shortest periodic
    /// images and their weights, as integer lattice vectors in the unit-cell
    /// basis.
    std::vector<std::vector<std::pair<std::array<int, 3>, double>>> shortest_vectors_{};
};

}  // namespace fastatoms::phonon
