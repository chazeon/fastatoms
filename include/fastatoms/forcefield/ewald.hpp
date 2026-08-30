// SPDX-License-Identifier: MIT
#pragma once

/// \file ewald.hpp
/// \brief Ewald summation for the electrostatics of a periodic system.
///
/// A lattice sum of `1/r` is conditionally convergent: its value depends on
/// the order of summation, so truncating it at a cutoff is not an
/// approximation but a different quantity. Ewald's construction splits each
/// charge into a screened point and a compensating smooth cloud,
///
///     1/r = erfc(alpha r)/r  +  erf(alpha r)/r
///
/// leaving a short-ranged piece that a neighbour list handles and a smooth
/// piece that converges quickly in reciprocal space. The split is exact: the
/// total is independent of `alpha`, which is the sharpest check that an
/// implementation is right, and the test suite makes it.
///
/// Conducting ("tinfoil") boundary conditions are used, so the surface dipole
/// term is zero. That is the right choice for a polar liquid surrounded by a
/// conductor, and it is what nearly every water simulation assumes.

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/core/neighbor_list.hpp"
#include "fastatoms/core/vec3.hpp"
#include "fastatoms/forcefield/topology.hpp"

namespace fastatoms::ff {

/// How accurately, and how, to split the sum.
struct EwaldParameters {
    /// Target relative accuracy of the electrostatic energy. Everything below
    /// that is not given is derived from it.
    double accuracy{1.0e-6};

    /// Real-space cutoff in angstrom. Zero picks `min(half the shortest cell
    /// width, 10 A)`, which is the usual choice for a water box.
    double real_cutoff{0.0};

    /// Splitting parameter in 1/angstrom. Zero derives it from the cutoff and
    /// the accuracy: `alpha = sqrt(-ln(2 delta)) / r_c`.
    double alpha{0.0};

    /// Largest reciprocal vector to include, in 1/angstrom. Zero derives it as
    /// `2 alpha sqrt(-ln delta)`.
    double kcutoff{0.0};
};

/// The four pieces of an Ewald sum, kept apart because each one is separately
/// checkable and because their relative sizes say whether `alpha` is sensible.
struct EwaldEnergy {
    double real{0.0};        ///< screened point charges, within the cutoff
    double reciprocal{0.0};  ///< the smooth compensating clouds
    double self{0.0};        ///< each cloud's interaction with its own charge
    double excluded{0.0};    ///< undoing the intramolecular pairs in k-space

    [[nodiscard]] double total() const noexcept {
        return real + reciprocal + self + excluded;
    }
    [[nodiscard]] std::string to_string() const;
};

/// Energy, forces and stress from one evaluation.
struct EwaldResult {
    EwaldEnergy energy{};
    std::vector<double> forces{};                            ///< 3N, eV/angstrom
    std::array<double, 6> stress{{0, 0, 0, 0, 0, 0}};        ///< Voigt, eV/angstrom^3
};

/// The Ewald solver.
///
/// `prepare` fixes the splitting for a given cell and must be called again if
/// the cell changes shape; `compute` then evaluates the sum. The reciprocal
/// vectors and the per-atom phase factors are cached between calls, which is
/// what makes this usable inside a molecular dynamics loop.
class Ewald {
public:
    explicit Ewald(EwaldParameters parameters = {});

    /// Choose `alpha`, the real-space cutoff and the reciprocal vectors for
    /// this cell. Cheap to call again; it rebuilds only when the cell moved.
    /// \throws ForcefieldError for a non-periodic or singular cell.
    void prepare(const Atoms& atoms);

    /// Evaluate the sum. `pairs` must be a full neighbour list built with at
    /// least `real_cutoff`; sharing it with the Lennard-Jones loop is the
    /// point of passing it in rather than building one here.
    [[nodiscard]] EwaldResult compute(const Atoms& atoms, const Topology& topology,
                                      const NeighborList& pairs, bool need_forces = true,
                                      bool need_stress = false);

    /// \name What the splitting came out as
    /// @{
    [[nodiscard]] double alpha() const noexcept { return alpha_; }
    [[nodiscard]] double real_cutoff() const noexcept { return real_cutoff_; }
    [[nodiscard]] double kcutoff() const noexcept { return kcutoff_; }
    [[nodiscard]] std::size_t kvector_count() const noexcept { return kvectors_.size(); }
    [[nodiscard]] std::array<int, 3> kmax() const noexcept { return kmax_; }
    [[nodiscard]] const EwaldParameters& parameters() const noexcept { return parameters_; }
    [[nodiscard]] bool prepared() const noexcept { return prepared_; }
    /// @}

    /// An estimate of the error left in the real-space and reciprocal sums, in
    /// eV, from the standard Kolafa-Perram expressions.
    [[nodiscard]] double estimated_error(const Topology& topology, double volume) const;

    [[nodiscard]] std::string to_string() const;

private:
    void build_kvectors(const Mat3& cell);

    EwaldParameters parameters_{};
    double alpha_{0.0};
    double real_cutoff_{0.0};
    double kcutoff_{0.0};
    std::array<int, 3> kmax_{{0, 0, 0}};

    /// One entry per reciprocal vector kept: the vector itself and the
    /// prefactor `exp(-k^2/4 alpha^2) / k^2`.
    std::vector<Vec3> kvectors_{};
    std::vector<double> kfactors_{};
    /// Integer indices of each kept vector, for the recursive phase factors.
    std::vector<std::array<int, 3>> kindices_{};

    Mat3 prepared_cell_{};
    bool prepared_{false};

    /// Scratch, kept between calls so a dynamics run does not reallocate.
    std::vector<double> cos_kr_{};
    std::vector<double> sin_kr_{};
    std::vector<double> phase_real_{};
    std::vector<double> phase_imag_{};
};

}  // namespace fastatoms::ff
