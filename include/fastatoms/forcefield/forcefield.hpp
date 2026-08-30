// SPDX-License-Identifier: MIT
#pragma once

/// \file forcefield.hpp
/// \brief A classical empirical force field: Lennard-Jones plus Ewald
///        electrostatics, on a topology.

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "fastatoms/calculators/calculator.hpp"
#include "fastatoms/forcefield/ewald.hpp"
#include "fastatoms/forcefield/topology.hpp"
#include "fastatoms/forcefield/water.hpp"

namespace fastatoms::ff {

/// The pieces of a force-field energy, kept apart for reporting and testing.
struct EnergyBreakdown {
    double lennard_jones{0.0};
    double lennard_jones_tail{0.0};  ///< isotropic correction beyond the cutoff
    EwaldEnergy coulomb{};
    double self{0.0};                ///< per-molecule constants, e.g. SPC/E polarisation

    [[nodiscard]] double total() const noexcept {
        return lennard_jones + lennard_jones_tail + coulomb.total() + self;
    }
    [[nodiscard]] std::string to_string() const;
};

/// A classical force field for rigid molecular liquids.
///
/// The energy is
///
///     E = sum_ij 4 eps_ij [(sig_ij/r)^12 - (sig_ij/r)^6]  +  E_Ewald  +  E_self
///
/// with Lorentz-Berthelot mixing, intramolecular pairs excluded, and the
/// electrostatics summed by Ewald rather than truncated. Truncating a `1/r`
/// sum is not an approximation to it - the result depends on the order of
/// summation - which is why a plain cutoff is not offered for the Coulomb
/// part. A structure that is not periodic has no such problem and gets the
/// direct sum instead.
///
/// The usual way in is a water model by name:
///
///     atoms.calc = fa.Forcefield(model="spce")
///
/// which detects the molecules, assigns the parameters, excludes the
/// intramolecular pairs and sets up the three distance constraints that keep
/// each molecule rigid. `constraints()` hands those to the dynamics.
///
/// Parameters:
///   - `model`         - "spc", "spce" or "tip3p"; omit for a bare topology
///   - `cutoff`        - real-space cutoff in angstrom (default: the smaller of
///                       half the shortest cell width and 10 A)
///   - `accuracy`      - target relative accuracy of the Ewald sum (1e-6)
///   - `alpha`         - splitting parameter, 1/angstrom; 0 derives it
///   - `kcutoff`       - reciprocal-space cutoff, 1/angstrom; 0 derives it
///   - `shift`         - shift the Lennard-Jones potential to zero at the
///                       cutoff, so no pair crosses it with a step in the
///                       energy (true)
///   - `tail`          - add the isotropic Lennard-Jones tail correction (true)
///   - `self_energy`   - include the model's per-molecule constant (true)
class ForcefieldCalculator : public Calculator {
public:
    explicit ForcefieldCalculator(Value parameters = {});

    [[nodiscard]] std::string name() const override { return "forcefield"; }
    [[nodiscard]] std::vector<Property> implemented_properties() const override;
    [[nodiscard]] std::string to_string() const override;

    /// Build the topology for this structure now, rather than on first use.
    /// Called automatically; call it directly to inspect what was detected.
    void prepare(const Atoms& atoms);

    /// The topology in use. \throws ForcefieldError if none has been built.
    [[nodiscard]] const Topology& topology() const;

    /// Use a topology built elsewhere - a mixed system, a solute, a model this
    /// class does not know. Its size must match the structure.
    void set_topology(Topology topology);

    /// The water model in use, or `nullptr` when the topology came from
    /// somewhere else.
    [[nodiscard]] const WaterModel* model() const noexcept { return model_; }

    /// The Ewald solver, for its `alpha`, k-vector count and error estimate.
    [[nodiscard]] const Ewald& ewald() const noexcept;

    /// The most recent energy, term by term.
    [[nodiscard]] const EnergyBreakdown& breakdown() const noexcept { return breakdown_; }

    /// The real-space cutoff actually in use, in angstrom.
    [[nodiscard]] double cutoff() const noexcept { return cutoff_; }

    /// The rigid-geometry constraints this model implies, ready to hand to the
    /// dynamics. Empty when the topology declares none.
    [[nodiscard]] const std::vector<DistanceConstraint>& constraints() const;

    /// The same list, through the base-class hook the dynamics uses. Null
    /// until a structure has been seen, because the constraints follow from
    /// the molecules that were detected in it.
    [[nodiscard]] const std::vector<DistanceConstraint>* rigid_constraints() const override;

protected:
    void compute(const Atoms& atoms, const std::vector<Property>& properties) override;

private:
    /// Lennard-Jones over the neighbour list, skipping excluded pairs.
    void lennard_jones(const Atoms& atoms, const NeighborList& pairs, bool need_forces,
                       bool need_stress, std::vector<double>& forces, double* virial);

    /// The plain `1/r` sum, for a structure with no periodic images. Ewald
    /// exists because a lattice sum is conditionally convergent; a cluster has
    /// no such problem, so it gets the sum itself.
    [[nodiscard]] EwaldEnergy direct_coulomb(const Atoms& atoms, const NeighborList& pairs,
                                             std::vector<double>& forces,
                                             bool need_forces) const;

    Topology topology_{};
    const WaterModel* model_{nullptr};
    Ewald ewald_{};
    EnergyBreakdown breakdown_{};
    double cutoff_{0.0};
    bool prepared_{false};
    bool periodic_{true};
    std::size_t prepared_size_{0};
};

}  // namespace fastatoms::ff
