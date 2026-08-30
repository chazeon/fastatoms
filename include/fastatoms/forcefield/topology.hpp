// SPDX-License-Identifier: MIT
#pragma once

/// \file topology.hpp
/// \brief Who is bonded to whom, what charge each site carries, and which
///        pairs must not see each other.
///
/// A classical force field needs more than positions and species: it needs to
/// know that these three atoms are one molecule, that this O-H pair is held
/// rigid, and that the two hydrogens of the same molecule must not feel each
/// other's charge. That bookkeeping is the topology, and it is kept separate
/// from both the parameters and the energy expression so that a different
/// water model - or a solute dropped into the box - changes only one of the
/// three.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/core/constraint.hpp"

namespace fastatoms::ff {

/// Raised when a topology cannot be built or a model is misconfigured.
class ForcefieldError : public std::runtime_error {
public:
    explicit ForcefieldError(const std::string& message) : std::runtime_error(message) {}
};

/// Lennard-Jones parameters of one site, in eV and angstrom.
struct LennardJonesSite {
    double epsilon{0.0};
    double sigma{0.0};

    [[nodiscard]] bool interacts() const noexcept { return epsilon > 0.0 && sigma > 0.0; }
};

/// A group of atoms treated as one molecule.
struct Molecule {
    std::vector<std::size_t> atoms{};
    std::string name{};
};

/// Everything the energy expression needs beyond the structure itself.
///
/// Charges and Lennard-Jones parameters are stored per atom rather than per
/// type: a flat array is what the inner loops want, and a type table would
/// only be dereferenced back into one.
class Topology {
public:
    Topology() = default;

    /// Start an empty topology sized for `natoms`.
    explicit Topology(std::size_t natoms);

    [[nodiscard]] std::size_t size() const noexcept { return charges_.size(); }
    [[nodiscard]] bool empty() const noexcept { return charges_.empty(); }

    /// \name Per-atom parameters
    /// @{
    [[nodiscard]] const std::vector<double>& charges() const noexcept { return charges_; }
    [[nodiscard]] const std::vector<double>& epsilons() const noexcept { return epsilons_; }
    [[nodiscard]] const std::vector<double>& sigmas() const noexcept { return sigmas_; }

    void set_charge(std::size_t atom, double charge);
    void set_lennard_jones(std::size_t atom, const LennardJonesSite& site);
    /// @}

    /// \name Molecules and constraints
    /// @{
    [[nodiscard]] const std::vector<Molecule>& molecules() const noexcept { return molecules_; }
    [[nodiscard]] const std::vector<DistanceConstraint>& constraints() const noexcept {
        return constraints_;
    }

    void add_molecule(Molecule molecule);
    void add_constraint(const DistanceConstraint& constraint);
    /// @}

    /// \name Exclusions
    ///
    /// Pairs that must not interact through the non-bonded terms, because the
    /// bonded terms - or, for a rigid molecule, the constraints - already
    /// account for them. Stored per atom as a sorted list of partners, which
    /// is what a pair loop wants: the test is a short linear scan over three
    /// or four entries, not a hash lookup.
    /// @{
    void exclude(std::size_t i, std::size_t j);

    /// Exclude every pair inside every molecule. This is what a rigid model
    /// wants: the geometry is fixed, so the intramolecular non-bonded energy
    /// is a constant that would only add noise to the forces.
    void exclude_within_molecules();

    [[nodiscard]] bool excluded(std::size_t i, std::size_t j) const noexcept;

    /// Every excluded pair once, with `i < j`. This is the list the Ewald sum
    /// has to undo in reciprocal space.
    [[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> excluded_pairs() const;

    [[nodiscard]] std::size_t exclusion_count() const noexcept;
    /// @}

    /// Total charge, which must be zero (or nearly so) for a periodic system.
    [[nodiscard]] double net_charge() const noexcept;

    /// Largest sigma present, for choosing a sensible Lennard-Jones cutoff.
    [[nodiscard]] double largest_sigma() const noexcept;

    /// Check the topology against a structure it is meant to describe.
    /// \throws ForcefieldError on a size mismatch or an out-of-range index.
    void validate(const Atoms& atoms) const;

    [[nodiscard]] std::string to_string() const;

private:
    std::vector<double> charges_{};
    std::vector<double> epsilons_{};
    std::vector<double> sigmas_{};
    std::vector<Molecule> molecules_{};
    std::vector<DistanceConstraint> constraints_{};
    std::vector<std::vector<std::int32_t>> exclusions_{};
};

/// Group atoms into molecules by distance-based bonding.
///
/// Two atoms are bonded when they are closer than `scale` times the sum of
/// their covalent radii; the connected components of that graph are the
/// molecules. Periodic images count, so a molecule straddling a cell boundary
/// is still one molecule.
[[nodiscard]] std::vector<Molecule> detect_molecules(const Atoms& atoms, double scale = 1.2);

}  // namespace fastatoms::ff
