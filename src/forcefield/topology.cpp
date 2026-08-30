// SPDX-License-Identifier: MIT
#include "fastatoms/forcefield/topology.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>

#include "fastatoms/core/elements.hpp"
#include "fastatoms/core/neighbor_list.hpp"

namespace fastatoms::ff {

Topology::Topology(std::size_t natoms)
    : charges_(natoms, 0.0), epsilons_(natoms, 0.0), sigmas_(natoms, 0.0),
      exclusions_(natoms) {}

void Topology::set_charge(std::size_t atom, double charge) {
    if (atom >= charges_.size()) {
        throw ForcefieldError("topology: atom index " + std::to_string(atom) +
                              " is out of range for " + std::to_string(charges_.size()) +
                              " atoms");
    }
    charges_[atom] = charge;
}

void Topology::set_lennard_jones(std::size_t atom, const LennardJonesSite& site) {
    if (atom >= epsilons_.size()) {
        throw ForcefieldError("topology: atom index " + std::to_string(atom) +
                              " is out of range for " + std::to_string(epsilons_.size()) +
                              " atoms");
    }
    if (site.epsilon < 0.0 || site.sigma < 0.0) {
        throw ForcefieldError("topology: epsilon and sigma must not be negative");
    }
    epsilons_[atom] = site.epsilon;
    sigmas_[atom] = site.sigma;
}

void Topology::add_molecule(Molecule molecule) {
    for (const std::size_t atom : molecule.atoms) {
        if (atom >= charges_.size()) {
            throw ForcefieldError("topology: molecule refers to atom " + std::to_string(atom) +
                                  ", which does not exist");
        }
    }
    molecules_.push_back(std::move(molecule));
}

void Topology::add_constraint(const DistanceConstraint& constraint) {
    if (constraint.i >= charges_.size() || constraint.j >= charges_.size()) {
        throw ForcefieldError("topology: constraint refers to an atom that does not exist");
    }
    if (constraint.i == constraint.j) {
        throw ForcefieldError("topology: a constraint needs two different atoms");
    }
    if (!(constraint.length > 0.0)) {
        throw ForcefieldError("topology: a constrained distance must be positive");
    }
    constraints_.push_back(constraint);
}

void Topology::exclude(std::size_t i, std::size_t j) {
    if (i == j || i >= exclusions_.size() || j >= exclusions_.size()) {
        return;
    }
    const auto insert = [this](std::size_t a, std::size_t b) {
        std::vector<std::int32_t>& list = exclusions_[a];
        const auto value = static_cast<std::int32_t>(b);
        const auto where = std::lower_bound(list.begin(), list.end(), value);
        if (where == list.end() || *where != value) {
            list.insert(where, value);
        }
    };
    insert(i, j);
    insert(j, i);
}

void Topology::exclude_within_molecules() {
    for (const Molecule& molecule : molecules_) {
        for (std::size_t a = 0; a < molecule.atoms.size(); ++a) {
            for (std::size_t b = a + 1; b < molecule.atoms.size(); ++b) {
                exclude(molecule.atoms[a], molecule.atoms[b]);
            }
        }
    }
}

bool Topology::excluded(std::size_t i, std::size_t j) const noexcept {
    if (i >= exclusions_.size()) {
        return false;
    }
    const std::vector<std::int32_t>& list = exclusions_[i];
    return std::binary_search(list.begin(), list.end(), static_cast<std::int32_t>(j));
}

std::vector<std::pair<std::size_t, std::size_t>> Topology::excluded_pairs() const {
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    pairs.reserve(exclusion_count() / 2);
    for (std::size_t i = 0; i < exclusions_.size(); ++i) {
        for (const std::int32_t partner : exclusions_[i]) {
            const auto j = static_cast<std::size_t>(partner);
            if (i < j) {
                pairs.emplace_back(i, j);
            }
        }
    }
    return pairs;
}

std::size_t Topology::exclusion_count() const noexcept {
    std::size_t total = 0;
    for (const std::vector<std::int32_t>& list : exclusions_) {
        total += list.size();
    }
    return total;
}

double Topology::net_charge() const noexcept {
    return std::accumulate(charges_.begin(), charges_.end(), 0.0);
}

double Topology::largest_sigma() const noexcept {
    double largest = 0.0;
    for (std::size_t i = 0; i < sigmas_.size(); ++i) {
        if (epsilons_[i] > 0.0) {
            largest = std::max(largest, sigmas_[i]);
        }
    }
    return largest;
}

void Topology::validate(const Atoms& atoms) const {
    if (atoms.size() != size()) {
        throw ForcefieldError("topology: built for " + std::to_string(size()) +
                              " atoms but the structure has " + std::to_string(atoms.size()) +
                              "; rebuild it after adding or removing atoms");
    }
    for (const DistanceConstraint& constraint : constraints_) {
        if (constraint.i >= size() || constraint.j >= size()) {
            throw ForcefieldError("topology: a constraint refers to a missing atom");
        }
    }
}

std::string Topology::to_string() const {
    std::ostringstream out;
    out << "Topology(" << size() << " atoms, " << molecules_.size() << " molecules, "
        << constraints_.size() << " constraints, " << exclusion_count() / 2
        << " excluded pairs, net charge " << net_charge() << ")";
    return out.str();
}

// ---------------------------------------------------------------------------
// Molecule detection
// ---------------------------------------------------------------------------
std::vector<Molecule> detect_molecules(const Atoms& atoms, double scale) {
    if (atoms.empty()) {
        return {};
    }
    if (!(scale > 0.0)) {
        throw ForcefieldError("detect_molecules: the bond scale must be positive");
    }

    // Per-atom radii turn "closer than the sum of the covalent radii" into a
    // single neighbour-list query, which is O(N) rather than O(N^2).
    std::vector<double> radii(atoms.size());
    double largest = 0.0;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        radii[i] = scale * covalent_radius(atoms.number(i));
        largest = std::max(largest, radii[i]);
    }
    if (!(largest > 0.0)) {
        throw ForcefieldError("detect_molecules: no atom has a covalent radius to bond with");
    }

    NeighborListOptions options;
    options.per_atom_cutoffs = radii;
    options.full = false;
    options.store_vectors = false;
    options.store_distances = false;
    const NeighborList bonds = build_neighbor_list(atoms, options);

    // Union-find over the bond graph; the connected components are molecules.
    std::vector<std::size_t> parent(atoms.size());
    std::iota(parent.begin(), parent.end(), std::size_t{0});
    const std::function<std::size_t(std::size_t)> find = [&](std::size_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    for (std::size_t pair = 0; pair < bonds.size(); ++pair) {
        const std::size_t a = find(static_cast<std::size_t>(bonds.i[pair]));
        const std::size_t b = find(static_cast<std::size_t>(bonds.j[pair]));
        if (a != b) {
            parent[a] = b;
        }
    }

    // Collect in order of first appearance, so molecule 0 contains atom 0.
    std::vector<std::size_t> label(atoms.size(), static_cast<std::size_t>(-1));
    std::vector<Molecule> molecules;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        const std::size_t root = find(i);
        if (label[root] == static_cast<std::size_t>(-1)) {
            label[root] = molecules.size();
            molecules.push_back(Molecule{});
        }
        molecules[label[root]].atoms.push_back(i);
    }
    return molecules;
}

}  // namespace fastatoms::ff
