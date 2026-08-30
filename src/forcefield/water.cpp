// SPDX-License-Identifier: MIT
#include "fastatoms/forcefield/water.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <random>
#include <sstream>

#include "fastatoms/core/elements.hpp"
#include "fastatoms/core/units.hpp"

namespace fastatoms::ff {
namespace {

constexpr double kDegree = 3.14159265358979323846 / 180.0;

/// Debye per e*angstrom, for reporting a dipole in the usual unit.
constexpr double kDebyePerEAngstrom = 4.803204712570263;

std::string canonical(std::string_view name) {
    std::string key;
    for (const char c : name) {
        if (c == '-' || c == '_' || c == ' ' || c == '/') {
            continue;
        }
        key += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return key;
}

/// The published parameters, in the units they were published in.
const std::vector<WaterModel>& table() {
    static const std::vector<WaterModel> models = [] {
        const double kcal = units::kcal_per_mol;
        std::vector<WaterModel> entries;

        // Berendsen, Postma, van Gunsteren and Hermans, "Interaction models for
        // water in relation to protein hydration" (1981).
        entries.push_back(WaterModel{"spc", -0.82, 0.41, 0.1554 * kcal, 3.166, 1.0, 109.47, 0.0,
                                     "Berendsen et al., Intermolecular Forces (1981) 331"});

        // Berendsen, Grigera and Straatsma, "The missing term in effective pair
        // potentials" (1987). The missing term is the polarisation self-energy.
        entries.push_back(WaterModel{"spce", -0.8476, 0.4238, 0.1553 * kcal, 3.166, 1.0, 109.47,
                                     5.22 * kcal,
                                     "Berendsen, Grigera and Straatsma, "
                                     "J. Phys. Chem. 91 (1987) 6269"});

        // Jorgensen et al., "Comparison of simple potential functions for
        // simulating liquid water" (1983); the rigid variant.
        entries.push_back(WaterModel{"tip3p", -0.834, 0.417, 0.1521 * kcal, 3.15061, 0.9572,
                                     104.52, 0.0,
                                     "Jorgensen et al., J. Chem. Phys. 79 (1983) 926"});
        return entries;
    }();
    return models;
}

}  // namespace

double WaterModel::hydrogen_distance() const noexcept {
    return 2.0 * bond_length * std::sin(0.5 * angle * kDegree);
}

double WaterModel::dipole_moment() const noexcept {
    // Oxygen at the origin, hydrogens symmetric about the bisector: the dipole
    // is along the bisector and the components across it cancel.
    const double half = 0.5 * angle * kDegree;
    return std::abs(2.0 * q_hydrogen * bond_length * std::cos(half)) * kDebyePerEAngstrom;
}

std::string WaterModel::to_string() const {
    std::ostringstream out;
    out << "WaterModel('" << name << "', q_O=" << q_oxygen << ", q_H=" << q_hydrogen
        << ", epsilon=" << epsilon_oxygen << " eV, sigma=" << sigma_oxygen
        << " A, r_OH=" << bond_length << " A, angle=" << angle << " deg)";
    return out.str();
}

const WaterModel& water_model(std::string_view name) {
    const std::string key = canonical(name);
    for (const WaterModel& model : table()) {
        if (canonical(model.name) == key) {
            return model;
        }
    }
    std::ostringstream message;
    message << "unknown water model '" << name << "'; known models are";
    for (const WaterModel& model : table()) {
        message << " " << model.name;
    }
    throw ForcefieldError(message.str());
}

std::vector<std::string> water_model_names() {
    std::vector<std::string> names;
    for (const WaterModel& model : table()) {
        names.push_back(model.name);
    }
    std::sort(names.begin(), names.end());
    return names;
}

Atoms water_molecule(const WaterModel& model, double box) {
    Atoms atoms;
    // Oxygen at the origin, molecule in the xy plane, bisector along +x.
    const double half = 0.5 * model.angle * kDegree;
    const double x = model.bond_length * std::cos(half);
    const double y = model.bond_length * std::sin(half);
    atoms.append(8, Vec3{0.0, 0.0, 0.0});
    atoms.append(1, Vec3{x, y, 0.0});
    atoms.append(1, Vec3{x, -y, 0.0});
    if (box > 0.0) {
        atoms.set_cell(Mat3::diagonal(box, box, box));
        atoms.set_pbc({true, true, true});
        // Put the molecule in the middle, away from the boundary.
        const Vec3 centre{0.5 * box, 0.5 * box, 0.5 * box};
        for (std::size_t i = 0; i < atoms.size(); ++i) {
            atoms.set_position(i, atoms.position(i) + centre);
        }
    }
    return atoms;
}

Atoms water_box(const WaterModel& model, int molecules, double density, std::uint64_t seed) {
    if (molecules <= 0) {
        throw ForcefieldError("water_box: the number of molecules must be positive");
    }
    if (!(density > 0.0)) {
        throw ForcefieldError("water_box: the density must be positive");
    }

    // Round up to a full fcc lattice: 4, 32, 108, 256, ... Filling it only
    // partly would leave the molecules that are there closer together than the
    // density calls for, close enough to be detected as one molecule.
    int cells = 1;
    while (4 * cells * cells * cells < molecules) {
        ++cells;
    }
    const int count = 4 * cells * cells * cells;

    // g/cm^3 to amu/angstrom^3 is 1 / 1.66053906660.
    const double mass_per_molecule = atomic_mass(8) + 2.0 * atomic_mass(1);
    const double volume = static_cast<double>(count) * mass_per_molecule /
                          (density / 1.66053906660);
    const double side = std::cbrt(volume);
    const double spacing = side / static_cast<double>(cells);

    const double half = 0.5 * model.angle * kDegree;
    const double bond_x = model.bond_length * std::cos(half);
    const double bond_y = model.bond_length * std::sin(half);

    std::mt19937_64 rng(seed == 0 ? 20260830ull : seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    Atoms atoms;
    atoms.reserve(3 * static_cast<std::size_t>(count));
    const double basis[4][3] = {{0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5},
                                {0.5, 0.5, 0.0}};
    for (int i = 0; i < cells; ++i) {
        for (int j = 0; j < cells; ++j) {
            for (int k = 0; k < cells; ++k) {
                for (const auto& site : basis) {
                    const Vec3 centre{(static_cast<double>(i) + site[0]) * spacing,
                                      (static_cast<double>(j) + site[1]) * spacing,
                                      (static_cast<double>(k) + site[2]) * spacing};

                    // A uniformly random orientation, from the Shoemake
                    // quaternion construction, so no direction is favoured.
                    const double u1 = uniform(rng);
                    const double u2 = uniform(rng);
                    const double u3 = uniform(rng);
                    const double s1 = std::sqrt(1.0 - u1);
                    const double s2 = std::sqrt(u1);
                    const double two_pi = 2.0 * 3.14159265358979323846;
                    const double qx = s1 * std::sin(two_pi * u2);
                    const double qy = s1 * std::cos(two_pi * u2);
                    const double qz = s2 * std::sin(two_pi * u3);
                    const double qw = s2 * std::cos(two_pi * u3);

                    const Mat3 rotation(
                        1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),
                        2 * (qx * qz + qy * qw), 2 * (qx * qy + qz * qw),
                        1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
                        2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw),
                        1 - 2 * (qx * qx + qy * qy));

                    atoms.append(8, centre);
                    atoms.append(1, centre + rotation * Vec3{bond_x, bond_y, 0.0});
                    atoms.append(1, centre + rotation * Vec3{bond_x, -bond_y, 0.0});
                }
            }
        }
    }

    atoms.set_cell(Mat3::diagonal(side, side, side));
    atoms.set_pbc({true, true, true});
    return atoms;
}

Topology water_topology(const Atoms& atoms, const WaterModel& model) {
    if (atoms.empty()) {
        throw ForcefieldError("water_topology: the structure has no atoms");
    }

    Topology topology(atoms.size());
    const std::vector<Molecule> molecules = detect_molecules(atoms);
    const double hh = model.hydrogen_distance();

    std::size_t index = 0;
    for (const Molecule& found : molecules) {
        ++index;
        std::size_t oxygen = 0;
        std::vector<std::size_t> hydrogens;
        for (const std::size_t atom : found.atoms) {
            const int z = atoms.number(atom);
            if (z == 8) {
                oxygen = atom;
            } else if (z == 1) {
                hydrogens.push_back(atom);
            }
        }
        if (found.atoms.size() != 3 || hydrogens.size() != 2) {
            std::ostringstream message;
            message << "water_topology: molecule " << index << " has " << found.atoms.size()
                    << " atoms (";
            for (const std::size_t atom : found.atoms) {
                message << symbol_from_number(atoms.number(atom));
            }
            message << "), but a rigid water model needs one oxygen and two hydrogens. "
                       "Check the geometry, or build the topology yourself for a mixed system.";
            throw ForcefieldError(message.str());
        }

        topology.set_charge(oxygen, model.q_oxygen);
        topology.set_lennard_jones(oxygen,
                                   LennardJonesSite{model.epsilon_oxygen, model.sigma_oxygen});
        for (const std::size_t hydrogen : hydrogens) {
            topology.set_charge(hydrogen, model.q_hydrogen);
            topology.set_lennard_jones(hydrogen, LennardJonesSite{0.0, 0.0});
        }

        Molecule molecule;
        molecule.atoms = {oxygen, hydrogens[0], hydrogens[1]};
        molecule.name = "water";
        topology.add_molecule(molecule);

        // Two bonds and the H-H distance: three constraints remove the three
        // internal degrees of freedom and leave the molecule rigid.
        topology.add_constraint(DistanceConstraint{oxygen, hydrogens[0], model.bond_length});
        topology.add_constraint(DistanceConstraint{oxygen, hydrogens[1], model.bond_length});
        topology.add_constraint(DistanceConstraint{hydrogens[0], hydrogens[1], hh});
    }

    topology.exclude_within_molecules();
    return topology;
}

}  // namespace fastatoms::ff
