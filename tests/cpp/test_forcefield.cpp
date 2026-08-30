// SPDX-License-Identifier: MIT
//
// The classical force field: Ewald electrostatics, the rigid water models, and
// the SHAKE/RATTLE constraints that keep them rigid.
//
// The Ewald sum is checked three ways, which between them leave nowhere for an
// error to hide: against the Madelung constant of rock salt, which is known
// exactly; against itself with different splitting parameters, which must
// agree because the split is an identity; and against finite differences of
// its own energy, which pins down the forces and the stress.

#include <cmath>
#include <string>
#include <vector>

#include "fastatoms/core/units.hpp"
#include "fastatoms/forcefield/ewald.hpp"
#include "fastatoms/forcefield/forcefield.hpp"
#include "fastatoms/forcefield/water.hpp"
#include "fastatoms/md/constraints.hpp"
#include "fastatoms/md/dynamics.hpp"
#include "fastatoms/md/velocity.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::ff;

namespace {

/// A displacement along one axis, written out rather than indexing `&vec.x`.
Vec3 axis_shift(int axis, double amount) {
    return axis == 0   ? Vec3{amount, 0.0, 0.0}
           : axis == 1 ? Vec3{0.0, amount, 0.0}
                       : Vec3{0.0, 0.0, amount};
}

/// Rock salt: the conventional cubic cell, four formula units.
Atoms rocksalt(double a = 5.64) {
    Atoms atoms;
    const double cation[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
    const double anion[4][3] = {{0.5, 0, 0}, {0, 0.5, 0}, {0, 0, 0.5}, {0.5, 0.5, 0.5}};
    for (const auto& site : cation) {
        atoms.append(11, Vec3{site[0] * a, site[1] * a, site[2] * a});
    }
    for (const auto& site : anion) {
        atoms.append(17, Vec3{site[0] * a, site[1] * a, site[2] * a});
    }
    atoms.set_cell(Mat3::diagonal(a, a, a));
    atoms.set_pbc({true, true, true});
    return atoms;
}

/// Unit charges on the sodium sites, minus one on the chlorines.
Topology rocksalt_topology(const Atoms& atoms) {
    Topology topology(atoms.size());
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        topology.set_charge(i, atoms.number(i) == 11 ? 1.0 : -1.0);
    }
    return topology;
}

NeighborList pairs_for(const Atoms& atoms, double cutoff) {
    NeighborListOptions options;
    options.cutoff = cutoff;
    options.full = true;
    options.store_vectors = false;
    options.store_distances = false;
    return build_neighbor_list(atoms, options);
}

double ewald_energy(const Atoms& atoms, const Topology& topology, double alpha = 0.0,
                    double cutoff = 0.0) {
    EwaldParameters parameters;
    parameters.accuracy = 1.0e-10;
    parameters.alpha = alpha;
    parameters.real_cutoff = cutoff;
    Ewald solver(parameters);
    solver.prepare(atoms);
    return solver.compute(atoms, topology, pairs_for(atoms, solver.real_cutoff()), false, false)
        .energy.total();
}

}  // namespace

// ---------------------------------------------------------------------------
// Ewald against a number nobody can argue with
// ---------------------------------------------------------------------------
FA_TEST(nacl_madelung_constant) {
    // The Madelung constant of rock salt is 1.747564594633..., defined by
    //     E per ion pair = -M * k_e * q^2 / r_nn.
    // The cell below holds four pairs, so the whole sum is fixed with no free
    // parameters at all. Nothing but a correct Ewald implementation lands here.
    const double a = 5.64;
    const Atoms atoms = rocksalt(a);
    const Topology topology = rocksalt_topology(atoms);

    const double nearest = 0.5 * a;
    const double expected = -4.0 * 1.7475645946331821 * units::coulomb / nearest;
    const double energy = ewald_energy(atoms, topology);

    std::printf("    Ewald %.9f eV, Madelung %.9f eV, difference %.2e\n", energy, expected,
                std::abs(energy - expected));
    FA_CHECK_CLOSE(energy, expected, 1e-9);

    // And the constant read back out of it, which is the number in the tables.
    const double madelung = -energy * nearest / (4.0 * units::coulomb);
    FA_CHECK_CLOSE(madelung, 1.7475645946331821, 1e-9);
}

FA_TEST(ewald_is_independent_of_the_splitting) {
    // alpha only decides how the identity 1/r = erfc/r + erf/r is divided
    // between the two sums. Move it, and as long as both sums are carried far
    // enough the total cannot change. That catches sign and factor errors the
    // Madelung test would not, because it varies the balance between the
    // terms rather than only their sum.
    //
    // "Far enough" is the point: the real-space cutoff has to follow alpha,
    // since it is erfc(alpha r_c) that has to be small. Holding the cutoff
    // fixed while alpha shrinks does not test the identity, it tests a
    // truncation.
    const Atoms atoms = rocksalt();
    const Topology topology = rocksalt_topology(atoms);

    const double a = 5.64;
    const double expected = -4.0 * 1.7475645946331821 * units::coulomb / (0.5 * a);
    const double tolerance = std::sqrt(-std::log(2.0e-12));  // erfc(alpha r_c) < 1e-12

    for (const double alpha : {0.8, 1.0, 1.2, 1.5, 2.0}) {
        const double cutoff = tolerance / alpha;
        const double energy = ewald_energy(atoms, topology, alpha, cutoff);
        FA_CHECK_CLOSE(energy, expected, 1e-8);
    }
}

FA_TEST(ewald_forces_match_finite_differences) {
    Atoms atoms = rocksalt();
    // Displace two ions so nothing sits on a symmetry point.
    atoms.set_position(0, atoms.position(0) + Vec3{0.21, -0.13, 0.07});
    atoms.set_position(5, atoms.position(5) + Vec3{-0.11, 0.09, 0.15});
    const Topology topology = rocksalt_topology(atoms);

    EwaldParameters parameters;
    parameters.accuracy = 1.0e-12;
    parameters.real_cutoff = 8.0;
    Ewald solver(parameters);
    solver.prepare(atoms);
    const EwaldResult result =
        solver.compute(atoms, topology, pairs_for(atoms, 8.0), true, false);

    const double step = 1.0e-6;
    for (const std::size_t atom : {std::size_t{0}, std::size_t{3}, std::size_t{5}}) {
        for (int axis = 0; axis < 3; ++axis) {
            Atoms plus = atoms;
            Atoms minus = atoms;
            plus.set_position(atom, plus.position(atom) + axis_shift(axis, step));
            minus.set_position(atom, minus.position(atom) - axis_shift(axis, step));

            const double e_plus = ewald_energy(plus, topology, 0.0, 8.0);
            const double e_minus = ewald_energy(minus, topology, 0.0, 8.0);
            const double numerical = -(e_plus - e_minus) / (2.0 * step);
            FA_CHECK_CLOSE(result.forces[3 * atom + static_cast<std::size_t>(axis)], numerical,
                           1e-6);
        }
    }
}

FA_TEST(ewald_stress_matches_finite_differences) {
    // The reciprocal-space stress has a term nobody guesses right first time -
    // the k_a k_b piece with its 1/k^2 + 1/(4 alpha^2) coefficient - so it is
    // checked by straining the cell and differentiating the energy.
    Atoms atoms = rocksalt();
    atoms.set_position(0, atoms.position(0) + Vec3{0.17, -0.09, 0.05});
    const Topology topology = rocksalt_topology(atoms);

    EwaldParameters parameters;
    parameters.accuracy = 1.0e-12;
    parameters.real_cutoff = 8.0;
    Ewald solver(parameters);
    solver.prepare(atoms);
    const EwaldResult result =
        solver.compute(atoms, topology, pairs_for(atoms, 8.0), false, true);

    const double volume = std::abs(atoms.cell().matrix().determinant());
    const double step = 1.0e-6;
    const int voigt[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};
    for (int component = 0; component < 6; ++component) {
        const int a = voigt[component][0];
        const int b = voigt[component][1];

        const auto strained = [&](double amount) {
            Mat3 deformation = Mat3::identity();
            deformation.m[static_cast<std::size_t>(3 * a + b)] += 0.5 * amount;
            deformation.m[static_cast<std::size_t>(3 * b + a)] += 0.5 * amount;

            Atoms copy = atoms;
            copy.set_cell(atoms.cell().matrix() * deformation);
            for (std::size_t i = 0; i < copy.size(); ++i) {
                copy.set_position(i, atoms.position(i) * deformation);
            }
            // A fresh solver, because the cell changed.
            return ewald_energy(copy, topology, 0.0, 8.0);
        };

        const double numerical = (strained(step) - strained(-step)) / (2.0 * step) / volume;
        FA_CHECK_CLOSE(result.stress[static_cast<std::size_t>(component)], numerical, 2e-6);
    }
}

FA_TEST(ewald_reports_its_pieces) {
    const Atoms atoms = rocksalt();
    const Topology topology = rocksalt_topology(atoms);
    EwaldParameters parameters;
    parameters.real_cutoff = 8.0;
    Ewald solver(parameters);
    solver.prepare(atoms);
    const EwaldResult result =
        solver.compute(atoms, topology, pairs_for(atoms, 8.0), true, true);

    // The self term is always negative and always the largest single piece;
    // the reciprocal term is a sum of squares and so never negative.
    FA_CHECK(result.energy.self < 0.0);
    FA_CHECK(result.energy.reciprocal >= 0.0);
    FA_CHECK(result.energy.excluded == 0.0);  // nothing is excluded here
    FA_CHECK(solver.kvector_count() > 0);
    FA_CHECK(solver.alpha() > 0.0);
    FA_CHECK(solver.estimated_error(topology, 5.64 * 5.64 * 5.64) >= 0.0);

    // Momentum conservation: the electrostatic forces must sum to zero.
    double sum[3] = {0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            sum[axis] += result.forces[3 * i + static_cast<std::size_t>(axis)];
        }
    }
    for (const double component : sum) {
        FA_CHECK_CLOSE(component, 0.0, 1e-10);
    }
}

FA_TEST(ewald_refuses_what_it_cannot_do) {
    Atoms molecule;
    molecule.append(8, Vec3{0.0, 0.0, 0.0});
    molecule.append(1, Vec3{1.0, 0.0, 0.0});
    Topology topology(2);
    Ewald solver;
    FA_CHECK_THROWS(solver.prepare(molecule), ForcefieldError);  // not periodic

    EwaldParameters bad;
    bad.accuracy = 0.0;
    FA_CHECK_THROWS(Ewald(bad), ForcefieldError);
}

// ---------------------------------------------------------------------------
// The water models
// ---------------------------------------------------------------------------
FA_TEST(water_models_carry_the_published_numbers) {
    const WaterModel& spc = water_model("spc");
    FA_CHECK(spc.name == "spc");
    FA_CHECK_CLOSE(spc.q_oxygen, -0.82, 1e-12);
    FA_CHECK_CLOSE(spc.q_hydrogen, 0.41, 1e-12);
    FA_CHECK_CLOSE(spc.bond_length, 1.0, 1e-12);
    FA_CHECK_CLOSE(spc.angle, 109.47, 1e-12);
    FA_CHECK_CLOSE(spc.self_energy, 0.0, 1e-12);

    const WaterModel& spce = water_model("spc/e");
    FA_CHECK(spce.name == "spce");
    FA_CHECK_CLOSE(spce.q_oxygen, -0.8476, 1e-12);
    FA_CHECK_CLOSE(spce.q_hydrogen, 0.4238, 1e-12);
    // The polarisation self-energy, 5.22 kcal/mol, is what distinguishes
    // SPC/E from SPC beyond the charges.
    FA_CHECK_CLOSE(spce.self_energy, 5.22 * units::kcal_per_mol, 1e-12);

    // Both models are electrically neutral per molecule.
    FA_CHECK_CLOSE(spc.q_oxygen + 2.0 * spc.q_hydrogen, 0.0, 1e-12);
    FA_CHECK_CLOSE(spce.q_oxygen + 2.0 * spce.q_hydrogen, 0.0, 1e-12);

    // The dipole moments these parameters imply are the ones in the papers:
    // 2.27 D for SPC and 2.35 D for SPC/E.
    FA_CHECK_CLOSE(spc.dipole_moment(), 2.27, 2e-3);
    FA_CHECK_CLOSE(spce.dipole_moment(), 2.35, 2e-3);
    FA_CHECK_CLOSE(water_model("tip3p").dipole_moment(), 2.35, 5e-3);

    // The H-H distance that makes the angle rigid.
    FA_CHECK_CLOSE(spce.hydrogen_distance(), 2.0 * std::sin(0.5 * 109.47 * M_PI / 180.0), 1e-12);

    FA_CHECK_THROWS(water_model("mercury"), ForcefieldError);
    FA_CHECK(water_model_names().size() >= 3);
}

FA_TEST(water_topology_finds_molecules_and_constraints) {
    const WaterModel& model = water_model("spce");
    const Atoms atoms = water_box(model, 32, 0.997, 42);
    FA_CHECK(atoms.size() == 96);

    const Topology topology = water_topology(atoms, model);
    FA_CHECK(topology.size() == 96);
    FA_CHECK(topology.molecules().size() == 32);
    FA_CHECK(topology.constraints().size() == 96);        // three per molecule
    FA_CHECK(topology.excluded_pairs().size() == 96);     // three pairs per molecule
    FA_CHECK_CLOSE(topology.net_charge(), 0.0, 1e-12);
    FA_CHECK_CLOSE(topology.largest_sigma(), model.sigma_oxygen, 1e-12);

    // Only the oxygens carry dispersion.
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        if (atoms.number(i) == 8) {
            FA_CHECK_CLOSE(topology.charges()[i], model.q_oxygen, 1e-12);
            FA_CHECK(topology.epsilons()[i] > 0.0);
        } else {
            FA_CHECK_CLOSE(topology.charges()[i], model.q_hydrogen, 1e-12);
            FA_CHECK_CLOSE(topology.epsilons()[i], 0.0, 1e-12);
        }
    }

    // The geometry the builder produced already satisfies the constraints.
    const md::Constraints constraints(topology.constraints());
    FA_CHECK(constraints.max_deviation(atoms) < 1e-12);
}

FA_TEST(a_structure_that_is_not_water_is_rejected) {
    Atoms atoms;
    atoms.append(8, Vec3{0.0, 0.0, 0.0});
    atoms.append(1, Vec3{1.0, 0.0, 0.0});
    atoms.set_cell(Mat3::diagonal(15.0, 15.0, 15.0));
    atoms.set_pbc({true, true, true});
    FA_CHECK_THROWS(water_topology(atoms, water_model("spc")), ForcefieldError);
}

// ---------------------------------------------------------------------------
// The calculator
// ---------------------------------------------------------------------------
FA_TEST(forcefield_forces_match_finite_differences) {
    const WaterModel& model = water_model("spce");
    Atoms atoms = water_box(model, 32, 0.997, 7);
    // Break the lattice symmetry so no force is zero by accident.
    atoms.set_position(0, atoms.position(0) + Vec3{0.13, -0.07, 0.05});
    atoms.set_position(4, atoms.position(4) + Vec3{-0.09, 0.11, -0.06});

    Value parameters(Value::Map{});
    parameters.insert("model", Value(std::string("spce")));
    parameters.insert("accuracy", Value(1.0e-12));
    parameters.insert("cutoff", Value(4.5));
    auto calculator = std::make_shared<ForcefieldCalculator>(parameters);
    atoms.set_calculator(calculator);

    const std::vector<double> forces = atoms.get_forces();
    const double step = 1.0e-6;
    for (const std::size_t atom : {std::size_t{0}, std::size_t{1}, std::size_t{4}}) {
        for (int axis = 0; axis < 3; ++axis) {
            Atoms plus = atoms;
            Atoms minus = atoms;
            plus.set_position(atom, plus.position(atom) + axis_shift(axis, step));
            minus.set_position(atom, minus.position(atom) - axis_shift(axis, step));

            auto probe = std::make_shared<ForcefieldCalculator>(parameters);
            plus.set_calculator(probe);
            const double e_plus = plus.get_potential_energy();
            auto other = std::make_shared<ForcefieldCalculator>(parameters);
            minus.set_calculator(other);
            const double e_minus = minus.get_potential_energy();

            const double numerical = -(e_plus - e_minus) / (2.0 * step);
            FA_CHECK_CLOSE(forces[3 * atom + static_cast<std::size_t>(axis)], numerical, 1e-5);
        }
    }
}

FA_TEST(forcefield_reports_its_terms) {
    const WaterModel& model = water_model("spce");
    Atoms atoms = water_box(model, 32, 0.997, 3);

    Value parameters(Value::Map{});
    parameters.insert("model", Value(std::string("spce")));
    auto calculator = std::make_shared<ForcefieldCalculator>(parameters);
    atoms.set_calculator(calculator);

    const double energy = atoms.get_potential_energy();
    const EnergyBreakdown& breakdown = calculator->breakdown();
    FA_CHECK_CLOSE(breakdown.total(), energy, 1e-12);
    FA_CHECK(calculator->model() != nullptr);
    FA_CHECK(calculator->model()->name == "spce");
    FA_CHECK(calculator->constraints().size() == 96);

    // SPC/E's polarisation correction is a positive constant per molecule.
    FA_CHECK_CLOSE(breakdown.self, 32.0 * model.self_energy, 1e-12);
    std::printf("    %s\n", breakdown.to_string().c_str());
    std::printf("    %s\n", calculator->ewald().to_string().c_str());

    // Turning it off changes the energy by exactly that amount.
    Value without = parameters;
    without.insert("self_energy", Value(false));
    Atoms same = water_box(model, 32, 0.997, 3);
    same.set_calculator(std::make_shared<ForcefieldCalculator>(without));
    FA_CHECK_CLOSE(same.get_potential_energy(), energy - 32.0 * model.self_energy, 1e-10);

    // The charges reach the structure, as they do from a DFT calculator.
    FA_CHECK(atoms.has_charges());
    FA_CHECK_CLOSE(atoms.charges()[0], model.q_oxygen, 1e-12);
}

FA_TEST(an_isolated_molecule_barely_interacts_with_its_images) {
    // Every pair inside a water molecule is excluded, so a single molecule
    // alone in a box has no electrostatic energy of its own: what is left is
    // only its interaction with its periodic images, which must fall off like
    // 1/L^3 for a neutral dipole. This is the test of the exclusion
    // correction - get its sign or its erf/erfc wrong and what is left over is
    // an enormous intramolecular Coulomb energy instead of a small residue.
    const WaterModel& model = water_model("spce");
    double previous = 0.0;
    for (const double box : {15.0, 30.0}) {
        Atoms atoms = water_molecule(model, box);
        Value parameters(Value::Map{});
        parameters.insert("model", Value(std::string("spce")));
        parameters.insert("accuracy", Value(1.0e-10));
        auto calculator = std::make_shared<ForcefieldCalculator>(parameters);
        atoms.set_calculator(calculator);
        (void)atoms.get_potential_energy();

        const double coulomb = std::abs(calculator->breakdown().coulomb.total());
        std::printf("    box %4.1f A: residual Coulomb %.3e eV\n", box, coulomb);
        // Compare against the intramolecular energy that a missing exclusion
        // would leave behind: of order q^2 / 1 A ~ 10 eV.
        FA_CHECK(coulomb < 1.0e-2);
        if (previous > 0.0) {
            // Doubling the box divides a dipole-image term by eight.
            FA_CHECK_CLOSE(previous / coulomb, 8.0, 0.15);
        }
        previous = coulomb;
    }
}

FA_TEST(a_dimer_matches_a_direct_pairwise_sum) {
    // Two molecules in a large box, against the sum every intermolecular pair
    // contributes with no periodicity at all. The two agree in the limit of a
    // large box, and the difference falls off as 1/L^3 - so this checks the
    // Lennard-Jones term, the charges, the exclusions and the Ewald sum
    // against an expression short enough to read.
    const WaterModel& model = water_model("spce");
    const Atoms single = water_molecule(model, 0.0);

    double previous = 0.0;
    for (const double box : {25.0, 40.0}) {
        Atoms dimer;
        const Vec3 centre{0.5 * box, 0.5 * box, 0.5 * box};
        const Vec3 offset{3.0, 2.2, 0.0};
        for (std::size_t i = 0; i < 3; ++i) {
            dimer.append(single.number(i), single.position(i) + centre);
        }
        for (std::size_t i = 0; i < 3; ++i) {
            dimer.append(single.number(i), single.position(i) + centre + offset);
        }
        dimer.set_cell(Mat3::diagonal(box, box, box));
        dimer.set_pbc({true, true, true});

        Value parameters(Value::Map{});
        parameters.insert("model", Value(std::string("spce")));
        parameters.insert("accuracy", Value(1.0e-10));
        parameters.insert("tail", Value(false));
        dimer.set_calculator(std::make_shared<ForcefieldCalculator>(parameters));
        const double energy = dimer.get_potential_energy() - 2.0 * model.self_energy;

        const double charges[6] = {model.q_oxygen,  model.q_hydrogen, model.q_hydrogen,
                                   model.q_oxygen,  model.q_hydrogen, model.q_hydrogen};
        double direct = 0.0;
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 3; j < 6; ++j) {
                const Vec3 r = dimer.position(j) - dimer.position(i);
                direct += units::coulomb * charges[i] * charges[j] / r.norm();
            }
        }
        const double oo = (dimer.position(3) - dimer.position(0)).norm();
        const double ratio6 = std::pow(model.sigma_oxygen / oo, 6.0);
        direct += 4.0 * model.epsilon_oxygen * (ratio6 * ratio6 - ratio6);

        const double difference = std::abs(energy - direct);
        std::printf("    box %4.1f A: Ewald %.6f eV, direct %.6f eV, difference %.2e\n", box,
                    energy, direct, difference);
        FA_CHECK(difference < 2.0e-3);
        if (previous > 0.0) {
            FA_CHECK(difference < previous);  // and it shrinks with the box
        }
        previous = difference;
    }
}

FA_TEST(forcefield_stress_matches_finite_differences) {
    const WaterModel& model = water_model("spce");
    Atoms atoms = water_box(model, 32, 0.997, 11);
    atoms.set_position(0, atoms.position(0) + Vec3{0.10, -0.05, 0.03});

    // The cutoff is pinned: with an automatic one, straining the cell would
    // move the truncation as well as the physics, and the finite difference
    // would be measuring both.
    Value parameters(Value::Map{});
    parameters.insert("model", Value(std::string("spce")));
    parameters.insert("accuracy", Value(1.0e-12));
    parameters.insert("cutoff", Value(4.5));
    atoms.set_calculator(std::make_shared<ForcefieldCalculator>(parameters));

    const std::array<double, 6> stress = atoms.get_stress();
    const double volume = std::abs(atoms.cell().matrix().determinant());
    const double step = 1.0e-6;
    const int voigt[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};

    for (int component = 0; component < 6; ++component) {
        const int a = voigt[component][0];
        const int b = voigt[component][1];
        const auto strained = [&](double amount) {
            Mat3 deformation = Mat3::identity();
            deformation.m[static_cast<std::size_t>(3 * a + b)] += 0.5 * amount;
            deformation.m[static_cast<std::size_t>(3 * b + a)] += 0.5 * amount;
            Atoms copy = atoms;
            copy.set_cell(atoms.cell().matrix() * deformation);
            for (std::size_t i = 0; i < copy.size(); ++i) {
                copy.set_position(i, atoms.position(i) * deformation);
            }
            copy.set_calculator(std::make_shared<ForcefieldCalculator>(parameters));
            return copy.get_potential_energy();
        };
        const double numerical = (strained(step) - strained(-step)) / (2.0 * step) / volume;
        FA_CHECK_CLOSE(stress[static_cast<std::size_t>(component)], numerical, 5e-5);
    }
}

// ---------------------------------------------------------------------------
// SHAKE and RATTLE
// ---------------------------------------------------------------------------
FA_TEST(shake_restores_the_geometry) {
    const WaterModel& model = water_model("spce");
    Atoms atoms = water_box(model, 32, 0.997, 5);
    const Topology topology = water_topology(atoms, model);
    const md::Constraints constraints(topology.constraints());

    const std::vector<double> reference = atoms.positions();
    // Push every atom off its site by up to a hundredth of an angstrom.
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        const double phase = 0.37 * static_cast<double>(i);
        atoms.set_position(i, atoms.position(i) + Vec3{0.01 * std::sin(phase),
                                                       0.01 * std::cos(phase),
                                                       0.01 * std::sin(2.0 * phase)});
    }
    FA_CHECK(constraints.max_deviation(atoms) > 1e-4);

    const md::ConstraintReport report = constraints.shake(atoms, reference, nullptr, 0.0);
    FA_CHECK(report.converged);
    FA_CHECK(constraints.max_deviation(atoms) < 1e-9);
    std::printf("    %s\n", report.to_string().c_str());
}

FA_TEST(rattle_removes_velocity_along_the_bonds) {
    const WaterModel& model = water_model("spce");
    Atoms atoms = water_box(model, 32, 0.997, 9);
    const Topology topology = water_topology(atoms, model);
    const md::Constraints constraints(topology.constraints());

    md::set_maxwell_boltzmann_velocities(atoms, 300.0, 1234, true, true);
    std::vector<double> velocities = atoms.velocities();
    FA_CHECK(constraints.max_velocity_deviation(atoms, velocities) > 1e-4);

    const md::ConstraintReport report = constraints.rattle(atoms, velocities);
    FA_CHECK(report.converged);
    FA_CHECK(constraints.max_velocity_deviation(atoms, velocities) < 1e-9);

    // The projection removes energy - it can only take the constrained
    // components away - but leaves the centre of mass alone.
    double momentum[3] = {0.0, 0.0, 0.0};
    const std::vector<double> masses = atoms.masses();
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            momentum[axis] += masses[i] * velocities[3 * i + static_cast<std::size_t>(axis)];
        }
    }
    for (const double component : momentum) {
        FA_CHECK_CLOSE(component, 0.0, 1e-9);
    }
}

FA_TEST(constraints_reject_nonsense) {
    FA_CHECK_THROWS(md::Constraints({DistanceConstraint{0, 0, 1.0}}), md::DynamicsError);
    FA_CHECK_THROWS(md::Constraints({DistanceConstraint{0, 1, -1.0}}), md::DynamicsError);
    md::Constraints empty;
    FA_CHECK(empty.empty());
    FA_CHECK(empty.removed_degrees_of_freedom() == 0);
}

FA_TEST(rigid_water_stays_rigid_and_conserves_energy) {
    // The whole module in one run: SPC/E forces from Ewald, velocity Verlet,
    // and SHAKE/RATTLE holding every molecule rigid.
    const WaterModel& model = water_model("spce");
    Atoms atoms = water_box(model, 32, 0.997, 2024);

    Value calculator_parameters(Value::Map{});
    calculator_parameters.insert("model", Value(std::string("spce")));
    atoms.set_calculator(std::make_shared<ForcefieldCalculator>(calculator_parameters));

    Value settings(Value::Map{});
    settings.insert("timestep", Value(2.0));          // fs, which needs the constraints
    settings.insert("temperature", Value(300.0));
    settings.insert("initialize_velocities", Value(true));
    settings.insert("seed", Value(std::int64_t{7}));
    md::Dynamics dynamics(settings);
    dynamics.prepare(atoms);

    FA_CHECK(dynamics.constraints() != nullptr);
    FA_CHECK(dynamics.constraints()->size() == 96);   // three per molecule

    const md::Constraints& constraints = *dynamics.constraints();
    FA_CHECK(constraints.max_deviation(atoms) < 1e-9);

    const md::DynamicsResult result = dynamics.run(atoms, 100);
    FA_CHECK(result.reason == md::DynamicsStopReason::Completed);

    // Rigid means rigid: after a hundred steps every bond is still its model
    // length to within the solver tolerance.
    const double deviation = constraints.max_deviation(atoms);
    std::printf("    largest bond error after 100 steps: %.3e (relative)\n", deviation);
    FA_CHECK(deviation < 1e-8);

    // And the velocities are still on the constraint surface.
    FA_CHECK(constraints.max_velocity_deviation(atoms, atoms.velocities()) < 1e-8);

    // Energy conservation. The interesting claim is not that the drift is
    // small - that depends on the box, the cutoff and the timestep - but that
    // it is second order in the timestep. A constrained integrator that has
    // its impulse bookkeeping wrong loses energy at first order, or does not
    // converge at all, so halving the step and watching the drift fall by four
    // is what actually demonstrates SHAKE and RATTLE are doing their job.
    const double coarse = std::abs(result.energy_drift_per_atom(atoms.size()));
    std::printf("    energy drift at 2.0 fs: %.3e eV/atom\n", coarse);
    FA_CHECK(coarse < 1e-3);

    Atoms fine_atoms = water_box(model, 32, 0.997, 2024);
    fine_atoms.set_calculator(std::make_shared<ForcefieldCalculator>(calculator_parameters));
    Value fine_settings = settings;
    fine_settings.insert("timestep", Value(1.0));
    md::Dynamics fine(fine_settings);
    const double refined =
        std::abs(fine.run(fine_atoms, 200).energy_drift_per_atom(fine_atoms.size()));
    std::printf("    energy drift at 1.0 fs: %.3e eV/atom (ratio %.2f)\n", refined,
                coarse / refined);
    FA_CHECK(refined < coarse);
    FA_CHECK(coarse / refined > 2.5);   // second order would give four
    FA_CHECK(coarse / refined < 6.0);
}

FA_TEST(constraints_are_counted_in_the_temperature) {
    const WaterModel& model = water_model("spce");
    Atoms atoms = water_box(model, 32, 0.997, 13);
    Value calculator_parameters(Value::Map{});
    calculator_parameters.insert("model", Value(std::string("spce")));
    atoms.set_calculator(std::make_shared<ForcefieldCalculator>(calculator_parameters));

    Value settings(Value::Map{});
    settings.insert("timestep", Value(1.0));
    settings.insert("temperature", Value(300.0));
    settings.insert("initialize_velocities", Value(true));
    settings.insert("seed", Value(std::int64_t{3}));

    md::Dynamics constrained(settings);
    constrained.prepare(atoms);
    // 96 atoms: 288 momentum components, minus 3 for the centre of mass, minus
    // 96 constraints, leaves 189 - six per molecule, less the three removed
    // globally.
    FA_CHECK(constrained.constraints()->removed_degrees_of_freedom() == 96);

    Value flexible_settings = settings;
    flexible_settings.insert("constraints", Value(false));
    Atoms flexible = water_box(model, 32, 0.997, 13);
    flexible.set_calculator(std::make_shared<ForcefieldCalculator>(calculator_parameters));
    md::Dynamics loose(flexible_settings);
    loose.prepare(flexible);
    FA_CHECK(loose.constraints() == nullptr);
}

FA_TEST(every_integrator_keeps_the_molecules_rigid) {
    const WaterModel& model = water_model("spce");
    for (const char* scheme : {"velocity_verlet", "leapfrog", "euler"}) {
        Atoms atoms = water_box(model, 32, 0.997, 17);
        Value calculator_parameters(Value::Map{});
        calculator_parameters.insert("model", Value(std::string("spce")));
        atoms.set_calculator(std::make_shared<ForcefieldCalculator>(calculator_parameters));

        Value settings(Value::Map{});
        settings.insert("timestep", Value(1.0));
        settings.insert("temperature", Value(250.0));
        settings.insert("integrator", Value(std::string(scheme)));
        settings.insert("initialize_velocities", Value(true));
        settings.insert("seed", Value(std::int64_t{5}));

        md::Dynamics dynamics(settings);
        dynamics.run(atoms, 20);

        const double deviation = dynamics.constraints()->max_deviation(atoms);
        std::printf("    %-16s largest bond error %.2e\n", scheme, deviation);
        FA_CHECK(deviation < 1e-8);
    }
}

FA_TEST_MAIN()
