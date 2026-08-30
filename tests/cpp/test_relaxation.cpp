// SPDX-License-Identifier: MIT
//
// Relaxing real structures: the mapping between an Atoms object and the
// optimizer's coordinate vector, position relaxation, and cell relaxation.

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "fastatoms/calculators/lennard_jones.hpp"
#include "fastatoms/core/atoms.hpp"
#include "fastatoms/optimize/registry.hpp"
#include "fastatoms/optimize/variables.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::optimize;

namespace {

Value settings(std::initializer_list<std::pair<const char*, Value>> entries) {
    Value map{Value::Map{}};
    for (const auto& [key, value] : entries) {
        map.insert(key, value);
    }
    return map;
}

constexpr double kEpsilon = 0.0103;
constexpr double kSigma = 3.40;

std::shared_ptr<Calculator> lennard_jones(double cutoff = 8.5) {
    return std::make_shared<LennardJonesCalculator>(settings(
        {{"epsilon", Value(kEpsilon)}, {"sigma", Value(kSigma)}, {"cutoff", Value(cutoff)}}));
}

/// Face-centred-cubic argon, optionally jittered so nothing is zero by
/// symmetry.
Atoms fcc_argon(double a, int repeat = 1, double jitter = 0.0, unsigned seed = 7) {
    const double basis[4][3] = {
        {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};

    std::vector<int> numbers(4, 18);
    std::vector<double> scaled;
    for (const auto& site : basis) {
        scaled.insert(scaled.end(), {site[0], site[1], site[2]});
    }

    Atoms cell(std::move(numbers), std::vector<double>(12, 0.0), Cell::cubic(a));
    cell.set_scaled_positions(scaled);
    Atoms atoms = repeat > 1 ? cell.repeat(repeat, repeat, repeat) : cell;

    if (jitter > 0.0) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> noise(-jitter, jitter);
        std::vector<double> positions = atoms.positions();
        for (double& value : positions) {
            value += noise(rng);
        }
        atoms.set_positions(positions);
    }
    return atoms;
}

}  // namespace

// ---------------------------------------------------------------------------
// The coordinate mapping
// ---------------------------------------------------------------------------

FA_TEST(atomic_variables_are_just_the_positions) {
    Atoms atoms = fcc_argon(5.26, 1, 0.1);
    atoms.set_calculator(lennard_jones());

    const AtomicVariables variables(atoms.size());
    FA_CHECK(variables.size() == 3 * atoms.size());
    FA_CHECK(variables.atom_rows() == atoms.size());
    FA_CHECK(variables.pack(atoms) == atoms.positions());

    const Evaluation evaluation = variables.evaluate(atoms);
    const std::vector<double> forces = atoms.get_forces();
    for (std::size_t k = 0; k < forces.size(); ++k) {
        FA_CHECK_CLOSE(evaluation.gradient[k], -forces[k], 1e-12);
    }
}

FA_TEST(cell_variables_round_trip_a_structure) {
    Atoms atoms = fcc_argon(5.26, 1, 0.15);
    atoms.set_calculator(lennard_jones());

    CellVariables variables(atoms);
    FA_CHECK(variables.size() == 3 * atoms.size() + 9);
    FA_CHECK(variables.atom_rows() == atoms.size());
    FA_CHECK_CLOSE(variables.cell_factor(), static_cast<double>(atoms.size()), 1e-15);

    const std::vector<double> x = variables.pack(atoms);
    // Straight after construction the deformation is the identity.
    for (std::size_t k = 3 * atoms.size(); k < x.size(); ++k) {
        FA_CHECK_CLOSE(x[k], 0.0, 1e-12);
    }

    const std::vector<double> original = atoms.positions();
    const Mat3 original_cell = atoms.cell().matrix();
    variables.unpack(x, atoms);
    for (std::size_t k = 0; k < original.size(); ++k) {
        FA_CHECK_CLOSE(atoms.positions()[k], original[k], 1e-12);
    }
    FA_CHECK(atoms.cell().matrix() == original_cell);
}

FA_TEST(cell_variables_deform_the_cell_and_carry_the_atoms) {
    Atoms atoms = fcc_argon(5.0);
    atoms.set_calculator(lennard_jones());

    CellVariables variables(atoms);
    std::vector<double> x = variables.pack(atoms);

    // A 10% stretch along x. eps is scaled by cell_factor = natoms.
    x[3 * atoms.size() + 0] = 0.1 * variables.cell_factor();
    variables.unpack(x, atoms);

    FA_CHECK_CLOSE(atoms.cell().lengths().x, 5.5, 1e-12);
    FA_CHECK_CLOSE(atoms.cell().lengths().y, 5.0, 1e-12);
    // Atom 1 sits at (0, 0.5, 0.5) in fractional coordinates and rides along.
    FA_CHECK_CLOSE(atoms.position(2).x, 0.5 * 5.5, 1e-12);
}

FA_TEST(cell_variables_gradient_matches_finite_differences) {
    // The whole cell-relaxation derivation rests on
    //     dE/dS = V S^-T sigma      and      dE/dy_i = -F_i S^T
    // so check both against numerical derivatives of the objective itself.
    Atoms atoms = fcc_argon(5.1, 1, 0.2, 11);
    atoms.set_calculator(lennard_jones(7.0));

    CellVariables variables(atoms);
    std::vector<double> x = variables.pack(atoms);
    variables.unpack(x, atoms);
    const Evaluation analytic = variables.evaluate(atoms);

    const double h = 1e-6;
    for (std::size_t k = 0; k < x.size(); ++k) {
        const double original = x[k];

        x[k] = original + h;
        variables.unpack(x, atoms);
        const double plus = variables.evaluate(atoms).energy;

        x[k] = original - h;
        variables.unpack(x, atoms);
        const double minus = variables.evaluate(atoms).energy;

        x[k] = original;
        FA_CHECK_CLOSE(analytic.gradient[k], (plus - minus) / (2.0 * h), 2e-5);
    }
}

FA_TEST(cell_variables_gradient_matches_finite_differences_under_pressure) {
    // The same check with an external pressure, which turns the objective into
    // the enthalpy E + pV.
    Atoms atoms = fcc_argon(5.1, 1, 0.15, 13);
    atoms.set_calculator(lennard_jones(7.0));

    CellVariablesOptions options;
    options.pressure = 0.002;  // eV/A^3, around 0.3 GPa
    CellVariables variables(atoms, options);

    std::vector<double> x = variables.pack(atoms);
    variables.unpack(x, atoms);
    const Evaluation analytic = variables.evaluate(atoms);
    FA_CHECK_CLOSE(analytic.energy,
                   atoms.get_potential_energy() + options.pressure * atoms.volume(), 1e-12);

    const double h = 1e-6;
    for (std::size_t k = 3 * atoms.size(); k < x.size(); ++k) {
        const double original = x[k];
        x[k] = original + h;
        variables.unpack(x, atoms);
        const double plus = variables.evaluate(atoms).energy;
        x[k] = original - h;
        variables.unpack(x, atoms);
        const double minus = variables.evaluate(atoms).energy;
        x[k] = original;
        FA_CHECK_CLOSE(analytic.gradient[k], (plus - minus) / (2.0 * h), 2e-5);
    }
}

FA_TEST(cell_relaxation_needs_a_periodic_cell) {
    Atoms molecule(std::vector<std::string>{"Ar", "Ar"}, {0.0, 0.0, 0.0, 4.0, 0.0, 0.0});
    FA_CHECK_THROWS(CellVariables(molecule), OptimizerError);

    Atoms boxed = molecule;
    boxed.set_cell(Mat3::diagonal(20.0, 20.0, 20.0));
    boxed.set_pbc(false);
    FA_CHECK_THROWS(CellVariables(boxed), OptimizerError);
}

// ---------------------------------------------------------------------------
// Relaxing positions
// ---------------------------------------------------------------------------

FA_TEST(a_dimer_relaxes_to_the_analytic_minimum) {
    // Two Lennard-Jones atoms have their minimum at 2^(1/6) sigma with energy
    // -epsilon, which is an exact answer to check against.
    const double expected_distance = std::pow(2.0, 1.0 / 6.0) * kSigma;

    for (const char* algorithm : {"sd", "cg", "bfgs", "lbfgs"}) {
        Atoms dimer(std::vector<std::string>{"Ar", "Ar"}, {0.0, 0.0, 0.0, 4.2, 0.0, 0.0});
        dimer.set_calculator(std::make_shared<LennardJonesCalculator>(
            settings({{"epsilon", Value(kEpsilon)},
                      {"sigma", Value(kSigma)},
                      {"cutoff", Value(20.0)},
                      {"shift", Value(false)}})));

        auto optimizer = make_optimizer(
            algorithm, settings({{"fmax", Value(1e-6)}, {"max_steps", Value(500)}}));
        const OptimizationResult result = optimizer->run(dimer);

        FA_CHECK(result.converged);
        FA_CHECK_CLOSE(dimer.distance(0, 1), expected_distance, 1e-5);
        FA_CHECK_CLOSE(result.energy, -kEpsilon, 1e-8);
        // The structure is left at the point the result describes.
        FA_CHECK_CLOSE(dimer.get_potential_energy(), result.energy, 1e-12);
    }
}

FA_TEST(all_algorithms_relax_a_disordered_crystal_to_the_same_place) {
    double reference_energy = 0.0;
    for (const char* algorithm : {"sd", "cg", "bfgs", "lbfgs"}) {
        Atoms atoms = fcc_argon(5.26, 2, 0.25, 3);
        atoms.set_calculator(lennard_jones());

        auto optimizer = make_optimizer(
            algorithm, settings({{"fmax", Value(0.005)}, {"max_steps", Value(2000)}}));
        const OptimizationResult result = optimizer->run(atoms);

        FA_CHECK(result.converged);
        FA_CHECK(max_row_norm(scaled(-1.0, atoms.get_forces())) <= 0.005);

        if (reference_energy == 0.0) {
            reference_energy = result.energy;
        } else {
            // All four find the same basin from the same start.
            FA_CHECK_CLOSE(result.energy, reference_energy, 1e-4);
        }
    }
}

FA_TEST(quasi_newton_needs_fewer_force_evaluations_than_steepest_descent) {
    const auto evaluations_for = [](const char* algorithm) {
        Atoms atoms = fcc_argon(5.26, 2, 0.25, 5);
        atoms.set_calculator(lennard_jones());
        auto optimizer = make_optimizer(
            algorithm, settings({{"fmax", Value(0.01)}, {"max_steps", Value(5000)}}));
        const OptimizationResult result = optimizer->run(atoms);
        FA_CHECK(result.converged);
        return result.evaluations;
    };

    const int sd = evaluations_for("sd");
    const int bfgs = evaluations_for("bfgs");
    const int lbfgs = evaluations_for("lbfgs");
    FA_CHECK(bfgs < sd);
    FA_CHECK(lbfgs < sd);
}

FA_TEST(the_energy_falls_monotonically) {
    Atoms atoms = fcc_argon(5.26, 2, 0.3, 17);
    atoms.set_calculator(lennard_jones());

    auto optimizer = make_optimizer("lbfgs", settings({{"fmax", Value(0.01)},
                                                       {"max_steps", Value(500)}}));
    const OptimizationResult result = optimizer->run(atoms);
    FA_CHECK(result.converged);
    FA_CHECK(result.history.size() > 3);

    for (std::size_t k = 1; k < result.history.size(); ++k) {
        FA_CHECK(result.history[k].energy <= result.history[k - 1].energy + 1e-12);
        FA_CHECK(result.history[k].energy_change <= 1e-12);
    }
}

FA_TEST(relaxation_reports_useful_failures) {
    Atoms atoms = fcc_argon(5.26, 1, 0.2);
    FA_CHECK_THROWS(make_optimizer("bfgs")->run(atoms), OptimizerError);  // no calculator

    atoms.set_calculator(lennard_jones());
    Atoms empty;
    empty.set_calculator(lennard_jones());
    FA_CHECK_THROWS(make_optimizer("bfgs")->run(empty), OptimizerError);

    // A budget of zero steps stops immediately without converging.
    auto optimizer = make_optimizer(
        "bfgs", settings({{"fmax", Value(1e-9)}, {"max_steps", Value(0)}}));
    const OptimizationResult result = optimizer->run(atoms);
    FA_CHECK(!result.converged);
    FA_CHECK(result.reason == StopReason::MaxSteps);
    FA_CHECK(result.steps == 0);
    FA_CHECK(result.evaluations == 1);
}

// ---------------------------------------------------------------------------
// Relaxing the cell
// ---------------------------------------------------------------------------

FA_TEST(cell_relaxation_drives_the_stress_to_zero) {
    // Start from a lattice constant well away from equilibrium and let the
    // cell find its own size.
    Atoms atoms = fcc_argon(5.6, 1, 0.02, 23);
    atoms.set_calculator(lennard_jones());
    const double initial_volume = atoms.volume();

    auto optimizer = make_optimizer("lbfgs", settings({{"fmax", Value(1e-4)},
                                                       {"max_steps", Value(500)},
                                                       {"relax_cell", Value(true)}}));
    const OptimizationResult result = optimizer->run(atoms);

    FA_CHECK(result.converged);
    FA_CHECK(atoms.volume() < initial_volume);  // argon at 5.6 A is stretched

    const std::array<double, 6> stress = atoms.get_stress();
    for (const double component : stress) {
        FA_CHECK(std::abs(component) < 1e-5);
    }
    FA_CHECK(std::abs(pressure_from_stress(stress)) < 1e-5);
    // The relaxed cell stays cubic, as fcc symmetry requires.
    const Vec3 lengths = atoms.cell().lengths();
    FA_CHECK_CLOSE(lengths.x, lengths.y, 1e-3);
    FA_CHECK_CLOSE(lengths.y, lengths.z, 1e-3);
}

FA_TEST(external_pressure_compresses_the_cell) {
    const auto relaxed_volume = [](double pressure) {
        Atoms atoms = fcc_argon(5.3);
        atoms.set_calculator(lennard_jones());
        auto optimizer = make_optimizer("lbfgs", settings({{"fmax", Value(1e-4)},
                                                           {"max_steps", Value(500)},
                                                           {"relax_cell", Value(true)},
                                                           {"pressure", Value(pressure)}}));
        const OptimizationResult result = optimizer->run(atoms);
        FA_CHECK(result.converged);
        return atoms.volume();
    };

    const double free_volume = relaxed_volume(0.0);
    const double squeezed_volume = relaxed_volume(0.005);  // about 0.8 GPa
    FA_CHECK(squeezed_volume < free_volume);

    // At equilibrium under pressure p the internal stress balances it.
    Atoms atoms = fcc_argon(5.3);
    atoms.set_calculator(lennard_jones());
    auto optimizer = make_optimizer("lbfgs", settings({{"fmax", Value(1e-5)},
                                                       {"max_steps", Value(500)},
                                                       {"relax_cell", Value(true)},
                                                       {"pressure", Value(0.005)}}));
    (void)optimizer->run(atoms);
    FA_CHECK_CLOSE(pressure_from_stress(atoms.get_stress()), 0.005, 1e-2);
}

FA_TEST(a_mask_freezes_the_selected_cell_directions) {
    Atoms atoms = fcc_argon(5.6);
    atoms.set_calculator(lennard_jones());
    const double original_c = atoms.cell().lengths().z;

    // Relax a and b only, as one would for a supported slab.
    Value mask{Value::List{Value(1), Value(1), Value(0), Value(0), Value(0), Value(0)}};
    auto optimizer = make_optimizer("lbfgs", settings({{"fmax", Value(1e-4)},
                                                       {"max_steps", Value(500)},
                                                       {"relax_cell", Value(true)},
                                                       {"mask", mask}}));
    (void)optimizer->run(atoms);

    FA_CHECK_CLOSE(atoms.cell().lengths().z, original_c, 1e-9);
    FA_CHECK(atoms.cell().lengths().x < original_c);
}

FA_TEST(a_concave_start_still_converges) {
    // A crystal stretched far past the pair-potential inflection point sits on
    // a concave stretch of the energy surface: every quasi-Newton curvature
    // update is legitimately discarded there. Without a line search that can
    // grow the step, the method degenerates to a fixed -g/alpha crawl and never
    // arrives, which is why the default line search is a Wolfe search.
    for (const char* algorithm : {"cg", "bfgs", "lbfgs"}) {
        // Simple cubic argon at 5.6 A: nearest neighbours are far beyond the
        // 3.82 A minimum, so the lattice wants to collapse by a third.
        Atoms atoms(std::vector<int>(1, 18), {0.1, 0.0, 0.0}, Cell::cubic(5.6));
        atoms = atoms.repeat(2, 2, 2);
        atoms.set_calculator(lennard_jones());

        auto optimizer = make_optimizer(algorithm, settings({{"fmax", Value(1e-4)},
                                                             {"max_steps", Value(300)},
                                                             {"relax_cell", Value(true)}}));
        const OptimizationResult result = optimizer->run(atoms);
        FA_CHECK(result.converged);
        FA_CHECK(result.steps < 100);
        // The lattice constant of the repeated cell, back to one unit.
        FA_CHECK(atoms.cell().lengths().x / 2.0 < 4.5);
        FA_CHECK(std::abs(pressure_from_stress(atoms.get_stress())) < 1e-5);
    }

    // With the line search switched off the same run stalls: this is the
    // behaviour the default avoids, not a bug in the algorithms.
    Atoms atoms(std::vector<int>(1, 18), {0.1, 0.0, 0.0}, Cell::cubic(5.6));
    atoms = atoms.repeat(2, 2, 2);
    atoms.set_calculator(lennard_jones());
    auto stalled = make_optimizer("lbfgs", settings({{"fmax", Value(1e-4)},
                                                     {"max_steps", Value(100)},
                                                     {"relax_cell", Value(true)},
                                                     {"line_search", Value("none")}}));
    const OptimizationResult result = stalled->run(atoms);
    FA_CHECK(!result.converged);
}

FA_TEST(cell_relaxation_reports_both_force_measures) {
    Atoms atoms = fcc_argon(5.5, 1, 0.15, 29);
    atoms.set_calculator(lennard_jones());

    auto optimizer = make_optimizer("bfgs", settings({{"fmax", Value(1e-3)},
                                                      {"max_steps", Value(500)},
                                                      {"relax_cell", Value(true)}}));
    const OptimizationResult result = optimizer->run(atoms);
    FA_CHECK(result.converged);

    // fmax covers the cell rows too; the atomic-only figure is reported
    // separately and can never be larger.
    for (const OptimizationStep& step : result.history) {
        FA_CHECK(step.atomic_fmax <= step.fmax + 1e-12);
    }
    FA_CHECK(result.history.back().fmax <= 1e-3);
}

FA_TEST_MAIN()
