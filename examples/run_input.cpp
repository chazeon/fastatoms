// SPDX-License-Identifier: MIT
//
// A small driver that reads a FastAtoms YAML file and reports what it found.
// It is the C++ mirror of `python/examples/read_input.py`.
//
//     fastatoms-run examples/silicon.yaml [cutoff]

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "fastatoms/fastatoms.hpp"

using namespace fastatoms;

namespace {

void print_structure(const Atoms& atoms) {
    std::printf("structure   : %s\n", atoms.to_string().c_str());
    std::printf("formula     : %s\n", atoms.formula().c_str());
    std::printf("atoms       : %zu\n", atoms.size());
    if (atoms.cell().is_invertible()) {
        const Vec3 lengths = atoms.cell().lengths();
        const Vec3 angles = atoms.cell().angles();
        std::printf("cell        : a=%.4f b=%.4f c=%.4f  alpha=%.2f beta=%.2f gamma=%.2f\n",
                    lengths.x, lengths.y, lengths.z, angles.x, angles.y, angles.z);
        std::printf("volume      : %.4f A^3  (%.4f A^3 / atom)\n", atoms.volume(),
                    atoms.size() ? atoms.volume() / static_cast<double>(atoms.size()) : 0.0);
    }
    const Vec3 com = atoms.center_of_mass();
    std::printf("centre of mass: [%.4f, %.4f, %.4f]\n", com.x, com.y, com.z);
    if (atoms.has_momenta()) {
        std::printf("temperature : %.2f K\n", atoms.temperature());
    }
}

void print_simulation(const SimulationParameters& simulation) {
    std::printf("\ntask        : %s\n", simulation.task_name.c_str());
    if (!simulation.calculator.name.empty()) {
        std::printf("calculator  : %s %s\n", simulation.calculator.name.c_str(),
                    simulation.calculator.parameters.to_string().c_str());
    }
    if (simulation.task == Task::MolecularDynamics) {
        std::printf("md          : %s, %lld steps of %.3f fs at %.1f K\n",
                    simulation.md.ensemble.c_str(),
                    static_cast<long long>(simulation.md.steps), simulation.md.timestep_fs,
                    simulation.md.temperature_K);
    }
    if (simulation.task == Task::Optimization) {
        std::printf("optimizer   : %s, fmax=%.4f eV/A, max %lld steps\n",
                    simulation.optimizer.algorithm.c_str(), simulation.optimizer.fmax,
                    static_cast<long long>(simulation.optimizer.max_steps));
    }
    if (simulation.task == Task::Phonon) {
        const std::vector<double>& n = simulation.phonon.supercell;
        std::printf("phonons     : supercell %gx%gx%g, displacement %.4f A, symmetry %s\n",
                    n.size() > 2 ? n[0] : 1.0, n.size() > 2 ? n[1] : 1.0,
                    n.size() > 2 ? n[2] : 1.0, simulation.phonon.displacement,
                    simulation.phonon.symmetry ? "on" : "off");
    }
    if (simulation.task == Task::NEB) {
        const BandParameters& band = simulation.neb;
        if (band.band.empty()) {
            const char* from = !band.initial_structure.is_null() ? "an inline structure"
                               : band.initial.empty()            ? "structure:"
                                                                 : band.initial.c_str();
            const char* to = !band.final_structure.is_null() ? "an inline structure"
                                                             : band.final.c_str();
            std::printf("neb         : %lld images, %s to %s\n",
                        static_cast<long long>(band.images), from, to);
        } else {
            std::printf("neb         : the band in '%s'\n", band.band.c_str());
        }
        std::printf("              k=%g eV/A^2, tangent %s, climbing %s\n", band.spring_constant,
                    band.tangent.c_str(), band.climbing ? "on" : "off");
        if (band.automatic) {
            std::printf("              autoneb, growing to %lld images (coarse fmax %g)\n",
                        static_cast<long long>(band.max_images), band.coarse_fmax);
        }
        if (!band.fixed_atoms.empty()) {
            std::printf("              %zu atoms held\n", band.fixed_atoms.size());
        }
    }
    if (!simulation.output.trajectory.empty()) {
        std::printf("trajectory  : %s every %lld steps\n", simulation.output.trajectory.c_str(),
                    static_cast<long long>(simulation.output.interval));
    }
}

void print_neighbors(const Atoms& atoms, double cutoff) {
    const NeighborList list = build_neighbor_list(atoms, cutoff);
    const std::vector<std::int32_t> coordination = list.coordination_numbers();

    double mean = 0.0;
    std::int32_t lowest = coordination.empty() ? 0 : coordination.front();
    std::int32_t highest = lowest;
    for (const std::int32_t n : coordination) {
        mean += static_cast<double>(n);
        lowest = std::min(lowest, n);
        highest = std::max(highest, n);
    }
    mean = coordination.empty() ? 0.0 : mean / static_cast<double>(coordination.size());

    double shortest = cutoff;
    for (const double d : list.distances) {
        shortest = std::min(shortest, d);
    }

    std::printf("\nneighbour list (cutoff %.2f A)\n", cutoff);
    std::printf("pairs       : %zu\n", list.size());
    std::printf("coordination: mean %.2f, range %d..%d\n", mean, lowest, highest);
    if (!list.distances.empty()) {
        std::printf("shortest    : %.4f A\n", shortest);
    }
}

/// Attach the calculator the file names and relax the structure with the
/// optimizer it names. Skipped when the file asks for neither.
void run_optimization(Atoms& atoms, const SimulationParameters& simulation) {
    if (simulation.task != Task::Optimization) {
        return;
    }
    if (simulation.calculator.name.empty()) {
        std::printf("\nthe task is 'optimize' but no calculator is configured\n");
        return;
    }

    // This driver's first job is to show what is in a file. A file that names
    // a calculator this build does not have is worth reporting, not worth
    // aborting over - everything above still printed.
    try {
        atoms.set_calculator(make_calculator(simulation.calculator));
    } catch (const CalculatorError& error) {
        std::printf("\nskipping the relaxation: %s\n", error.what());
        return;
    }
    auto optimizer = optimize::make_optimizer(simulation.optimizer);

    std::printf("\nrelaxation with %s\n", optimizer->to_string().c_str());
    std::printf("%6s %16s %14s %14s\n", "step", "energy [eV]", "fmax [eV/A]", "dE [eV]");
    optimizer->attach([](const optimize::OptimizationStep& step) {
        std::printf("%6d %16.6f %14.6f %14.3e\n", step.step, step.energy, step.fmax,
                    step.energy_change);
        return true;
    });

    const optimize::OptimizationResult result = optimizer->run(atoms);
    std::printf("\n%s\n", result.message.c_str());
    std::printf("force calls : %d\n", result.evaluations);
    if (simulation.optimizer.relax_cell) {
        const Vec3 lengths = atoms.cell().lengths();
        std::printf("final cell  : a=%.4f b=%.4f c=%.4f  (%.3f A^3)\n", lengths.x, lengths.y,
                    lengths.z, atoms.volume());
    }
}

/// Attach the calculator the file names and run the dynamics it describes.
void run_dynamics(Atoms& atoms, const SimulationParameters& simulation) {
    if (simulation.task != Task::MolecularDynamics) {
        return;
    }
    if (simulation.calculator.name.empty()) {
        std::printf("\nthe task is 'md' but no calculator is configured\n");
        return;
    }

    try {
        atoms.set_calculator(make_calculator(simulation.calculator));
    } catch (const CalculatorError& error) {
        std::printf("\nskipping the dynamics: %s\n", error.what());
        return;
    }
    auto dynamics = md::make_dynamics(simulation);

    std::printf("\n%s\n", dynamics->to_string().c_str());
    std::printf("%8s %12s %10s %16s %16s\n", "step", "time [fs]", "T [K]", "E_tot [eV]",
                "E_cons [eV]");
    const int report = std::max(1, static_cast<int>(simulation.md.steps) / 10);
    dynamics->attach(
        [](const Atoms&, const md::DynamicsStep& step) {
            std::printf("%8d %12.1f %10.2f %16.6f %16.6f\n", step.step, step.time,
                        step.temperature, step.total_energy, step.conserved_energy);
            return true;
        },
        report);

    const md::DynamicsResult result = dynamics->run(atoms);
    std::printf("\n%s\n", result.message.c_str());
    std::printf("force calls : %d\n", result.evaluations);
    std::printf("drift       : %.3e eV/atom\n", result.energy_drift_per_atom(atoms.size()));
}

/// Attach the calculator the file names, run the displacements it asks for,
/// and print the frequencies along a q-point path.
void run_phonons(Atoms& atoms, const SimulationParameters& simulation) {
    if (simulation.task != Task::Phonon) {
        return;
    }
    if (simulation.calculator.name.empty()) {
        std::printf("\nthe task is 'phonon' but no calculator is configured\n");
        return;
    }

    try {
        atoms.set_calculator(make_calculator(simulation.calculator));
    } catch (const CalculatorError& error) {
        std::printf("\nskipping the phonons: %s\n", error.what());
        return;
    }
    auto phonons = phonon::make_phonons(atoms, simulation);

    std::printf("\n%s\n", phonons->to_string().c_str());
    std::printf("displacements: %zu, against %zu without symmetry\n",
                phonons->calculation_count(), phonons->calculation_count_without_symmetry());

    phonons->run();
    std::printf("residual force: %.3e eV/A\n", phonons->residual_force());

    // The zone centre, the two zone-boundary points along the axes, and the
    // corner. Which letters those carry depends on the lattice, so they are
    // printed as the fractional coordinates they are: enough to show the
    // acoustic branches going to zero at the centre and the dispersion after.
    const std::vector<Vec3> path{
        Vec3{0.0, 0.0, 0.0},
        Vec3{0.5, 0.0, 0.0},
        Vec3{0.5, 0.5, 0.0},
        Vec3{0.5, 0.5, 0.5},
    };
    std::printf("\n%-20s %s\n", "q", "frequencies [THz]");
    for (const Vec3& q : path) {
        std::printf("[%.2f, %.2f, %.2f]  ", q.x, q.y, q.z);
        for (const double f : phonons->frequencies(q)) {
            std::printf(" %9.4f", f);
        }
        std::printf("\n");
    }
}

/// Print an energy profile the way `python/examples/neb.py` does.
void print_profile(const neb::BandProfile& profile) {
    std::printf("\n%4s %12s %14s %12s\n", "image", "s [A]", "E-E0 [eV]", "fmax [eV/A]");
    for (std::size_t k = 0; k < profile.energies.size(); ++k) {
        const bool interior = k > 0 && k + 1 < profile.energies.size();
        std::printf("%4zu %12.4f %14.6f", k, profile.distances[k],
                    profile.energies[k] - profile.energies.front());
        if (interior) {
            std::printf(" %12.3e", profile.image_fmax[k - 1]);
        } else {
            std::printf(" %12s", "fixed");
        }
        std::printf("%s\n", k == profile.highest ? "   <- highest" : "");
    }
    std::printf("\nforward barrier : %.6f eV\n", profile.forward_barrier);
    std::printf("reverse barrier : %.6f eV\n", profile.reverse_barrier);
    std::printf("reaction energy : %.6f eV\n", profile.reaction_energy);
}

/// Attach the calculator the file names, build the band it describes, and
/// relax it with the optimizer from the `optimizer:` section.
void run_neb(Atoms& atoms, const SimulationParameters& simulation) {
    if (simulation.task != Task::NEB) {
        return;
    }
    if (simulation.calculator.name.empty()) {
        std::printf("\nthe task is 'neb' but no calculator is configured\n");
        return;
    }
    try {
        atoms.set_calculator(make_calculator(simulation.calculator));
    } catch (const CalculatorError& error) {
        std::printf("\nskipping the band: %s\n", error.what());
        return;
    }

    // FIRE unless the file asked for something else: the band force is not the
    // gradient of anything, so an optimizer with a line search will be
    // refused, and "bfgs" is the deck-wide default.
    OptimizerParameters spec = simulation.optimizer;
    if (simulation.optimizer.parameters.is_null()) {
        spec.algorithm = "fire";
    }

    if (neb::wants_autoneb(simulation)) {
        neb::AutoNEB automatic = neb::make_autoneb(atoms, simulation);
        std::printf("\n%s\n", automatic.band().to_string().c_str());

        const neb::AutoNEBResult result = automatic.run(spec.algorithm, spec.parameters);
        std::printf("\n%s\n", result.to_string().c_str());
        print_profile(result.profile);
        if (!simulation.output.trajectory.empty()) {
            io::write(simulation.output.trajectory, automatic.band().images());
            std::printf("\nband written to %s\n", simulation.output.trajectory.c_str());
        }
        return;
    }

    neb::NEB band = neb::make_band(atoms, simulation);
    std::printf("\n%s\n", band.to_string().c_str());

    auto optimizer = optimize::make_optimizer(spec);
    std::printf("relaxing with %s\n", optimizer->to_string().c_str());
    std::printf("%6s %16s %14s\n", "step", "sum E [eV]", "fmax [eV/A]");
    optimizer->attach([](const optimize::OptimizationStep& step) {
        std::printf("%6d %16.6f %14.6f\n", step.step, step.energy, step.fmax);
        return true;
    });

    const optimize::OptimizationResult result = band.run(*optimizer);
    std::printf("\n%s\n", result.message.c_str());
    std::printf("force calls : %d\n", result.evaluations);
    print_profile(band.profile());

    if (!simulation.output.trajectory.empty()) {
        io::write(simulation.output.trajectory, band.images());
        std::printf("\nband written to %s\n", simulation.output.trajectory.c_str());
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <input.yaml> [neighbour cutoff in A]\n", argv[0]);
        return 2;
    }

    try {
        io::InputDeck deck = io::read_yaml_file(argv[1]);
        std::printf("FastAtoms %s\n\n", version());
        print_structure(deck.atoms);
        print_simulation(deck.simulation);

        const double cutoff = argc > 2 ? std::atof(argv[2]) : 0.0;
        if (cutoff > 0.0) {
            print_neighbors(deck.atoms, cutoff);
        }

        run_optimization(deck.atoms, deck.simulation);
        run_dynamics(deck.atoms, deck.simulation);
        run_phonons(deck.atoms, deck.simulation);
        run_neb(deck.atoms, deck.simulation);
    } catch (const InputError& error) {
        std::fprintf(stderr, "input error: %s\n", error.what());
        return 1;
    } catch (const neb::NEBError& error) {
        std::fprintf(stderr, "neb error: %s\n", error.what());
        return 1;
    } catch (const phonon::PhononError& error) {
        std::fprintf(stderr, "phonon error: %s\n", error.what());
        return 1;
    } catch (const md::DynamicsError& error) {
        std::fprintf(stderr, "dynamics error: %s\n", error.what());
        return 1;
    } catch (const optimize::OptimizerError& error) {
        std::fprintf(stderr, "optimizer error: %s\n", error.what());
        return 1;
    } catch (const CalculatorError& error) {
        std::fprintf(stderr, "calculator error: %s\n", error.what());
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
