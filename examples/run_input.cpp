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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <input.yaml> [neighbour cutoff in A]\n", argv[0]);
        return 2;
    }

    try {
        const io::InputDeck deck = io::read_yaml_file(argv[1]);
        std::printf("FastAtoms %s\n\n", version());
        print_structure(deck.atoms);
        print_simulation(deck.simulation);

        const double cutoff = argc > 2 ? std::atof(argv[2]) : 0.0;
        if (cutoff > 0.0) {
            print_neighbors(deck.atoms, cutoff);
        }
    } catch (const InputError& error) {
        std::fprintf(stderr, "input error: %s\n", error.what());
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
