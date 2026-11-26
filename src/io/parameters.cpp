// SPDX-License-Identifier: MIT
#include "fastatoms/io/parameters.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "fastatoms/core/units.hpp"


namespace fastatoms {

Task task_from_string(std::string_view name) {
    std::string key(name);
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::replace(key.begin(), key.end(), '-', '_');

    if (key == "single_point" || key == "singlepoint" || key == "energy" || key == "static") {
        return Task::SinglePoint;
    }
    if (key == "md" || key == "molecular_dynamics" || key == "dynamics") {
        return Task::MolecularDynamics;
    }
    if (key == "optimize" || key == "optimization" || key == "relax" || key == "minimize" ||
        key == "minimization") {
        return Task::Optimization;
    }
    if (key == "phonon" || key == "phonons" || key == "vibrations" || key == "frequencies") {
        return Task::Phonon;
    }
    if (key == "neb" || key == "band" || key == "reaction_path" || key == "saddle") {
        return Task::NEB;
    }
    return Task::Custom;
}

std::string_view task_to_string(Task task) {
    switch (task) {
        case Task::SinglePoint:        return "single_point";
        case Task::MolecularDynamics:  return "md";
        case Task::Optimization:       return "optimize";
        case Task::Phonon:             return "phonon";
        case Task::NEB:                return "neb";
        case Task::Custom:             return "custom";
    }
    return "custom";
}

double MolecularDynamicsParameters::timestep_internal() const { return timestep_fs * units::fs; }

}  // namespace fastatoms
