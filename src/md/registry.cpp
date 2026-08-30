// SPDX-License-Identifier: MIT
#include "fastatoms/md/registry.hpp"

namespace fastatoms::md {
namespace {

/// Copy `key` from `spec` into `target` unless the file already set it.
void fill(Value& target, const std::string& key, Value value) {
    if (!target.contains(key)) {
        target.insert(key, std::move(value));
    }
}

}  // namespace

std::shared_ptr<Dynamics> make_dynamics(const Value& parameters) {
    return std::make_shared<Dynamics>(parameters);
}

std::shared_ptr<Dynamics> make_dynamics(const MolecularDynamicsParameters& spec) {
    // The unparsed subtree carries every key the file set, including the ones
    // the typed struct does not know about; the typed values fill the gaps.
    Value parameters = spec.parameters.is_map() ? spec.parameters : Value(Value::Map{});

    fill(parameters, "timestep", Value(spec.timestep_fs));
    fill(parameters, "steps", Value(spec.steps));
    fill(parameters, "ensemble", Value(spec.ensemble));
    fill(parameters, "seed", Value(spec.seed));
    if (spec.temperature_K > 0.0) {
        fill(parameters, "temperature", Value(spec.temperature_K));
    }
    if (spec.pressure_GPa != 0.0) {
        fill(parameters, "pressure", Value(spec.pressure_GPa));
    }
    if (!spec.thermostat.empty()) {
        fill(parameters, "thermostat", Value(spec.thermostat));
    }
    if (spec.initialize_velocities) {
        fill(parameters, "initialize_velocities", Value(true));
    }
    // The file's `friction` is a rate in 1/fs; the thermostats want a time.
    if (spec.friction > 0.0 && !parameters.contains("tau")) {
        fill(parameters, "tau", Value(1.0 / spec.friction));
    }

    return make_dynamics(parameters);
}

std::shared_ptr<Dynamics> make_dynamics(const SimulationParameters& simulation) {
    auto dynamics = make_dynamics(simulation.md);

    // The output section is written once for the whole run, so fold it in.
    Value parameters = dynamics->parameters();
    const OutputParameters& output = simulation.output;
    if (!output.trajectory.empty()) {
        fill(parameters, "trajectory", Value(output.trajectory));
    }
    if (!output.log.empty()) {
        fill(parameters, "log", Value(output.log));
    }
    if (output.interval > 0) {
        fill(parameters, "interval", Value(output.interval));
    }
    dynamics->set_parameters(std::move(parameters));
    return dynamics;
}

}  // namespace fastatoms::md
