// SPDX-License-Identifier: MIT
#include "bindings.hpp"

namespace fastatoms::bindings {

py::class_<io::InputDeck> bind_io(py::module_& m) {
    py::enum_<Task>(m, "Task", "What an input file asks the run to do.")
        .value("single_point", Task::SinglePoint)
        .value("md", Task::MolecularDynamics)
        .value("optimize", Task::Optimization)
        .value("phonon", Task::Phonon)
        .value("custom", Task::Custom);

    py::class_<CalculatorParameters>(m, "CalculatorParameters",
                                     "Which force model to attach, and its settings.")
        .def(py::init<>())
        .def_readwrite("name", &CalculatorParameters::name)
        .def_readwrite("parameters", &CalculatorParameters::parameters,
                       "Calculator-specific settings as a plain dict.")
        .def("__repr__", [](const CalculatorParameters& self) {
            return "CalculatorParameters(name='" + self.name + "')";
        });

    py::class_<MolecularDynamicsParameters>(m, "MolecularDynamicsParameters",
                                            "Settings from the ``simulation.md`` section.")
        .def(py::init<>())
        .def_readwrite("ensemble", &MolecularDynamicsParameters::ensemble)
        .def_readwrite("timestep_fs", &MolecularDynamicsParameters::timestep_fs)
        .def_readwrite("steps", &MolecularDynamicsParameters::steps)
        .def_readwrite("temperature_K", &MolecularDynamicsParameters::temperature_K)
        .def_readwrite("pressure_GPa", &MolecularDynamicsParameters::pressure_GPa)
        .def_readwrite("thermostat", &MolecularDynamicsParameters::thermostat)
        .def_readwrite("friction", &MolecularDynamicsParameters::friction)
        .def_readwrite("seed", &MolecularDynamicsParameters::seed)
        .def_readwrite("initialize_velocities",
                       &MolecularDynamicsParameters::initialize_velocities)
        .def_property_readonly("timestep", &MolecularDynamicsParameters::timestep_internal,
                               "The timestep converted to the internal time unit.")
        .def("__repr__", [](const MolecularDynamicsParameters& self) {
            return "MolecularDynamicsParameters(ensemble='" + self.ensemble +
                   "', timestep_fs=" + std::to_string(self.timestep_fs) +
                   ", steps=" + std::to_string(self.steps) + ")";
        });

    py::class_<OptimizerParameters>(m, "OptimizerParameters",
                                    "Settings from the ``simulation.optimizer`` section.")
        .def(py::init<>())
        .def_readwrite("algorithm", &OptimizerParameters::algorithm)
        .def_readwrite("fmax", &OptimizerParameters::fmax)
        .def_readwrite("max_steps", &OptimizerParameters::max_steps)
        .def_readwrite("relax_cell", &OptimizerParameters::relax_cell)
        .def("__repr__", [](const OptimizerParameters& self) {
            return "OptimizerParameters(algorithm='" + self.algorithm +
                   "', fmax=" + std::to_string(self.fmax) + ")";
        });

    py::class_<OutputParameters>(m, "OutputParameters",
                                 "Settings from the ``simulation.output`` section.")
        .def(py::init<>())
        .def_readwrite("trajectory", &OutputParameters::trajectory)
        .def_readwrite("log", &OutputParameters::log)
        .def_readwrite("interval", &OutputParameters::interval)
        .def("__repr__", [](const OutputParameters& self) {
            return "OutputParameters(trajectory='" + self.trajectory +
                   "', interval=" + std::to_string(self.interval) + ")";
        });

    py::class_<SimulationParameters>(m, "SimulationParameters",
                                     "The parsed ``simulation:`` section of an input file.")
        .def(py::init<>())
        .def_readwrite("task", &SimulationParameters::task)
        .def_readwrite("task_name", &SimulationParameters::task_name,
                       "The task exactly as spelled in the file.")
        .def_readwrite("calculator", &SimulationParameters::calculator)
        .def_readwrite("md", &SimulationParameters::md)
        .def_readwrite("optimizer", &SimulationParameters::optimizer)
        .def_readwrite("phonon", &SimulationParameters::phonon)
        .def_readwrite("output", &SimulationParameters::output)
        .def_readwrite("raw", &SimulationParameters::raw,
                       "The whole section as a dict, including keys FastAtoms ignores.")
        .def("__repr__", [](const SimulationParameters& self) {
            return "SimulationParameters(task='" + self.task_name + "', calculator='" +
                   self.calculator.name + "')";
        });

    py::class_<io::InputDeck> deck(m, "InputDeck", "A parsed input document.");
    deck
        .def(py::init<>())
        .def_readwrite("atoms", &io::InputDeck::atoms)
        .def_readwrite("simulation", &io::InputDeck::simulation)
        .def_readwrite("document", &io::InputDeck::document,
                       "The complete document as a nested dict.")
        .def("__repr__", [](const io::InputDeck& self) {
            return "InputDeck(" + self.atoms.to_string() + ", " + self.simulation.task_name + ")";
        });


    m.def(
        "read", [](const std::string& path) { return io::read_yaml_file(path); }, py::arg("path"),
        "Parse a YAML input file into an :class:`InputDeck`.");

    m.def(
        "read_string", [](const std::string& text) { return io::read_yaml_string(text); },
        py::arg("text"), "Parse a YAML document held in a string.");

    m.def(
        "read_atoms",
        [](const std::string& path) { return io::read_yaml_file(path).atoms; }, py::arg("path"),
        "Parse only the structure from a YAML file.");

    m.def(
        "parse_yaml", [](const std::string& text) { return io::parse_yaml_string(text); },
        py::arg("text"), "Parse arbitrary YAML into plain Python data.");

    m.def(
        "parse_yaml_file", [](const std::string& path) { return io::parse_yaml_file(path); },
        py::arg("path"), "Parse an arbitrary YAML file into plain Python data.");

    m.def(
        "write",
        [](const std::string& path, const Atoms& atoms, bool scaled) {
            io::write_yaml_file(path, atoms, scaled);
        },
        py::arg("path"), py::arg("atoms"), py::arg("scaled") = false,
        "Write a structure to a YAML file that :func:`read` accepts.");

    m.def(
        "to_yaml",
        [](const Atoms& atoms, bool scaled) { return io::write_yaml_string(atoms, scaled); },
        py::arg("atoms"), py::arg("scaled") = false,
        "Serialise a structure to a YAML string.");

    m.def(
        "atoms_from_dict",
        [](const Value& structure) { return io::atoms_from_value(structure); },
        py::arg("structure"),
        "Build an :class:`Atoms` from a ``structure:`` mapping given as a dict.");

    m.def(
        "simulation_from_dict",
        [](const Value& simulation) { return io::simulation_from_value(simulation); },
        py::arg("simulation"),
        "Build :class:`SimulationParameters` from a dict.");

    return deck;
}

}  // namespace fastatoms::bindings
