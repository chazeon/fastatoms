// SPDX-License-Identifier: MIT
#include "bindings.hpp"

#include <optional>

#include "fastatoms/calculators/single_point.hpp"

namespace fastatoms::bindings {

py::class_<io::InputDeck> bind_io(py::module_& m) {
    py::enum_<Task>(m, "Task", "What an input file asks the run to do.")
        .value("single_point", Task::SinglePoint)
        .value("md", Task::MolecularDynamics)
        .value("optimize", Task::Optimization)
        .value("phonon", Task::Phonon)
        .value("neb", Task::NEB)
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

    py::class_<BandParameters>(m, "BandParameters",
                               "Settings from the ``simulation.neb`` section.")
        .def(py::init<>())
        .def_readwrite("initial", &BandParameters::initial, "Path to the first endpoint.")
        .def_readwrite("final", &BandParameters::final, "Path to the last endpoint.")
        .def_readwrite("band", &BandParameters::band,
                       "Path to a prepared multi-frame band, which overrides the endpoints.")
        .def_readwrite("initial_structure", &BandParameters::initial_structure,
                       "The first endpoint written inline, as a dict.")
        .def_readwrite("final_structure", &BandParameters::final_structure,
                       "The last endpoint written inline, as a dict.")
        .def_readwrite("images", &BandParameters::images, "Images, endpoints included.")
        .def_readwrite("spring_constant", &BandParameters::spring_constant)
        .def_readwrite("climbing", &BandParameters::climbing)
        .def_readwrite("climb_threshold", &BandParameters::climb_threshold)
        .def_readwrite("tangent", &BandParameters::tangent)
        .def_readwrite("variable_springs", &BandParameters::variable_springs)
        .def_readwrite("spring_softening", &BandParameters::spring_softening)
        .def_readwrite("minimum_image", &BandParameters::minimum_image)
        .def_readwrite("fixed_atoms", &BandParameters::fixed_atoms)
        .def_readwrite("automatic", &BandParameters::automatic, "Use AutoNEB.")
        .def_readwrite("max_images", &BandParameters::max_images)
        .def_readwrite("coarse_fmax", &BandParameters::coarse_fmax)
        .def_readwrite("insert", &BandParameters::insert)
        .def_readwrite("prune", &BandParameters::prune)
        .def_readwrite("prune_tolerance", &BandParameters::prune_tolerance)
        .def_readwrite("redistribute", &BandParameters::redistribute)
        .def_readwrite("parameters", &BandParameters::parameters,
                       "The unparsed ``neb:`` subtree.")
        .def("__repr__", [](const BandParameters& self) {
            return "BandParameters(images=" + std::to_string(self.images) +
                   ", climbing=" + (self.climbing ? "True" : "False") + ")";
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
        .def_readwrite("neb", &SimulationParameters::neb)
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


    // -----------------------------------------------------------------
    // Structures. Extended XYZ is the default format; the extension picks.
    // -----------------------------------------------------------------
    m.def(
        "read",
        [](const std::string& path, const py::object& index, bool results) -> py::object {
            const auto attach = [results](Atoms& atoms) {
                if (results) {
                    attach_stored_results(atoms);
                }
            };

            // A slice - `index=":"` or `index=slice(None)` - asks for frames
            // rather than a frame, and gets a list back. One index gets one
            // structure, which is what the common case wants unwrapped.
            std::optional<py::slice> slice;
            if (py::isinstance<py::slice>(index)) {
                slice = index.cast<py::slice>();
            } else if (py::isinstance<py::str>(index)) {
                const std::string text = index.cast<std::string>();
                if (text.find(':') == std::string::npos) {
                    throw py::value_error("read(): index='" + text +
                                          "' is not a slice; write ':' for every frame, "
                                          "'-10:' for the last ten, or pass a number");
                }
                slice = py::module_::import("builtins")
                            .attr("eval")(py::str("__import__('numpy').s_[" + text + "]"))
                            .cast<py::slice>();
            }

            if (slice) {
                std::vector<Atoms> frames = io::read_frames(path);
                std::size_t start = 0;
                std::size_t stop = 0;
                std::size_t step = 0;
                std::size_t length = 0;
                if (!slice->compute(frames.size(), &start, &stop, &step, &length)) {
                    throw py::error_already_set();
                }
                py::list out;
                for (std::size_t k = 0; k < length; ++k) {
                    Atoms& frame = frames[start + k * step];
                    attach(frame);
                    out.append(frame);
                }
                return out;
            }

            Atoms atoms = io::read(path, index.cast<std::ptrdiff_t>());
            attach(atoms);
            return py::cast(std::move(atoms));
        },
        py::arg("path"), py::arg("index") = -1, py::kw_only(), py::arg("results") = true,
        R"doc(
Read a structure.

    atoms  = fa.read("relaxed.extxyz")        # the last frame
    first  = fa.read("md.extxyz", index=0)    # the first
    every  = fa.read("md.extxyz", index=":")  # a list of all of them
    last10 = fa.read("md.extxyz", index="-10:")

Extended XYZ unless the extension says YAML, and a negative index counts from
the end - so the default is the final frame of a trajectory, which is the
relaxed geometry or the last configuration of a run.

To load an input file describing a *run*, rather than a structure, use
:func:`read_input`: extended XYZ has nowhere to say which calculator to attach.

A frame carrying an energy, forces or a stress comes back with a
:class:`SinglePointCalculator` holding them, so ``atoms.get_potential_energy()``
returns what the file recorded rather than raising. Pass ``results=False`` to
get the bare structure.
)doc");

    m.def(
        "read_frames",
        [](const std::string& path, bool results) {
            std::vector<Atoms> frames = io::read_frames(path);
            if (results) {
                for (Atoms& frame : frames) {
                    attach_stored_results(frame);
                }
            }
            return frames;
        },
        py::arg("path"), py::kw_only(), py::arg("results") = true,
        "Every frame of a trajectory, as a list of :class:`Atoms`, each carrying "
        "whatever results the file recorded.");

    m.def(
        "read_input", [](const std::string& path) { return io::read_yaml_file(path); },
        py::arg("path"), "Parse a YAML input file into an :class:`InputDeck`.");

    m.def(
        "read_input_string", [](const std::string& text) { return io::read_yaml_string(text); },
        py::arg("text"), "Parse a YAML input document held in a string.");

    m.def(
        "read_extxyz",
        [](const std::string& path, std::ptrdiff_t index) {
            return io::read_extxyz(path, index);
        },
        py::arg("path"), py::arg("index") = -1,
        "Read a frame from an extended XYZ file, whatever the extension says.");

    m.def("read_extxyz_frames", &io::read_extxyz_frames, py::arg("path"),
          "Every frame of an extended XYZ file.");

    m.def(
        "parse_extxyz",
        [](const std::string& text, std::ptrdiff_t index) {
            return io::parse_extxyz(text, index);
        },
        py::arg("text"), py::arg("index") = -1,
        "Read a frame from extended XYZ text held in a string.");

    m.def("parse_extxyz_frames", &io::parse_extxyz_frames, py::arg("text"),
          "Every frame in extended XYZ text held in a string.");

    m.def(
        "read_atoms",
        [](const std::string& path) { return io::read(path); }, py::arg("path"),
        "The structure from a file, whatever its format. An alias for :func:`read`.");

    m.def(
        "parse_yaml", [](const std::string& text) { return io::parse_yaml_string(text); },
        py::arg("text"), "Parse arbitrary YAML into plain Python data.");

    m.def(
        "parse_yaml_file", [](const std::string& path) { return io::parse_yaml_file(path); },
        py::arg("path"), "Parse an arbitrary YAML file into plain Python data.");

    // Keyword arguments become the writer's options, so a call reads the same
    // as the file it produces.
    const auto extxyz_options = [](const py::kwargs& kwargs) {
        io::ExtxyzOptions options;
        for (const auto item : kwargs) {
            const std::string key = py::cast<std::string>(item.first);
            const py::handle value = item.second;
            if (key == "append") {
                options.append = value.cast<bool>();
            } else if (key == "velocities") {
                options.velocities = value.cast<bool>();
            } else if (key == "forces") {
                options.forces = value.cast<bool>();
            } else if (key == "masses") {
                options.masses = value.cast<bool>();
            } else if (key == "charges") {
                options.charges = value.cast<bool>();
            } else if (key == "precision") {
                options.precision = value.cast<int>();
            } else if (key == "arrays") {
                options.arrays = value.cast<std::vector<std::string>>();
            } else if (key == "info") {
                options.info = value.cast<Value>();
            } else {
                throw std::invalid_argument("write: unknown option '" + key + "'");
            }
        }
        return options;
    };

    m.def(
        "write",
        [extxyz_options](const std::string& path, const py::object& structure,
                         const py::kwargs& kwargs) {
            const io::ExtxyzOptions options = extxyz_options(kwargs);
            if (py::isinstance<Atoms>(structure)) {
                io::write(path, structure.cast<Atoms>(), options);
            } else {
                io::write(path, structure.cast<std::vector<Atoms>>(), options);
            }
        },
        py::arg("path"), py::arg("atoms"),
        R"doc(
Write a structure, or a list of them, to a file.

    fa.write("relaxed.extxyz", atoms)
    fa.write("md.extxyz", frames)                 # a whole trajectory
    fa.write("md.extxyz", atoms, append=True)     # one more frame

Extended XYZ unless the extension says YAML. Real numbers are written in the
shortest form that reads back as exactly the same double; pass ``precision`` to
trade that for a smaller file.

Options: ``append``, ``velocities``, ``forces``, ``masses``, ``charges``,
``arrays``, ``precision``, ``info``.
)doc");

    m.def(
        "append",
        [extxyz_options](const std::string& path, const Atoms& atoms,
                         const py::kwargs& kwargs) {
            io::append(path, atoms, extxyz_options(kwargs));
        },
        py::arg("path"), py::arg("atoms"),
        "Append one frame to a trajectory, creating the file if needed.");

    m.def(
        "to_extxyz",
        [extxyz_options](const Atoms& atoms, const py::kwargs& kwargs) {
            return io::extxyz_string(atoms, extxyz_options(kwargs));
        },
        py::arg("atoms"), "Serialise a structure to an extended XYZ string.");

    m.def(
        "write_yaml",
        [](const std::string& path, const Atoms& atoms, bool scaled) {
            io::write_yaml_file(path, atoms, scaled);
        },
        py::arg("path"), py::arg("atoms"), py::arg("scaled") = false,
        "Write a structure to a YAML file, whatever the extension says.");

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
