// SPDX-License-Identifier: MIT
#include "bindings.hpp"

#include "fastatoms/calculators/lennard_jones.hpp"
#include "fastatoms/calculators/quantum_espresso.hpp"
#include "fastatoms/calculators/registry.hpp"
#include "fastatoms/calculators/siesta.hpp"
#include "fastatoms/calculators/vasp.hpp"

namespace fastatoms::bindings {
namespace {

/// Names registered from Python, so their factories - which hold Python
/// objects - can be dropped while the interpreter is still alive.
std::vector<std::string>& python_registered_names() {
    static std::vector<std::string> names;
    return names;
}

py::object optional_double(const std::optional<double>& value) {
    return value.has_value() ? py::cast(*value) : py::none();
}

py::object forces_array(const std::vector<double>& forces) {
    if (forces.empty()) {
        return py::none();
    }
    std::vector<double> copy = forces;
    const auto natoms = static_cast<py::ssize_t>(forces.size() / 3);
    return move_to_array(std::move(copy), {natoms, static_cast<py::ssize_t>(3)});
}

py::object stress_array(const std::optional<std::array<double, 6>>& stress) {
    if (!stress.has_value()) {
        return py::none();
    }
    std::vector<double> copy(stress->begin(), stress->end());
    return move_to_array(std::move(copy), {static_cast<py::ssize_t>(6)});
}

/// Accept a calculator, or a name with keyword arguments, wherever a
/// calculator is expected.
std::shared_ptr<Calculator> coerce_calculator(const py::object& value) {
    if (value.is_none()) {
        return nullptr;
    }
    if (py::isinstance<Calculator>(value)) {
        return value.cast<std::shared_ptr<Calculator>>();
    }
    if (py::isinstance<py::str>(value)) {
        return make_calculator(value.cast<std::string>());
    }
    throw py::type_error(
        "expected a Calculator, a calculator name, or None, got " +
        std::string(py::str(py::type::of(value).attr("__name__"))));
}

}  // namespace

std::vector<Property> parse_properties(const py::object& value) {
    if (value.is_none()) {
        return default_properties();
    }
    const auto one = [](const py::handle& item) {
        if (py::isinstance<Property>(item)) {
            return item.cast<Property>();
        }
        return property_from_string(py::cast<std::string>(item));
    };
    if (py::isinstance<py::str>(value) || py::isinstance<Property>(value)) {
        return {one(value)};
    }

    std::vector<Property> properties;
    for (const py::handle item : value) {
        properties.push_back(one(item));
    }
    if (properties.empty()) {
        throw py::value_error("properties must name at least one quantity");
    }
    return properties;
}

void bind_calculators(py::module_& m, py::class_<Atoms>& atoms_class,
                      py::class_<io::InputDeck>& deck_class) {
    // ---------------------------------------------------------------------
    // Properties and results
    // ---------------------------------------------------------------------
    py::enum_<Property>(m, "Property", "A quantity a calculator can produce.")
        .value("energy", Property::Energy)
        .value("free_energy", Property::FreeEnergy)
        .value("forces", Property::Forces)
        .value("stress", Property::Stress)
        .value("charges", Property::Charges)
        .value("magmom", Property::Magmom);

    py::class_<Results>(m, "Results", R"doc(
The outcome of one calculation.

Units follow ASE: eV for energies, eV/angstrom for forces, and eV/angstrom^3
for the stress, which is given in Voigt order ``(xx, yy, zz, yz, xz, xy)``
with the sign convention ``sigma = (1/V) dE/dstrain``.

Absent quantities are ``None`` rather than zero.
)doc")
        .def_property_readonly(
            "energy", [](const Results& self) { return optional_double(self.energy); },
            "Total energy in eV, extrapolated to zero smearing where the code reports it.")
        .def_property_readonly(
            "free_energy",
            [](const Results& self) { return optional_double(self.free_energy); },
            "The energy the forces are consistent with, in eV.")
        .def_property_readonly(
            "forces", [](const Results& self) { return forces_array(self.forces); },
            "Forces in eV/angstrom, shape (N, 3), or None.")
        .def_property_readonly(
            "stress", [](const Results& self) { return stress_array(self.stress); },
            "Stress in eV/angstrom^3, Voigt order, shape (6,), or None.")
        .def_property_readonly(
            "stress_matrix",
            [](const Results& self) -> py::object {
                if (!self.stress.has_value()) {
                    return py::none();
                }
                return py::cast(voigt_to_matrix(*self.stress));
            },
            "The stress as a symmetric 3x3 matrix, or None.")
        .def_property_readonly(
            "pressure",
            [](const Results& self) -> py::object {
                if (!self.stress.has_value()) {
                    return py::none();
                }
                return py::cast(pressure_from_stress(*self.stress));
            },
            "Hydrostatic pressure in eV/angstrom^3, or None.")
        .def_property_readonly(
            "charges",
            [](const Results& self) -> py::object {
                if (self.charges.empty()) {
                    return py::none();
                }
                std::vector<double> copy = self.charges;
                const auto n = static_cast<py::ssize_t>(copy.size());
                return move_to_array(std::move(copy), {n});
            })
        .def_property_readonly(
            "magmom", [](const Results& self) { return optional_double(self.magmom); })
        .def_readonly("extra", &Results::extra,
                      "Code-specific extras as a dict: Fermi level, convergence flags, "
                      "iteration counts.")
        .def("__repr__", [](const Results& self) {
            std::string out = "Results(";
            out += self.energy ? "energy=" + std::to_string(*self.energy) : "energy=None";
            out += self.forces.empty() ? ", forces=None"
                                       : ", forces=(" + std::to_string(self.forces.size() / 3) +
                                             ", 3)";
            out += self.stress ? ", stress=(6,)" : ", stress=None";
            return out + ")";
        });

    m.def("voigt_to_matrix", &voigt_to_matrix, py::arg("voigt"),
          "Expand a Voigt stress vector into a symmetric 3x3 matrix.");
    m.def("matrix_to_voigt", &matrix_to_voigt, py::arg("matrix"),
          "Compress a symmetric 3x3 matrix into Voigt order.");
    m.def("pressure_from_stress", &pressure_from_stress, py::arg("stress"),
          "Hydrostatic pressure: -(1/3) trace(sigma).");

    // ---------------------------------------------------------------------
    // Calculator hierarchy
    // ---------------------------------------------------------------------
    py::class_<Calculator, std::shared_ptr<Calculator>> calculator(m, "Calculator", R"doc(
Base class for anything that evaluates energies, forces and stresses.

A calculator caches its results and only re-runs the underlying engine when
the structure it is given has actually changed, so asking for the energy, then
the forces, then the stress costs one evaluation, not three.
)doc");

    calculator
        .def_property_readonly("name", &Calculator::name)
        .def_property_readonly(
            "implemented_properties",
            [](const Calculator& self) {
                py::list names;
                for (const Property property : self.implemented_properties()) {
                    names.append(py::str(std::string(property_name(property))));
                }
                return names;
            },
            "Names of the quantities this calculator can produce.")
        .def_property(
            "parameters", [](const Calculator& self) { return self.parameters(); },
            [](Calculator& self, const Value& value) { self.set_parameters(value); },
            "Configuration as a plain dict. Assigning to it clears the cached results.")
        .def("set", &Calculator::set_parameter, py::arg("key"), py::arg("value"),
             "Set one parameter, clearing the cached results.")
        .def_property_readonly("results", &Calculator::results,
                               py::return_value_policy::reference_internal,
                               "Results of the most recent evaluation.")
        .def_property_readonly("call_count", &Calculator::call_count,
                               "How many times the engine has actually been invoked.")
        .def("reset", &Calculator::reset, "Discard cached results.")

        .def("get_potential_energy", &Calculator::get_potential_energy, py::arg("atoms"),
             py::arg("force_consistent") = false, "Potential energy in eV.")
        .def(
            "get_forces",
            [](Calculator& self, Atoms& atoms) {
                std::vector<double> forces = self.get_forces(atoms);
                const auto n = static_cast<py::ssize_t>(atoms.size());
                return move_to_array(std::move(forces), {n, static_cast<py::ssize_t>(3)});
            },
            py::arg("atoms"), "Forces in eV/angstrom, shape (N, 3).")
        .def(
            "get_stress",
            [](Calculator& self, Atoms& atoms) {
                const std::array<double, 6> stress = self.get_stress(atoms);
                std::vector<double> copy(stress.begin(), stress.end());
                return move_to_array(std::move(copy), {static_cast<py::ssize_t>(6)});
            },
            py::arg("atoms"), "Stress in eV/angstrom^3, Voigt order, shape (6,).")
        .def(
            "calculate",
            [](Calculator& self, Atoms& atoms, const py::object& properties) {
                self.calculate(atoms, parse_properties(properties));
            },
            py::arg("atoms"), py::arg("properties") = py::none(),
            "Evaluate the named properties, reusing cached results where possible.")
        .def(
            "needs_calculation",
            [](const Calculator& self, const Atoms& atoms, const py::object& properties) {
                return self.needs_calculation(atoms, parse_properties(properties));
            },
            py::arg("atoms"), py::arg("properties") = py::none(),
            "True when `calculate` would invoke the engine.")
        .def("__repr__", &Calculator::to_string);

    py::class_<FileIOCalculator, Calculator, std::shared_ptr<FileIOCalculator>> file_io(
        m, "FileIOCalculator", R"doc(
A calculator that drives an external program through files on disk.

The two halves are usable on their own: :meth:`write_inputs` stages a
directory for a job you will submit yourself, and :meth:`read_results` parses a
run that has already finished elsewhere.
)doc");

    file_io
        .def_property(
            "directory", [](const FileIOCalculator& self) { return self.directory().string(); },
            [](FileIOCalculator& self, const std::string& value) { self.set_directory(value); },
            "Working directory; created if it does not exist.")
        .def_property(
            "command", [](const FileIOCalculator& self) { return self.command(); },
            [](FileIOCalculator& self, const std::string& value) { self.set_command(value); },
            "The shell command that launches the engine, after falling back to the "
            "code's environment variables and then to a built-in default.")
        .def(
            "write_inputs",
            [](FileIOCalculator& self, const Atoms& atoms, const py::object& directory,
               const py::object& properties) {
                const std::filesystem::path target =
                    directory.is_none() ? std::filesystem::path{}
                                        : std::filesystem::path(directory.cast<std::string>());
                self.write_inputs(atoms, target, parse_properties(properties));
            },
            py::arg("atoms"), py::arg("directory") = py::none(),
            py::arg("properties") = py::none(),
            "Write the input files without running anything.")
        .def(
            "read_results",
            [](FileIOCalculator& self, const py::object& directory) {
                const std::filesystem::path source =
                    directory.is_none() ? std::filesystem::path{}
                                        : std::filesystem::path(directory.cast<std::string>());
                self.read_results(source);
            },
            py::arg("directory") = py::none(),
            "Parse a finished run. The structure cache is left invalid, so a later "
            "`calculate` still runs the engine.")
        .def_property_readonly(
            "last_process",
            [](const FileIOCalculator& self) {
                const util::ProcessResult& process = self.last_process();
                py::dict out;
                out["exit_code"] = process.exit_code;
                out["timed_out"] = process.timed_out;
                out["signalled"] = process.signalled;
                out["ok"] = process.ok();
                out["log_path"] = process.log_path.string();
                out["log_tail"] = process.log_tail;
                return out;
            },
            "Outcome of the most recent engine invocation, as a dict.");

    // ---------------------------------------------------------------------
    // Concrete calculators
    // ---------------------------------------------------------------------
    // Keyword arguments become the parameter mapping, so a calculator reads the
    // same whether it is built in Python or from a YAML file.
    const auto init_from_kwargs = [](const py::kwargs& kwargs) {
        return py::cast<Value>(py::dict(kwargs));
    };

    py::class_<VaspCalculator, FileIOCalculator, std::shared_ptr<VaspCalculator>>(m, "Vasp",
                                                                                 R"doc(
VASP, driven through POSCAR / INCAR / KPOINTS / POTCAR and OUTCAR.

    atoms.calc = fa.Vasp(xc="pbe", encut=400, kpts=[4, 4, 4], directory="run")
    energy = atoms.get_potential_energy()

Any keyword that FastAtoms does not interpret itself becomes an INCAR tag, so
``encut=400`` writes ``ENCUT = 400``. POTCAR files are concatenated from
``pp_path`` (or ``$VASP_PP_PATH``) in POSCAR species order.
)doc")
        .def(py::init([&init_from_kwargs](const py::kwargs& kwargs) {
                 return std::make_shared<VaspCalculator>(init_from_kwargs(kwargs));
             }))
        .def("poscar_string",
             [](const VaspCalculator& self, const Atoms& atoms) {
                 return self.poscar_string(atoms, vasp_species_layout(atoms));
             },
             py::arg("atoms"), "The POSCAR this calculator would write.")
        .def(
            "incar_string",
            [](const VaspCalculator& self, const py::object& properties) {
                return self.incar_string(parse_properties(properties));
            },
            py::arg("properties") = py::none(), "The INCAR this calculator would write.")
        .def("kpoints_string", &VaspCalculator::kpoints_string,
             "The KPOINTS file, or an empty string when KSPACING is in use.")
        .def_static("parse_outcar", &VaspCalculator::parse_outcar, py::arg("text"),
                    "Parse the contents of an OUTCAR into a :class:`Results`.");

    m.def(
        "vasp_species_layout",
        [](const Atoms& atoms) {
            const VaspSpeciesLayout layout = vasp_species_layout(atoms);
            py::dict out;
            out["symbols"] = layout.symbols;
            out["counts"] = layout.counts;
            out["order"] = layout.order;
            return out;
        },
        py::arg("atoms"),
        "How the atoms are grouped by species in a POSCAR, and where each row came from.");

    py::class_<QuantumEspressoCalculator, FileIOCalculator,
               std::shared_ptr<QuantumEspressoCalculator>>(m, "Espresso", R"doc(
Quantum ESPRESSO's pw.x.

    atoms.calc = fa.Espresso(
        pseudopotentials={"Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF"},
        ecutwfc=40.0, kpts=[4, 4, 4],
    )

Keywords are routed to the namelist that owns them - ``ecutwfc`` to &SYSTEM,
``conv_thr`` to &ELECTRONS - so a flat call works as well as passing
``system={...}`` and ``electrons={...}`` explicitly.
)doc")
        .def(py::init([&init_from_kwargs](const py::kwargs& kwargs) {
                 return std::make_shared<QuantumEspressoCalculator>(init_from_kwargs(kwargs));
             }))
        .def(
            "input_string",
            [](const QuantumEspressoCalculator& self, const Atoms& atoms,
               const py::object& properties) {
                return self.input_string(atoms, parse_properties(properties));
            },
            py::arg("atoms"), py::arg("properties") = py::none(),
            "The pw.x input deck this calculator would write.")
        .def_static("parse_output_text", &QuantumEspressoCalculator::parse_output_text,
                    py::arg("text"), "Parse the contents of a pw.x output file.")
        .def_static("namelist_for", &QuantumEspressoCalculator::namelist_for, py::arg("keyword"),
                    "Which namelist a pw.x keyword belongs to.");

    m.attr("QuantumEspresso") = m.attr("Espresso");

    py::class_<SiestaCalculator, FileIOCalculator, std::shared_ptr<SiestaCalculator>>(m, "Siesta",
                                                                                     R"doc(
SIESTA, driven through an .fdf deck.

    atoms.calc = fa.Siesta(xc="pbe", mesh_cutoff=250.0, kpts=[4, 4, 4])

Keywords become fdf directives with underscores turned into dots, so
``dm_tolerance=1e-4`` writes ``dm.tolerance 0.0001``. Pseudopotentials are
copied from ``pp_path`` (or ``$SIESTA_PP_PATH``) into the run directory.
)doc")
        .def(py::init([&init_from_kwargs](const py::kwargs& kwargs) {
                 return std::make_shared<SiestaCalculator>(init_from_kwargs(kwargs));
             }))
        .def("fdf_string", &SiestaCalculator::fdf_string, py::arg("atoms"),
             "The .fdf deck this calculator would write.")
        .def_static("parse_output_text", &SiestaCalculator::parse_output_text, py::arg("text"),
                    "Parse SIESTA's text output.");

    py::class_<LennardJonesCalculator, Calculator, std::shared_ptr<LennardJonesCalculator>>(
        m, "LennardJones", R"doc(
Cut-and-shifted Lennard-Jones, evaluated in C++ on the neighbour list.

Needs no external program, which makes it useful for checking a workflow
before pointing it at a DFT code.

    atoms.calc = fa.LennardJones(epsilon=0.0103, sigma=3.40, cutoff=8.5)
)doc")
        .def(py::init([&init_from_kwargs](const py::kwargs& kwargs) {
            return std::make_shared<LennardJonesCalculator>(init_from_kwargs(kwargs));
        }));

    // ---------------------------------------------------------------------
    // Registry
    // ---------------------------------------------------------------------
    m.def(
        "make_calculator",
        [](const py::object& source, const py::object& parameters, const py::kwargs& kwargs) {
            Value settings = parameters.is_none() ? Value(Value::Map{}) : parameters.cast<Value>();
            for (const auto item : kwargs) {
                settings.insert(py::cast<std::string>(item.first), item.second.cast<Value>());
            }
            if (py::isinstance<CalculatorParameters>(source)) {
                CalculatorParameters spec = source.cast<CalculatorParameters>();
                if (settings.size() != 0) {
                    // Explicit arguments win over what the input file said.
                    Value merged = spec.parameters.is_map() ? spec.parameters : Value(Value::Map{});
                    for (const auto& [key, value] : settings.as_map()) {
                        merged.insert(key, value);
                    }
                    spec.parameters = std::move(merged);
                }
                return make_calculator(spec);
            }
            return make_calculator(py::cast<std::string>(source), settings);
        },
        py::arg("name_or_spec"), py::arg("parameters") = py::none(),
        R"doc(
Build a calculator by name, or from an input file's calculator section.

    calc = fa.make_calculator("vasp", encut=400)
    calc = fa.make_calculator(deck.simulation.calculator)

Names are matched ignoring case, hyphens and underscores.
)doc");

    m.def("registered_calculators", &registered_calculators,
          "Names `make_calculator` recognises.");

    m.def(
        "register_calculator",
        [](const std::string& name, const py::object& factory) {
            register_calculator(name, [factory](const Value& parameters) {
                const py::gil_scoped_acquire gil;
                py::object built = factory(parameters);
                return built.cast<std::shared_ptr<Calculator>>();
            });
            python_registered_names().push_back(name);
        },
        py::arg("name"), py::arg("factory"),
        "Make a calculator available to `make_calculator`. `factory` takes a "
        "parameter dict and returns a Calculator.");

    // A factory written in Python keeps a reference to a Python object. The C++
    // registry is a static, so without this its destructor would decref that
    // object from __cxa_finalize, long after the interpreter is gone - a
    // segfault at exit. This capsule is destroyed while the module dict is
    // cleared, which is early enough for the decref to be safe.
    m.add_object("_calculator_registry_cleanup", py::capsule([] {
                     for (const std::string& name : python_registered_names()) {
                         unregister_calculator(name);
                     }
                     python_registered_names().clear();
                 }));

    // ---------------------------------------------------------------------
    // The Atoms side of the link
    // ---------------------------------------------------------------------
    atoms_class
        .def_property(
            "calc",
            [](const Atoms& self) -> py::object {
                if (!self.has_calculator()) {
                    return py::none();
                }
                return py::cast(self.calculator());
            },
            [](Atoms& self, const py::object& value) {
                self.set_calculator(coerce_calculator(value));
            },
            "The attached calculator, or None. Assigning a name such as "
            "``'lennard_jones'`` builds one with default settings.")
        .def_property(
            "calculator",
            [](const Atoms& self) -> py::object {
                if (!self.has_calculator()) {
                    return py::none();
                }
                return py::cast(self.calculator());
            },
            [](Atoms& self, const py::object& value) {
                self.set_calculator(coerce_calculator(value));
            },
            "Alias for :attr:`calc`.")
        .def(
            "set_calculator",
            [](Atoms& self, const py::object& value) {
                self.set_calculator(coerce_calculator(value));
            },
            py::arg("calculator"))
        .def("get_calculator", [](const Atoms& self) -> py::object {
                 if (!self.has_calculator()) {
                     return py::none();
                 }
                 return py::cast(self.calculator());
             })
        .def("get_potential_energy", &Atoms::get_potential_energy,
             py::arg("force_consistent") = false,
             "Potential energy in eV from the attached calculator.")
        .def(
            "get_forces",
            [](Atoms& self) {
                std::vector<double> forces = self.get_forces();
                const auto n = static_cast<py::ssize_t>(self.size());
                return move_to_array(std::move(forces), {n, static_cast<py::ssize_t>(3)});
            },
            "Forces in eV/angstrom, shape (N, 3). Also stored as the 'forces' array.")
        .def(
            "get_stress",
            [](Atoms& self) {
                const std::array<double, 6> stress = self.get_stress();
                std::vector<double> copy(stress.begin(), stress.end());
                return move_to_array(std::move(copy), {static_cast<py::ssize_t>(6)});
            },
            "Stress in eV/angstrom^3, Voigt order (xx, yy, zz, yz, xz, xy).");

    // ---------------------------------------------------------------------
    // The YAML side of the link
    // ---------------------------------------------------------------------
    deck_class.def(
        "make_calculator",
        [](const io::InputDeck& self, const py::kwargs& overrides) {
            CalculatorParameters spec = self.simulation.calculator;
            if (py::len(overrides) != 0) {
                Value merged = spec.parameters.is_map() ? spec.parameters : Value(Value::Map{});
                for (const auto item : overrides) {
                    merged.insert(py::cast<std::string>(item.first), item.second.cast<Value>());
                }
                spec.parameters = std::move(merged);
            }
            return make_calculator(spec);
        },
        R"doc(
Build the calculator described by this file's ``simulation.calculator`` section.

    deck = fa.read("dft.yaml")
    atoms = deck.atoms
    atoms.calc = deck.make_calculator(directory="run-01")

Keyword arguments override what the file said, which is how one input file
drives a series of runs.
)doc");
}

}  // namespace fastatoms::bindings
