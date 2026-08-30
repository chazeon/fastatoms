// SPDX-License-Identifier: MIT
#include "bindings.hpp"

#include "fastatoms/md/registry.hpp"
#include "fastatoms/md/velocity.hpp"

namespace fastatoms::bindings {
namespace {

using namespace fastatoms::md;

}  // namespace

void bind_md(py::module_& m, py::class_<Atoms>& atoms_class,
             py::class_<io::InputDeck>& deck_class) {
    // ---------------------------------------------------------------------
    // Results
    // ---------------------------------------------------------------------
    py::enum_<DynamicsStopReason>(m, "DynamicsStopReason", "Why a run stopped.")
        .value("completed", DynamicsStopReason::Completed)
        .value("interrupted", DynamicsStopReason::Interrupted)
        .value("unstable", DynamicsStopReason::Unstable);

    py::class_<DynamicsStep>(m, "DynamicsStep", "One sampled frame of a run.")
        .def_readonly("step", &DynamicsStep::step)
        .def_readonly("time", &DynamicsStep::time, "Femtoseconds since the run started.")
        .def_readonly("potential_energy", &DynamicsStep::potential_energy, "eV")
        .def_readonly("kinetic_energy", &DynamicsStep::kinetic_energy, "eV")
        .def_readonly("total_energy", &DynamicsStep::total_energy, "eV")
        .def_readonly("conserved_energy", &DynamicsStep::conserved_energy,
                      "Potential plus kinetic plus the extended-system terms of any "
                      "thermostat and barostat. This is the quantity that should stay flat.")
        .def_readonly("temperature", &DynamicsStep::temperature, "K")
        .def_readonly("pressure", &DynamicsStep::pressure,
                      "eV/angstrom^3, including the kinetic term.")
        .def_readonly("volume", &DynamicsStep::volume, "angstrom^3")
        .def("__repr__", [](const DynamicsStep& self) {
            char buffer[192];
            std::snprintf(buffer, sizeof(buffer),
                          "DynamicsStep(step=%d, time=%.1f fs, T=%.1f K, E=%.6f eV)", self.step,
                          self.time, self.temperature, self.total_energy);
            return std::string(buffer);
        });

    py::class_<DynamicsResult>(m, "DynamicsResult", "The outcome of a run.")
        .def_readonly("reason", &DynamicsResult::reason)
        .def_readonly("steps", &DynamicsResult::steps)
        .def_readonly("evaluations", &DynamicsResult::evaluations,
                      "Force evaluations: one per step for NVE and NVT, two for NPT.")
        .def_readonly("time", &DynamicsResult::time, "Femtoseconds simulated.")
        .def_readonly("message", &DynamicsResult::message)
        .def_readonly("history", &DynamicsResult::history,
                      "Frames at the sampling interval, starting with the initial state.")
        .def_property_readonly(
            "completed",
            [](const DynamicsResult& self) {
                return self.reason == DynamicsStopReason::Completed;
            })
        .def("energy_drift_per_atom", &DynamicsResult::energy_drift_per_atom, py::arg("natoms"),
             "Drift of the conserved quantity over the run, in eV per atom.")
        .def("__bool__",
             [](const DynamicsResult& self) {
                 return self.reason == DynamicsStopReason::Completed;
             })
        .def("__repr__", &DynamicsResult::to_string);

    // ---------------------------------------------------------------------
    // Integrators
    // ---------------------------------------------------------------------
    py::class_<Integrator, std::shared_ptr<Integrator>> integrator(m, "Integrator", R"doc(
Base class for the time integration schemes.

An integrator does one plain Newtonian step. Temperature and pressure control
are applied around it by :class:`Dynamics`, so any thermostat combines with any
integrator.
)doc");
    integrator.def_property_readonly("name", &Integrator::name)
        .def_property_readonly("is_symplectic", &Integrator::is_symplectic,
                               "Whether the scheme conserves energy for a sensible timestep.")
        .def("__repr__", &Integrator::to_string);

    py::class_<EulerIntegrator, Integrator, std::shared_ptr<EulerIntegrator>>(m, "Euler", R"doc(
Explicit Euler: `r += v dt`, `v += F/m dt`.

First order and not symplectic - the energy of an oscillator grows every
period, so a run heats up without bound. Here because it is the scheme every
derivation starts from; do not run production dynamics with it.
)doc")
        .def(py::init<>());

    py::class_<VelocityVerletIntegrator, Integrator,
               std::shared_ptr<VelocityVerletIntegrator>>(m, "VelocityVerlet", R"doc(
Velocity Verlet: half kick, drift, half kick.

Second order, symplectic and time reversible, one force evaluation per step,
positions and velocities defined at the same instant. The default.
)doc")
        .def(py::init<>());

    py::class_<LeapfrogIntegrator, Integrator, std::shared_ptr<LeapfrogIntegrator>>(
        m, "Leapfrog", R"doc(
Leapfrog: velocities live half a step away from the positions.

Generates exactly the same positions as velocity Verlet - the two schemes are
algebraically identical - and differs only in where the velocities are defined.
The structure is given the on-step average so that temperatures and thermostats
stay in step with the positions.
)doc")
        .def(py::init<>());

    m.def("make_integrator", &make_integrator, py::arg("name"),
          "Build an integrator by name: euler, velocity_verlet or leapfrog.");
    m.def("integrator_names", &integrator_names);

    // ---------------------------------------------------------------------
    // Thermostats
    // ---------------------------------------------------------------------
    py::class_<Thermostat, std::shared_ptr<Thermostat>> thermostat(m, "Thermostat", R"doc(
Base class for temperature control.

Applied as an operator for half a timestep on each side of the integrator's
step, so it costs no extra force evaluations: it only ever rescales velocities
and advances its own variables.
)doc");
    thermostat.def_property_readonly("name", &Thermostat::name)
        .def_property_readonly("parameters", [](const Thermostat& self) { return self.parameters(); })
        .def_property_readonly("time_constant", &Thermostat::time_constant_fs,
                               "Coupling time constant in femtoseconds.")
        .def_property_readonly("samples_canonical", &Thermostat::samples_canonical,
                               "Whether it samples the canonical distribution rather than "
                               "merely holding the mean temperature.")
        .def("__repr__", &Thermostat::to_string);

    const auto from_kwargs = [](const py::kwargs& kwargs) {
        return py::cast<Value>(py::dict(kwargs));
    };

    py::class_<BerendsenThermostat, Thermostat, std::shared_ptr<BerendsenThermostat>>(
        m, "BerendsenThermostat", R"doc(
Berendsen weak coupling: rescale velocities towards the target.

Cheap and very stable, and the usual way to bring a structure to temperature.
It does **not** sample the canonical distribution - it suppresses the kinetic
energy fluctuations, and over long runs drains energy into the centre-of-mass
motion. Equilibrate with it, then switch to Nose-Hoover.
)doc")
        .def(py::init([&from_kwargs](const py::kwargs& kwargs) {
            return std::make_shared<BerendsenThermostat>(from_kwargs(kwargs));
        }));

    py::class_<LangevinThermostat, Thermostat, std::shared_ptr<LangevinThermostat>>(
        m, "Langevin", R"doc(
Langevin dynamics: friction plus a matching random force.

    md.thermostat = fa.Langevin(friction=0.01, seed=1)   # 1/fs

The friction drains energy and the noise puts it back in the ratio the
fluctuation-dissipation theorem fixes, so the stationary distribution is
canonical - and unlike a single Nose-Hoover variable it is ergodic by
construction, because every degree of freedom is thermostatted independently.

The Ornstein-Uhlenbeck half of the step is solved exactly rather than
discretised, so all the timestep error comes from the Verlet part. Being
stochastic it conserves nothing of its own, but it tracks the heat exchanged
with the bath, so `potential + kinetic + heat` stays flat and can be checked
for drift like the deterministic thermostats.
)doc")
        .def(py::init([&from_kwargs](const py::kwargs& kwargs) {
                 return std::make_shared<LangevinThermostat>(from_kwargs(kwargs));
             }))
        .def_property_readonly("friction", &LangevinThermostat::friction,
                               "The friction coefficient in 1/fs.")
        .def_property_readonly("heat", &LangevinThermostat::heat,
                               "Energy handed to the bath so far, in eV.");

    py::class_<NoseThermostat, Thermostat, std::shared_ptr<NoseThermostat>>(m, "Nose", R"doc(
Nose's original extended-system thermostat.

Couples the system to one extra degree of freedom `s` with momentum `p_s`,
giving a Hamiltonian whose microcanonical dynamics reproduce the canonical
distribution. Integrated in real time through Hoover's change of variables,
keeping Nose's own count of degrees of freedom, `g = N_f + 1` - the extra one
comes from the time scaling in the original derivation. That count is right for
sampling in virtual time and slightly off in real time, which is exactly why
Hoover's version superseded it. Prefer :class:`NoseHoover` for production work.
)doc")
        .def(py::init([&from_kwargs](const py::kwargs& kwargs) {
                 return std::make_shared<NoseThermostat>(from_kwargs(kwargs));
             }))
        .def_property_readonly("s", &NoseThermostat::s, "The extended coordinate.")
        .def_property_readonly("friction", &NoseThermostat::friction, "xi = p_s / Q.");

    py::class_<NoseHooverThermostat, Thermostat, std::shared_ptr<NoseHooverThermostat>>(
        m, "NoseHoover", R"doc(
Nose-Hoover, optionally as a chain.

The real-time formulation with `g = N_f`, integrated with the symmetric Trotter
splitting of Martyna, Tuckerman and Klein, so the extended-system energy is
conserved to second order.

A single Nose-Hoover variable is non-ergodic for small or stiff systems - a lone
harmonic oscillator never explores its canonical distribution. `chain_length`
above one thermostats the thermostat, and three or four fixes it.
)doc")
        .def(py::init([&from_kwargs](const py::kwargs& kwargs) {
                 return std::make_shared<NoseHooverThermostat>(from_kwargs(kwargs));
             }))
        .def_property_readonly("chain_length", &NoseHooverThermostat::chain_length)
        .def_property_readonly(
            "friction",
            [](const NoseHooverThermostat& self) {
                std::vector<double> values = self.friction();
                const auto n = static_cast<py::ssize_t>(values.size());
                return move_to_array(std::move(values), {n});
            },
            "The friction variables, innermost first.");

    m.def("make_thermostat", &make_thermostat, py::arg("name"),
          py::arg("parameters") = Value{},
          "Build a thermostat by name: berendsen, langevin, nose or nose_hoover. An "
          "empty name "
          "gives None, i.e. NVE.");
    m.def("thermostat_names", &thermostat_names);

    // ---------------------------------------------------------------------
    // Barostats
    // ---------------------------------------------------------------------
    py::class_<Barostat, std::shared_ptr<Barostat>> barostat(m, "Barostat", R"doc(
Base class for pressure control.

Changes the simulation box and carries the atoms with it. Because the box moves
between force evaluations, an NPT step costs two force calls rather than one.
)doc");
    barostat.def_property_readonly("name", &Barostat::name)
        .def_property_readonly("parameters", [](const Barostat& self) { return self.parameters(); })
        .def_property_readonly("time_constant", &Barostat::time_constant_fs,
                               "Coupling time constant in femtoseconds.")
        .def_property_readonly("is_isotropic", &Barostat::is_isotropic,
                               "Whether only the volume may change, not the cell shape.")
        .def_property_readonly("samples_isobaric", &Barostat::samples_isobaric)
        .def("__repr__", &Barostat::to_string);

    py::class_<BerendsenBarostat, Barostat, std::shared_ptr<BerendsenBarostat>>(
        m, "BerendsenBarostat", R"doc(
Berendsen weak coupling: scale the box towards the target pressure.

Robust and forgiving, and the usual way to bring a cell to the right density.
Like the Berendsen thermostat it does **not** sample the correct distribution -
the volume fluctuations come out too small - so switch to
:class:`ParrinelloRahman` for production NPT.
)doc")
        .def(py::init([&from_kwargs](const py::kwargs& kwargs) {
                 return std::make_shared<BerendsenBarostat>(from_kwargs(kwargs));
             }))
        .def_property_readonly("compressibility", &BerendsenBarostat::compressibility,
                               "Isothermal compressibility in 1/GPa.");

    py::class_<ParrinelloRahmanBarostat, Barostat, std::shared_ptr<ParrinelloRahmanBarostat>>(
        m, "ParrinelloRahman", R"doc(
Parrinello-Rahman, in the strain-rate formulation.

The cell carries a velocity of its own, driven by the imbalance between the
internal and applied stress, and the atoms and their velocities transform
affinely with it. The volume is a genuine dynamical variable, so its
fluctuations are physical.

`isotropic=True` (the default) changes the size but not the shape; set it false
for the full cell dynamics. Always pair this with a thermostat - the cell degree
of freedom heats up otherwise.
)doc")
        .def(py::init([&from_kwargs](const py::kwargs& kwargs) {
                 return std::make_shared<ParrinelloRahmanBarostat>(from_kwargs(kwargs));
             }))
        .def_property_readonly("strain_rate", [](const ParrinelloRahmanBarostat& self) {
            return self.strain_rate();
        });

    m.def("make_barostat", &make_barostat, py::arg("name"), py::arg("parameters") = Value{},
          "Build a barostat by name: berendsen or parrinello_rahman. An empty name gives "
          "None, i.e. a fixed cell.");
    m.def("barostat_names", &barostat_names);

    // ---------------------------------------------------------------------
    // The loop
    // ---------------------------------------------------------------------
    py::class_<Dynamics, std::shared_ptr<Dynamics>>(m, "Dynamics", R"doc(
Runs molecular dynamics on a structure.

    atoms.calc = fa.LennardJones()
    md = fa.Dynamics(timestep=2.0, temperature=300.0, thermostat="nose_hoover")
    result = md.run(atoms, 1000)

The physics of one step is composed from three pluggable pieces: an
:class:`Integrator`, an optional :class:`Thermostat` for NVT, and an optional
:class:`Barostat` for NPT. They are applied in a symmetric Trotter splitting, so
an NVE or NVT step costs exactly one force evaluation and NPT costs two.
)doc")
        .def(py::init([&from_kwargs](const py::kwargs& kwargs) {
            return std::make_shared<Dynamics>(from_kwargs(kwargs));
        }))

        .def("run", &Dynamics::run, py::arg("atoms"), py::arg("steps") = -1,
             "Run `steps` steps, or the configured number when omitted.")
        .def("step", &Dynamics::step, py::arg("atoms"),
             "Advance by a single timestep. Call `prepare` first.")
        .def("prepare", &Dynamics::prepare, py::arg("atoms"),
             "Set up the context, the pieces and the initial velocities.")
        .def("sample", &Dynamics::sample, py::arg("atoms"),
             "The current state as a log record, without advancing anything.")

        .def_property(
            "integrator", &Dynamics::integrator,
            [](Dynamics& self, const py::object& value) {
                if (value.is_none()) {
                    self.set_integrator(nullptr);
                } else if (py::isinstance<py::str>(value)) {
                    self.set_integrator(make_integrator(value.cast<std::string>()));
                } else {
                    self.set_integrator(value.cast<std::shared_ptr<Integrator>>());
                }
            })
        .def_property(
            "thermostat", &Dynamics::thermostat,
            [](Dynamics& self, const py::object& value) {
                if (value.is_none()) {
                    self.set_thermostat(nullptr);
                } else if (py::isinstance<py::str>(value)) {
                    self.set_thermostat(make_thermostat(value.cast<std::string>()));
                } else {
                    self.set_thermostat(value.cast<std::shared_ptr<Thermostat>>());
                }
            })
        .def_property(
            "constraints", &Dynamics::constraints,
            [](Dynamics& self, const py::object& value) {
                // `None` means "no constraints", including the ones the
                // calculator would otherwise have supplied, which is how a
                // rigid model is run flexibly for comparison.
                self.set_constraints(value.is_none()
                                         ? nullptr
                                         : value.cast<std::shared_ptr<Constraints>>());
            },
            "Rigid distances to enforce. Left alone, these come from the calculator: a "
            "water model knows its own geometry. Each one removes a degree of freedom "
            "from the temperature.")
        .def_property(
            "barostat", &Dynamics::barostat,
            [](Dynamics& self, const py::object& value) {
                if (value.is_none()) {
                    self.set_barostat(nullptr);
                } else if (py::isinstance<py::str>(value)) {
                    self.set_barostat(make_barostat(value.cast<std::string>()));
                } else {
                    self.set_barostat(value.cast<std::shared_ptr<Barostat>>());
                }
            })
        .def_property_readonly("ensemble", &Dynamics::ensemble,
                               "'nve', 'nvt' or 'npt', from what is attached.")

        .def_property(
            "parameters", [](const Dynamics& self) { return self.parameters(); },
            [](Dynamics& self, const Value& value) { self.set_parameters(value); })
        .def("set", &Dynamics::set_parameter, py::arg("key"), py::arg("value"))
        .def_property_readonly("timestep", &Dynamics::timestep_fs, "Femtoseconds.")
        .def_property_readonly("temperature", &Dynamics::temperature, "Target, in kelvin.")
        .def_property_readonly("pressure", &Dynamics::pressure,
                               "Target, in eV/angstrom^3.")
        .def_property_readonly("time", &Dynamics::time_fs,
                               "Femtoseconds simulated since `prepare`.")

        .def(
            "attach",
            [](Dynamics& self, const py::object& callback, int interval) {
                self.attach(
                    [callback](const Atoms& atoms, const DynamicsStep& step) {
                        py::gil_scoped_acquire gil;
                        const py::object keep_going = callback(atoms, step);
                        return keep_going.is_none() || keep_going.cast<bool>();
                    },
                    interval);
            },
            py::arg("callback"), py::arg("interval") = 1,
            "Call `callback(atoms, step)` every `interval` steps. Returning False stops "
            "the run.")
        .def("clear_callbacks", &Dynamics::clear_callbacks)
        .def("__repr__", &Dynamics::to_string);

    m.def(
        "make_dynamics",
        [](const py::object& source, const py::kwargs& kwargs) {
            Value settings = py::cast<Value>(py::dict(kwargs));
            if (py::isinstance<MolecularDynamicsParameters>(source)) {
                MolecularDynamicsParameters spec =
                    source.cast<MolecularDynamicsParameters>();
                Value merged = spec.parameters.is_map() ? spec.parameters : Value(Value::Map{});
                for (const auto& [key, value] : settings.as_map()) {
                    merged.insert(key, value);
                }
                spec.parameters = std::move(merged);
                return make_dynamics(spec);
            }
            if (py::isinstance<SimulationParameters>(source)) {
                return make_dynamics(source.cast<SimulationParameters>());
            }
            if (!source.is_none()) {
                Value merged = source.cast<Value>();
                for (const auto& [key, value] : settings.as_map()) {
                    merged.insert(key, value);
                }
                return make_dynamics(merged);
            }
            return make_dynamics(settings);
        },
        py::arg("source") = py::none(),
        "Build a run from a parameter dict, or from an input file's md or simulation "
        "section.");

    m.def(
        "run_md",
        [](Atoms& atoms, int steps, const py::kwargs& kwargs) {
            auto dynamics = std::make_shared<Dynamics>(py::cast<Value>(py::dict(kwargs)));
            return dynamics->run(atoms, steps);
        },
        py::arg("atoms"), py::arg("steps"),
        R"doc(
Run molecular dynamics in one call.

    result = fa.run_md(atoms, 1000, timestep=2.0, temperature=300.0,
                       thermostat="nose_hoover")
)doc");

    // ---------------------------------------------------------------------
    // Velocities and observables
    // ---------------------------------------------------------------------
    m.def("maxwell_boltzmann", &set_maxwell_boltzmann_velocities, py::arg("atoms"),
          py::arg("temperature"), py::arg("seed") = 0,
          py::arg("remove_center_of_mass") = true, py::arg("exact") = true,
          R"doc(
Draw velocities from a Maxwell-Boltzmann distribution, in place.

With `exact` the result is rescaled so the instantaneous temperature is exactly
what was asked for; set it false for a genuine draw. A non-zero `seed` makes the
draw reproducible.
)doc");

    m.def("rescale_to_temperature", &rescale_to_temperature, py::arg("atoms"),
          py::arg("temperature"), py::arg("degrees_of_freedom"),
          "Rescale velocities so the instantaneous temperature is exactly the target.");

    m.def("momentum_degrees_of_freedom", &momentum_degrees_of_freedom, py::arg("atoms"),
          py::arg("remove_center_of_mass") = true,
          "3N, less three when the centre-of-mass motion is removed.");

    m.def("kinetic_stress", &kinetic_stress, py::arg("atoms"),
          "The ideal-gas contribution to the stress, in eV/angstrom^3.");
    m.def("total_stress", &total_stress, py::arg("atoms"),
          "Potential stress from the calculator plus the kinetic term.");
    m.def("instantaneous_pressure", &instantaneous_pressure, py::arg("atoms"),
          "Pressure in eV/angstrom^3, including the kinetic term.");

    // ---------------------------------------------------------------------
    // Trajectories
    // ---------------------------------------------------------------------
    py::class_<TrajectoryWriter>(m, "TrajectoryWriter", R"doc(
Writes frames in extended XYZ, the format ASE, OVITO and VMD all read.
)doc")
        .def(py::init([](const std::string& path, int interval, bool velocities, bool forces,
                         bool append) {
                 TrajectoryOptions options;
                 options.interval = interval;
                 options.velocities = velocities;
                 options.forces = forces;
                 options.append = append;
                 return std::make_unique<TrajectoryWriter>(path, options);
             }),
             py::arg("path"), py::arg("interval") = 1, py::arg("velocities") = true,
             py::arg("forces") = true, py::arg("append") = false)
        .def("write", &TrajectoryWriter::write, py::arg("atoms"), py::arg("step"))
        .def_property_readonly("frames_written", &TrajectoryWriter::frames_written)
        .def_property_readonly(
            "path", [](const TrajectoryWriter& self) { return self.path().string(); });

    // ---------------------------------------------------------------------
    // The Atoms and input-file sides
    // ---------------------------------------------------------------------
    atoms_class.def(
        "run_md",
        [](Atoms& self, int steps, const py::kwargs& kwargs) {
            auto dynamics = std::make_shared<Dynamics>(py::cast<Value>(py::dict(kwargs)));
            return dynamics->run(self, steps);
        },
        py::arg("steps"),
        "Run dynamics on this structure: `atoms.run_md(1000, temperature=300.0)`.");

    deck_class.def(
        "make_dynamics",
        [](const io::InputDeck& self, const py::kwargs& overrides) {
            SimulationParameters simulation = self.simulation;
            Value merged = simulation.md.parameters.is_map() ? simulation.md.parameters
                                                             : Value(Value::Map{});
            for (const auto item : overrides) {
                merged.insert(py::cast<std::string>(item.first), item.second.cast<Value>());
            }
            simulation.md.parameters = std::move(merged);
            return make_dynamics(simulation);
        },
        R"doc(
Build the run described by this file's ``simulation.md`` and ``simulation.output``
sections.

    deck = fa.read("md.yaml")
    atoms = deck.atoms
    atoms.calc = deck.make_calculator()
    deck.make_dynamics().run(atoms)

Keyword arguments override what the file said.
)doc");
}

}  // namespace fastatoms::bindings
