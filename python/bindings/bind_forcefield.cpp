// SPDX-License-Identifier: MIT
#include "bindings.hpp"

#include <memory>
#include <string>
#include <vector>

#include "fastatoms/forcefield/ewald.hpp"
#include "fastatoms/forcefield/forcefield.hpp"
#include "fastatoms/forcefield/topology.hpp"
#include "fastatoms/forcefield/water.hpp"
#include "fastatoms/md/constraints.hpp"

namespace fastatoms::bindings {
namespace {

using namespace fastatoms::ff;

Value from_kwargs(const py::kwargs& kwargs) { return py::cast<Value>(py::dict(kwargs)); }

py::array_t<double> as_array(const std::vector<double>& values) {
    std::vector<double> copy = values;
    const auto n = static_cast<py::ssize_t>(copy.size());
    return move_to_array(std::move(copy), {n});
}

}  // namespace

void bind_forcefield(py::module_& m) {
    // ---------------------------------------------------------------------
    // Constraints
    // ---------------------------------------------------------------------
    py::class_<DistanceConstraint>(m, "DistanceConstraint",
                                   "One rigid distance, `|r_i - r_j| = length`.")
        .def(py::init([](std::size_t i, std::size_t j, double length) {
                 return DistanceConstraint{i, j, length};
             }),
             py::arg("i"), py::arg("j"), py::arg("length"))
        .def_readwrite("i", &DistanceConstraint::i)
        .def_readwrite("j", &DistanceConstraint::j)
        .def_readwrite("length", &DistanceConstraint::length, "Angstrom.")
        .def("__repr__", [](const DistanceConstraint& self) {
            return "DistanceConstraint(" + std::to_string(self.i) + ", " +
                   std::to_string(self.j) + ", " + std::to_string(self.length) + ")";
        });

    py::class_<md::ConstraintReport>(m, "ConstraintReport", "What one constraint solve did.")
        .def_readonly("iterations", &md::ConstraintReport::iterations)
        .def_readonly("residual", &md::ConstraintReport::residual)
        .def_readonly("converged", &md::ConstraintReport::converged)
        .def("__repr__", &md::ConstraintReport::to_string);

    py::class_<md::Constraints, std::shared_ptr<md::Constraints>>(m, "Constraints", R"doc(
Holonomic distance constraints, enforced by SHAKE and RATTLE.

Freezing the O-H stretch and the H-O-H bend is what lets a water simulation
take a 2 fs step instead of half a femtosecond. SHAKE fixes the positions,
RATTLE then projects the velocities so the constrained distances have zero rate
of change as well.

    constraints = fa.Constraints(calc.constraints)
    constraints.max_deviation(atoms)     # relative, should be ~1e-10

`Dynamics` picks these up from the calculator on its own, so this class is
usually only touched to inspect what is being enforced.
)doc")
        .def(py::init([](const std::vector<DistanceConstraint>& bonds, double tolerance,
                         int max_iterations) {
                 return std::make_shared<md::Constraints>(bonds, tolerance, max_iterations);
             }),
             py::arg("bonds"), py::arg("tolerance") = 1.0e-10,
             py::arg("max_iterations") = 500)
        .def_property_readonly("bonds", &md::Constraints::bonds)
        .def_property("tolerance", &md::Constraints::tolerance, &md::Constraints::set_tolerance)
        .def_property("max_iterations", &md::Constraints::max_iterations,
                      &md::Constraints::set_max_iterations)
        .def_property_readonly("removed_degrees_of_freedom",
                               &md::Constraints::removed_degrees_of_freedom,
                               "One per constraint; the temperature has to allow for them.")
        .def("enforce", &md::Constraints::enforce, py::arg("atoms"),
             "Move the atoms onto the constraint surface.")
        .def("max_deviation", &md::Constraints::max_deviation, py::arg("atoms"),
             "Largest relative violation of a constrained distance.")
        .def(
            "max_velocity_deviation",
            [](const md::Constraints& self, const Atoms& atoms, const py::object& velocities) {
                const std::vector<double> values = as_flat_doubles(
                    velocities, static_cast<py::ssize_t>(atoms.size()), 3, "velocities");
                return self.max_velocity_deviation(atoms, values);
            },
            py::arg("atoms"), py::arg("velocities"),
            "Largest rate of change left in a constrained distance.")
        .def("__len__", &md::Constraints::size)
        .def("__repr__", &md::Constraints::to_string);

    // ---------------------------------------------------------------------
    // Topology
    // ---------------------------------------------------------------------
    py::class_<Molecule>(m, "Molecule", "A group of atoms treated as one molecule.")
        .def_readonly("atoms", &Molecule::atoms)
        .def_readonly("name", &Molecule::name)
        .def("__len__", [](const Molecule& self) { return self.atoms.size(); })
        .def("__repr__", [](const Molecule& self) {
            return "Molecule('" + self.name + "', " + std::to_string(self.atoms.size()) +
                   " atoms)";
        });

    py::class_<Topology>(m, "Topology", R"doc(
Charges, dispersion parameters, molecules, exclusions and constraints.

Everything a classical energy expression needs that the structure itself does
not carry. Build one with :func:`water_topology`, or assemble it by hand for a
system the built-in models do not cover.
)doc")
        .def(py::init<std::size_t>(), py::arg("natoms"))
        .def_property_readonly("charges",
                               [](const Topology& self) { return as_array(self.charges()); })
        .def_property_readonly("epsilons",
                               [](const Topology& self) { return as_array(self.epsilons()); })
        .def_property_readonly("sigmas",
                               [](const Topology& self) { return as_array(self.sigmas()); })
        .def_property_readonly("molecules", &Topology::molecules)
        .def_property_readonly("constraints", &Topology::constraints)
        .def_property_readonly("net_charge", &Topology::net_charge)
        .def("set_charge", &Topology::set_charge, py::arg("atom"), py::arg("charge"))
        .def(
            "set_lennard_jones",
            [](Topology& self, std::size_t atom, double epsilon, double sigma) {
                self.set_lennard_jones(atom, LennardJonesSite{epsilon, sigma});
            },
            py::arg("atom"), py::arg("epsilon"), py::arg("sigma"))
        .def("exclude", &Topology::exclude, py::arg("i"), py::arg("j"),
             "Stop two atoms from interacting through the non-bonded terms.")
        .def("excluded", &Topology::excluded, py::arg("i"), py::arg("j"))
        .def("exclude_within_molecules", &Topology::exclude_within_molecules)
        .def("excluded_pairs", &Topology::excluded_pairs)
        .def(
            "add_constraint",
            [](Topology& self, std::size_t i, std::size_t j, double length) {
                self.add_constraint(DistanceConstraint{i, j, length});
            },
            py::arg("i"), py::arg("j"), py::arg("length"))
        .def("__len__", &Topology::size)
        .def("__repr__", &Topology::to_string);

    m.def("detect_molecules", &detect_molecules, py::arg("atoms"), py::arg("scale") = 1.2,
          "Group atoms into molecules by distance-based bonding.");

    // ---------------------------------------------------------------------
    // Water models
    // ---------------------------------------------------------------------
    py::class_<WaterModel>(m, "WaterModel", R"doc(
The parameters of a rigid three-site water model.

    fa.water_model("spce").dipole_moment      # 2.35 debye
)doc")
        .def_readonly("name", &WaterModel::name)
        .def_readonly("q_oxygen", &WaterModel::q_oxygen, "Elementary charges.")
        .def_readonly("q_hydrogen", &WaterModel::q_hydrogen)
        .def_readonly("epsilon_oxygen", &WaterModel::epsilon_oxygen, "eV.")
        .def_readonly("sigma_oxygen", &WaterModel::sigma_oxygen, "Angstrom.")
        .def_readonly("bond_length", &WaterModel::bond_length, "O-H, angstrom.")
        .def_readonly("angle", &WaterModel::angle, "H-O-H, degrees.")
        .def_readonly("self_energy", &WaterModel::self_energy,
                      "eV per molecule; SPC/E's polarisation correction, zero for SPC.")
        .def_readonly("reference", &WaterModel::reference)
        .def_property_readonly("hydrogen_distance", &WaterModel::hydrogen_distance,
                               "The H-H distance that makes the angle rigid.")
        .def_property_readonly("dipole_moment", &WaterModel::dipole_moment, "Debye.")
        .def("__repr__", &WaterModel::to_string);

    m.def("water_model", &water_model, py::arg("name"),
          py::return_value_policy::reference,
          "Look up 'spc', 'spce' or 'tip3p'. Case, hyphens and slashes are ignored.");
    m.def("water_model_names", &water_model_names, "Every model this build knows.");

    m.def(
        "water_molecule",
        [](const py::object& model, double box) {
            return water_molecule(py::isinstance<WaterModel>(model)
                                      ? model.cast<WaterModel>()
                                      : water_model(model.cast<std::string>()),
                                  box);
        },
        py::arg("model") = "spce", py::arg("box") = 20.0,
        "One water molecule at its model geometry, oxygen first.");

    m.def(
        "water_box",
        [](const py::object& model, int molecules, double density, std::uint64_t seed) {
            return water_box(py::isinstance<WaterModel>(model)
                                 ? model.cast<WaterModel>()
                                 : water_model(model.cast<std::string>()),
                             molecules, density, seed);
        },
        py::arg("model") = "spce", py::arg("molecules") = 32, py::arg("density") = 0.997,
        py::arg("seed") = 0,
        R"doc(
A cubic box of randomly oriented water molecules at a given density.

The count is rounded up to fill a face-centred cubic lattice - 4, 32, 108,
256 - because a partly filled one would put the molecules closer together than
the density asks for.

    atoms = fa.water_box("spce", 108)     # 324 atoms at 0.997 g/cm^3
)doc");

    m.def(
        "water_topology",
        [](const Atoms& atoms, const py::object& model) {
            return water_topology(atoms, py::isinstance<WaterModel>(model)
                                             ? model.cast<WaterModel>()
                                             : water_model(model.cast<std::string>()));
        },
        py::arg("atoms"), py::arg("model") = "spce",
        "Charges, exclusions and rigid-geometry constraints for a box of water.");

    // ---------------------------------------------------------------------
    // Ewald
    // ---------------------------------------------------------------------
    py::class_<EwaldEnergy>(m, "EwaldEnergy", "The four pieces of an Ewald sum, in eV.")
        .def_readonly("real", &EwaldEnergy::real, "Screened point charges, within the cutoff.")
        .def_readonly("reciprocal", &EwaldEnergy::reciprocal, "The smooth compensating clouds.")
        .def_readonly("self", &EwaldEnergy::self, "Each cloud on its own charge.")
        .def_readonly("excluded", &EwaldEnergy::excluded,
                      "Undoing the intramolecular pairs the reciprocal sum included.")
        .def_property_readonly("total", &EwaldEnergy::total)
        .def("__repr__", &EwaldEnergy::to_string);

    py::class_<Ewald>(m, "Ewald", R"doc(
Ewald summation for a periodic system of point charges.

A lattice sum of 1/r is only conditionally convergent, so truncating it at a
cutoff computes a different quantity rather than an approximate one. Ewald
splits it into a short-ranged part and a smooth part that converges in
reciprocal space; the total does not depend on where the split is made.

    ewald = fa.Ewald(accuracy=1e-6)
    ewald.prepare(atoms)
    ewald.alpha, ewald.kvector_count
)doc")
        .def(py::init([](double accuracy, double cutoff, double alpha, double kcutoff) {
                 EwaldParameters parameters;
                 parameters.accuracy = accuracy;
                 parameters.real_cutoff = cutoff;
                 parameters.alpha = alpha;
                 parameters.kcutoff = kcutoff;
                 return std::make_unique<Ewald>(parameters);
             }),
             py::arg("accuracy") = 1.0e-6, py::arg("cutoff") = 0.0, py::arg("alpha") = 0.0,
             py::arg("kcutoff") = 0.0)
        .def("prepare", &Ewald::prepare, py::arg("atoms"),
             "Fix the splitting and build the reciprocal vectors for this cell.")
        .def_property_readonly("alpha", &Ewald::alpha, "Splitting parameter, 1/angstrom.")
        .def_property_readonly("cutoff", &Ewald::real_cutoff)
        .def_property_readonly("kcutoff", &Ewald::kcutoff)
        .def_property_readonly("kvector_count", &Ewald::kvector_count)
        .def_property_readonly("prepared", &Ewald::prepared)
        .def("estimated_error", &Ewald::estimated_error, py::arg("topology"), py::arg("volume"),
             "Estimated error left in the two truncations, in eV.")
        .def(
            "compute",
            [](Ewald& self, const Atoms& atoms, const Topology& topology) {
                self.prepare(atoms);
                NeighborListOptions options;
                options.cutoff = self.real_cutoff();
                options.full = true;
                options.store_vectors = false;
                options.store_distances = false;
                const EwaldResult result =
                    self.compute(atoms, topology, build_neighbor_list(atoms, options), true, true);
                py::dict out;
                out["energy"] = result.energy;
                std::vector<double> forces = result.forces;
                const auto n = static_cast<py::ssize_t>(atoms.size());
                out["forces"] =
                    move_to_array(std::move(forces), {n, static_cast<py::ssize_t>(3)});
                out["stress"] = result.stress;
                return out;
            },
            py::arg("atoms"), py::arg("topology"),
            "Energy, forces and stress from the full sum.")
        .def("__repr__", &Ewald::to_string);

    // ---------------------------------------------------------------------
    // The calculator
    // ---------------------------------------------------------------------
    py::class_<EnergyBreakdown>(m, "EnergyBreakdown", "A force-field energy, term by term.")
        .def_readonly("lennard_jones", &EnergyBreakdown::lennard_jones)
        .def_readonly("lennard_jones_tail", &EnergyBreakdown::lennard_jones_tail)
        .def_readonly("coulomb", &EnergyBreakdown::coulomb)
        .def_readonly("self", &EnergyBreakdown::self,
                      "Per-molecule constants, such as SPC/E's polarisation correction.")
        .def_property_readonly("total", &EnergyBreakdown::total)
        .def("__repr__", &EnergyBreakdown::to_string);

    py::class_<ForcefieldCalculator, Calculator, std::shared_ptr<ForcefieldCalculator>>(
        m, "Forcefield", R"doc(
A classical empirical force field: Lennard-Jones plus Ewald electrostatics.

    atoms = fa.water_box("spce", 108)
    atoms.calc = fa.Forcefield(model="spce")
    fa.run_md(atoms, 1000, timestep=2.0, temperature=300.0, thermostat="nose_hoover")

Naming a water model is enough: the molecules are found by connectivity, the
charges and dispersion parameters assigned, the intramolecular pairs excluded,
and the three distance constraints that keep each molecule rigid set up.
`Dynamics` takes those constraints automatically, which is what makes the 2 fs
timestep above legitimate.
)doc")
        .def(py::init([](const py::kwargs& kwargs) {
            return std::make_shared<ForcefieldCalculator>(from_kwargs(kwargs));
        }))
        .def("prepare", &ForcefieldCalculator::prepare, py::arg("atoms"),
             "Detect the molecules and set up the Ewald sum now rather than on first use.")
        .def_property_readonly("topology", &ForcefieldCalculator::topology,
                               py::return_value_policy::reference_internal)
        .def("set_topology", &ForcefieldCalculator::set_topology, py::arg("topology"),
             "Use a topology built elsewhere, for a system the models do not cover.")
        .def_property_readonly(
            "model",
            [](const ForcefieldCalculator& self) -> py::object {
                const WaterModel* model = self.model();
                return model == nullptr ? py::none() : py::cast(*model);
            },
            "The water model in use, or None for a hand-built topology.")
        .def_property_readonly("ewald", &ForcefieldCalculator::ewald,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("breakdown", &ForcefieldCalculator::breakdown,
                               py::return_value_policy::reference_internal,
                               "The most recent energy, term by term.")
        .def_property_readonly("cutoff", &ForcefieldCalculator::cutoff)
        .def_property_readonly("constraints", &ForcefieldCalculator::constraints,
                               "The rigid distances this model implies.");
}

}  // namespace fastatoms::bindings
