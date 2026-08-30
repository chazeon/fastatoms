// SPDX-License-Identifier: MIT
#include "bindings.hpp"

#include <complex>

#include "fastatoms/phonon/registry.hpp"

namespace fastatoms::bindings {
namespace {

using namespace fastatoms::phonon;

/// A square complex matrix as a numpy array.
py::array_t<std::complex<double>> to_array(const ComplexMatrix& matrix) {
    const auto n = static_cast<py::ssize_t>(matrix.size());
    std::vector<std::complex<double>> flat = matrix.data();
    auto* holder = new std::vector<std::complex<double>>(std::move(flat));
    py::capsule owner(holder, [](void* p) {
        delete static_cast<std::vector<std::complex<double>>*>(p);
    });
    return py::array_t<std::complex<double>>({n, n}, holder->data(), owner);
}

Vec3 to_qpoint(const py::object& value) {
    const std::vector<double> entries =
        as_flat_doubles(value, 3, 1, "q must have three fractional components");
    return Vec3{entries[0], entries[1], entries[2]};
}

}  // namespace

void bind_phonon(py::module_& m, py::class_<io::InputDeck>& deck_class) {

    // ---------------------------------------------------------------------
    // Supercells
    // ---------------------------------------------------------------------
    py::class_<SupercellMap>(m, "SupercellMap", R"doc(
A supercell and the bookkeeping that ties it back to the unit cell.

Atoms are ordered so the first ``unit_cell_size`` of them are the original
cell at the zero translation, which is the block a phonon calculation needs
rows for.
)doc")
        .def_readonly("atoms", &SupercellMap::atoms)
        .def_readonly("transformation", &SupercellMap::transformation)
        .def_readonly("unit_cell_size", &SupercellMap::unit_cell_size)
        .def_readonly("cell_count", &SupercellMap::cell_count,
                      "Unit cells in the supercell, i.e. |det P|.")
        .def_property_readonly(
            "unit_cell_index",
            [](const SupercellMap& self) {
                std::vector<std::int64_t> values(self.unit_cell_index.begin(),
                                                 self.unit_cell_index.end());
                const auto n = static_cast<py::ssize_t>(values.size());
                return move_to_array(std::move(values), {n});
            },
            "For each supercell atom, which unit-cell atom it is a copy of.")
        .def_property_readonly(
            "lattice_points",
            [](const SupercellMap& self) {
                std::vector<std::int64_t> values;
                values.reserve(3 * self.lattice_point.size());
                for (const std::array<int, 3>& point : self.lattice_point) {
                    values.insert(values.end(), {point[0], point[1], point[2]});
                }
                const auto n = static_cast<py::ssize_t>(self.lattice_point.size());
                return move_to_array(std::move(values), {n, static_cast<py::ssize_t>(3)});
            },
            "Integer lattice translation of each atom, in unit-cell coordinates.")
        .def("__len__", &SupercellMap::size)
        .def("__repr__", &SupercellMap::to_string);

    m.def(
        "make_supercell",
        [](const Atoms& unit_cell, const py::object& transformation) {
            if (py::isinstance<py::int_>(transformation)) {
                const int n = transformation.cast<int>();
                return make_supercell(unit_cell, n, n, n);
            }
            auto array =
                py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
                    transformation);
            if (array && array.size() == 3) {
                const double* d = array.data();
                return make_supercell(unit_cell, static_cast<int>(d[0]),
                                      static_cast<int>(d[1]), static_cast<int>(d[2]));
            }
            return make_supercell(unit_cell, transformation.cast<Mat3>());
        },
        py::arg("atoms"), py::arg("transformation"),
        R"doc(
Build a supercell from an integer transformation matrix.

    fa.make_supercell(atoms, 2)              # 2x2x2
    fa.make_supercell(atoms, [2, 2, 3])
    fa.make_supercell(atoms, [[-1, 1, 1], [1, -1, 1], [1, 1, -1]])

The last form turns an fcc primitive cell into its conventional cubic one.
)doc");

    // ---------------------------------------------------------------------
    // Symmetry
    // ---------------------------------------------------------------------
    py::class_<SymmetryOperation>(m, "SymmetryOperation",
                                  "One space group operation, `s -> s W + t`.")
        .def_readonly("rotation", &SymmetryOperation::rotation,
                      "The rotation in the lattice basis; integer valued.")
        .def_readonly("translation", &SymmetryOperation::translation, "Fractional.")
        .def_readonly("cartesian_rotation", &SymmetryOperation::cartesian_rotation,
                      "The same rotation in Cartesian coordinates; orthogonal.")
        .def_property_readonly(
            "atom_map",
            [](const SymmetryOperation& self) {
                std::vector<std::int64_t> values(self.atom_map.begin(), self.atom_map.end());
                const auto n = static_cast<py::ssize_t>(values.size());
                return move_to_array(std::move(values), {n});
            },
            "Where each atom is sent.")
        .def("is_pure_translation", &SymmetryOperation::is_pure_translation,
             py::arg("tolerance") = 1e-8)
        .def("__repr__", &SymmetryOperation::to_string);

    py::class_<SymmetryDataset>(m, "SymmetryDataset",
                                "The space group of a structure, and what it implies.")
        .def_readonly("operations", &SymmetryDataset::operations)
        .def_readonly("point_group", &SymmetryDataset::point_group,
                      "The distinct rotations.")
        .def_readonly("tolerance", &SymmetryDataset::tolerance)
        .def_property_readonly(
            "equivalent_atoms",
            [](const SymmetryDataset& self) {
                std::vector<std::int64_t> values(self.equivalent_atoms.begin(),
                                                 self.equivalent_atoms.end());
                const auto n = static_cast<py::ssize_t>(values.size());
                return move_to_array(std::move(values), {n});
            },
            "For each atom, the lowest index in its orbit.")
        .def_property_readonly(
            "independent_atoms",
            [](const SymmetryDataset& self) {
                std::vector<std::int64_t> values(self.independent_atoms.begin(),
                                                 self.independent_atoms.end());
                const auto n = static_cast<py::ssize_t>(values.size());
                return move_to_array(std::move(values), {n});
            },
            "The orbit representatives - the atoms that actually need displacing.")
        .def_property_readonly("order", &SymmetryDataset::order)
        .def("summary", &SymmetryDataset::summary)
        .def("__len__", &SymmetryDataset::order)
        .def("__repr__",
             [](const SymmetryDataset& self) { return "SymmetryDataset(" + self.summary() + ")"; });

    m.def("find_symmetry", &find_symmetry, py::arg("atoms"), py::arg("tolerance") = 1e-5,
          R"doc(
Find the space group operations of a periodic structure.

Every rotation is an integer matrix preserving the metric tensor; the
translations are those that carry the atoms onto themselves. `tolerance` is a
distance in angstrom.
)doc");

    m.def(
        "lattice_point_group",
        [](const py::object& cell, double tolerance) {
            // A `Cell`, a 3x3 matrix, or three lengths all name a lattice.
            const Cell lattice = py::isinstance<Cell>(cell) ? cell.cast<Cell>()
                                                            : Cell(cell.cast<Mat3>());
            return lattice_point_group(lattice, tolerance);
        },
        py::arg("cell"), py::arg("tolerance") = 1e-5,
        "The point group of a lattice, ignoring the atoms: 48 operations for a cubic "
        "lattice, 24 for hexagonal, 2 for triclinic.");

    m.def("site_symmetry", &site_symmetry, py::arg("symmetry"), py::arg("atom"),
          "The operations that leave one atom where it is.");

    // ---------------------------------------------------------------------
    // Force constants and modes
    // ---------------------------------------------------------------------
    py::class_<ForceConstants>(m, "ForceConstants", R"doc(
The harmonic force constants, in eV/angstrom^2.

Only rows for the first unit cell are stored; periodicity makes the rest
redundant and the dynamical matrix never needs them.
)doc")
        .def_property_readonly("rows", &ForceConstants::rows)
        .def_property_readonly("columns", &ForceConstants::columns)
        .def_property_readonly(
            "array",
            [](const ForceConstants& self) {
                std::vector<double> flat = self.to_array();
                return move_to_array(std::move(flat),
                                     {static_cast<py::ssize_t>(3 * self.rows()),
                                      static_cast<py::ssize_t>(3 * self.columns())});
            },
            "As a (3*rows, 3*columns) array.")
        .def(
            "block",
            [](const ForceConstants& self, std::size_t i, std::size_t j) {
                if (i >= self.rows() || j >= self.columns()) {
                    throw py::index_error("force constant index out of range");
                }
                return self.block(i, j);
            },
            py::arg("i"), py::arg("j"), "The 3x3 block coupling atoms i and j.")
        .def("acoustic_sum_violation", &ForceConstants::acoustic_sum_violation,
             "Largest entry of sum_j Phi(i,j); zero once the sum rule is applied.")
        .def("__repr__", &ForceConstants::to_string);

    py::class_<Displacement>(m, "Displacement", "One displaced configuration.")
        .def_readonly("atom", &Displacement::atom, "Index into the supercell.")
        .def_readonly("vector", &Displacement::vector, "The displacement, in angstrom.")
        .def("__repr__", [](const Displacement& self) {
            char buffer[128];
            std::snprintf(buffer, sizeof(buffer),
                          "Displacement(atom=%zu, vector=[%.4f, %.4f, %.4f])", self.atom,
                          self.vector.x, self.vector.y, self.vector.z);
            return std::string(buffer);
        });

    py::class_<PhononModes>(m, "PhononModes", "Frequencies and polarisation vectors at one q.")
        .def_property_readonly(
            "frequencies",
            [](const PhononModes& self) {
                std::vector<double> values = self.frequencies;
                const auto n = static_cast<py::ssize_t>(values.size());
                return move_to_array(std::move(values), {n});
            },
            "THz, ascending. Negative means an imaginary frequency.")
        .def_property_readonly(
            "squared_frequencies",
            [](const PhononModes& self) {
                std::vector<double> values = self.squared_frequencies;
                const auto n = static_cast<py::ssize_t>(values.size());
                return move_to_array(std::move(values), {n});
            },
            "Eigenvalues of the dynamical matrix, in eV/(angstrom^2 amu).")
        .def_property_readonly(
            "eigenvectors",
            [](const PhononModes& self) { return to_array(self.eigenvectors); },
            "Polarisation vectors as columns, mass-weighted and orthonormal.")
        .def("__repr__", [](const PhononModes& self) {
            return "PhononModes(" + std::to_string(self.frequencies.size()) + " branches)";
        });

    // ---------------------------------------------------------------------
    // The driver
    // ---------------------------------------------------------------------
    py::class_<Phonons, std::shared_ptr<Phonons>>(m, "Phonons", R"doc(
Harmonic phonons by finite displacements.

    atoms.calc = fa.LennardJones()
    ph = fa.Phonons(atoms, supercell=[2, 2, 2], displacement=0.01)
    ph.run()
    ph.frequencies([0.5, 0.0, 0.0])          # THz

Space group symmetry decides which atoms, and which directions, actually have
to be displaced: for silicon in a 2x2x2 supercell that is two calculations
rather than forty-eight. Pass ``symmetry=False`` to displace everything, which
is what the test suite compares against.

Frequencies come back in THz; multiply by ``Phonons.THZ_TO_WAVENUMBER`` for
cm^-1 or ``Phonons.THZ_TO_MEV`` for meV.
)doc")
        .def(py::init([](const Atoms& atoms, const py::kwargs& kwargs) {
                 return std::make_shared<Phonons>(atoms, py::cast<Value>(py::dict(kwargs)));
             }),
             py::arg("atoms"))

        .def_property_readonly("unit_cell", &Phonons::unit_cell)
        .def_property_readonly("supercell", &Phonons::supercell,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("symmetry", &Phonons::symmetry,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("supercell_matrix", &Phonons::supercell_matrix)
        .def_property_readonly("displacement_distance", &Phonons::displacement_distance,
                               "The finite displacement, in angstrom.")
        .def_property_readonly("uses_symmetry", &Phonons::use_symmetry)
        .def_property_readonly("central_differences", &Phonons::central_differences)

        .def_property_readonly("displacements", &Phonons::displacements,
                               "The configurations that have to be evaluated, in order.")
        .def("displaced_supercells", &Phonons::displaced_supercells,
             "One supercell per displacement, for running the forces elsewhere.")
        .def_property_readonly("calculation_count", &Phonons::calculation_count,
                               "Force evaluations the reduced set needs.")
        .def_property_readonly("calculation_count_without_symmetry",
                               &Phonons::calculation_count_without_symmetry,
                               "What it would have cost without the symmetry reduction.")

        .def("run", &Phonons::run, py::arg("calculator") = nullptr,
             "Displace, evaluate and solve for the force constants.")
        .def(
            "set_forces",
            [](Phonons& self, const py::object& forces, const py::object& equilibrium) {
                const std::size_t total = self.supercell().size();
                std::vector<std::vector<double>> sets;
                for (const py::handle item : forces) {
                    sets.push_back(as_flat_doubles(py::reinterpret_borrow<py::object>(item),
                                                   static_cast<py::ssize_t>(total), 3,
                                                   "forces"));
                }
                std::vector<double> reference;
                if (!equilibrium.is_none()) {
                    reference = as_flat_doubles(equilibrium, static_cast<py::ssize_t>(total), 3,
                                                "equilibrium_forces");
                }
                self.set_forces(sets, reference);
            },
            py::arg("forces"), py::arg("equilibrium_forces") = py::none(),
            "Supply forces computed elsewhere, one (N, 3) array per displacement.")

        .def_property_readonly("has_force_constants", &Phonons::has_force_constants)
        .def_property_readonly("force_constants", &Phonons::force_constants,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("residual_force", &Phonons::residual_force,
                               "Largest force on the undisplaced supercell, in eV/angstrom. "
                               "A large value means the structure was not relaxed.")

        .def(
            "dynamical_matrix",
            [](const Phonons& self, const py::object& q) {
                return to_array(self.dynamical_matrix(to_qpoint(q)));
            },
            py::arg("q"), "The dynamical matrix at q, in fractional reciprocal coordinates.")
        .def(
            "frequencies",
            [](const Phonons& self, const py::object& q) {
                std::vector<double> values = self.frequencies(to_qpoint(q));
                const auto n = static_cast<py::ssize_t>(values.size());
                return move_to_array(std::move(values), {n});
            },
            py::arg("q"), "Frequencies in THz at q, ascending.")
        .def(
            "modes", [](const Phonons& self, const py::object& q) {
                return self.modes(to_qpoint(q));
            },
            py::arg("q"), "Frequencies and polarisation vectors at q.")
        .def(
            "band_structure",
            [](const Phonons& self, const py::object& path) {
                std::vector<Vec3> points;
                for (const py::handle item : path) {
                    points.push_back(to_qpoint(py::reinterpret_borrow<py::object>(item)));
                }
                std::vector<std::vector<double>> bands = self.band_structure(points);
                std::vector<double> flat;
                const std::size_t branches = bands.empty() ? 0 : bands.front().size();
                for (const std::vector<double>& row : bands) {
                    flat.insert(flat.end(), row.begin(), row.end());
                }
                return move_to_array(std::move(flat),
                                     {static_cast<py::ssize_t>(bands.size()),
                                      static_cast<py::ssize_t>(branches)});
            },
            py::arg("path"), "Frequencies along a list of q-points, one row per point.")
        .def(
            "mesh_frequencies",
            [](const Phonons& self, int n1, int n2, int n3) {
                std::vector<double> values = self.mesh_frequencies(n1, n2, n3);
                const auto n = static_cast<py::ssize_t>(values.size());
                return move_to_array(std::move(values), {n});
            },
            py::arg("n1"), py::arg("n2"), py::arg("n3"),
            "Frequencies on a uniform Gamma-centred mesh, flattened - the input for a "
            "density of states.")

        .def_readonly_static("THZ_TO_WAVENUMBER", &Phonons::kTHzToWavenumber)
        .def_readonly_static("THZ_TO_MEV", &Phonons::kTHzToMilliElectronVolt)
        .def("__repr__", &Phonons::to_string);

    // ---------------------------------------------------------------------
    // Input files
    // ---------------------------------------------------------------------
    py::class_<PhononParameters>(m, "PhononParameters",
                                 "Settings from the ``simulation.phonons`` section.")
        .def(py::init<>())
        .def_readwrite("supercell", &PhononParameters::supercell,
                       "Three diagonal entries, or all nine of the transformation matrix.")
        .def_readwrite("displacement", &PhononParameters::displacement, "Angstrom.")
        .def_readwrite("symmetry", &PhononParameters::symmetry)
        .def_readwrite("symmetry_tolerance", &PhononParameters::symmetry_tolerance)
        .def_readwrite("parameters", &PhononParameters::parameters,
                       "The whole section as a dict, including keys the fields above "
                       "do not cover.")
        .def("__repr__", [](const PhononParameters& self) {
            std::string text = "PhononParameters(supercell=[";
            for (std::size_t k = 0; k < self.supercell.size(); ++k) {
                text += (k != 0 ? ", " : "") +
                        std::to_string(static_cast<long long>(self.supercell[k]));
            }
            return text + "], displacement=" + std::to_string(self.displacement) + ")";
        });

    m.def(
        "make_phonons",
        [](const Atoms& atoms, const py::object& source, const py::kwargs& kwargs) {
            Value settings = py::cast<Value>(py::dict(kwargs));
            if (py::isinstance<PhononParameters>(source)) {
                PhononParameters spec = source.cast<PhononParameters>();
                Value merged = spec.parameters.is_map() ? spec.parameters : Value(Value::Map{});
                for (const auto& [key, value] : settings.as_map()) {
                    merged.insert(key, value);
                }
                spec.parameters = std::move(merged);
                return make_phonons(atoms, spec);
            }
            if (py::isinstance<SimulationParameters>(source)) {
                return make_phonons(atoms, source.cast<SimulationParameters>());
            }
            if (!source.is_none()) {
                Value merged = source.cast<Value>();
                for (const auto& [key, value] : settings.as_map()) {
                    merged.insert(key, value);
                }
                return make_phonons(atoms, merged);
            }
            return make_phonons(atoms, settings);
        },
        py::arg("atoms"), py::arg("source") = py::none(),
        "Build a calculation from a parameter dict, or from an input file's phonons or "
        "simulation section.");

    deck_class.def(
        "make_phonons",
        [](const io::InputDeck& self, const py::object& atoms, const py::kwargs& overrides) {
            PhononParameters spec = self.simulation.phonon;
            Value merged = spec.parameters.is_map() ? spec.parameters : Value(Value::Map{});
            for (const auto item : overrides) {
                merged.insert(py::cast<std::string>(item.first), item.second.cast<Value>());
            }
            spec.parameters = std::move(merged);
            return make_phonons(atoms.is_none() ? self.atoms : atoms.cast<Atoms>(), spec);
        },
        py::arg("atoms") = py::none(),
        R"doc(
Build the calculation described by this file's ``simulation.phonons`` section.

    deck = fa.read("phonons.yaml")
    atoms = deck.atoms
    atoms.calc = deck.make_calculator()
    ph = deck.make_phonons(atoms)
    ph.run()

Keyword arguments override what the file said. The structure defaults to the
file's own, which has no calculator attached.
)doc");
}

}  // namespace fastatoms::bindings
