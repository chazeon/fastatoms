// SPDX-License-Identifier: MIT
#include "bindings.hpp"

namespace fastatoms::bindings {

void bind_elements(py::module_& m) {
    m.attr("MAX_ATOMIC_NUMBER") = kMaxAtomicNumber;

    m.def(
        "symbol_from_number",
        [](int z) { return std::string(symbol_from_number(z)); }, py::arg("number"),
        "Chemical symbol for an atomic number.");

    m.def(
        "number_from_symbol",
        [](const std::string& symbol) { return number_from_symbol(symbol); }, py::arg("symbol"),
        "Atomic number for a chemical symbol (case tolerant).");

    m.def("atomic_mass", &atomic_mass, py::arg("number"),
          "Standard atomic weight in atomic mass units.");

    m.def("covalent_radius", &covalent_radius, py::arg("number"),
          "Cordero (2008) covalent radius in angstrom; 0.0 when not tabulated.");

    m.def(
        "natural_cutoffs",
        [](const py::object& source, double mult, double fallback) {
            std::vector<int> numbers;
            if (py::isinstance<Atoms>(source)) {
                numbers = source.cast<const Atoms&>().numbers();
            } else {
                numbers = py::cast<std::vector<int>>(source);
            }
            std::vector<double> cutoffs = natural_cutoffs(numbers, mult, fallback);
            const auto n = static_cast<py::ssize_t>(cutoffs.size());
            return move_to_array(std::move(cutoffs), {n});
        },
        py::arg("atoms_or_numbers"), py::arg("mult") = 1.0, py::arg("fallback") = 0.2,
        R"doc(
Per-atom radii derived from covalent radii, for bond detection.

Pass the result as ``per_atom_cutoffs`` to :func:`neighbor_list`: two atoms
count as bonded when their distance is below the sum of their cutoffs.
)doc");

    // A units namespace mirroring ASE's, so scripts can convert without
    // hard-coding constants.
    py::module_ units = m.def_submodule("units", "Physical constants in the ASE unit system.");
    units.attr("kB") = units::kB;
    units.attr("fs") = units::fs;
    units.attr("ps") = units::ps;
    units.attr("second") = units::second;
    units.attr("hbar") = units::hbar;
    units.attr("Bohr") = units::Bohr;
    units.attr("Hartree") = units::Hartree;
    units.attr("Rydberg") = units::Rydberg;
    units.attr("kcal_per_mol") = units::kcal_per_mol;
    units.attr("kJ_per_mol") = units::kJ_per_mol;
    units.attr("GPa") = units::GPa;
}

}  // namespace fastatoms::bindings
