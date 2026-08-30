// SPDX-License-Identifier: MIT
///
/// Crystallographic symmetry: point groups, space groups and cell
/// standardization.

#include "bindings.hpp"

#include <string>
#include <vector>

#include "fastatoms/symmetry/spacegroup.hpp"

namespace fastatoms::bindings {

void bind_symmetry(py::module_& parent, py::class_<Atoms>& atoms_class) {
    py::module_ m = parent.def_submodule("symmetry", "Crystallographic symmetry.");
    namespace sym = fastatoms::symmetry;

    py::enum_<sym::CrystalSystem>(m, "CrystalSystem", "One of the seven crystal systems.")
        .value("triclinic", sym::CrystalSystem::Triclinic)
        .value("monoclinic", sym::CrystalSystem::Monoclinic)
        .value("orthorhombic", sym::CrystalSystem::Orthorhombic)
        .value("tetragonal", sym::CrystalSystem::Tetragonal)
        .value("trigonal", sym::CrystalSystem::Trigonal)
        .value("hexagonal", sym::CrystalSystem::Hexagonal)
        .value("cubic", sym::CrystalSystem::Cubic);

    py::enum_<sym::Centring>(m, "Centring", "The centring of a conventional cell.")
        .value("P", sym::Centring::P).value("A", sym::Centring::A)
        .value("B", sym::Centring::B).value("C", sym::Centring::C)
        .value("I", sym::Centring::I).value("F", sym::Centring::F)
        .value("R", sym::Centring::R);

    py::class_<sym::PointGroup>(m, "PointGroup", "One of the 32 crystallographic point groups.")
        .def_readonly("symbol", &sym::PointGroup::symbol, "Hermann-Mauguin, e.g. '4/mmm'.")
        .def_readonly("schoenflies", &sym::PointGroup::schoenflies, "e.g. 'D4h'.")
        .def_readonly("order", &sym::PointGroup::order)
        .def_readonly("system", &sym::PointGroup::system)
        .def_readonly("centrosymmetric", &sym::PointGroup::centrosymmetric)
        .def_readonly("enantiomorphic", &sym::PointGroup::enantiomorphic,
                      "True when nothing improper is present, so the structure can be chiral.")
        .def("__repr__", &sym::PointGroup::to_string);

    py::class_<sym::WyckoffPosition>(m, "WyckoffPosition", "One orbit of symmetry-related atoms.")
        .def_property_readonly("letter",
                               [](const sym::WyckoffPosition& self) {
                                   return std::string(1, self.letter);
                               })
        .def_readonly("multiplicity", &sym::WyckoffPosition::multiplicity,
                      "Atoms in the orbit, counted in the conventional cell.")
        .def_readonly("site_symmetry", &sym::WyckoffPosition::site_symmetry,
                      "The point group of the site, e.g. '-43m'.")
        .def_readonly("atoms", &sym::WyckoffPosition::atoms,
                      "Indices into SpaceGroup.primitive.")
        .def_readonly("representative", &sym::WyckoffPosition::representative)
        .def("__repr__", &sym::WyckoffPosition::to_string);

    py::class_<sym::SpaceGroup>(m, "SpaceGroup", "What a structure's symmetry turned out to be.")
        .def_readonly("number", &sym::SpaceGroup::number,
                      "International Tables number, 1 to 230.")
        .def_readonly("hermann_mauguin", &sym::SpaceGroup::hermann_mauguin,
                      "The short symbol, e.g. 'Fd-3m'.")
        .def_readonly("hermann_mauguin_full", &sym::SpaceGroup::hermann_mauguin_full,
                      "Every direction written out, e.g. 'F 4_1/d -3 2/m'.")
        .def_readonly("schoenflies", &sym::SpaceGroup::schoenflies, "e.g. 'Oh-7'.")
        .def_readonly("point_group", &sym::SpaceGroup::point_group)
        .def_readonly("system", &sym::SpaceGroup::system)
        .def_readonly("centring", &sym::SpaceGroup::centring)
        .def_readonly("primitive", &sym::SpaceGroup::primitive,
                      "The primitive cell the analysis was carried out on.")
        .def_readonly("wyckoff", &sym::SpaceGroup::wyckoff)
        .def_readonly("primitive_atoms", &sym::SpaceGroup::primitive_atoms)
        .def_readonly("conventional_atoms", &sym::SpaceGroup::conventional_atoms)
        .def_readonly("tolerance", &sym::SpaceGroup::tolerance)
        .def_property_readonly("operations",
                               [](const sym::SpaceGroup& self) { return self.operations; },
                               "Every operation, in the basis of `primitive`.")
        .def_property_readonly("order", &sym::SpaceGroup::order,
                               "How many operations the group has.")
        .def("summary", &sym::SpaceGroup::summary, "A readable multi-line report.")
        .def("__len__", &sym::SpaceGroup::order)
        .def("__repr__", &sym::SpaceGroup::to_string);

    py::class_<sym::CellAnalysis>(m, "CellAnalysis", "What a cell turned out to be.")
        .def_readonly("point_group", &sym::CellAnalysis::point_group)
        .def_readonly("system", &sym::CellAnalysis::system)
        .def_readonly("centring", &sym::CellAnalysis::centring)
        .def_readonly("primitive_atoms", &sym::CellAnalysis::primitive_atoms)
        .def_readonly("conventional_atoms", &sym::CellAnalysis::conventional_atoms)
        .def_readonly("input_multiplicity", &sym::CellAnalysis::input_multiplicity,
                      "How many primitive cells the given cell holds.")
        .def_readonly("to_primitive", &sym::CellAnalysis::to_primitive)
        .def_readonly("to_conventional", &sym::CellAnalysis::to_conventional)
        .def("__repr__", &sym::CellAnalysis::to_string);

    m.def("analyse", &sym::analyse, py::arg("atoms"), py::kw_only(),
          py::arg("tolerance") = 1e-5,
          R"doc(
Determine the space group of a structure.

    group = fa.symmetry.analyse(atoms)
    print(group.number, group.hermann_mauguin)

`tolerance` is a distance in angstrom: too tight and real symmetry is missed,
too loose and operations appear that are not there, which shows up as a point
group that is not one of the thirty-two.
)doc");
    m.attr("analyze") = m.attr("analyse");   // for the other spelling

    m.def("analyse_cell", &sym::analyse_cell, py::arg("atoms"), py::kw_only(),
          py::arg("tolerance") = 1e-5, "Work out what cell a structure is given in.");
    m.attr("analyze_cell") = m.attr("analyse_cell");

    m.def("primitive_cell", &sym::primitive_cell, py::arg("atoms"), py::kw_only(),
          py::arg("tolerance") = 1e-5,
          "The smallest cell that tiles the crystal.");
    m.def("conventional_cell", &sym::conventional_cell, py::arg("atoms"), py::kw_only(),
          py::arg("tolerance") = 1e-5,
          "The conventional cell in its standard setting.");
    m.def("niggli_reduce", &sym::niggli_reduce, py::arg("atoms"), py::kw_only(),
          py::arg("tolerance") = 1e-5,
          "The unique shortest, most orthogonal basis for the same lattice.");

    m.def(
        "point_group",
        [](const Atoms& atoms, double tolerance) {
            return sym::identify_point_group(phonon::find_symmetry(atoms, tolerance).point_group);
        },
        py::arg("atoms"), py::kw_only(), py::arg("tolerance") = 1e-5,
        "The crystallographic point group of a structure.");

    m.def("point_groups", &sym::crystallographic_point_groups,
          "All thirty-two crystallographic point groups.");
    m.def("space_group_number", &sym::space_group_number, py::arg("hermann_mauguin"),
          "The International Tables number for a symbol, or 0 if it is not one of the 230.");
    m.def("space_group_symbol", &sym::space_group_symbol, py::arg("number"),
          "The short Hermann-Mauguin symbol for a number between 1 and 230.");
    m.def("space_group_table", &sym::space_group_table,
          "All 230, as (number, symbol) pairs in International Tables order.");

    // Convenience on Atoms itself, which is where people will look for it.
    atoms_class.def(
        "get_space_group",
        [](const Atoms& self, double tolerance) { return sym::analyse(self, tolerance); },
        py::kw_only(), py::arg("tolerance") = 1e-5,
        "Determine this structure's space group.");
    atoms_class.def(
        "get_primitive_cell",
        [](const Atoms& self, double tolerance) { return sym::primitive_cell(self, tolerance); },
        py::kw_only(), py::arg("tolerance") = 1e-5, "Reduce to the primitive cell.");
    atoms_class.def(
        "get_conventional_cell",
        [](const Atoms& self, double tolerance) {
            return sym::conventional_cell(self, tolerance);
        },
        py::kw_only(), py::arg("tolerance") = 1e-5, "Raise to the conventional cell.");
}

}  // namespace fastatoms::bindings
