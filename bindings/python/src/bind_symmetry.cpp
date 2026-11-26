// SPDX-License-Identifier: MIT
/// \file bind_symmetry.cpp
/// \brief `fastatoms._core.symmetry`: spglib-backed space groups and cells.
///
/// Compiled only with FASTATOMS_WITH_SYMMETRY. Results follow spglib's Python
/// API: get_symmetry_dataset returns an object whose attributes carry the
/// spglib names, with numpy arrays for the per-atom and per-operation fields.

#include "common.hpp"

#ifdef FASTATOMS_WITH_SYMMETRY

#include <string>
#include <vector>

#include "fastatoms/symmetry/pointgroup.hpp"
#include "fastatoms/symmetry/spacegroup.hpp"
#include "fastatoms/symmetry/standardize.hpp"

namespace sy = fastatoms::symmetry;

namespace fastatoms::bindings {
namespace {

using Index = py::ssize_t;

py::array_t<int> ints_to_py(std::vector<int> data) {
    const auto n = static_cast<Index>(data.size());
    return move_to_array(std::move(data), {n});
}

py::array_t<double> vec3s_to_py(const std::vector<Vec3>& points) {
    std::vector<double> flat;
    flat.reserve(3 * points.size());
    for (const Vec3& p : points) {
        flat.insert(flat.end(), {p.x, p.y, p.z});
    }
    return move_to_array(std::move(flat), {static_cast<Index>(points.size()), 3});
}

py::array_t<double> vec3_to_py(const Vec3& p) {
    return move_to_array(std::vector<double>{p.x, p.y, p.z}, {3});
}

py::array_t<int> rotations_to_py(const std::vector<sy::IntMatrix3>& rotations) {
    std::vector<int> flat;
    flat.reserve(9 * rotations.size());
    for (const auto& r : rotations) {
        for (const auto& row : r) {
            flat.insert(flat.end(), row.begin(), row.end());
        }
    }
    const auto n = static_cast<Index>(rotations.size());
    return move_to_array(std::move(flat), {n, 3, 3});
}

std::vector<std::string> letters_of(const std::vector<char>& letters) {
    std::vector<std::string> out;
    out.reserve(letters.size());
    for (char c : letters) {
        out.emplace_back(1, c);
    }
    return out;
}

/// The SymmetryDataset class, with spglib's attribute names.
void bind_dataset(py::module_& sym) {
    py::class_<sy::SymmetryDataset>(sym, "SymmetryDataset",
                                    "spglib's dataset: space group, operations, Wyckoff sites "
                                    "and the standardised cell. Attribute names follow spglib.")
        .def_property_readonly("number", [](const sy::SymmetryDataset& d) { return d.number; })
        .def_property_readonly("international",
                               [](const sy::SymmetryDataset& d) { return d.international; })
        .def_property_readonly("hall", [](const sy::SymmetryDataset& d) { return d.hall; })
        .def_property_readonly("hall_number",
                               [](const sy::SymmetryDataset& d) { return d.hall_number; })
        .def_property_readonly("choice", [](const sy::SymmetryDataset& d) { return d.choice; })
        .def_property_readonly("pointgroup",
                               [](const sy::SymmetryDataset& d) { return d.pointgroup.symbol; })
        .def_property_readonly("crystal_system",
                               [](const sy::SymmetryDataset& d) {
                                   return std::string(sy::crystal_system_name(d.crystal_system));
                               })
        .def_property_readonly("rotations",
                               [](const sy::SymmetryDataset& d) { return rotations_to_py(d.rotations); })
        .def_property_readonly("translations",
                               [](const sy::SymmetryDataset& d) { return vec3s_to_py(d.translations); })
        .def_property_readonly("wyckoffs",
                               [](const sy::SymmetryDataset& d) { return letters_of(d.wyckoffs); })
        .def_property_readonly("site_symmetry_symbols",
                               [](const sy::SymmetryDataset& d) { return d.site_symmetry_symbols; })
        .def_property_readonly("equivalent_atoms",
                               [](const sy::SymmetryDataset& d) { return ints_to_py(d.equivalent_atoms); })
        .def_property_readonly("crystallographic_orbits",
                               [](const sy::SymmetryDataset& d) {
                                   return ints_to_py(d.crystallographic_orbits);
                               })
        .def_property_readonly("std_lattice",
                               [](const sy::SymmetryDataset& d) { return mat3_to_py(d.std_lattice); })
        .def_property_readonly("std_types",
                               [](const sy::SymmetryDataset& d) { return ints_to_py(d.std_types); })
        .def_property_readonly("std_positions",
                               [](const sy::SymmetryDataset& d) { return vec3s_to_py(d.std_positions); })
        .def_property_readonly("transformation_matrix",
                               [](const sy::SymmetryDataset& d) {
                                   return mat3_to_py(d.transformation_matrix);
                               })
        .def_property_readonly("origin_shift",
                               [](const sy::SymmetryDataset& d) { return vec3_to_py(d.origin_shift); })
        .def_property_readonly("primitive_lattice",
                               [](const sy::SymmetryDataset& d) { return mat3_to_py(d.primitive_lattice); })
        .def_property_readonly("mapping_to_primitive",
                               [](const sy::SymmetryDataset& d) {
                                   return ints_to_py(d.mapping_to_primitive);
                               })
        .def_property_readonly("std_mapping_to_primitive",
                               [](const sy::SymmetryDataset& d) {
                                   return ints_to_py(d.std_mapping_to_primitive);
                               });
}

void bind_space_group(py::module_& sym) {
    py::class_<sy::WyckoffPosition>(sym, "WyckoffPosition")
        .def_property_readonly("letter",
                               [](const sy::WyckoffPosition& w) { return std::string(1, w.letter); })
        .def_property_readonly("site_symmetry",
                               [](const sy::WyckoffPosition& w) { return w.site_symmetry; })
        .def_property_readonly("atoms", [](const sy::WyckoffPosition& w) { return w.atoms; });

    py::class_<sy::SpaceGroup>(sym, "SpaceGroup", "The space group and its Wyckoff positions.")
        .def_property_readonly("number", [](const sy::SpaceGroup& g) { return g.number; })
        .def_property_readonly("international",
                               [](const sy::SpaceGroup& g) { return g.international; })
        .def_property_readonly("pointgroup",
                               [](const sy::SpaceGroup& g) { return g.pointgroup.symbol; })
        .def_property_readonly("crystal_system",
                               [](const sy::SpaceGroup& g) {
                                   return std::string(sy::crystal_system_name(g.crystal_system));
                               })
        .def_property_readonly("wyckoff", [](const sy::SpaceGroup& g) { return g.wyckoff; });
}

}  // namespace

void bind_symmetry(py::module_& m) {
    py::module_ sym = m.def_submodule(
        "symmetry", "Space groups, standardised cells and primitive cells (spglib).");
    py::register_exception<sy::SymmetryError>(sym, "SymmetryError", PyExc_ValueError);

    bind_dataset(sym);
    bind_space_group(sym);

    sym.def(
        "get_symmetry_dataset",
        [](const Atoms& atoms, double symprec, double angle_tolerance, bool use_labels) {
            return sy::get_symmetry_dataset(atoms, symprec, angle_tolerance, use_labels);
        },
        py::arg("atoms"), py::arg("symprec") = 1e-5, py::arg("angle_tolerance") = -1.0,
        py::arg("use_labels") = false,
        "The spglib dataset of a structure. Labels are the atom types when use_labels is set.");

    sym.def(
        "standardize_cell",
        [](const Atoms& atoms, bool to_primitive, bool no_idealize, double symprec,
           double angle_tolerance, bool use_labels) {
            return sy::standardize_cell(atoms, to_primitive, no_idealize, symprec, angle_tolerance,
                                        use_labels);
        },
        py::arg("atoms"), py::arg("to_primitive") = false, py::arg("no_idealize") = false,
        py::arg("symprec") = 1e-5, py::arg("angle_tolerance") = -1.0, py::arg("use_labels") = false,
        "spglib's standardised cell as Atoms.");

    sym.def(
        "find_primitive",
        [](const Atoms& atoms, double symprec, double angle_tolerance, bool use_labels) {
            return sy::find_primitive(atoms, symprec, angle_tolerance, use_labels);
        },
        py::arg("atoms"), py::arg("symprec") = 1e-5, py::arg("angle_tolerance") = -1.0,
        py::arg("use_labels") = false, "The idealised primitive cell as Atoms.");

    sym.def(
        "niggli_reduce",
        [](const Atoms& atoms, double symprec) { return sy::niggli_reduce(atoms, symprec); },
        py::arg("atoms"), py::arg("symprec") = 1e-5,
        "The Niggli-reduced cell of the same lattice, as Atoms.");

    sym.def(
        "analyse",
        [](const Atoms& atoms, double symprec, double angle_tolerance, bool use_labels) {
            return sy::analyse(atoms, symprec, angle_tolerance, use_labels);
        },
        py::arg("atoms"), py::arg("symprec") = 1e-5, py::arg("angle_tolerance") = -1.0,
        py::arg("use_labels") = false, "The space group and Wyckoff positions of a structure.");

    sym.def("space_group_number", &sy::space_group_number, py::arg("hermann_mauguin"),
            "International Tables number for a short Hermann-Mauguin symbol, or 0.");

    sym.def("space_group_symbol", &sy::space_group_symbol, py::arg("number"),
            "Short Hermann-Mauguin symbol for a number between 1 and 230.");
}

}  // namespace fastatoms::bindings

#endif  // FASTATOMS_WITH_SYMMETRY
