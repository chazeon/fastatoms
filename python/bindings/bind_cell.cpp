// SPDX-License-Identifier: MIT
#include "bindings.hpp"

namespace fastatoms::bindings {

void bind_cell(py::module_& m) {
    py::class_<Cell>(m, "Cell", R"doc(
A periodic simulation cell.

The three lattice vectors are the *rows* of the 3x3 matrix, so that
``cartesian = scaled @ cell.array``, matching ASE. Boundary conditions live on
the cell, and :class:`Atoms` forwards to them.
)doc")
        .def(py::init<>(), "An empty (zero) cell with no periodicity.")
        .def(py::init([](const Mat3& matrix, const py::object& pbc) {
                 return Cell(matrix, pbc.is_none() ? std::array<bool, 3>{true, true, true}
                                                   : parse_pbc(pbc));
             }),
             py::arg("matrix"), py::arg("pbc") = py::none(),
             "Build a cell from a 3x3 matrix, 9 numbers, or 3 edge lengths.")

        .def_static("cubic", &Cell::cubic, py::arg("a"), "Cubic cell of edge ``a``.")
        .def_static("orthorhombic", &Cell::orthorhombic, py::arg("a"), py::arg("b"), py::arg("c"),
                    "Orthorhombic cell with the given edge lengths.")
        .def_static(
            "from_parameters",
            [](double a, double b, double c, double alpha, double beta, double gamma,
               const py::object& pbc) {
                return Cell::from_parameters(a, b, c, alpha, beta, gamma,
                                             pbc.is_none()
                                                 ? std::array<bool, 3>{true, true, true}
                                                 : parse_pbc(pbc));
            },
            py::arg("a"), py::arg("b"), py::arg("c"), py::arg("alpha") = 90.0,
            py::arg("beta") = 90.0, py::arg("gamma") = 90.0, py::arg("pbc") = py::none(),
            "Build a cell from the six crystallographic parameters (angles in degrees).")

        .def_property(
            "array", [](const Cell& self) { return self.matrix(); },
            [](Cell& self, const Mat3& matrix) { self.set_matrix(matrix); },
            "The 3x3 lattice matrix, one lattice vector per row.")
        .def_property(
            "pbc", [](const Cell& self) { return pbc_to_array(self.pbc()); },
            [](Cell& self, const py::object& pbc) { self.set_pbc(parse_pbc(pbc)); },
            "Periodic boundary conditions as a length-3 boolean array.")

        .def_property_readonly("volume", &Cell::volume, "Cell volume in cubic angstrom.")
        .def_property_readonly("lengths", &Cell::lengths, "Lattice vector lengths |a|, |b|, |c|.")
        .def_property_readonly("angles", &Cell::angles,
                               "Lattice angles alpha, beta, gamma in degrees.")
        .def_property_readonly("heights", &Cell::heights,
                               "Perpendicular width of the cell along each lattice direction.")
        .def_property_readonly("reciprocal", &Cell::reciprocal,
                               "Reciprocal lattice vectors as rows, without the 2*pi factor.")
        .def_property_readonly("is_invertible", &Cell::is_invertible,
                               "True when the cell spans three dimensions.")

        .def("is_orthorhombic", &Cell::is_orthorhombic, py::arg("tol") = 1e-10,
             "True when all three lattice vectors are mutually orthogonal.")
        .def("scaled", &Cell::scaled, py::arg("position"),
             "Convert a Cartesian position to fractional coordinates.")
        .def("cartesian", &Cell::cartesian, py::arg("scaled"),
             "Convert fractional coordinates to a Cartesian position.")
        .def("wrap", &Cell::wrap, py::arg("position"), py::arg("eps") = 1e-7,
             "Fold a Cartesian position into the cell along the periodic directions.")
        .def(
            "minimum_image",
            [](const Cell& self, const Vec3& dr, bool return_offset) -> py::object {
                std::array<int, 3> offset{};
                const Vec3 image = self.minimum_image(dr, offset);
                if (!return_offset) {
                    return py::cast(image);
                }
                return py::make_tuple(py::cast(image), py::cast(offset));
            },
            py::arg("dr"), py::arg("return_offset") = false,
            "Shortest periodic image of a displacement vector.")
        .def("maximum_unique_cutoff", &Cell::maximum_unique_cutoff,
             "Largest cutoff for which the minimum-image convention is unambiguous.")

        .def("__array__",
             [](const Cell& self, const py::object&, const py::object&) { return self.matrix(); },
             py::arg("dtype") = py::none(), py::arg("copy") = py::none(),
             "numpy interoperability: ``np.asarray(cell)`` gives the 3x3 matrix.")
        .def("__repr__", &Cell::to_string)
        .def("__copy__", [](const Cell& self) { return Cell(self); })
        .def("__deepcopy__", [](const Cell& self, const py::dict&) { return Cell(self); },
             py::arg("memo"))
        .def(
            "__eq__",
            [](const Cell& self, const py::object& other) {
                if (!py::isinstance<Cell>(other)) {
                    return false;
                }
                const Cell& rhs = other.cast<const Cell&>();
                return self.matrix() == rhs.matrix() && self.pbc() == rhs.pbc();
            },
            py::arg("other"), py::is_operator())
        .def(
            "__ne__",
            [](const Cell& self, const py::object& other) {
                if (!py::isinstance<Cell>(other)) {
                    return true;
                }
                const Cell& rhs = other.cast<const Cell&>();
                return !(self.matrix() == rhs.matrix() && self.pbc() == rhs.pbc());
            },
            py::arg("other"), py::is_operator());
}

}  // namespace fastatoms::bindings
