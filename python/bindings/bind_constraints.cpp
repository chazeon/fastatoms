// SPDX-License-Identifier: MIT
///
/// The constraint hierarchy, and the `Atoms` methods that carry it.

#include "bindings.hpp"

#include <memory>
#include <string>
#include <vector>

#include "fastatoms/core/constraints.hpp"

namespace fastatoms::bindings {
namespace {

/// Accept indices, a boolean mask, or both spellings of the same thing.
///
/// `FixAtoms([0, 1, 2])`, `FixAtoms(range(16))` and
/// `FixAtoms(mask=[True, True, False])` all describe a set of atoms, and which
/// one reads better depends entirely on where the set came from.
std::vector<std::size_t> index_set(const py::object& indices, const py::object& mask,
                                   const char* what) {
    const bool has_indices = !indices.is_none();
    const bool has_mask = !mask.is_none();
    if (has_indices == has_mask) {
        throw ConstraintError(std::string(what) +
                              ": give either indices or mask=, and not both");
    }
    if (has_indices) {
        return indices.cast<std::vector<std::size_t>>();
    }
    std::vector<std::size_t> out;
    std::size_t position = 0;
    for (const py::handle item : mask) {
        if (item.cast<bool>()) {
            out.push_back(position);
        }
        ++position;
    }
    return out;
}

Vec3 axis(const py::object& value, const char* what) {
    if (py::isinstance<py::str>(value)) {
        const std::string name = value.cast<std::string>();
        if (name == "x") return Vec3{1.0, 0.0, 0.0};
        if (name == "y") return Vec3{0.0, 1.0, 0.0};
        if (name == "z") return Vec3{0.0, 0.0, 1.0};
        throw ConstraintError(std::string(what) + ": '" + name +
                             "' is not an axis; use 'x', 'y', 'z' or a vector");
    }
    return value.cast<Vec3>();
}

}  // namespace

void bind_constraints(py::module_& m, py::class_<Atoms>& atoms) {
    py::class_<Constraint, std::shared_ptr<Constraint>>(m, "Constraint", R"doc(
Base class of the constraints that restrict how atoms may move.

A constraint answers the same three questions everywhere it is used - what may
a force do, what may a displacement do, what may a velocity do - and the whole
codebase asks: ``get_forces`` returns forces with the forbidden components
already removed, the optimizers project every proposed step, the integrators
project the velocities too, and the nudged elastic band projects again after
its own tangent projection.

Not constructible from Python; use :class:`FixAtoms`, :class:`FixLine` or
:class:`FixPlane`.
)doc")
        .def_property_readonly("name", &Constraint::name)
        .def_property_readonly(
            "indices",
            [](const Constraint& self) {
                return std::vector<std::size_t>(self.indices().begin(), self.indices().end());
            },
            "The atoms this constraint acts on.")
        .def("removed_degrees_of_freedom", &Constraint::removed_degrees_of_freedom,
             py::arg("atoms"), "How many degrees of freedom this alone removes.")
        .def("adjust_forces",
             [](const Constraint& self, const Atoms& atoms, std::vector<double> forces) {
                 self.adjust_forces(atoms, forces);
                 return move_to_array(std::move(forces),
                                      {static_cast<py::ssize_t>(atoms.size()), 3});
             },
             py::arg("atoms"), py::arg("forces"),
             "The forces with the forbidden components projected out.")
        .def("adjust_momenta",
             [](const Constraint& self, const Atoms& atoms, std::vector<double> momenta) {
                 self.adjust_momenta(atoms, momenta);
                 return move_to_array(std::move(momenta),
                                      {static_cast<py::ssize_t>(atoms.size()), 3});
             },
             py::arg("atoms"), py::arg("momenta"),
             "The momenta or velocities with the forbidden components projected out.")
        .def("__repr__", &Constraint::to_string);

    py::class_<FixAtoms, Constraint, std::shared_ptr<FixAtoms>>(m, "FixAtoms", R"doc(
Atoms that do not move at all.

The substrate of a surface calculation, the far end of a cluster, the bulk
around a defect. Three degrees of freedom go per atom.

    atoms.constraints = [fa.FixAtoms(range(16))]
    atoms.constraints = [fa.FixAtoms(mask=[z < 5.0 for z in heights])]
)doc")
        .def(py::init([](const py::object& indices, const py::object& mask) {
                 return std::make_shared<FixAtoms>(index_set(indices, mask, "FixAtoms"));
             }),
             py::arg("indices") = py::none(), py::kw_only(), py::arg("mask") = py::none());

    py::class_<FixLine, Constraint, std::shared_ptr<FixLine>>(m, "FixLine", R"doc(
Atoms confined to a line: motion is allowed *along* ``direction`` only.

An atom pulled through a channel, or one held to a reaction coordinate. Two
degrees of freedom go per atom, leaving one. The line passes through wherever
the atom already is, because it is the displacement that is constrained, not
the position.

    atoms.constraints = [fa.FixLine([4], "z")]
    atoms.constraints = [fa.FixLine([4], [1, 1, 0])]
)doc")
        .def(py::init([](const py::object& indices, const py::object& direction,
                         const py::object& mask) {
                 return std::make_shared<FixLine>(index_set(indices, mask, "FixLine"),
                                                  axis(direction, "FixLine"));
             }),
             py::arg("indices") = py::none(), py::arg("direction") = py::none(), py::kw_only(),
             py::arg("mask") = py::none())
        .def_property_readonly("direction", &FixLine::direction,
                               "The allowed direction, normalised.");

    py::class_<FixPlane, Constraint, std::shared_ptr<FixPlane>>(m, "FixPlane", R"doc(
Atoms confined to a plane: motion perpendicular to ``normal`` is forbidden.

An adsorbate kept at a fixed height, or a slab layer free to relax in-plane but
not to buckle. One degree of freedom goes per atom, leaving two. The plane
passes through wherever the atom already is.

    atoms.constraints = [fa.FixPlane([16], "z")]
)doc")
        .def(py::init([](const py::object& indices, const py::object& normal,
                         const py::object& mask) {
                 return std::make_shared<FixPlane>(index_set(indices, mask, "FixPlane"),
                                                   axis(normal, "FixPlane"));
             }),
             py::arg("indices") = py::none(), py::arg("normal") = py::none(), py::kw_only(),
             py::arg("mask") = py::none())
        .def_property_readonly("normal", &FixPlane::normal, "The forbidden direction, normalised.");

    // ----------------------------------------------------------------------
    // The Atoms side
    // ----------------------------------------------------------------------
    atoms.def_property(
        "constraints",
        [](const Atoms& self) {
            py::list out;
            for (const std::shared_ptr<Constraint>& constraint : self.constraints()) {
                out.append(constraint);
            }
            return out;
        },
        [](Atoms& self, const py::object& value) {
            if (value.is_none()) {
                self.clear_constraints();
                return;
            }
            ConstraintList list;
            // One constraint on its own is what people write, and refusing it
            // would be pedantry.
            if (py::isinstance<Constraint>(value)) {
                list.push_back(value.cast<std::shared_ptr<Constraint>>());
            } else {
                for (const py::handle item : value) {
                    list.push_back(item.cast<std::shared_ptr<Constraint>>());
                }
            }
            self.set_constraints(std::move(list));
        },
        "The constraints restricting how these atoms may move.");

    atoms.def("add_constraint", &Atoms::add_constraint, py::arg("constraint"),
              "Attach one more constraint, keeping the ones already there.");
    atoms.def("clear_constraints", &Atoms::clear_constraints, "Remove every constraint.");
    atoms.def("constrained_degrees_of_freedom", &Atoms::constrained_degrees_of_freedom,
              R"doc(
Degrees of freedom the constraints remove.

Counted from the projection itself rather than summed over the constraints, so
an atom named by two of them is not charged twice: a ``FixLine`` and a
``FixPlane`` sharing an atom together leave one direction, not none.
)doc");
    atoms.def(
        "move_to",
        [](Atoms& self, const py::object& positions) {
            self.move_to(as_flat_doubles(positions, static_cast<py::ssize_t>(self.size()), 3,
                                         "move_to"));
        },
        py::arg("positions"),
        R"doc(
Set the positions, projected onto what the constraints allow.

This is what an optimizer or an integrator does; plain ``positions =`` stays
unconstrained, because building a structure is not a move.
)doc");
}

}  // namespace fastatoms::bindings
