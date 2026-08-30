// SPDX-License-Identifier: MIT
#include "bindings.hpp"

namespace fastatoms::bindings {

void bind_neighbor_list(py::module_& m) {
    py::class_<NeighborList>(m, "NeighborList", R"doc(
A pair list in compressed-sparse-row form.

Pairs are ordered by ``i``, so ``first[a]:first[a + 1]`` is the block of
neighbours of atom ``a``. All arrays are zero-copy views into the C++ storage
and stay valid as long as this object does.

``shifts`` holds the integer cell offset of the image that was used::

    r_ij = positions[j] + shifts @ cell - positions[i]
)doc")
        .def_property_readonly(
            "i",
            [](py::object self) {
                NeighborList& nl = self.cast<NeighborList&>();
                return make_view(nl.i.data(), {static_cast<py::ssize_t>(nl.i.size())}, self,
                                 false);
            },
            "First atom of each pair, shape (npairs,).")
        .def_property_readonly(
            "j",
            [](py::object self) {
                NeighborList& nl = self.cast<NeighborList&>();
                return make_view(nl.j.data(), {static_cast<py::ssize_t>(nl.j.size())}, self,
                                 false);
            },
            "Second atom of each pair, shape (npairs,).")
        .def_property_readonly(
            "shifts",
            [](py::object self) {
                NeighborList& nl = self.cast<NeighborList&>();
                return make_view(nl.shifts.data(),
                                 {static_cast<py::ssize_t>(nl.size()),
                                  static_cast<py::ssize_t>(3)},
                                 self, false);
            },
            "Integer cell offsets, shape (npairs, 3).")
        .def_property_readonly(
            "vectors",
            [](py::object self) -> py::object {
                NeighborList& nl = self.cast<NeighborList&>();
                if (nl.vectors.empty()) {
                    return py::none();
                }
                return make_view(nl.vectors.data(),
                                 {static_cast<py::ssize_t>(nl.size()),
                                  static_cast<py::ssize_t>(3)},
                                 self, false);
            },
            "Minimum-image displacement of each pair, shape (npairs, 3), or None.")
        .def_property_readonly(
            "distances",
            [](py::object self) -> py::object {
                NeighborList& nl = self.cast<NeighborList&>();
                if (nl.distances.empty()) {
                    return py::none();
                }
                return make_view(nl.distances.data(),
                                 {static_cast<py::ssize_t>(nl.distances.size())}, self, false);
            },
            "Pair distances, shape (npairs,), or None.")
        .def_property_readonly(
            "first",
            [](py::object self) {
                NeighborList& nl = self.cast<NeighborList&>();
                return make_view(nl.first.data(),
                                 {static_cast<py::ssize_t>(nl.first.size())}, self, false);
            },
            "CSR row offsets, shape (natoms + 1,).")
        .def_readonly("cutoff", &NeighborList::cutoff)
        .def_readonly("full", &NeighborList::full,
                      "True when both (i, j) and (j, i) are stored.")
        .def("__len__", &NeighborList::size)
        .def(
            "get_neighbors",
            [](py::object self, std::size_t index) {
                NeighborList& nl = self.cast<NeighborList&>();
                if (index >= nl.natoms()) {
                    throw py::index_error("atom index out of range");
                }
                const std::size_t begin = nl.first[index];
                const std::size_t count = nl.count(index);
                auto neighbors = make_view(nl.j.data() + begin,
                                           {static_cast<py::ssize_t>(count)}, self, false);
                auto offsets = make_view(nl.shifts.data() + 3 * begin,
                                         {static_cast<py::ssize_t>(count),
                                          static_cast<py::ssize_t>(3)},
                                         self, false);
                return py::make_tuple(neighbors, offsets);
            },
            py::arg("index"),
            "The neighbours of one atom as ``(indices, cell_offsets)``.")
        .def(
            "coordination_numbers",
            [](const NeighborList& self) {
                std::vector<std::int32_t> counts = self.coordination_numbers();
                const auto n = static_cast<py::ssize_t>(counts.size());
                return move_to_array(std::move(counts), {n});
            },
            "Number of neighbours of every atom.")
        .def("__repr__", [](const NeighborList& self) {
            return "NeighborList(npairs=" + std::to_string(self.size()) +
                   ", natoms=" + std::to_string(self.natoms()) +
                   ", cutoff=" + std::to_string(self.cutoff) +
                   (self.full ? ", full)" : ", half)");
        });

    m.def(
        "neighbor_list",
        [](const Atoms& atoms, const py::object& cutoff, const py::object& per_atom_cutoffs,
           bool full, bool self_interaction, bool store_vectors, bool store_distances) {
            NeighborListOptions options;
            if (!per_atom_cutoffs.is_none()) {
                options.per_atom_cutoffs = as_flat_doubles(
                    per_atom_cutoffs, static_cast<py::ssize_t>(atoms.size()), 1,
                    "per_atom_cutoffs");
            } else if (!cutoff.is_none()) {
                options.cutoff = cutoff.cast<double>();
            } else {
                throw py::value_error("neighbor_list: pass either cutoff or per_atom_cutoffs");
            }
            options.full = full;
            options.self_interaction = self_interaction;
            options.store_vectors = store_vectors;
            options.store_distances = store_distances;

            py::gil_scoped_release release;
            return build_neighbor_list(atoms, options);
        },
        py::arg("atoms"), py::arg("cutoff") = py::none(), py::kw_only(),
        py::arg("per_atom_cutoffs") = py::none(), py::arg("full") = true,
        py::arg("self_interaction") = false, py::arg("store_vectors") = true,
        py::arg("store_distances") = true,
        R"doc(
Build a neighbour list in O(N) time.

Handles any mix of periodic and free directions, triclinic cells, and cutoffs
larger than the cell (multiple images of a pair are then returned with
different shifts). Releases the GIL while it runs.

Pass ``per_atom_cutoffs`` (see :func:`natural_cutoffs`) instead of ``cutoff``
to accept pairs closer than the sum of their two radii.
)doc");
}

}  // namespace fastatoms::bindings
