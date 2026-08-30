// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <string>

#include "bindings.hpp"

namespace fastatoms::bindings {
namespace {

constexpr double kPi = 3.14159265358979323846;

/// Expand a formula such as "H2O" or "CH3CH2OH" into atomic numbers.
std::vector<int> parse_formula(const std::string& formula) {
    std::vector<int> numbers;
    std::size_t i = 0;
    while (i < formula.size()) {
        if (std::isspace(static_cast<unsigned char>(formula[i])) != 0) {
            ++i;
            continue;
        }
        if (std::isupper(static_cast<unsigned char>(formula[i])) == 0) {
            throw py::value_error("could not parse chemical formula '" + formula +
                                  "': expected an element symbol at position " +
                                  std::to_string(i));
        }
        std::size_t end = i + 1;
        while (end < formula.size() &&
               std::islower(static_cast<unsigned char>(formula[end])) != 0) {
            ++end;
        }
        const std::string symbol = formula.substr(i, end - i);
        i = end;

        std::size_t count = 0;
        while (i < formula.size() && std::isdigit(static_cast<unsigned char>(formula[i])) != 0) {
            count = count * 10 + static_cast<std::size_t>(formula[i] - '0');
            ++i;
        }
        if (count == 0) {
            count = 1;
        }
        const int z = number_from_symbol(symbol);
        numbers.insert(numbers.end(), count, z);
    }
    if (numbers.empty()) {
        throw py::value_error("could not parse chemical formula '" + formula + "'");
    }
    return numbers;
}

/// Accept a formula string, a list of symbols, or a list of atomic numbers.
std::vector<int> species_to_numbers(const py::object& source) {
    if (py::isinstance<py::str>(source)) {
        return parse_formula(source.cast<std::string>());
    }
    std::vector<int> numbers;
    for (const py::handle item : source) {
        if (py::isinstance<py::str>(item)) {
            numbers.push_back(number_from_symbol(item.cast<std::string>()));
        } else {
            numbers.push_back(item.cast<int>());
        }
    }
    return numbers;
}

/// `atoms[i]`: a live handle on one atom of a parent structure.
struct AtomProxy {
    py::object owner;
    std::size_t index{0};

    [[nodiscard]] Atoms& atoms() const { return owner.cast<Atoms&>(); }
};

/// A write-through view of `Atoms::info()`.
///
/// Returning a plain dict would make `atoms.info["energy"] = -1.5` look like it
/// worked and quietly change nothing, which for the field extended XYZ exists
/// to carry is worse than not offering it at all.
struct InfoProxy {
    py::object owner;

    [[nodiscard]] Atoms& atoms() const { return owner.cast<Atoms&>(); }

    [[nodiscard]] Value& map() const {
        Atoms& target = atoms();
        if (!target.info().is_map()) {
            target.set_info(Value(Value::Map{}));
        }
        return target.info();
    }
};

Cell cell_from_object(const py::object& source) {
    if (source.is_none()) {
        return Cell{};
    }
    if (py::isinstance<Cell>(source)) {
        return source.cast<Cell>();
    }
    if (py::isinstance<py::float_>(source) || py::isinstance<py::int_>(source)) {
        return Cell::cubic(source.cast<double>());
    }
    return Cell(source.cast<Mat3>(), {true, true, true});
}

}  // namespace

py::class_<Atoms> bind_atoms(py::module_& m) {
    py::class_<AtomProxy>(m, "Atom", R"doc(
A view on a single atom of an :class:`Atoms` object.

Writes go straight through to the parent structure, so ``atoms[3].position[0] = 1.0``
moves the atom. The view is invalidated if the parent changes size.
)doc")
        .def_readonly("index", &AtomProxy::index)
        .def_property_readonly("atoms", [](const AtomProxy& self) { return self.owner; },
                               "The parent structure.")
        .def_property(
            "symbol", [](const AtomProxy& self) { return self.atoms().symbol(self.index); },
            [](AtomProxy& self, const std::string& symbol) {
                self.atoms().set_number(self.index, number_from_symbol(symbol));
            })
        .def_property(
            "number", [](const AtomProxy& self) { return self.atoms().number(self.index); },
            [](AtomProxy& self, int z) { self.atoms().set_number(self.index, z); })
        .def_property(
            "position",
            [](const AtomProxy& self) {
                return make_view(self.atoms().positions_data() + 3 * self.index,
                                 {static_cast<py::ssize_t>(3)}, self.owner);
            },
            [](AtomProxy& self, const Vec3& r) { self.atoms().set_position(self.index, r); })
        .def_property(
            "mass", [](const AtomProxy& self) { return self.atoms().mass(self.index); },
            [](AtomProxy& self, double value) {
                Atoms& atoms = self.atoms();
                std::vector<double> masses = atoms.masses();
                masses[self.index] = value;
                atoms.set_masses(masses);
            })
        .def_property(
            "charge",
            [](const AtomProxy& self) { return self.atoms().charges()[self.index]; },
            [](AtomProxy& self, double value) {
                Atoms& atoms = self.atoms();
                std::vector<double> charges = atoms.charges();
                charges[self.index] = value;
                atoms.set_charges(charges);
            })
        .def_property(
            "tag", [](const AtomProxy& self) { return self.atoms().tags()[self.index]; },
            [](AtomProxy& self, int value) {
                Atoms& atoms = self.atoms();
                std::vector<int> tags = atoms.tags();
                tags[self.index] = value;
                atoms.set_tags(tags);
            })
        .def("__repr__", [](const AtomProxy& self) {
            const Atoms& atoms = self.atoms();
            const Vec3 r = atoms.position(self.index);
            char buffer[160];
            std::snprintf(buffer, sizeof(buffer),
                          "Atom('%s', [%.6f, %.6f, %.6f], index=%zu)",
                          atoms.symbol(self.index).c_str(), r.x, r.y, r.z, self.index);
            return std::string(buffer);
        });

    py::class_<InfoProxy>(m, "AtomsInfo",
                          "A write-through view of a structure's frame information.")
        .def("__getitem__",
             [](const InfoProxy& self, const std::string& key) {
                 const Value* found = self.map().find(key);
                 if (found == nullptr) {
                     throw py::key_error(key);
                 }
                 return *found;
             })
        .def("__setitem__",
             [](const InfoProxy& self, const std::string& key, const Value& value) {
                 self.map().insert(key, value);
             })
        .def("__delitem__",
             [](const InfoProxy& self, const std::string& key) {
                 Value& info = self.map();
                 Value::Map remaining;
                 bool removed = false;
                 for (const auto& [name, value] : info.as_map()) {
                     if (name == key) {
                         removed = true;
                     } else {
                         remaining.emplace_back(name, value);
                     }
                 }
                 if (!removed) {
                     throw py::key_error(key);
                 }
                 self.atoms().set_info(Value(std::move(remaining)));
             })
        .def("__contains__",
             [](const InfoProxy& self, const std::string& key) {
                 return self.map().find(key) != nullptr;
             })
        .def("__len__", [](const InfoProxy& self) { return self.map().as_map().size(); })
        .def("__iter__",
             [](const InfoProxy& self) {
                 py::list keys;
                 for (const auto& [name, value] : self.map().as_map()) {
                     (void)value;
                     keys.append(py::str(name));
                 }
                 return py::iter(keys);
             })
        .def("keys",
             [](const InfoProxy& self) {
                 py::list keys;
                 for (const auto& [name, value] : self.map().as_map()) {
                     (void)value;
                     keys.append(py::str(name));
                 }
                 return keys;
             })
        .def("items",
             [](const InfoProxy& self) {
                 py::list items;
                 for (const auto& [name, value] : self.map().as_map()) {
                     items.append(py::make_tuple(py::str(name), py::cast(value)));
                 }
                 return items;
             })
        .def("get",
             [](const InfoProxy& self, const std::string& key, const py::object& fallback) {
                 const Value* found = self.map().find(key);
                 return found != nullptr ? py::cast(*found) : fallback;
             },
             py::arg("key"), py::arg("default") = py::none())
        .def("update",
             [](const InfoProxy& self, const Value& other) {
                 if (other.is_map()) {
                     for (const auto& [name, value] : other.as_map()) {
                         self.map().insert(name, value);
                     }
                 }
             })
        .def("clear", [](const InfoProxy& self) { self.atoms().set_info(Value(Value::Map{})); })
        .def("__repr__", [](const InfoProxy& self) {
            return "AtomsInfo(" + self.map().to_string() + ")";
        });

    py::class_<Atoms> atoms_class(m, "Atoms", R"doc(
A collection of atoms in an optionally periodic cell.

The API mirrors ASE's ``Atoms`` closely enough that most scripts port over
unchanged, but every array lives in contiguous C++ storage. ``atoms.positions``
and ``atoms.numbers`` are zero-copy numpy views into that storage: writing to
them writes straight into the C++ buffers.

.. warning::
   A view is invalidated when the structure changes size (``append``,
   ``extend``, ``del``). Re-read the property after such a call.
)doc");

    atoms_class.def(py::init([](const py::object& symbols, const py::object& positions,
                         const py::object& numbers, const py::object& cell,
                         const py::object& pbc, const py::object& masses,
                         const py::object& charges, const py::object& tags,
                         const py::object& momenta, const py::object& velocities,
                         const py::object& scaled_positions) {
                 std::vector<int> z;
                 if (!numbers.is_none()) {
                     z = py::cast<std::vector<int>>(numbers);
                 } else if (!symbols.is_none()) {
                     z = species_to_numbers(symbols);
                 }

                 const py::object& coords =
                     scaled_positions.is_none() ? positions : scaled_positions;
                 std::size_t natoms = z.size();
                 if (z.empty() && !coords.is_none()) {
                     auto array =
                         py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
                             coords);
                     if (!array || array.size() % 3 != 0) {
                         throw py::value_error("positions must have shape (N, 3)");
                     }
                     natoms = static_cast<std::size_t>(array.size() / 3);
                     z.assign(natoms, 0);
                 }

                 std::vector<double> r(3 * natoms, 0.0);
                 if (!coords.is_none()) {
                     r = as_flat_doubles(coords, static_cast<py::ssize_t>(natoms), 3, "positions");
                 }

                 Cell c = cell_from_object(cell);
                 if (!pbc.is_none()) {
                     c.set_pbc(parse_pbc(pbc));
                 } else if (c.is_zero()) {
                     c.set_pbc(false);
                 }

                 Atoms atoms(std::move(z), std::move(r), c);
                 if (!scaled_positions.is_none()) {
                     const std::vector<double> fractional = atoms.positions();
                     atoms.set_scaled_positions(fractional);
                 }
                 if (!masses.is_none()) {
                     atoms.set_masses(
                         as_flat_doubles(masses, static_cast<py::ssize_t>(natoms), 1, "masses"));
                 }
                 if (!charges.is_none()) {
                     atoms.set_charges(
                         as_flat_doubles(charges, static_cast<py::ssize_t>(natoms), 1, "charges"));
                 }
                 if (!tags.is_none()) {
                     atoms.set_tags(
                         as_flat_ints(tags, static_cast<py::ssize_t>(natoms), "tags"));
                 }
                 if (!momenta.is_none()) {
                     atoms.set_momenta(
                         as_flat_doubles(momenta, static_cast<py::ssize_t>(natoms), 3, "momenta"));
                 } else if (!velocities.is_none()) {
                     atoms.set_velocities(as_flat_doubles(
                         velocities, static_cast<py::ssize_t>(natoms), 3, "velocities"));
                 }
                 return atoms;
             }),
             py::arg("symbols") = py::none(), py::arg("positions") = py::none(),
             py::kw_only(), py::arg("numbers") = py::none(), py::arg("cell") = py::none(),
             py::arg("pbc") = py::none(), py::arg("masses") = py::none(),
             py::arg("charges") = py::none(), py::arg("tags") = py::none(),
             py::arg("momenta") = py::none(), py::arg("velocities") = py::none(),
             py::arg("scaled_positions") = py::none(),
             R"doc(
Create a structure.

``symbols`` accepts a formula string (``"H2O"``), a list of symbols, or a list
of atomic numbers. ``cell`` accepts a :class:`Cell`, a 3x3 matrix, three edge
lengths, or a single number for a cubic cell.
)doc")

        // --- size and identity --------------------------------------------
        .def("__len__", &Atoms::size)
        .def("__repr__", &Atoms::to_string)
        // As in ASE, a copy does not carry the calculator over: a stored
        // snapshot should not keep a live engine alive.
        .def("copy", [](const Atoms& self) {
                 Atoms out(self);
                 out.set_calculator(nullptr);
                 return out;
             },
             "An independent copy of this structure, without the calculator.")
        .def("__copy__", [](const Atoms& self) {
                 Atoms out(self);
                 out.set_calculator(nullptr);
                 return out;
             })
        .def("__deepcopy__", [](const Atoms& self, const py::dict&) {
                 Atoms out(self);
                 out.set_calculator(nullptr);
                 return out;
             },
             py::arg("memo"))

        // --- positions ------------------------------------------------------
        .def_property(
            "positions",
            [](py::object self) {
                Atoms& atoms = self.cast<Atoms&>();
                return make_view(atoms.positions_data(),
                                 {static_cast<py::ssize_t>(atoms.size()),
                                  static_cast<py::ssize_t>(3)},
                                 self);
            },
            [](Atoms& self, const py::object& value) {
                self.set_positions(as_flat_doubles(
                    value, static_cast<py::ssize_t>(self.size()), 3, "positions"));
            },
            "Cartesian positions as a zero-copy (N, 3) view.")
        .def(
            "get_positions",
            [](const Atoms& self) {
                std::vector<double> r = self.positions();
                return move_to_array(std::move(r),
                                     {static_cast<py::ssize_t>(self.size()),
                                      static_cast<py::ssize_t>(3)});
            },
            "A copy of the Cartesian positions, shape (N, 3).")
        .def(
            "set_positions",
            [](Atoms& self, const py::object& value) {
                self.set_positions(as_flat_doubles(
                    value, static_cast<py::ssize_t>(self.size()), 3, "positions"));
            },
            py::arg("positions"))

        // --- species ---------------------------------------------------------
        .def_property(
            "numbers",
            [](py::object self) {
                Atoms& atoms = self.cast<Atoms&>();
                return make_view(atoms.numbers_data(),
                                 {static_cast<py::ssize_t>(atoms.size())}, self);
            },
            [](Atoms& self, const py::object& value) {
                self.set_numbers(
                    as_flat_ints(value, static_cast<py::ssize_t>(self.size()), "numbers"));
            },
            "Atomic numbers as a zero-copy (N,) view.")
        .def("get_atomic_numbers", [](const Atoms& self) { return self.numbers(); })
        .def(
            "set_atomic_numbers",
            [](Atoms& self, const py::object& value) {
                self.set_numbers(
                    as_flat_ints(value, static_cast<py::ssize_t>(self.size()), "numbers"));
            },
            py::arg("numbers"))
        .def_property(
            "symbols", [](const Atoms& self) { return self.symbols(); },
            [](Atoms& self, const py::object& value) {
                const std::vector<int> z = species_to_numbers(value);
                self.set_numbers(z);
            },
            "Chemical symbols as a list of strings.")
        .def("get_chemical_symbols", &Atoms::symbols)
        .def("set_chemical_symbols", &Atoms::set_symbols, py::arg("symbols"))
        .def("get_chemical_formula", &Atoms::formula, "Formula in Hill notation, e.g. 'H2O'.")

        // --- cell and boundary conditions -------------------------------------
        .def_property(
            "cell", [](Atoms& self) -> Cell& { return self.cell(); },
            [](Atoms& self, const py::object& value) { self.set_cell(cell_from_object(value)); },
            py::return_value_policy::reference_internal,
            "The simulation :class:`Cell`; mutating it changes the structure.")
        .def("get_cell", [](const Atoms& self) { return self.cell(); },
             "A copy of the simulation cell.")
        .def(
            "set_cell",
            [](Atoms& self, const py::object& value, bool scale_atoms) {
                self.set_cell(cell_from_object(value), scale_atoms);
            },
            py::arg("cell"), py::arg("scale_atoms") = false,
            "Replace the cell, optionally keeping the fractional coordinates fixed.")
        .def_property(
            "pbc", [](const Atoms& self) { return pbc_to_array(self.pbc()); },
            [](Atoms& self, const py::object& value) { self.set_pbc(parse_pbc(value)); },
            "Periodic boundary conditions as a length-3 boolean array.")
        .def("get_pbc", [](const Atoms& self) { return pbc_to_array(self.pbc()); })
        .def(
            "set_pbc", [](Atoms& self, const py::object& value) { self.set_pbc(parse_pbc(value)); },
            py::arg("pbc"))
        .def("get_volume", &Atoms::volume, "Cell volume in cubic angstrom.")

        // --- fractional coordinates -------------------------------------------
        .def(
            "get_scaled_positions",
            [](const Atoms& self, bool wrap) {
                std::vector<double> s = self.scaled_positions(wrap);
                return move_to_array(std::move(s), {static_cast<py::ssize_t>(self.size()),
                                                    static_cast<py::ssize_t>(3)});
            },
            py::arg("wrap") = true, "Fractional coordinates, shape (N, 3).")
        .def(
            "set_scaled_positions",
            [](Atoms& self, const py::object& value) {
                self.set_scaled_positions(as_flat_doubles(
                    value, static_cast<py::ssize_t>(self.size()), 3, "scaled_positions"));
            },
            py::arg("scaled_positions"))
        .def("wrap", &Atoms::wrap, py::arg("eps") = 1e-7,
             "Fold every atom into the cell along the periodic directions.")

        // --- optional arrays ----------------------------------------------------
        .def(
            "get_masses",
            [](const Atoms& self) {
                std::vector<double> masses = self.masses();
                return move_to_array(std::move(masses),
                                     {static_cast<py::ssize_t>(self.size())});
            },
            "Masses in amu, falling back to standard atomic weights.")
        .def(
            "set_masses",
            [](Atoms& self, const py::object& value) {
                if (value.is_none()) {
                    self.set_masses({});
                    return;
                }
                self.set_masses(
                    as_flat_doubles(value, static_cast<py::ssize_t>(self.size()), 1, "masses"));
            },
            py::arg("masses"))
        .def_property(
            "masses",
            [](const Atoms& self) {
                std::vector<double> masses = self.masses();
                return move_to_array(std::move(masses),
                                     {static_cast<py::ssize_t>(self.size())});
            },
            [](Atoms& self, const py::object& value) {
                self.set_masses(
                    as_flat_doubles(value, static_cast<py::ssize_t>(self.size()), 1, "masses"));
            })
        .def(
            "get_momenta",
            [](const Atoms& self) {
                std::vector<double> p = self.momenta();
                return move_to_array(std::move(p), {static_cast<py::ssize_t>(self.size()),
                                                    static_cast<py::ssize_t>(3)});
            })
        .def(
            "set_momenta",
            [](Atoms& self, const py::object& value) {
                if (value.is_none()) {
                    self.set_momenta({});
                    return;
                }
                self.set_momenta(as_flat_doubles(value, static_cast<py::ssize_t>(self.size()), 3,
                                                 "momenta"));
            },
            py::arg("momenta"))
        .def(
            "get_velocities",
            [](const Atoms& self) {
                std::vector<double> v = self.velocities();
                return move_to_array(std::move(v), {static_cast<py::ssize_t>(self.size()),
                                                    static_cast<py::ssize_t>(3)});
            })
        .def(
            "set_velocities",
            [](Atoms& self, const py::object& value) {
                if (value.is_none()) {
                    self.set_momenta({});
                    return;
                }
                self.set_velocities(as_flat_doubles(value, static_cast<py::ssize_t>(self.size()),
                                                    3, "velocities"));
            },
            py::arg("velocities"))
        .def(
            "get_charges",
            [](const Atoms& self) {
                std::vector<double> q = self.charges();
                return move_to_array(std::move(q), {static_cast<py::ssize_t>(self.size())});
            })
        .def(
            "set_charges",
            [](Atoms& self, const py::object& value) {
                if (value.is_none()) {
                    self.set_charges({});
                    return;
                }
                self.set_charges(
                    as_flat_doubles(value, static_cast<py::ssize_t>(self.size()), 1, "charges"));
            },
            py::arg("charges"))
        .def(
            "get_tags",
            [](const Atoms& self) {
                std::vector<int> tags = self.tags();
                return move_to_array(std::move(tags), {static_cast<py::ssize_t>(self.size())});
            })
        .def(
            "set_tags",
            [](Atoms& self, const py::object& value) {
                if (value.is_none()) {
                    self.set_tags({});
                    return;
                }
                self.set_tags(as_flat_ints(value, static_cast<py::ssize_t>(self.size()), "tags"));
            },
            py::arg("tags"))

        .def(
            "set_array",
            [](Atoms& self, const std::string& name, const py::object& value) {
                auto array =
                    py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
                if (!array) {
                    throw py::type_error("set_array: expected an array of floats");
                }
                const auto n = static_cast<py::ssize_t>(self.size());
                if (n == 0 || array.size() % n != 0) {
                    throw py::value_error("set_array: " + std::to_string(array.size()) +
                                          " values do not divide evenly among " +
                                          std::to_string(self.size()) + " atoms");
                }
                const auto ncols = static_cast<std::size_t>(array.size() / n);
                std::vector<double> data(array.data(), array.data() + array.size());
                self.set_array(name, data, ncols);
            },
            py::arg("name"), py::arg("data"),
            "Attach an arbitrary per-atom array; it follows the atoms through "
            "``extend``, slicing and ``repeat``.")
        .def(
            "get_array",
            [](const Atoms& self, const std::string& name) {
                const Atoms::Array& array = self.array(name);
                std::vector<double> data = array.data;
                if (array.ncols == 1) {
                    return move_to_array(std::move(data),
                                         {static_cast<py::ssize_t>(self.size())});
                }
                return move_to_array(std::move(data),
                                     {static_cast<py::ssize_t>(self.size()),
                                      static_cast<py::ssize_t>(array.ncols)});
            },
            py::arg("name"))
        .def("has_array", &Atoms::has_array, py::arg("name"))
        .def("remove_array", &Atoms::remove_array, py::arg("name"))
        .def("array_names", &Atoms::array_names, "Names of the user-defined per-atom arrays.")

        // --- geometry -------------------------------------------------------------
        .def("get_distance", &Atoms::distance, py::arg("a"), py::arg("b"), py::arg("mic") = false,
             "Distance between two atoms, optionally minimum-image.")
        .def(
            "get_distances",
            [](const Atoms& self, std::size_t a, const py::object& indices, bool mic) {
                std::vector<double> all = self.distances_from(a, mic);
                if (indices.is_none()) {
                    return move_to_array(std::move(all),
                                         {static_cast<py::ssize_t>(self.size())});
                }
                const std::vector<std::size_t> chosen = resolve_indices(indices, self.size());
                std::vector<double> out;
                out.reserve(chosen.size());
                for (const std::size_t index : chosen) {
                    out.push_back(all[index]);
                }
                const auto n = static_cast<py::ssize_t>(out.size());
                return move_to_array(std::move(out), {n});
            },
            py::arg("a"), py::arg("indices") = py::none(), py::arg("mic") = false,
            "Distances from atom ``a`` to the selected atoms.")
        .def(
            "get_all_distances",
            [](const Atoms& self, bool mic) {
                std::vector<double> d = self.all_distances(mic);
                const auto n = static_cast<py::ssize_t>(self.size());
                return move_to_array(std::move(d), {n, n});
            },
            py::arg("mic") = false, "The full N x N distance matrix.")
        .def(
            "get_distance_vector",
            [](const Atoms& self, std::size_t a, std::size_t b, bool mic) {
                return self.distance_vector(a, b, mic);
            },
            py::arg("a"), py::arg("b"), py::arg("mic") = false,
            "Displacement ``r_b - r_a``.")
        .def("get_center_of_mass", &Atoms::center_of_mass)
        .def("get_center_of_geometry", &Atoms::center_of_geometry)
        .def("get_moments_of_inertia_tensor", &Atoms::moments_of_inertia_tensor,
             "The symmetric 3x3 inertia tensor about the centre of mass.")

        .def(
            "translate",
            [](Atoms& self, const py::object& shift) {
                auto array =
                    py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(shift);
                if (array && array.size() == static_cast<py::ssize_t>(3 * self.size()) &&
                    self.size() != 1) {
                    self.translate(std::vector<double>(array.data(), array.data() + array.size()));
                    return;
                }
                self.translate(shift.cast<Vec3>());
            },
            py::arg("displacement"),
            "Shift every atom by one vector, or by a per-atom (N, 3) array.")
        .def(
            "rotate",
            [](Atoms& self, double angle, const Vec3& axis, const py::object& center,
               bool degrees) {
                const double radians = degrees ? angle * kPi / 180.0 : angle;
                if (center.is_none()) {
                    self.rotate(radians, axis);
                } else {
                    self.rotate(radians, axis, center.cast<Vec3>());
                }
            },
            py::arg("angle"), py::arg("axis"), py::arg("center") = py::none(),
            py::arg("degrees") = true,
            "Rotate about ``axis`` through ``center`` (the centre of mass by default).")
        .def(
            "center",
            [](Atoms& self, const py::object& vacuum, const py::object& axis) {
                std::array<bool, 3> axes{true, true, true};
                if (!axis.is_none()) {
                    axes = {false, false, false};
                    if (py::isinstance<py::int_>(axis)) {
                        axes[static_cast<std::size_t>(axis.cast<int>())] = true;
                    } else {
                        for (const py::handle item : axis) {
                            axes[static_cast<std::size_t>(item.cast<int>())] = true;
                        }
                    }
                }
                self.center(vacuum.is_none() ? -1.0 : vacuum.cast<double>(), axes);
            },
            py::arg("vacuum") = py::none(), py::arg("axis") = py::none(),
            "Centre the atoms in the cell, optionally resizing it to leave "
            "``vacuum`` angstrom of space on each side.")

        // --- dynamics ---------------------------------------------------------------
        .def("get_kinetic_energy", &Atoms::kinetic_energy, "Kinetic energy in eV.")
        .def(
            "get_temperature",
            [](const Atoms& self, const py::object& dof) {
                return self.temperature(dof.is_none() ? -1 : dof.cast<int>());
            },
            py::arg("dof") = py::none(), "Instantaneous temperature in kelvin.")
        .def("get_total_momentum", &Atoms::total_momentum)
        .def("zero_center_of_mass_momentum", &Atoms::zero_center_of_mass_momentum,
             "Remove the net drift from the momenta.")

        // --- composition --------------------------------------------------------------
        .def(
            "append",
            [](Atoms& self, const py::object& species, const Vec3& position) {
                if (py::isinstance<py::str>(species)) {
                    self.append(species.cast<std::string>(), position);
                } else {
                    self.append(species.cast<int>(), position);
                }
            },
            py::arg("symbol_or_number"), py::arg("position"),
            "Add one atom. Invalidates existing array views.")
        .def("extend", &Atoms::extend, py::arg("other"),
             "Append another structure's atoms; its cell is ignored.")
        .def("repeat", &Atoms::repeat, py::arg("nx"), py::arg("ny"), py::arg("nz"),
             "Build an (nx, ny, nz) supercell.")
        .def(
            "__mul__",
            [](const Atoms& self, const py::object& reps) {
                if (py::isinstance<py::int_>(reps)) {
                    const int n = reps.cast<int>();
                    return self.repeat(n, n, n);
                }
                const auto r = py::cast<std::vector<int>>(reps);
                if (r.size() != 3) {
                    throw py::value_error("repetitions must be an int or a sequence of 3 ints");
                }
                return self.repeat(r[0], r[1], r[2]);
            },
            py::arg("repetitions"))
        .def("__rmul__",
             [](py::object self, const py::object& reps) {
                 return self.attr("__mul__")(reps);
             },
             py::arg("repetitions"))
        .def(
            "__add__",
            [](const Atoms& self, const Atoms& other) {
                Atoms out(self);
                out.extend(other);
                return out;
            },
            py::arg("other"))
        .def(
            "__iadd__",
            [](Atoms& self, const Atoms& other) -> Atoms& {
                self.extend(other);
                return self;
            },
            py::arg("other"))

        .def(
            "__getitem__",
            [](py::object self, const py::object& key) -> py::object {
                Atoms& atoms = self.cast<Atoms&>();
                if (py::isinstance<py::int_>(key)) {
                    py::ssize_t index = key.cast<py::ssize_t>();
                    if (index < 0) {
                        index += static_cast<py::ssize_t>(atoms.size());
                    }
                    if (index < 0 || index >= static_cast<py::ssize_t>(atoms.size())) {
                        throw py::index_error("atom index out of range");
                    }
                    return py::cast(AtomProxy{self, static_cast<std::size_t>(index)});
                }
                return py::cast(atoms.subset(resolve_indices(key, atoms.size())));
            },
            py::arg("key"),
            "``atoms[i]`` gives an :class:`Atom` view; a slice, mask or index list "
            "gives a new :class:`Atoms`.")
        .def(
            "__delitem__",
            [](Atoms& self, const py::object& key) {
                self.remove(resolve_indices(key, self.size()));
            },
            py::arg("key"))
        .def(
            "__iter__",
            [](py::object self) {
                // Each proxy holds its own reference to the parent, so the list
                // needs no keep_alive (and a list_iterator has no weak
                // references for one to attach to).
                Atoms& atoms = self.cast<Atoms&>();
                py::list items;
                for (std::size_t i = 0; i < atoms.size(); ++i) {
                    items.append(py::cast(AtomProxy{self, i}));
                }
                return py::iter(items);
            })
        .def("subset", &Atoms::subset, py::arg("indices"),
             "A new structure holding only the given atoms, in that order.")

        .def_property(
            "info",
            [](py::object self) { return InfoProxy{std::move(self)}; },
            [](Atoms& self, const Value& value) { self.set_info(value); },
            R"doc(
Per-frame scalars, as a write-through mapping.

    atoms.info["energy"] = -1.5
    atoms.info["step"]

This is what an extended XYZ comment line carries beyond the structure - the
energy of a frame, its timestep, a label - and it survives a write and a read.
Named and shaped after ASE's ``atoms.info``.
)doc")

        .def_property_readonly("arrays", [](py::object self) {
            Atoms& atoms = self.cast<Atoms&>();
            py::dict out;
            out["positions"] = self.attr("positions");
            out["numbers"] = self.attr("numbers");
            for (const std::string& name : atoms.array_names()) {
                out[py::str(name)] = self.attr("get_array")(name);
            }
            return out;
        }, "The per-atom arrays as a dict; 'positions' and 'numbers' are live views.");

    return atoms_class;
}

}  // namespace fastatoms::bindings
