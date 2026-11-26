// SPDX-License-Identifier: MIT
#include <cstdint>
#include <string>
#include <vector>

#include "common.hpp"
#include "fastatoms/core/elements.hpp"

namespace fastatoms::bindings {
namespace {

using Index = py::ssize_t;

/// Species from a list of atomic numbers and/or chemical symbols.
std::vector<int> species_from_py(const py::object& obj) {
    std::vector<int> numbers;
    for (auto item : py::reinterpret_borrow<py::iterable>(obj)) {
        if (py::isinstance<py::str>(item)) {
            numbers.push_back(number_from_symbol(item.cast<std::string>()));
        } else {
            numbers.push_back(item.cast<int>());
        }
    }
    return numbers;
}

py::array_t<std::int64_t> numbers_to_py(const Atoms& a) {
    std::vector<std::int64_t> data(a.numbers().begin(), a.numbers().end());
    return move_to_array(std::move(data), {static_cast<Index>(a.size())});
}

py::array_t<double> vec3s_to_py(std::vector<double> data, Index n) {
    return move_to_array(std::move(data), {n, 3});
}

/// Build a numpy array for an extra per-atom array with `ncols` columns.
template <typename T>
py::array_t<T> per_atom_to_py(std::vector<T> data, Index n, std::size_t ncols) {
    if (ncols == 1) {
        return move_to_array(std::move(data), {n});
    }
    return move_to_array(std::move(data), {n, static_cast<Index>(ncols)});
}

/// The `arrays` dict: built-in per-atom arrays that are set, then user arrays.
/// Every value is a fresh numpy array (or a list of str for string arrays).
py::dict arrays_to_py(const Atoms& a) {
    const auto n = static_cast<Index>(a.size());
    py::dict out;
    out["numbers"] = numbers_to_py(a);
    out["positions"] = vec3s_to_py(std::vector<double>(a.positions()), n);
    if (a.has_masses()) {
        out["masses"] = move_to_array(a.masses(), {n});
    }
    if (a.has_momenta()) {
        out["momenta"] = vec3s_to_py(a.momenta(), n);
    }
    if (a.has_tags()) {
        const std::vector<int> tags = a.tags();
        out["tags"] = move_to_array(std::vector<std::int64_t>(tags.begin(), tags.end()), {n});
    }
    if (a.has_charges()) {
        out["charges"] = move_to_array(a.charges(), {n});
    }
    for (const auto& name : a.array_names()) {
        const Atoms::Array& arr = a.array(name);
        out[py::str(name)] = per_atom_to_py(std::vector<double>(arr.data), n, arr.ncols);
    }
    for (const auto& name : a.int_array_names()) {
        const Atoms::IntArray& arr = a.int_array(name);
        out[py::str(name)] = per_atom_to_py(std::vector<std::int64_t>(arr.data), n, arr.ncols);
    }
    for (const auto& name : a.string_array_names()) {
        out[py::str(name)] = py::cast(a.string_array(name));
    }
    return out;
}

/// Set an extra array from numpy-like data, choosing the storage by dtype:
/// floats -> double array, ints/bools -> int array, anything else -> strings.
void set_array_from_py(Atoms& a, const std::string& name, const py::object& obj) {
    auto arr = py::array::ensure(obj);
    if (!arr) {
        throw py::type_error("set_array: expected an array-like");
    }
    const auto n = static_cast<Index>(a.size());
    const Index ncols = arr.ndim() == 2 ? arr.shape(1) : 1;
    const char kind = arr.dtype().kind();
    if (kind == 'f') {
        auto data = as_flat<double>(obj, n, ncols, "set_array");
        a.set_array(name, data, static_cast<std::size_t>(ncols));
    } else if (kind == 'i' || kind == 'u' || kind == 'b') {
        auto data = as_flat<std::int64_t>(obj, n, ncols, "set_array");
        a.set_int_array(name, data, static_cast<std::size_t>(ncols));
    } else {
        if (ncols != 1 || arr.ndim() > 1) {
            throw py::value_error("set_array: string arrays must be one-dimensional");
        }
        std::vector<std::string> strings;
        for (auto item : py::reinterpret_borrow<py::iterable>(obj)) {
            strings.push_back(py::str(py::reinterpret_borrow<py::object>(item)).cast<std::string>());
        }
        a.set_string_array(name, strings);
    }
}

}  // namespace

void bind_atoms(py::module_& m) {
    py::class_<Atoms>(m, "Atoms")
        .def(py::init([](const py::object& numbers, const py::object& positions,
                         const py::object& cell, const py::object& pbc) {
                 if (numbers.is_none()) {
                     return Atoms(std::vector<int>{}, std::vector<double>{},
                                  Cell(mat3_from_py(cell), pbc_from_py(pbc)));
                 }
                 std::vector<int> z = species_from_py(numbers);
                 const auto n = static_cast<Index>(z.size());
                 std::vector<double> r = positions.is_none()
                                             ? std::vector<double>{}
                                             : as_flat<double>(positions, n, 3, "positions");
                 return Atoms(std::move(z), std::move(r),
                              Cell(mat3_from_py(cell), pbc_from_py(pbc)));
             }),
             py::arg("numbers") = py::none(), py::arg("positions") = py::none(),
             py::arg("cell") = py::none(), py::arg("pbc") = false,
             "Atoms from atomic numbers or symbols, (N, 3) positions, a (3, 3) cell and pbc.")

        .def("__len__", &Atoms::size)
        .def("__repr__", &Atoms::to_string)
        .def("copy", [](const Atoms& a) { return a; })

        // Positions, species and cell. Getters return copies (see move_to_array).
        .def_property(
            "numbers", [](const Atoms& a) { return numbers_to_py(a); },
            [](Atoms& a, const py::object& obj) {
                auto data = as_flat<std::int64_t>(obj, static_cast<Index>(a.size()), 1, "numbers");
                a.set_numbers(std::vector<int>(data.begin(), data.end()));
            })
        .def_property(
            "positions",
            [](const Atoms& a) {
                return vec3s_to_py(std::vector<double>(a.positions()), static_cast<Index>(a.size()));
            },
            [](Atoms& a, const py::object& obj) {
                a.set_positions(as_flat<double>(obj, static_cast<Index>(a.size()), 3, "positions"));
            })
        .def_property(
            "symbols", [](const Atoms& a) { return a.symbols(); },
            [](Atoms& a, const std::vector<std::string>& symbols) { a.set_symbols(symbols); })
        .def_property_readonly("labels", [](const Atoms& a) { return a.labels(); })
        .def_property_readonly("formula", &Atoms::formula)
        .def_property(
            "cell", [](const Atoms& a) { return mat3_to_py(a.cell().matrix()); },
            [](Atoms& a, const py::object& obj) { a.set_cell(mat3_from_py(obj), false); })
        .def_property(
            "pbc", [](const Atoms& a) { return pbc_to_py(a.pbc()); },
            [](Atoms& a, const py::object& obj) { a.set_pbc(pbc_from_py(obj)); })
        .def_property_readonly("volume", &Atoms::volume)

        .def("set_cell", [](Atoms& a, const py::object& cell, bool scale_atoms) {
            a.set_cell(mat3_from_py(cell), scale_atoms);
        }, py::arg("cell"), py::arg("scale_atoms") = false)

        .def("scaled_positions",
             [](const Atoms& a, bool wrap) {
                 const auto n = static_cast<Index>(a.size());
                 return vec3s_to_py(a.scaled_positions(wrap), n);
             },
             py::arg("wrap") = false, "Fractional coordinates, (N, 3).")
        .def("wrap", &Atoms::wrap, py::arg("eps") = 1e-7)

        .def("repeat",
             [](const Atoms& a, const py::object& reps) {
                 const auto r = triple_from_py(reps, "repeat");
                 return a.repeat(r[0], r[1], r[2]);
             },
             py::arg("reps"), "Supercell with reps = (nx, ny, nz) or a single int.")

        // Per-frame info (ASE atoms.info). A copy is returned each time.
        .def_property(
            "info", [](const Atoms& a) { return value_to_py(a.info()); },
            [](Atoms& a, const py::object& obj) { a.set_info(value_from_py(obj)); })

        // Extra arrays. `arrays` is a fresh dict of copies.
        .def_property_readonly("arrays", [](const Atoms& a) { return arrays_to_py(a); })
        .def("set_array", &set_array_from_py, py::arg("name"), py::arg("value"))
        .def("remove_array",
             [](Atoms& a, const std::string& name) {
                 if (a.has_array(name)) {
                     a.remove_array(name);
                 } else if (a.has_int_array(name)) {
                     a.remove_int_array(name);
                 } else if (a.has_string_array(name)) {
                     a.remove_string_array(name);
                 } else {
                     throw py::key_error(name);
                 }
             },
             py::arg("name"))

        // Built-in per-atom arrays. Getters return the defaults when unset
        // (masses from the periodic table, zeros for the others).
        .def_property(
            "masses", [](const Atoms& a) { return move_to_array(a.masses(), {static_cast<Index>(a.size())}); },
            [](Atoms& a, const py::object& obj) {
                a.set_masses(as_flat<double>(obj, static_cast<Index>(a.size()), 1, "masses"));
            })
        .def_property(
            "momenta", [](const Atoms& a) { return vec3s_to_py(a.momenta(), static_cast<Index>(a.size())); },
            [](Atoms& a, const py::object& obj) {
                a.set_momenta(as_flat<double>(obj, static_cast<Index>(a.size()), 3, "momenta"));
            })
        .def_property(
            "charges", [](const Atoms& a) { return move_to_array(a.charges(), {static_cast<Index>(a.size())}); },
            [](Atoms& a, const py::object& obj) {
                a.set_charges(as_flat<double>(obj, static_cast<Index>(a.size()), 1, "charges"));
            })
        .def_property(
            "tags",
            [](const Atoms& a) {
                const std::vector<int> tags = a.tags();
                return move_to_array(std::vector<std::int64_t>(tags.begin(), tags.end()),
                                     {static_cast<Index>(a.size())});
            },
            [](Atoms& a, const py::object& obj) {
                auto data = as_flat<std::int64_t>(obj, static_cast<Index>(a.size()), 1, "tags");
                a.set_tags(std::vector<int>(data.begin(), data.end()));
            });
}

}  // namespace fastatoms::bindings
