// SPDX-License-Identifier: MIT
#include "bindings.hpp"

#include <algorithm>
#include <cstring>

namespace fastatoms::bindings {

std::vector<double> as_flat_doubles(const py::object& obj, py::ssize_t rows, py::ssize_t cols,
                                    const char* what) {
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!array) {
        throw py::type_error(std::string(what) + ": expected an array of floats");
    }
    const py::ssize_t expected = rows * cols;
    if (array.size() != expected) {
        throw py::value_error(std::string(what) + ": expected " + std::to_string(expected) +
                              " values (shape (" + std::to_string(rows) + ", " +
                              std::to_string(cols) + ")), got " + std::to_string(array.size()));
    }
    std::vector<double> out(static_cast<std::size_t>(expected));
    if (expected > 0) {
        std::memcpy(out.data(), array.data(), static_cast<std::size_t>(expected) * sizeof(double));
    }
    return out;
}

std::vector<int> as_flat_ints(const py::object& obj, py::ssize_t rows, const char* what) {
    auto array = py::array_t<int, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!array) {
        throw py::type_error(std::string(what) + ": expected an array of integers");
    }
    if (array.size() != rows) {
        throw py::value_error(std::string(what) + ": expected " + std::to_string(rows) +
                              " values, got " + std::to_string(array.size()));
    }
    std::vector<int> out(static_cast<std::size_t>(rows));
    if (rows > 0) {
        std::memcpy(out.data(), array.data(), static_cast<std::size_t>(rows) * sizeof(int));
    }
    return out;
}

std::array<bool, 3> parse_pbc(const py::object& obj) {
    if (py::isinstance<py::bool_>(obj)) {
        const bool value = obj.cast<bool>();
        return {value, value, value};
    }
    if (py::isinstance<py::str>(obj)) {
        const std::string text = obj.cast<std::string>();
        std::array<bool, 3> pbc{false, false, false};
        for (const char c : text) {
            switch (c) {
                case 'x': case 'X': pbc[0] = true; break;
                case 'y': case 'Y': pbc[1] = true; break;
                case 'z': case 'Z': pbc[2] = true; break;
                default:
                    throw py::value_error(
                        "pbc: expected some combination of 'x', 'y' and 'z', got '" + text + "'");
            }
        }
        return pbc;
    }

    py::sequence seq;
    try {
        seq = py::reinterpret_borrow<py::sequence>(py::cast<py::sequence>(obj));
    } catch (const py::cast_error&) {
        throw py::type_error("pbc: expected a bool, a string such as 'xy', or a sequence of 3");
    }
    if (py::len(seq) != 3) {
        throw py::value_error("pbc: expected 3 values, got " + std::to_string(py::len(seq)));
    }
    return {py::cast<bool>(seq[0]), py::cast<bool>(seq[1]), py::cast<bool>(seq[2])};
}

py::array_t<bool> pbc_to_array(const std::array<bool, 3>& pbc) {
    py::array_t<bool> out(3);
    bool* data = out.mutable_data();
    data[0] = pbc[0];
    data[1] = pbc[1];
    data[2] = pbc[2];
    return out;
}

std::vector<std::size_t> resolve_indices(const py::object& key, std::size_t n) {
    const auto size = static_cast<py::ssize_t>(n);
    std::vector<std::size_t> indices;

    if (py::isinstance<py::slice>(key)) {
        py::ssize_t start = 0, stop = 0, step = 0, length = 0;
        if (!key.cast<py::slice>().compute(size, &start, &stop, &step, &length)) {
            throw py::error_already_set();
        }
        indices.reserve(static_cast<std::size_t>(length));
        for (py::ssize_t k = 0; k < length; ++k) {
            indices.push_back(static_cast<std::size_t>(start + k * step));
        }
        return indices;
    }

    // A boolean mask selects the atoms it marks.
    auto mask = py::array_t<bool, py::array::c_style>::ensure(key);
    if (mask && mask.ndim() == 1 && py::isinstance<py::array>(key) &&
        py::cast<py::array>(key).dtype().is(py::dtype::of<bool>())) {
        if (mask.size() != size) {
            throw py::value_error("boolean mask has " + std::to_string(mask.size()) +
                                  " entries but there are " + std::to_string(n) + " atoms");
        }
        const bool* data = mask.data();
        for (py::ssize_t k = 0; k < size; ++k) {
            if (data[k]) {
                indices.push_back(static_cast<std::size_t>(k));
            }
        }
        return indices;
    }

    if (py::isinstance<py::int_>(key)) {
        py::ssize_t index = key.cast<py::ssize_t>();
        if (index < 0) {
            index += size;
        }
        if (index < 0 || index >= size) {
            throw py::index_error("atom index out of range");
        }
        return {static_cast<std::size_t>(index)};
    }

    py::sequence seq;
    try {
        seq = py::cast<py::sequence>(key);
    } catch (const py::cast_error&) {
        throw py::type_error("indices must be an int, a slice, a boolean mask, or a sequence "
                             "of ints");
    }
    indices.reserve(static_cast<std::size_t>(py::len(seq)));
    for (const py::handle item : seq) {
        py::ssize_t index = py::cast<py::ssize_t>(item);
        if (index < 0) {
            index += size;
        }
        if (index < 0 || index >= size) {
            throw py::index_error("atom index out of range");
        }
        indices.push_back(static_cast<std::size_t>(index));
    }
    return indices;
}

}  // namespace fastatoms::bindings
