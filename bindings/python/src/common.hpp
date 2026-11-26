// SPDX-License-Identifier: MIT
#pragma once

/// \file common.hpp
/// \brief Helpers shared by the pybind11 test module: numpy ownership,
///        Python <-> Value and Python <-> Mat3 conversions.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/core/value.hpp"

namespace fastatoms::bindings {

namespace py = pybind11;

/// Hand a freshly computed buffer to numpy without a copy. The vector is moved
/// to the heap and freed by a capsule once the last array referring to it dies,
/// so the array can outlive the C++ call and the source object.
///
/// Only for data the binding owns (results of a computation). Views onto an
/// `Atoms` member would dangle as soon as the structure reallocates, so the
/// `Atoms` getters copy into a fresh vector and pass it here.
template <typename T>
py::array_t<T> move_to_array(std::vector<T>&& data, std::vector<py::ssize_t> shape) {
    if (data.empty()) {
        return py::array_t<T>(std::move(shape));
    }
    auto* holder = new std::vector<T>(std::move(data));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    return py::array_t<T>(std::move(shape), holder->data(), owner);
}

/// Copy `obj` into a flat row-major buffer. Accepts a (rows, cols) array or a
/// flat sequence of rows * cols values.
/// \throws py::value_error on a shape mismatch.
template <typename T>
std::vector<T> as_flat(const py::object& obj, py::ssize_t rows, py::ssize_t cols,
                       const char* what) {
    auto arr = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr) {
        throw py::type_error(std::string(what) + ": expected a numeric array or sequence");
    }
    const bool shaped = arr.ndim() == 2 && arr.shape(0) == rows && arr.shape(1) == cols;
    const bool flat = arr.ndim() == 1 && arr.shape(0) == rows * cols;
    if (!shaped && !flat) {
        throw py::value_error(std::string(what) + ": expected shape (" + std::to_string(rows) +
                              ", " + std::to_string(cols) + "), got " +
                              std::to_string(arr.ndim()) + " dimensions");
    }
    return std::vector<T>(arr.data(), arr.data() + arr.size());
}

/// Python -> Value. None, bool, int, float, str, list/tuple and str-keyed dict.
[[nodiscard]] Value value_from_py(const py::handle& obj);

/// Value -> plain Python objects (dict, list, str, int, float, bool, None).
[[nodiscard]] py::object value_to_py(const Value& value);

/// Keyword arguments of a format function, as a Value map.
[[nodiscard]] Value options_from_kwargs(const py::kwargs& kwargs);

/// A 3x3 cell from None (zero cell), a (3, 3) matrix or a (3,) diagonal.
[[nodiscard]] Mat3 mat3_from_py(const py::object& obj);

[[nodiscard]] py::array_t<double> mat3_to_py(const Mat3& m);

/// Boundary flags from a bool or a sequence of three bools.
[[nodiscard]] std::array<bool, 3> pbc_from_py(const py::object& obj);

[[nodiscard]] py::array_t<bool> pbc_to_py(const std::array<bool, 3>& pbc);

/// Three ints from an int or a sequence of three ints.
[[nodiscard]] std::array<int, 3> triple_from_py(const py::object& obj, const char* what);

/// Register the Atoms class and its free functions.
void bind_atoms(py::module_& m);
void bind_formats(py::module_& m);
void bind_neighbors(py::module_& m);
void bind_symmetry(py::module_& m);  // no-op unless FASTATOMS_WITH_SYMMETRY

}  // namespace fastatoms::bindings
