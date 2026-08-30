// SPDX-License-Identifier: MIT
#pragma once

/// \file bindings.hpp
/// \brief Shared helpers and per-module registration hooks for the pybind11
///        extension.

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "fastatoms/fastatoms.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Automatic conversion of the small linear-algebra types to and from numpy.
// ---------------------------------------------------------------------------
namespace pybind11::detail {

template <>
struct type_caster<fastatoms::Vec3> {
    PYBIND11_TYPE_CASTER(fastatoms::Vec3, const_name("numpy.ndarray[float64[3]]"));

    bool load(handle src, bool) {
        auto array = array_t<double, array::c_style | array::forcecast>::ensure(src);
        if (!array || array.ndim() != 1 || array.shape(0) != 3) {
            return false;
        }
        const double* data = array.data();
        value = fastatoms::Vec3{data[0], data[1], data[2]};
        return true;
    }

    static handle cast(const fastatoms::Vec3& v, return_value_policy, handle) {
        array_t<double> out(3);
        double* data = out.mutable_data();
        data[0] = v.x;
        data[1] = v.y;
        data[2] = v.z;
        return out.release();
    }
};

template <>
struct type_caster<fastatoms::Mat3> {
    PYBIND11_TYPE_CASTER(fastatoms::Mat3, const_name("numpy.ndarray[float64[3,3]]"));

    bool load(handle src, bool) {
        auto array = array_t<double, array::c_style | array::forcecast>::ensure(src);
        if (!array) {
            return false;
        }
        if (array.ndim() == 2 && array.shape(0) == 3 && array.shape(1) == 3) {
            const double* data = array.data();
            for (std::size_t k = 0; k < 9; ++k) {
                value.m[k] = data[k];
            }
            return true;
        }
        if (array.ndim() == 1 && array.shape(0) == 9) {
            const double* data = array.data();
            for (std::size_t k = 0; k < 9; ++k) {
                value.m[k] = data[k];
            }
            return true;
        }
        if (array.ndim() == 1 && array.shape(0) == 3) {
            // A list of three lengths means a diagonal cell.
            const double* data = array.data();
            value = fastatoms::Mat3::diagonal(data[0], data[1], data[2]);
            return true;
        }
        return false;
    }

    static handle cast(const fastatoms::Mat3& m, return_value_policy, handle) {
        array_t<double> out({static_cast<py::ssize_t>(3), static_cast<py::ssize_t>(3)});
        double* data = out.mutable_data();
        for (std::size_t k = 0; k < 9; ++k) {
            data[k] = m.m[k];
        }
        return out.release();
    }
};


/// `fastatoms::Value` <-> plain Python objects (dict, list, str, int, float,
/// bool, None). Configuration data therefore looks like ordinary Python data
/// on the Python side, with no wrapper class to learn.
template <>
struct type_caster<fastatoms::Value> {
    PYBIND11_TYPE_CASTER(fastatoms::Value, const_name("Union[None, bool, int, float, str, list, dict]"));

    bool load(handle src, bool convert) {
        value = from_python(src, convert);
        return true;
    }

    static handle cast(const fastatoms::Value& v, return_value_policy, handle) {
        return to_python(v).release();
    }

private:
    static fastatoms::Value from_python(handle src, bool convert) {
        using V = fastatoms::Value;
        // Names are fully qualified: inside pybind11::detail some of these
        // shadow unrelated helper templates.
        if (!src || src.is_none()) {
            return V{};
        }
        if (::pybind11::isinstance<::pybind11::bool_>(src)) {
            return V(src.cast<bool>());
        }
        if (::pybind11::isinstance<::pybind11::int_>(src)) {
            return V(src.cast<std::int64_t>());
        }
        if (::pybind11::isinstance<::pybind11::float_>(src)) {
            return V(src.cast<double>());
        }
        if (::pybind11::isinstance<::pybind11::str>(src)) {
            return V(src.cast<std::string>());
        }
        if (::pybind11::isinstance<::pybind11::dict>(src)) {
            V::Map items;
            for (auto item : ::pybind11::reinterpret_borrow<::pybind11::dict>(src)) {
                items.emplace_back(::pybind11::str(item.first).cast<std::string>(),
                                   from_python(item.second, convert));
            }
            return V(std::move(items));
        }
        if (::pybind11::isinstance<::pybind11::list>(src) ||
            ::pybind11::isinstance<::pybind11::tuple>(src)) {
            V::List items;
            for (auto item : ::pybind11::reinterpret_borrow<::pybind11::sequence>(src)) {
                items.push_back(from_python(item, convert));
            }
            return V(std::move(items));
        }
        if (::pybind11::isinstance<::pybind11::array>(src)) {
            V::List items;
            for (auto item : ::pybind11::reinterpret_borrow<::pybind11::sequence>(
                     src.attr("tolist")())) {
                items.push_back(from_python(item, convert));
            }
            return V(std::move(items));
        }
        // Anything else (a Path, a numpy scalar, ...) goes through str().
        return V(::pybind11::str(src).cast<std::string>());
    }

    static object to_python(const fastatoms::Value& v) {
        using Type = fastatoms::Value::Type;
        switch (v.type()) {
            case Type::Null:   return ::pybind11::none();
            case Type::Bool:   return ::pybind11::cast(v.as_bool());
            case Type::Int:    return ::pybind11::cast(v.as_int());
            case Type::Float:  return ::pybind11::cast(v.as_double());
            case Type::String: return ::pybind11::cast(v.as_string());
            case Type::List: {
                ::pybind11::list out;
                for (const fastatoms::Value& item : v.as_list()) {
                    out.append(to_python(item));
                }
                return out;
            }
            case Type::Map: {
                ::pybind11::dict out;
                for (const auto& [key, item] : v.as_map()) {
                    out[::pybind11::str(key)] = to_python(item);
                }
                return out;
            }
        }
        return ::pybind11::none();
    }
};

}  // namespace pybind11::detail

namespace fastatoms::bindings {

/// Hand a `std::vector` over to numpy without copying: the vector is moved to
/// the heap and freed by a capsule when the array goes away.
template <typename T>
py::array_t<T> move_to_array(std::vector<T>&& data, std::vector<py::ssize_t> shape) {
    auto* holder = new std::vector<T>(std::move(data));
    py::capsule owner(holder, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    return py::array_t<T>(std::move(shape), holder->data(), owner);
}

/// A read-only or writable numpy view onto memory owned by `base`.
///
/// The view keeps `base` alive, but it is only valid while the underlying
/// buffer is not reallocated; growing the structure invalidates it.
template <typename T>
py::array_t<T> make_view(T* data, std::vector<py::ssize_t> shape, py::handle base,
                         bool writable = true) {
    py::array_t<T> array(std::move(shape), data, base);
    if (!writable) {
        // numpy's own public API; clearing the flag is always permitted.
        array.attr("setflags")(py::arg("write") = false);
    }
    return array;
}

/// Copy `obj` into a flat row-major buffer, accepting either (rows, cols) or
/// (rows * cols,) shaped input.
[[nodiscard]] std::vector<double> as_flat_doubles(const py::object& obj, py::ssize_t rows,
                                                  py::ssize_t cols, const char* what);

/// As `as_flat_doubles`, but for integer arrays.
[[nodiscard]] std::vector<int> as_flat_ints(const py::object& obj, py::ssize_t rows,
                                            const char* what);

/// Accept `True`, `"xyz"`, or a 3-element sequence as periodic boundary flags.
[[nodiscard]] std::array<bool, 3> parse_pbc(const py::object& obj);

/// Expose boundary conditions the way ASE does: a length-3 bool array.
[[nodiscard]] py::array_t<bool> pbc_to_array(const std::array<bool, 3>& pbc);

/// Turn a Python index, slice, mask or index sequence into concrete indices.
[[nodiscard]] std::vector<std::size_t> resolve_indices(const py::object& key, std::size_t n);

void bind_elements(py::module_& m);
void bind_cell(py::module_& m);
void bind_atoms(py::module_& m);
void bind_neighbor_list(py::module_& m);
void bind_io(py::module_& m);

}  // namespace fastatoms::bindings
