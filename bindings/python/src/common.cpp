// SPDX-License-Identifier: MIT
#include "common.hpp"

#include <utility>

namespace fastatoms::bindings {

Value value_from_py(const py::handle& obj) {
    if (!obj || obj.is_none()) {
        return Value{};
    }
    // bool before int: bool is a subclass of int in Python.
    if (py::isinstance<py::bool_>(obj)) {
        return Value(obj.cast<bool>());
    }
    if (py::isinstance<py::int_>(obj)) {
        return Value(obj.cast<std::int64_t>());
    }
    if (py::isinstance<py::float_>(obj)) {
        return Value(obj.cast<double>());
    }
    if (py::isinstance<py::str>(obj)) {
        return Value(obj.cast<std::string>());
    }
    if (py::isinstance<py::dict>(obj)) {
        Value::Map items;
        for (auto kv : py::reinterpret_borrow<py::dict>(obj)) {
            if (!py::isinstance<py::str>(kv.first)) {
                throw py::type_error("fastatoms values need string keys");
            }
            items.emplace_back(kv.first.cast<std::string>(), value_from_py(kv.second));
        }
        return Value(std::move(items));
    }
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        Value::List items;
        for (auto item : py::reinterpret_borrow<py::iterable>(obj)) {
            items.push_back(value_from_py(item));
        }
        return Value(std::move(items));
    }
    throw py::type_error("cannot convert this Python object to a fastatoms value; "
                         "use None, bool, int, float, str, list or dict");
}

py::object value_to_py(const Value& value) {
    switch (value.type()) {
        case Value::Type::Null:
            return py::none();
        case Value::Type::Bool:
            return py::bool_(value.as_bool());
        case Value::Type::Int:
            return py::int_(value.as_int());
        case Value::Type::Float:
            return py::float_(value.as_double());
        case Value::Type::String:
            return py::str(value.as_string());
        case Value::Type::List: {
            py::list out;
            for (const Value& item : value.as_list()) {
                out.append(value_to_py(item));
            }
            return out;
        }
        case Value::Type::Map: {
            py::dict out;
            for (const auto& [key, item] : value.as_map()) {
                out[py::str(key)] = value_to_py(item);
            }
            return out;
        }
    }
    return py::none();
}

Value options_from_kwargs(const py::kwargs& kwargs) {
    Value::Map items;
    for (auto kv : kwargs) {
        items.emplace_back(py::str(py::reinterpret_borrow<py::object>(kv.first)).cast<std::string>(), value_from_py(kv.second));
    }
    return Value(std::move(items));
}

Mat3 mat3_from_py(const py::object& obj) {
    if (obj.is_none()) {
        return Mat3{};
    }
    auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr) {
        throw py::type_error("cell: expected a (3, 3) array, a (3,) diagonal or None");
    }
    if (arr.ndim() == 2 && arr.shape(0) == 3 && arr.shape(1) == 3) {
        Mat3 m;
        for (std::size_t k = 0; k < 9; ++k) {
            m.m[k] = arr.data()[k];
        }
        return m;
    }
    if (arr.ndim() == 1 && arr.shape(0) == 3) {
        const double* d = arr.data();
        return Mat3::diagonal(d[0], d[1], d[2]);
    }
    throw py::value_error("cell must have shape (3, 3) or (3,)");
}

py::array_t<double> mat3_to_py(const Mat3& m) {
    std::vector<double> data(m.m.begin(), m.m.end());
    return move_to_array(std::move(data), {3, 3});
}

std::array<bool, 3> pbc_from_py(const py::object& obj) {
    if (obj.is_none()) {
        return {false, false, false};
    }
    if (py::isinstance<py::sequence>(obj) && !py::isinstance<py::str>(obj)) {
        std::array<bool, 3> out{};
        py::ssize_t n = 0;
        for (auto item : py::reinterpret_borrow<py::iterable>(obj)) {
            if (n >= 3) {
                throw py::value_error("pbc must have three entries");
            }
            out[static_cast<std::size_t>(n++)] = item.cast<bool>();
        }
        if (n != 3) {
            throw py::value_error("pbc must have three entries");
        }
        return out;
    }
    const bool all = py::bool_(obj).cast<bool>();
    return {all, all, all};
}

py::array_t<bool> pbc_to_py(const std::array<bool, 3>& pbc) {
    py::array_t<bool> out(3);
    bool* data = out.mutable_data();
    for (std::size_t k = 0; k < 3; ++k) {
        data[k] = pbc[k];
    }
    return out;
}

std::array<int, 3> triple_from_py(const py::object& obj, const char* what) {
    if (py::isinstance<py::int_>(obj)) {
        const int n = obj.cast<int>();
        return {n, n, n};
    }
    auto seq = py::reinterpret_borrow<py::iterable>(obj);
    std::array<int, 3> out{};
    std::size_t n = 0;
    for (auto item : seq) {
        if (n >= 3) {
            throw py::value_error(std::string(what) + " must be an int or three ints");
        }
        out[n++] = item.cast<int>();
    }
    if (n != 3) {
        throw py::value_error(std::string(what) + " must be an int or three ints");
    }
    return out;
}

}  // namespace fastatoms::bindings
