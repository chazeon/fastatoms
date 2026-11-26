// SPDX-License-Identifier: MIT
/// \file module.cpp
/// \brief Entry point of the `fastatoms._core` test module.

#include "common.hpp"
#include "fastatoms/core/value.hpp"
#include "fastatoms/io/format.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Test-only pybind11 bindings for the fastatoms C++ core. Not for PyPI.";

    // Malformed input is a ValueError in Python, with a named subclass for
    // each C++ error type so tests can tell them apart.
    py::register_exception<fastatoms::InputError>(m, "InputError", PyExc_ValueError);
    py::register_exception<fastatoms::io::FormatError>(m, "FormatError", PyExc_ValueError);

    fastatoms::bindings::bind_atoms(m);
    fastatoms::bindings::bind_formats(m);
    fastatoms::bindings::bind_neighbors(m);
#ifdef FASTATOMS_WITH_SYMMETRY
    fastatoms::bindings::bind_symmetry(m);
#endif
}
