// SPDX-License-Identifier: MIT
///
/// The `fastatoms._core` extension module.
///
/// Registration is split across translation units so that each part of the API
/// can be compiled (and rebuilt) independently.

#include "bindings.hpp"

PYBIND11_MODULE(_core, m) {
    m.doc() = R"doc(
FastAtoms - a high-performance C++ core for atomistic simulations.

This is the compiled extension. Import the :mod:`fastatoms` package instead;
it re-exports everything here alongside a few pure-Python conveniences.
)doc";

    m.attr("__version__") = fastatoms::version();

    // Input problems get their own exception so scripts can catch a bad input
    // file without also catching genuine bugs.
    static py::exception<fastatoms::InputError> input_error(m, "InputError", PyExc_ValueError);
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const fastatoms::InputError& e) {
            py::set_error(input_error, e.what());
        }
    });

    fastatoms::bindings::bind_cell(m);
    fastatoms::bindings::bind_atoms(m);
    fastatoms::bindings::bind_neighbor_list(m);
    fastatoms::bindings::bind_elements(m);
    fastatoms::bindings::bind_io(m);
}
