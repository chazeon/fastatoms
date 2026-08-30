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
    //
    // These are deliberately never destroyed. A `py::exception` holds a Python
    // object, and a static one would decref it from a C++ static destructor at
    // process exit - after the interpreter has been finalised.
    static auto* const input_error =
        new py::exception<fastatoms::InputError>(m, "InputError", PyExc_ValueError);
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const fastatoms::InputError& e) {
            py::set_error(*input_error, e.what());
        }
    });

    static auto* const calculator_error =
        new py::exception<fastatoms::CalculatorError>(m, "CalculatorError", PyExc_RuntimeError);
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const fastatoms::CalculatorError& e) {
            py::set_error(*calculator_error, e.what());
        }
    });

    fastatoms::bindings::bind_cell(m);
    py::class_<fastatoms::Atoms> atoms_class = fastatoms::bindings::bind_atoms(m);
    fastatoms::bindings::bind_neighbor_list(m);
    fastatoms::bindings::bind_elements(m);
    py::class_<fastatoms::io::InputDeck> deck_class = fastatoms::bindings::bind_io(m);
    fastatoms::bindings::bind_calculators(m, atoms_class, deck_class);
}
