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
    static auto* const optimizer_error = new py::exception<fastatoms::optimize::OptimizerError>(
        m, "OptimizerError", PyExc_RuntimeError);
    static auto* const dynamics_error = new py::exception<fastatoms::md::DynamicsError>(
        m, "DynamicsError", PyExc_RuntimeError);
    static auto* const phonon_error = new py::exception<fastatoms::phonon::PhononError>(
        m, "PhononError", PyExc_RuntimeError);
    static auto* const neb_error =
        new py::exception<fastatoms::neb::NEBError>(m, "NEBError", PyExc_RuntimeError);
    static auto* const constraint_error =
        new py::exception<fastatoms::ConstraintError>(m, "ConstraintError", PyExc_ValueError);
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const fastatoms::CalculatorError& e) {
            py::set_error(*calculator_error, e.what());
        } catch (const fastatoms::optimize::OptimizerError& e) {
            py::set_error(*optimizer_error, e.what());
        } catch (const fastatoms::md::DynamicsError& e) {
            py::set_error(*dynamics_error, e.what());
        } catch (const fastatoms::phonon::PhononError& e) {
            py::set_error(*phonon_error, e.what());
        } catch (const fastatoms::neb::NEBError& e) {
            py::set_error(*neb_error, e.what());
        } catch (const fastatoms::ConstraintError& e) {
            py::set_error(*constraint_error, e.what());
        }
    });

    fastatoms::bindings::bind_cell(m);
    py::class_<fastatoms::Atoms> atoms_class = fastatoms::bindings::bind_atoms(m);
    fastatoms::bindings::bind_neighbor_list(m);
    fastatoms::bindings::bind_constraints(m, atoms_class);
    fastatoms::bindings::bind_elements(m);
    py::class_<fastatoms::io::InputDeck> deck_class = fastatoms::bindings::bind_io(m);
    fastatoms::bindings::bind_calculators(m, atoms_class, deck_class);
    fastatoms::bindings::bind_mlip(m);
    fastatoms::bindings::bind_forcefield(m);
    fastatoms::bindings::bind_neb(m);
    fastatoms::bindings::bind_optimize(m, atoms_class, deck_class);
    fastatoms::bindings::bind_md(m, atoms_class, deck_class);
    fastatoms::bindings::bind_phonon(m, deck_class);
}
