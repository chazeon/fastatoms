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
    // Derived from CalculatorError on both sides, so `except CalculatorError`
    // still catches them and code that wants to tell them apart can.
    static auto* const property_error = new py::exception<fastatoms::PropertyNotImplementedError>(
        m, "PropertyNotImplementedError", calculator_error->ptr());
    static auto* const mlip_error = new py::exception<fastatoms::mlip::MLIPError>(
        m, "MLIPError", calculator_error->ptr());
    static auto* const extxyz_error =
        new py::exception<fastatoms::io::ExtxyzError>(m, "ExtxyzError", PyExc_ValueError);
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
    static auto* const build_error =
        new py::exception<fastatoms::build::BuildError>(m, "BuildError", PyExc_ValueError);
    static auto* const symmetry_error = new py::exception<fastatoms::symmetry::SymmetryError>(
        m, "SymmetryError", PyExc_RuntimeError);
    static auto* const analysis_error = new py::exception<fastatoms::analysis::AnalysisError>(
        m, "AnalysisError", PyExc_ValueError);
    static auto* const electronic_error =
        new py::exception<fastatoms::electronic::ElectronicError>(m, "ElectronicError",
                                                                  PyExc_ValueError);
    static auto* const adsorption_error =
        new py::exception<fastatoms::adsorption::AdsorptionError>(m, "AdsorptionError",
                                                                  PyExc_ValueError);
    static auto* const global_error =
        new py::exception<fastatoms::globalopt::GlobalOptimizationError>(
            m, "GlobalOptimizationError", PyExc_RuntimeError);
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const fastatoms::mlip::MLIPError& e) {
            py::set_error(*mlip_error, e.what());
        } catch (const fastatoms::PropertyNotImplementedError& e) {
            py::set_error(*property_error, e.what());
        } catch (const fastatoms::CalculatorError& e) {
            py::set_error(*calculator_error, e.what());
        } catch (const fastatoms::io::ExtxyzError& e) {
            py::set_error(*extxyz_error, e.what());
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
        } catch (const fastatoms::build::BuildError& e) {
            py::set_error(*build_error, e.what());
        } catch (const fastatoms::symmetry::SymmetryError& e) {
            py::set_error(*symmetry_error, e.what());
        } catch (const fastatoms::analysis::AnalysisError& e) {
            py::set_error(*analysis_error, e.what());
        } catch (const fastatoms::electronic::ElectronicError& e) {
            py::set_error(*electronic_error, e.what());
        } catch (const fastatoms::adsorption::AdsorptionError& e) {
            py::set_error(*adsorption_error, e.what());
        } catch (const fastatoms::globalopt::GlobalOptimizationError& e) {
            py::set_error(*global_error, e.what());
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
    fastatoms::bindings::bind_build(m);
    fastatoms::bindings::bind_symmetry(m, atoms_class);
    fastatoms::bindings::bind_analysis(m);
    fastatoms::bindings::bind_electronic(m);
    fastatoms::bindings::bind_adsorption(m);
    fastatoms::bindings::bind_globalopt(m);
    fastatoms::bindings::bind_optimize(m, atoms_class, deck_class);
    fastatoms::bindings::bind_md(m, atoms_class, deck_class);
    fastatoms::bindings::bind_phonon(m, deck_class);
}
