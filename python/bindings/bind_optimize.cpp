// SPDX-License-Identifier: MIT
#include "bindings.hpp"

#include "fastatoms/optimize/bfgs.hpp"
#include "fastatoms/optimize/conjugate_gradient.hpp"
#include "fastatoms/optimize/fire.hpp"
#include "fastatoms/optimize/lbfgs.hpp"
#include "fastatoms/optimize/registry.hpp"
#include "fastatoms/optimize/steepest_descent.hpp"

namespace fastatoms::bindings {
namespace {

using namespace fastatoms::optimize;

/// Names registered from Python, dropped again at module teardown so that no
/// Python object outlives the interpreter inside a C++ static.
std::vector<std::string>& python_registered_optimizers() {
    static std::vector<std::string> names;
    return names;
}

std::vector<double> to_vector(const py::object& value, const char* what) {
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array) {
        throw py::type_error(std::string(what) + ": expected an array of floats");
    }
    return {array.data(), array.data() + array.size()};
}

py::array_t<double> to_array(std::vector<double> values) {
    const auto n = static_cast<py::ssize_t>(values.size());
    return move_to_array(std::move(values), {n});
}

}  // namespace

void bind_optimize(py::module_& m, py::class_<Atoms>& atoms_class,
                   py::class_<io::InputDeck>& deck_class) {
    // ---------------------------------------------------------------------
    // Results
    // ---------------------------------------------------------------------
    py::enum_<StopReason>(m, "StopReason", "Why a relaxation stopped.")
        .value("converged", StopReason::Converged)
        .value("max_steps", StopReason::MaxSteps)
        .value("line_search_failed", StopReason::LineSearchFailed)
        .value("no_progress", StopReason::NoProgress)
        .value("interrupted", StopReason::Interrupted);

    py::class_<OptimizationStep>(m, "OptimizationStep", "One line of the optimization log.")
        .def_readonly("step", &OptimizationStep::step, "0 is the starting geometry.")
        .def_readonly("evaluations", &OptimizationStep::evaluations,
                      "Force evaluations used so far - the cost that matters for DFT.")
        .def_readonly("energy", &OptimizationStep::energy, "eV")
        .def_readonly("fmax", &OptimizationStep::fmax,
                      "Largest gradient row norm in eV/angstrom; includes the cell rows "
                      "when the cell is being relaxed.")
        .def_readonly("atomic_fmax", &OptimizationStep::atomic_fmax,
                      "The same over the atomic rows only.")
        .def_readonly("energy_change", &OptimizationStep::energy_change, "eV since the last step")
        .def_readonly("max_displacement", &OptimizationStep::max_displacement,
                      "Furthest any atom moved, in angstrom.")
        .def_readonly("step_length", &OptimizationStep::step_length,
                      "Scale factor the line search accepted.")
        .def("__repr__", [](const OptimizationStep& self) {
            char buffer[192];
            std::snprintf(buffer, sizeof(buffer),
                          "OptimizationStep(step=%d, energy=%.6f, fmax=%.6f)", self.step,
                          self.energy, self.fmax);
            return std::string(buffer);
        });

    py::class_<OptimizationResult>(m, "OptimizationResult", "The outcome of a relaxation.")
        .def_readonly("converged", &OptimizationResult::converged)
        .def_readonly("reason", &OptimizationResult::reason)
        .def_readonly("steps", &OptimizationResult::steps)
        .def_readonly("evaluations", &OptimizationResult::evaluations)
        .def_readonly("energy", &OptimizationResult::energy, "eV at the final geometry")
        .def_readonly("fmax", &OptimizationResult::fmax, "eV/angstrom at the final geometry")
        .def_readonly("message", &OptimizationResult::message)
        .def_readonly("history", &OptimizationResult::history,
                      "Every step, starting with the initial geometry.")
        .def("__bool__", [](const OptimizationResult& self) { return self.converged; })
        .def("__repr__", &OptimizationResult::to_string);

    // ---------------------------------------------------------------------
    // Optimizers
    // ---------------------------------------------------------------------
    py::class_<Optimizer, std::shared_ptr<Optimizer>> optimizer(m, "Optimizer", R"doc(
Base class for the geometry optimizers.

    atoms.calc = fa.LennardJones()
    result = fa.BFGS(fmax=0.01).run(atoms)
    print(result.converged, result.steps, result.energy)

The structure is relaxed in place. Convergence means that the largest force on
any single atom - not the largest Cartesian component - has fallen below
``fmax``.
)doc");

    optimizer
        .def_property_readonly("name", &Optimizer::name)
        .def_property(
            "parameters", [](const Optimizer& self) { return self.parameters(); },
            [](Optimizer& self, const Value& value) { self.set_parameters(value); },
            "Configuration as a plain dict.")
        .def("set", &Optimizer::set_parameter, py::arg("key"), py::arg("value"),
             "Set one parameter.")
        .def_property_readonly("fmax", &Optimizer::fmax, "Force threshold in eV/angstrom.")
        .def_property_readonly("max_steps", &Optimizer::max_steps, "Step budget.")
        .def_property_readonly("max_step", &Optimizer::max_step,
                               "Trust radius in angstrom: the furthest any atom may move in "
                               "one step.")
        .def_property_readonly(
            "line_search",
            [](const Optimizer& self) {
                return std::string(line_search_name(self.line_search_kind()));
            })

        .def("run", &Optimizer::run, py::arg("atoms"),
             "Relax `atoms` in place using its attached calculator.")
        .def(
            "minimize",
            [](Optimizer& self, const py::object& objective, const py::object& x0) {
                std::vector<double> x = to_vector(x0, "x0");
                const ObjectiveFunction wrapped =
                    [&objective](const std::vector<double>& point) {
                        py::gil_scoped_acquire gil;
                        const py::object returned = objective(to_array(point));
                        if (!py::isinstance<py::tuple>(returned) &&
                            !py::isinstance<py::list>(returned)) {
                            throw py::type_error(
                                "the objective must return a (energy, gradient) pair, got " +
                                std::string(py::str(py::type::of(returned).attr("__name__"))));
                        }
                        const auto pair = py::cast<py::sequence>(returned);
                        if (py::len(pair) != 2) {
                            throw py::value_error(
                                "the objective must return exactly (energy, gradient), got " +
                                std::to_string(py::len(pair)) + " values");
                        }
                        Evaluation out;
                        out.energy = pair[0].cast<double>();
                        out.gradient = to_vector(pair[1].cast<py::object>(), "gradient");
                        return out;
                    };
                OptimizationResult result = self.minimize(wrapped, x);
                return py::make_tuple(std::move(result), to_array(std::move(x)));
            },
            py::arg("objective"), py::arg("x0"),
            R"doc(
Minimise an arbitrary function, with no structure involved.

``objective(x)`` takes an array and returns ``(energy, gradient)``. Returns
``(result, x)``. The length of ``x`` must be a multiple of three, because the
convergence test groups the gradient in threes.
)doc")

        .def(
            "attach",
            [](Optimizer& self, const py::object& callback) {
                self.attach([callback](const OptimizationStep& step) {
                    py::gil_scoped_acquire gil;
                    const py::object keep_going = callback(step);
                    // Returning nothing means "carry on", which is the useful
                    // default for a logging or trajectory-writing callback.
                    return keep_going.is_none() || keep_going.cast<bool>();
                });
            },
            py::arg("callback"),
            "Call `callback(step)` after every accepted step. Returning False stops the run.")
        .def("clear_callbacks", &Optimizer::clear_callbacks)
        .def("__repr__", &Optimizer::to_string);

    const auto from_kwargs = [](const py::kwargs& kwargs) {
        return py::cast<Value>(py::dict(kwargs));
    };

    py::class_<SteepestDescent, Optimizer, std::shared_ptr<SteepestDescent>>(m, "SteepestDescent",
                                                                            R"doc(
Steepest descent: step straight downhill, ``d = -g / alpha``.

The slowest of the four - on an ill-conditioned surface it zig-zags across the
valley rather than running along it - but completely robust, and the baseline
the others improve on.
)doc")
        .def(py::init([&from_kwargs](const py::kwargs& kwargs) {
            return std::make_shared<SteepestDescent>(from_kwargs(kwargs));
        }));

    py::class_<ConjugateGradient, Optimizer, std::shared_ptr<ConjugateGradient>>(
        m, "ConjugateGradient", R"doc(
Nonlinear conjugate gradient, ``d_k = -g_k / alpha + beta_k d_{k-1}``.

Mixing in the previous direction removes the zig-zag while storing only one
extra vector. ``variant`` selects the formula for ``beta``:
``polak_ribiere`` (default), ``fletcher_reeves`` or ``hestenes_stiefel``.
)doc")
        .def(py::init([&from_kwargs](const py::kwargs& kwargs) {
                 return std::make_shared<ConjugateGradient>(from_kwargs(kwargs));
             }))
        .def_property_readonly("variant", [](const ConjugateGradient& self) {
            return std::string(cg_variant_name(self.variant()));
        });

    py::class_<BFGS, Optimizer, std::shared_ptr<BFGS>>(m, "BFGS", R"doc(
BFGS, keeping a dense approximation to the inverse Hessian.

``d = -H g``, with ``H`` updated from each step and the gradient change it
produced. Storing the inverse rather than the Hessian means the direction is a
matrix-vector product: no linear solve is ever needed. Costs O(n^2) memory, so
use :class:`LBFGS` beyond a few thousand degrees of freedom.
)doc")
        .def(py::init([&from_kwargs](const py::kwargs& kwargs) {
                 return std::make_shared<BFGS>(from_kwargs(kwargs));
             }))
        .def_property_readonly("skipped_updates", &BFGS::skipped_updates,
                               "Updates dropped because the curvature condition failed.")
        .def_property_readonly(
            "inverse_hessian",
            [](const BFGS& self) {
                const SymmetricMatrix& h = self.inverse_hessian();
                const auto n = static_cast<py::ssize_t>(h.size());
                std::vector<double> flat(h.size() * h.size());
                for (std::size_t i = 0; i < h.size(); ++i) {
                    for (std::size_t j = 0; j < h.size(); ++j) {
                        flat[i * h.size() + j] = h(i, j);
                    }
                }
                return move_to_array(std::move(flat), {n, n});
            },
            "The current inverse Hessian approximation, in angstrom^2/eV.");

    py::class_<LBFGS, Optimizer, std::shared_ptr<LBFGS>>(m, "LBFGS", R"doc(
Limited-memory BFGS.

Keeps only the last ``memory`` step and gradient-change pairs (default 10) and
rebuilds the action of the inverse Hessian with the two-loop recursion, so cost
is O(memory * n) instead of O(n^2). The right choice for large structures.
)doc")
        .def(py::init([&from_kwargs](const py::kwargs& kwargs) {
                 return std::make_shared<LBFGS>(from_kwargs(kwargs));
             }))
        .def_property_readonly("memory", &LBFGS::memory)
        .def_property_readonly("history_size", &LBFGS::history_size,
                               "Correction pairs currently stored.")
        .def_property_readonly("skipped_updates", &LBFGS::skipped_updates);

    py::class_<Fire, Optimizer, std::shared_ptr<Fire>>(m, "Fire", R"doc(
FIRE: a ball rolling downhill with adaptive friction.

    fa.optimize(atoms, "fire", fmax=0.02)

The velocity is mixed towards the force while progress is being made and reset
the moment it turns uphill, with the timestep growing and collapsing to match.
No curvature model and no line search.

This is the optimizer for a nudged elastic band. The band force is not the
gradient of any energy - the springs are added by hand and the climbing image's
parallel component is reversed - so there is nothing for a line search to
decrease and nothing for a quasi-Newton method to model. FIRE asks only for a
direction to roll in.
)doc")
        .def(py::init([&from_kwargs](const py::kwargs& kwargs) {
                 return std::make_shared<Fire>(from_kwargs(kwargs));
             }))
        .def_property_readonly("timestep", &Fire::timestep,
                               "The adaptive timestep as it stands.")
        .def_property_readonly("mixing", &Fire::mixing,
                               "How strongly the velocity is bent towards the force.")
        .def_property_readonly("downhill_steps", &Fire::downhill_steps,
                               "Consecutive steps that have gone downhill.");

    // Short aliases, matching the names the input files accept.
    m.attr("SD") = m.attr("SteepestDescent");
    m.attr("CG") = m.attr("ConjugateGradient");

    // ---------------------------------------------------------------------
    // Registry and convenience
    // ---------------------------------------------------------------------
    m.def(
        "make_optimizer",
        [](const py::object& source, const py::kwargs& kwargs) {
            Value settings = py::cast<Value>(py::dict(kwargs));
            if (py::isinstance<OptimizerParameters>(source)) {
                OptimizerParameters spec = source.cast<OptimizerParameters>();
                Value merged =
                    spec.parameters.is_map() ? spec.parameters : Value(Value::Map{});
                for (const auto& [key, value] : settings.as_map()) {
                    merged.insert(key, value);
                }
                spec.parameters = std::move(merged);
                return make_optimizer(spec);
            }
            return make_optimizer(py::cast<std::string>(source), settings);
        },
        py::arg("name_or_spec"),
        R"doc(
Build an optimizer by name, or from an input file's optimizer section.

    opt = fa.make_optimizer("lbfgs", fmax=0.01)
    opt = fa.make_optimizer(deck.simulation.optimizer)

Names are matched ignoring case, hyphens and underscores.
)doc");

    m.def("registered_optimizers", &registered_optimizers,
          "Names `make_optimizer` recognises.");

    m.def(
        "register_optimizer",
        [](const std::string& name, const py::object& factory) {
            register_optimizer(name, [factory](const Value& parameters) {
                const py::gil_scoped_acquire gil;
                return factory(parameters).cast<std::shared_ptr<Optimizer>>();
            });
            python_registered_optimizers().push_back(name);
        },
        py::arg("name"), py::arg("factory"),
        "Make an optimizer available to `make_optimizer`.");

    m.add_object("_optimizer_registry_cleanup", py::capsule([] {
                     for (const std::string& name : python_registered_optimizers()) {
                         unregister_optimizer(name);
                     }
                     python_registered_optimizers().clear();
                 }));

    m.def(
        "optimize",
        [](Atoms& atoms, const std::string& algorithm, const py::kwargs& kwargs) {
            auto optimizer = make_optimizer(algorithm, py::cast<Value>(py::dict(kwargs)));
            return optimizer->run(atoms);
        },
        py::arg("atoms"), py::arg("algorithm") = "bfgs",
        R"doc(
Relax a structure in one call.

    result = fa.optimize(atoms, "lbfgs", fmax=0.01, relax_cell=True)

Equivalent to building the optimizer and calling `run`, for when you do not
need to keep it around.
)doc");

    // ---------------------------------------------------------------------
    // The Atoms and input-file sides
    // ---------------------------------------------------------------------
    atoms_class.def(
        "optimize",
        [](Atoms& self, const std::string& algorithm, const py::kwargs& kwargs) {
            auto optimizer = make_optimizer(algorithm, py::cast<Value>(py::dict(kwargs)));
            return optimizer->run(self);
        },
        py::arg("algorithm") = "bfgs",
        "Relax this structure in place: `atoms.optimize('lbfgs', fmax=0.01)`.");

    deck_class.def(
        "make_optimizer",
        [](const io::InputDeck& self, const py::kwargs& overrides) {
            OptimizerParameters spec = self.simulation.optimizer;
            Value merged = spec.parameters.is_map() ? spec.parameters : Value(Value::Map{});
            for (const auto item : overrides) {
                merged.insert(py::cast<std::string>(item.first), item.second.cast<Value>());
            }
            spec.parameters = std::move(merged);
            return make_optimizer(spec);
        },
        R"doc(
Build the optimizer described by this file's ``simulation.optimizer`` section.

    deck = fa.read("relax.yaml")
    atoms = deck.atoms
    atoms.calc = deck.make_calculator()
    deck.make_optimizer().run(atoms)

Keyword arguments override what the file said.
)doc");
}

}  // namespace fastatoms::bindings
