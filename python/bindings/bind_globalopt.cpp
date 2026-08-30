// SPDX-License-Identifier: MIT
///
/// Global optimization: basin hopping.

#include "bindings.hpp"

#include <string>
#include <vector>

#include "fastatoms/globalopt/basin_hopping.hpp"

namespace fastatoms::bindings {
namespace {

namespace go = fastatoms::globalopt;

go::Move move_of(const py::object& value) {
    if (py::isinstance<go::Move>(value)) {
        return value.cast<go::Move>();
    }
    return go::move_from_string(value.cast<std::string>());
}

}  // namespace

void bind_globalopt(py::module_& m) {
    py::enum_<go::Move>(m, "Move", "How to perturb a structure between relaxations.")
        .value("cartesian", go::Move::Cartesian, "Displace every atom independently.")
        .value("single", go::Move::Single, "Displace one atom.")
        .value("twist", go::Move::Twist, "Rotate half the cluster about a random axis.")
        .value("exchange", go::Move::Exchange, "Swap two atoms of different species.")
        .value("mixed", go::Move::Mixed, "Choose among the others each step.");

    py::class_<go::BasinHoppingStep>(m, "BasinHoppingStep", "What one step did.")
        .def_readonly("step", &go::BasinHoppingStep::step)
        .def_readonly("energy", &go::BasinHoppingStep::energy,
                      "The relaxed energy of the structure proposed this step.")
        .def_readonly("current_energy", &go::BasinHoppingStep::current_energy,
                      "The energy of the state the walk sits on after this step.")
        .def_readonly("best_energy", &go::BasinHoppingStep::best_energy)
        .def_readonly("accepted", &go::BasinHoppingStep::accepted)
        .def_readonly("improved", &go::BasinHoppingStep::improved,
                      "True when this step set a new global best.")
        .def_readonly("step_size", &go::BasinHoppingStep::step_size)
        .def_readonly("evaluations", &go::BasinHoppingStep::evaluations)
        .def("__repr__", &go::BasinHoppingStep::to_string);

    py::class_<go::BasinHoppingResult>(m, "BasinHoppingResult", "The outcome of a search.")
        .def_readonly("best", &go::BasinHoppingResult::best,
                      "The lowest structure found - what the search is for.")
        .def_readonly("best_energy", &go::BasinHoppingResult::best_energy)
        .def_readonly("initial_energy", &go::BasinHoppingResult::initial_energy,
                      "The energy handed in, after one relaxation.")
        .def_readonly("steps", &go::BasinHoppingResult::steps)
        .def_readonly("accepted", &go::BasinHoppingResult::accepted)
        .def_readonly("improved", &go::BasinHoppingResult::improved)
        .def_readonly("evaluations", &go::BasinHoppingResult::evaluations,
                      "Force evaluations over every relaxation - the cost that matters.")
        .def_readonly("acceptance_ratio", &go::BasinHoppingResult::acceptance_ratio)
        .def_readonly("best_step", &go::BasinHoppingResult::best_step)
        .def_readonly("history", &go::BasinHoppingResult::history)
        .def_readonly("message", &go::BasinHoppingResult::message)
        .def("__repr__", &go::BasinHoppingResult::to_string);

    py::class_<go::BasinHopping>(m, "BasinHopping", R"doc(
Basin hopping: the search for the global minimum, not a nearby one.

    search = fa.BasinHopping(steps=200, temperature=100.0, step_size=0.4)
    result = search.run(cluster)
    print(result.best_energy)        # `cluster` now holds the best structure

Every proposed structure is relaxed before its energy is looked at, which turns
the basin around each minimum into a plateau and leaves a Metropolis walk with
nothing to do but step between basins. The lowest structure ever seen is kept
separately from where the walk ends up, because the walk is allowed to go
uphill and often should.
)doc")
        .def(py::init([](int steps, double temperature, double step_size,
                         const py::object& move, std::uint64_t seed, bool adapt_step,
                         double target_acceptance, double adapt_factor, int adapt_interval,
                         const std::string& optimizer, double fmax, int max_steps,
                         int stall_limit, double minimum_distance,
                         double improvement_tolerance) {
                 go::BasinHoppingOptions options;
                 options.steps = steps;
                 options.temperature = temperature;
                 options.step_size = step_size;
                 options.move = move_of(move);
                 options.seed = seed;
                 options.adapt_step = adapt_step;
                 options.target_acceptance = target_acceptance;
                 options.adapt_factor = adapt_factor;
                 options.adapt_interval = adapt_interval;
                 options.optimizer = optimizer;
                 options.fmax = fmax;
                 options.max_steps = max_steps;
                 options.stall_limit = stall_limit;
                 options.minimum_distance = minimum_distance;
                 options.improvement_tolerance = improvement_tolerance;
                 return go::BasinHopping(options);
             }),
             py::kw_only(), py::arg("steps") = 100, py::arg("temperature") = 100.0,
             py::arg("step_size") = 0.4, py::arg("move") = "cartesian", py::arg("seed") = 0,
             py::arg("adapt_step") = true, py::arg("target_acceptance") = 0.5,
             py::arg("adapt_factor") = 0.9, py::arg("adapt_interval") = 25,
             py::arg("optimizer") = "lbfgs", py::arg("fmax") = 0.05,
             py::arg("max_steps") = 200, py::arg("stall_limit") = 0,
             py::arg("minimum_distance") = 0.0,
             py::arg("improvement_tolerance") = 1e-9)
        .def("run", &go::BasinHopping::run, py::arg("atoms"),
             "Search, leaving `atoms` holding the best structure found.")
        .def(
            "attach",
            [](go::BasinHopping& self, const py::object& callback) {
                self.attach([callback](const go::BasinHoppingStep& step) {
                    const py::object keep = callback(step);
                    return keep.is_none() || keep.cast<bool>();
                });
            },
            py::arg("callback"),
            "Watch the search. Returning False from the callback stops it.")
        .def("perturb",
             [](const go::BasinHopping& self, Atoms& atoms, std::uint64_t seed) {
                 self.perturb(atoms, seed);
             },
             py::arg("atoms"), py::arg("seed"),
             "Apply one perturbation without relaxing, to see what a move does.")
        .def("__repr__", [](const go::BasinHopping& self) {
            return "BasinHopping(steps=" + std::to_string(self.options().steps) +
                   ", temperature=" + std::to_string(self.options().temperature) + ")";
        });
}

}  // namespace fastatoms::bindings
