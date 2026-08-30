// SPDX-License-Identifier: MIT
#include "bindings.hpp"

#include <memory>
#include <string>
#include <vector>

#include "fastatoms/neb/autoneb.hpp"
#include "fastatoms/neb/idpp.hpp"
#include "fastatoms/neb/neb.hpp"
#include "fastatoms/neb/registry.hpp"
#include "fastatoms/optimize/registry.hpp"

namespace fastatoms::bindings {
namespace {

using namespace fastatoms::neb;

py::array_t<double> as_rows(std::vector<double> values, std::size_t rows) {
    const auto n = static_cast<py::ssize_t>(rows);
    return move_to_array(std::move(values), {n, static_cast<py::ssize_t>(3)});
}

/// Build NEBParameters from keyword arguments, so a band reads the same from
/// Python as it does from C++.
NEBParameters band_parameters(const py::kwargs& kwargs) {
    NEBParameters parameters;
    for (const auto item : kwargs) {
        const std::string key = py::cast<std::string>(item.first);
        const py::handle value = item.second;
        if (key == "k" || key == "spring_constant") {
            parameters.spring_constant = value.cast<double>();
        } else if (key == "climb" || key == "climbing") {
            parameters.climbing = value.cast<bool>();
        } else if (key == "climb_threshold") {
            parameters.climb_threshold = value.cast<double>();
        } else if (key == "tangent") {
            parameters.tangent = tangent_from_string(value.cast<std::string>());
        } else if (key == "variable_springs") {
            parameters.variable_springs = value.cast<bool>();
        } else if (key == "spring_softening") {
            parameters.spring_softening = value.cast<double>();
        } else if (key == "minimum_image") {
            parameters.minimum_image = value.cast<bool>();
        } else if (key == "fix" || key == "fixed_atoms") {
            parameters.fixed_atoms = value.cast<std::vector<std::size_t>>();
        } else {
            throw std::invalid_argument("neb: unknown option '" + key + "'");
        }
    }
    return parameters;
}

}  // namespace

void bind_neb(py::module_& m) {
    // ---------------------------------------------------------------------
    // The energy profile
    // ---------------------------------------------------------------------
    py::class_<BandProfile>(m, "BandProfile", R"doc(
The energy along a band, and where the top of it is.

    profile = band.profile()
    profile.forward_barrier      # eV, from the first image to the highest
    profile.energies             # one per image, endpoints included
    profile.distances            # cumulative path length, angstrom
)doc")
        .def_readonly("energies", &BandProfile::energies)
        .def_readonly("distances", &BandProfile::distances,
                      "Cumulative path length; the reaction coordinate to plot against.")
        .def_readonly("image_fmax", &BandProfile::image_fmax,
                      "Largest band force on each movable image.")
        .def_readonly("highest", &BandProfile::highest, "Index of the highest-energy image.")
        .def_readonly("forward_barrier", &BandProfile::forward_barrier, "eV.")
        .def_readonly("reverse_barrier", &BandProfile::reverse_barrier, "eV.")
        .def_readonly("reaction_energy", &BandProfile::reaction_energy,
                      "Last image minus first, in eV.")
        .def_readonly("climbing", &BandProfile::climbing,
                      "Whether the highest image was climbing when this was taken.")
        .def("__repr__", &BandProfile::to_string);

    // ---------------------------------------------------------------------
    // The band
    // ---------------------------------------------------------------------
    py::class_<NEB, std::shared_ptr<NEB>>(m, "NEB", R"doc(
A nudged elastic band between two fixed endpoints.

    band = fa.NEB(initial, final, images=7, climb=True)
    band.run("fire")
    band.profile().forward_barrier

Each image feels the true force with its along-path component removed, plus a
spring force with everything but that component removed. With ``climb=True``
the highest image loses its springs and has its parallel force inverted, so it
ends on the saddle point itself rather than near it.

The band force is not the gradient of any energy, so an optimizer with a line
search is refused rather than allowed to converge badly. ``fire`` is the usual
choice; anything else works with ``line_search="none"``.
)doc")
        .def(py::init([](const Atoms& initial, const Atoms& final, std::size_t images,
                         const py::kwargs& kwargs) {
                 if (images < 3) {
                     throw std::invalid_argument("neb: images must be at least 3");
                 }
                 return std::make_shared<NEB>(
                     NEB::from_endpoints(initial, final, images - 2, band_parameters(kwargs)));
             }),
             py::arg("initial"), py::arg("final"), py::arg("images") = 7,
             "Interpolate a band between two endpoints. `images` counts the endpoints.")
        .def(py::init([](const std::vector<Atoms>& images, const py::kwargs& kwargs) {
                 return std::make_shared<NEB>(images, band_parameters(kwargs));
             }),
             py::arg("images"), "Build a band from images you have already prepared.")

        .def_static(
            "interpolate",
            [](const Atoms& initial, const Atoms& final, std::size_t images,
               bool minimum_image) {
                if (images < 3) {
                    throw std::invalid_argument("neb: images must be at least 3");
                }
                return NEB::interpolate(initial, final, images - 2, minimum_image);
            },
            py::arg("initial"), py::arg("final"), py::arg("images") = 7,
            py::arg("minimum_image") = true,
            "A straight-line band between two structures, endpoints included.")

        .def("__len__", &NEB::size)
        .def_property_readonly("images", &NEB::images)
        .def_property_readonly("movable", &NEB::movable,
                               "Images free to move: everything but the endpoints.")
        .def(
            "image", [](NEB& self, std::ptrdiff_t index) -> Atoms& {
                const auto count = static_cast<std::ptrdiff_t>(self.size());
                return self.image(static_cast<std::size_t>(index < 0 ? count + index : index));
            },
            py::arg("index"), py::return_value_policy::reference_internal)

        .def_property_readonly(
            "positions",
            [](NEB& self) { return as_rows(self.positions(), self.natoms() * self.movable()); },
            "The movable images' coordinates, concatenated.")
        .def_property_readonly(
            "forces",
            [](NEB& self) { return as_rows(self.forces(), self.natoms() * self.movable()); },
            "The projected band forces on the movable images.")
        .def_property_readonly(
            "true_forces",
            [](NEB& self) { return as_rows(self.true_forces(), self.natoms() * self.size()); },
            "The unprojected forces on every image, endpoints included.")
        .def(
            "tangent",
            [](NEB& self, std::size_t index) {
                return as_rows(self.tangent(index), self.natoms());
            },
            py::arg("image"), "The unit tangent to the path at one movable image.")

        .def("profile", &NEB::profile, "Evaluate the band and describe its energy profile.")
        .def_property_readonly("fmax", &NEB::fmax,
                               "Largest band force on any movable image, eV/angstrom.")
        .def_property_readonly("highest_image", &NEB::highest_image)
        .def_property_readonly("climbing", &NEB::is_climbing,
                               "Whether the climbing image is active right now.")

        .def(
            "run",
            [](NEB& self, const py::object& optimizer, const py::kwargs& kwargs) {
                if (py::isinstance<optimize::Optimizer>(optimizer)) {
                    return self.run(*optimizer.cast<std::shared_ptr<optimize::Optimizer>>());
                }
                Value settings = py::cast<Value>(py::dict(kwargs));
                // Forced: the band force is not a gradient, so a line search
                // has nothing to search on.
                settings.insert("line_search", Value(std::string("none")));
                auto built = optimize::make_optimizer(
                    optimizer.is_none() ? "fire" : optimizer.cast<std::string>(), settings);
                return self.run(*built);
            },
            py::arg("optimizer") = "fire",
            R"doc(
Relax the band.

    band.run("fire", fmax=0.05, max_steps=500)
    band.run(my_optimizer)          # an Optimizer built yourself

Named algorithms get ``line_search="none"`` applied for them. An `Optimizer`
passed in is used as it stands, and is refused if it has a line search.
)doc")
        .def("__repr__", &NEB::to_string);

    // ---------------------------------------------------------------------
    // AutoNEB
    // ---------------------------------------------------------------------
    py::class_<AutoNEBResult>(m, "AutoNEBResult", "What an AutoNEB run did.")
        .def_readonly("profile", &AutoNEBResult::profile)
        .def_readonly("stages", &AutoNEBResult::stages, "Relaxations performed.")
        .def_readonly("steps", &AutoNEBResult::steps)
        .def_readonly("evaluations", &AutoNEBResult::evaluations)
        .def_readonly("converged", &AutoNEBResult::converged)
        .def_readonly("history", &AutoNEBResult::history,
                      "Image count after each stage, so the growth is visible.")
        .def("__repr__", &AutoNEBResult::to_string);

    py::class_<AutoNEB, std::shared_ptr<AutoNEB>>(m, "AutoNEB", R"doc(
A band that grows where the resolution is needed.

    auto_band = fa.AutoNEB(initial, final, initial_images=3, max_images=11)
    result = auto_band.run("fire")
    result.profile.forward_barrier

Starts with the fewest images that can describe a path, relaxes them loosely,
and repeatedly inserts an image across the largest energy jump - which is where
the saddle is - relaxing after each. Only the final stage runs to the tight
tolerance, and only that stage climbs: an image driven uphill before the band
knows where the path is can climb the wrong ridge.
)doc")
        .def(py::init([](const Atoms& initial, const Atoms& final, const py::kwargs& kwargs) {
                 AutoNEBParameters parameters;
                 py::kwargs remaining;
                 for (const auto item : kwargs) {
                     const std::string key = py::cast<std::string>(item.first);
                     const py::handle value = item.second;
                     // "images" is what `NEB` calls it; accepting both here
                     // means the two classes can be swapped for one another
                     // without renaming the argument.
                     if (key == "initial_images" || key == "images") {
                         parameters.initial_images = value.cast<std::size_t>();
                     } else if (key == "max_images") {
                         parameters.max_images = value.cast<std::size_t>();
                     } else if (key == "fmax") {
                         parameters.fmax = value.cast<double>();
                     } else if (key == "coarse_fmax") {
                         parameters.coarse_fmax = value.cast<double>();
                     } else if (key == "max_steps") {
                         parameters.max_steps = value.cast<int>();
                     } else if (key == "insert") {
                         parameters.insert = insertion_from_string(value.cast<std::string>());
                     } else if (key == "prune") {
                         parameters.prune = value.cast<bool>();
                     } else if (key == "prune_tolerance") {
                         parameters.prune_tolerance = value.cast<double>();
                     } else if (key == "redistribute") {
                         parameters.redistribute = value.cast<bool>();
                     } else {
                         // Anything else describes the band itself.
                         remaining[item.first] = value;
                     }
                 }
                 if (kwargs.contains("climb")) {
                     parameters.climb = kwargs["climb"].cast<bool>();
                     remaining.attr("pop")("climb", py::none());
                 }
                 return std::make_shared<AutoNEB>(initial, final, parameters,
                                                  band_parameters(remaining));
             }),
             py::arg("initial"), py::arg("final"))

        .def(
            "run",
            [](AutoNEB& self, const std::string& algorithm, const py::kwargs& kwargs) {
                return self.run(algorithm, py::cast<Value>(py::dict(kwargs)));
            },
            py::arg("optimizer") = "fire", "Grow and relax the band.")
        .def_property_readonly("band", py::overload_cast<>(&AutoNEB::band),
                               py::return_value_policy::reference_internal)
        .def("insert_image", &AutoNEB::insert_image,
             "Insert one image where the band is least resolved; returns where it went.")
        .def("prune_image", &AutoNEB::prune_image,
             "Remove the least informative interior image; returns which one went.")
        .def("redistribute", &AutoNEB::redistribute,
             "Respace the movable images evenly along the current path.")
        .def("__repr__", [](AutoNEB& self) {
            return "AutoNEB(" + std::to_string(self.band().size()) + " images)";
        });

    m.def(
        "idpp_interpolate",
        [](const Atoms& initial, const Atoms& final, std::size_t images, double fmax,
           int max_steps, double spring_constant, bool minimum_image) {
            IDPPOptions options;
            options.fmax = fmax;
            options.max_steps = max_steps;
            options.spring_constant = spring_constant;
            options.minimum_image = minimum_image;
            return idpp_interpolate(initial, final, images, options);
        },
        py::arg("initial"), py::arg("final"), py::arg("images") = 7, py::kw_only(),
        py::arg("fmax") = 0.1, py::arg("max_steps") = 200, py::arg("spring_constant") = 1.0,
        py::arg("minimum_image") = true,
        "Place a starting band by image-dependent pair potential.\n\n"
        "Interpolates the pair distances rather than the coordinates, so atoms\n"
        "go around each other instead of through. Use it whenever a straight\n"
        "line would put two atoms on top of one another - an exchange, a\n"
        "diffusion hop, a molecule turning over. `images` counts the endpoints.");

    m.def("closest_approach", &closest_approach, py::arg("atoms"), py::kw_only(),
          py::arg("minimum_image") = true,
          "The shortest distance between any two atoms, in angstrom.\n\n"
          "The one number that says whether an interpolated band is usable.");

    m.def(
        "make_band",
        [](const Atoms& structure, const SimulationParameters& simulation) {
            return make_band(structure, simulation);
        },
        py::arg("structure"), py::arg("simulation"),
        "Build the band an input deck's ``neb:`` section describes.");

    m.def(
        "make_autoneb",
        [](const Atoms& structure, const SimulationParameters& simulation) {
            return make_autoneb(structure, simulation);
        },
        py::arg("structure"), py::arg("simulation"),
        "Build the AutoNEB run an input deck's ``neb:`` section describes.");
}

}  // namespace fastatoms::bindings
