// SPDX-License-Identifier: MIT
///
/// Adsorption sites, and putting adsorbates on them.

#include "bindings.hpp"

#include <string>
#include <vector>

#include "fastatoms/adsorption/sites.hpp"

namespace fastatoms::bindings {
namespace {

namespace ads = fastatoms::adsorption;

ads::SiteType site_type_of(const py::object& value) {
    if (py::isinstance<ads::SiteType>(value)) {
        return value.cast<ads::SiteType>();
    }
    return ads::site_type_from_string(value.cast<std::string>());
}

/// An adsorbate given as an `Atoms` or as the name of a built-in molecule.
Atoms adsorbate_of(const py::object& value) {
    if (py::isinstance<Atoms>(value)) {
        return value.cast<Atoms>();
    }
    return ads::molecule(value.cast<std::string>());
}

}  // namespace

void bind_adsorption(py::module_& parent) {
    py::module_ m = parent.def_submodule("adsorption", "Adsorption sites and adsorbates.");

    py::enum_<ads::SiteType>(m, "SiteType", "How many surface atoms an adsorbate touches.")
        .value("atop", ads::SiteType::Atop)
        .value("bridge", ads::SiteType::Bridge)
        .value("hollow", ads::SiteType::Hollow)
        .value("four_fold", ads::SiteType::FourFold);

    py::class_<ads::AdsorptionSite>(m, "AdsorptionSite", "One place an adsorbate can bind.")
        .def_readonly("type", &ads::AdsorptionSite::type)
        .def_readonly("position", &ads::AdsorptionSite::position,
                      "Where the binding atom goes.")
        .def_readonly("normal", &ads::AdsorptionSite::normal, "The outward unit normal.")
        .def_readonly("centre", &ads::AdsorptionSite::centre,
                      "The site itself, on the surface rather than above it.")
        .def_readonly("atoms", &ads::AdsorptionSite::atoms,
                      "The surface atoms that form it.")
        .def_readonly("unique", &ads::AdsorptionSite::unique,
                      "Index into unique_sites() of this site's representative.")
        .def_property_readonly("name",
                               [](const ads::AdsorptionSite& self) {
                                   return std::string(ads::site_type_name(self.type));
                               })
        .def("__repr__", &ads::AdsorptionSite::to_string);

    py::class_<ads::SiteOptions>(m, "SiteOptions", "What to look for.")
        .def(py::init([](double height, double bond_scale, int surface_coordination, bool atop,
                         bool bridge, bool hollow, bool four_fold, double clearance,
                         double tolerance) {
                 ads::SiteOptions options;
                 options.height = height;
                 options.bond_scale = bond_scale;
                 options.surface_coordination = surface_coordination;
                 options.atop = atop;
                 options.bridge = bridge;
                 options.hollow = hollow;
                 options.four_fold = four_fold;
                 options.clearance = clearance;
                 options.tolerance = tolerance;
                 return options;
             }),
             py::kw_only(), py::arg("height") = 2.0, py::arg("bond_scale") = 1.25,
             py::arg("surface_coordination") = 0, py::arg("atop") = true,
             py::arg("bridge") = true, py::arg("hollow") = true, py::arg("four_fold") = true,
             py::arg("clearance") = 0.75, py::arg("tolerance") = 0.05)
        .def_readwrite("height", &ads::SiteOptions::height)
        .def_readwrite("bond_scale", &ads::SiteOptions::bond_scale)
        .def_readwrite("surface_coordination", &ads::SiteOptions::surface_coordination)
        .def_readwrite("clearance", &ads::SiteOptions::clearance)
        .def_readwrite("tolerance", &ads::SiteOptions::tolerance);

    py::class_<ads::AdsorptionFinder>(m, "AdsorptionFinder", R"doc(
Adsorption sites on a surface or a nanoparticle.

    finder = fa.adsorption.AdsorptionFinder(particle, height=1.9)
    print(finder.summary())

    frames = finder.trajectory("OH")        # one per unique site
    fa.write("oh_sites.extxyz", frames)

Surface atoms are the ones whose coordination falls short of the highest
present; "outward" is measured from where an atom's neighbours are, so a facet
of an icosahedron and the top of a slab work the same way.
)doc")
        .def(py::init([](const Atoms& surface, double height, double bond_scale,
                         int surface_coordination, bool atop, bool bridge, bool hollow,
                         bool four_fold, double clearance, double tolerance) {
                 ads::SiteOptions options;
                 options.height = height;
                 options.bond_scale = bond_scale;
                 options.surface_coordination = surface_coordination;
                 options.atop = atop;
                 options.bridge = bridge;
                 options.hollow = hollow;
                 options.four_fold = four_fold;
                 options.clearance = clearance;
                 options.tolerance = tolerance;
                 return ads::AdsorptionFinder(surface, options);
             }),
             py::arg("surface"), py::kw_only(), py::arg("height") = 2.0,
             py::arg("bond_scale") = 1.25, py::arg("surface_coordination") = 0,
             py::arg("atop") = true, py::arg("bridge") = true, py::arg("hollow") = true,
             py::arg("four_fold") = true, py::arg("clearance") = 0.75,
             py::arg("tolerance") = 0.05)
        .def_property_readonly("surface", &ads::AdsorptionFinder::surface)
        .def_property_readonly("sites", &ads::AdsorptionFinder::sites)
        .def_property_readonly("surface_atoms", &ads::AdsorptionFinder::surface_atoms,
                               "Which atoms were judged to be exposed.")
        .def_property_readonly("unique_sites", &ads::AdsorptionFinder::unique_sites,
                               "One representative of each distinct site.")
        .def("equivalent_sites", &ads::AdsorptionFinder::equivalent_sites, py::arg("unique"),
             "Every site equivalent to a unique one, including itself.")
        .def(
            "sites_of_type",
            [](const ads::AdsorptionFinder& self, const py::object& type, bool unique_only) {
                return self.sites_of_type(site_type_of(type), unique_only);
            },
            py::arg("type"), py::kw_only(), py::arg("unique_only") = false,
            "The sites of one kind, as indices.")
        .def("summary", &ads::AdsorptionFinder::summary)
        .def("place_atom", &ads::AdsorptionFinder::place_atom, py::arg("symbol"),
             py::arg("site"), "A single atom on one site.")
        .def(
            "place",
            [](const ads::AdsorptionFinder& self, const py::object& adsorbate, std::size_t site,
               std::size_t binding_atom, std::ptrdiff_t anchor) {
                return self.place(adsorbate_of(adsorbate), site, binding_atom, anchor);
            },
            py::arg("adsorbate"), py::arg("site"), py::kw_only(), py::arg("binding_atom") = 0,
            py::arg("anchor") = -1,
            R"doc(
Put a molecule on one site.

    finder.place("OH", site)                    # O down, H pointing out
    finder.place(molecule, site, binding_atom=2)

The molecule is moved so `binding_atom` sits at the site and turned so the rest
of it points away from the surface. `anchor` names the atom whose direction
from the binding atom is aligned with the normal; the default uses the centroid
of everything else, which is the O-to-H direction for a hydroxyl.
)doc")
        .def(
            "place_many",
            [](const ads::AdsorptionFinder& self, const py::object& adsorbate,
               const std::vector<std::size_t>& sites, std::size_t binding_atom,
               std::ptrdiff_t anchor) {
                return self.place_many(adsorbate_of(adsorbate), sites, binding_atom, anchor);
            },
            py::arg("adsorbate"), py::arg("sites"), py::kw_only(), py::arg("binding_atom") = 0,
            py::arg("anchor") = -1, "Several adsorbates at once.")
        .def(
            "trajectory",
            [](const ads::AdsorptionFinder& self, const py::object& adsorbate, bool unique_only,
               std::size_t binding_atom, std::ptrdiff_t anchor) {
                return self.trajectory(adsorbate_of(adsorbate), unique_only, binding_atom,
                                       anchor);
            },
            py::arg("adsorbate"), py::kw_only(), py::arg("unique_only") = true,
            py::arg("binding_atom") = 0, py::arg("anchor") = -1,
            R"doc(
One structure per site, ready to write as a trajectory.

    fa.write("sites.extxyz", finder.trajectory("OH"))

`unique_only` visits one representative of each distinct site, which on a
symmetric particle is a great many fewer calculations.
)doc")
        .def(
            "coverage_series",
            [](const ads::AdsorptionFinder& self, const py::object& adsorbate,
               const std::vector<std::size_t>& candidates, int steps, std::size_t binding_atom,
               std::ptrdiff_t anchor) {
                return self.coverage_series(adsorbate_of(adsorbate), candidates, steps,
                                            binding_atom, anchor);
            },
            py::arg("adsorbate"), py::kw_only(),
            py::arg("candidates") = std::vector<std::size_t>{}, py::arg("steps") = 0,
            py::arg("binding_atom") = 0, py::arg("anchor") = -1,
            R"doc(
A series of increasing coverage.

    frames = finder.coverage_series("H", candidates=hollows, steps=6)

Structure `k` carries `k + 1` adsorbates. Each new one goes on the candidate
furthest from those already placed, so the series spreads over the particle
rather than filling one facet and leaving the rest bare.
)doc")
        .def("coverage_fractions", &ads::AdsorptionFinder::coverage_fractions, py::kw_only(),
             py::arg("candidates") = std::vector<std::size_t>{}, py::arg("steps") = 0,
             "Adsorbates per surface atom, for each structure a series returns.")
        .def("__len__",
             [](const ads::AdsorptionFinder& self) { return self.sites().size(); })
        .def("__repr__", [](const ads::AdsorptionFinder& self) {
            return "AdsorptionFinder(" + std::to_string(self.sites().size()) + " sites, " +
                   std::to_string(self.unique_sites().size()) + " unique)";
        });

    m.def("molecule", &ads::molecule, py::arg("name"),
          "A small molecule ready to adsorb, with the binding atom first.");
    m.def("known_molecules", &ads::known_molecules, "Which molecules are built in.");
}

}  // namespace fastatoms::bindings
