// SPDX-License-Identifier: MIT
///
/// Trajectory analysis: g(r), S(q) and surface tension.
///
/// Every array comes back as a read-only numpy view onto the C++ vector that
/// holds it, with the owning object as the array's base. Nothing is copied,
/// and nothing can outlive its storage.

#include "bindings.hpp"

#include <string>
#include <vector>

#include "fastatoms/analysis/rdf.hpp"
#include "fastatoms/analysis/structure_factor.hpp"
#include "fastatoms/analysis/surface_tension.hpp"

namespace fastatoms::bindings {
namespace {

namespace an = fastatoms::analysis;

/// A read-only view of a `std::vector<double>` belonging to `owner`.
py::array_t<double> view_of(const py::object& owner, const std::vector<double>& data) {
    return make_view(const_cast<double*>(data.data()),
                     {static_cast<py::ssize_t>(data.size())}, owner, false);
}

/// Species selection: a symbol, a sequence of them, or None for all.
std::vector<std::string> species_of(const py::object& value) {
    if (value.is_none()) {
        return {};
    }
    if (py::isinstance<py::str>(value)) {
        return {value.cast<std::string>()};
    }
    return value.cast<std::vector<std::string>>();
}

an::RDFOptions rdf_options(double rmax, int bins, const py::object& first,
                           const py::object& second) {
    an::RDFOptions options;
    options.rmax = rmax;
    options.bins = bins;
    options.first = species_of(first);
    options.second = species_of(second);
    return options;
}

an::StructureFactorOptions sq_options(double qmin, double qmax, int points, bool lorch,
                                      double rmax) {
    an::StructureFactorOptions options;
    options.qmin = qmin;
    options.qmax = qmax;
    options.points = points;
    options.lorch = lorch;
    options.rmax = rmax;
    return options;
}

}  // namespace

void bind_analysis(py::module_& parent) {
    py::module_ m = parent.def_submodule("analysis", "Trajectory analysis.");

    // ------------------------------------------------------------------ g(r)
    py::class_<an::RadialDistribution>(m, "RadialDistribution", R"doc(
A computed pair correlation function.

The arrays are zero-copy views onto the C++ storage: reading them costs
nothing, and they keep this object alive for as long as they exist.
)doc")
        .def_property_readonly("r",
                               [](py::object self) {
                                   return view_of(self, self.cast<const an::RadialDistribution&>().r);
                               },
                               "Bin centres, in angstrom.")
        .def_property_readonly("g",
                               [](py::object self) {
                                   return view_of(self, self.cast<const an::RadialDistribution&>().g);
                               },
                               "The pair correlation, which tends to one at large r.")
        .def_property_readonly("counts",
                               [](py::object self) {
                                   return view_of(
                                       self, self.cast<const an::RadialDistribution&>().counts);
                               },
                               "Raw ordered-pair counts per bin, summed over frames.")
        .def_property_readonly(
            "coordination",
            [](py::object self) {
                return view_of(self, self.cast<const an::RadialDistribution&>().coordination);
            },
            "Running coordination number; its plateaux are the shells.")
        .def_readonly("rmax", &an::RadialDistribution::rmax)
        .def_readonly("bin_width", &an::RadialDistribution::bin_width)
        .def_readonly("frames", &an::RadialDistribution::frames)
        .def_readonly("first_atoms", &an::RadialDistribution::first_atoms)
        .def_readonly("second_atoms", &an::RadialDistribution::second_atoms)
        .def_readonly("density", &an::RadialDistribution::density)
        .def_readonly("volume", &an::RadialDistribution::volume)
        .def_readonly("reliable_rmax", &an::RadialDistribution::reliable_rmax,
                      "Half the smallest cell width: beyond it g(r) is biased by the "
                      "periodic images, which is worth knowing rather than being "
                      "protected from.")
        .def_readonly("label", &an::RadialDistribution::label)
        .def("__len__", &an::RadialDistribution::size)
        .def("__repr__", &an::RadialDistribution::to_string);

    py::class_<an::RadialDistributionAccumulator>(m, "RadialDistributionAccumulator", R"doc(
Accumulate g(r) one frame at a time.

A trajectory of ten thousand frames need not be held in memory: feed them in
as they are read, and ask for the result at the end.

    accumulator = fa.analysis.RadialDistributionAccumulator(rmax=10.0, bins=200)
    for frame in fa.read_frames("run.extxyz"):
        accumulator.add(frame)
    rdf = accumulator.result()
)doc")
        .def(py::init([](double rmax, int bins, const py::object& first,
                         const py::object& second) {
                 return an::RadialDistributionAccumulator(
                     rdf_options(rmax, bins, first, second));
             }),
             py::kw_only(), py::arg("rmax") = 10.0, py::arg("bins") = 200,
             py::arg("first") = py::none(), py::arg("second") = py::none())
        .def("add", &an::RadialDistributionAccumulator::add, py::arg("frame"),
             "Add one frame.")
        .def_property_readonly("frames", &an::RadialDistributionAccumulator::frames)
        .def("result", &an::RadialDistributionAccumulator::result,
             "The accumulated g(r).");

    m.def(
        "radial_distribution",
        [](const py::object& trajectory, double rmax, int bins, const py::object& first,
           const py::object& second) {
            an::RDFOptions options = rdf_options(rmax, bins, first, second);
            if (py::isinstance<Atoms>(trajectory)) {
                return an::radial_distribution(trajectory.cast<const Atoms&>(),
                                               std::move(options));
            }
            return an::radial_distribution(trajectory.cast<std::vector<Atoms>>(),
                                           std::move(options));
        },
        py::arg("trajectory"), py::kw_only(), py::arg("rmax") = 10.0, py::arg("bins") = 200,
        py::arg("first") = py::none(), py::arg("second") = py::none(),
        R"doc(
The radial distribution function of a structure or a trajectory.

    rdf = fa.analysis.radial_distribution(frames, rmax=10.0, bins=200)
    oh = fa.analysis.radial_distribution(frames, first="O", second="H")

`first` and `second` select which species a pair may be made of, by chemical
symbol; leaving them out gives the total g(r). Pairs come from the neighbour
list, so the cost is linear in the number of atoms and the minimum-image
convention is handled there.
)doc");

    // ------------------------------------------------------------------ S(q)
    py::class_<an::StructureFactor>(m, "StructureFactor", "A computed static structure factor.")
        .def_property_readonly("q",
                               [](py::object self) {
                                   return view_of(self, self.cast<const an::StructureFactor&>().q);
                               },
                               "Wavevector magnitudes, in inverse angstrom.")
        .def_property_readonly("s",
                               [](py::object self) {
                                   return view_of(self, self.cast<const an::StructureFactor&>().s);
                               },
                               "The structure factor, which tends to one at large q.")
        .def_property_readonly(
            "weight",
            [](py::object self) {
                return view_of(self, self.cast<const an::StructureFactor&>().weight);
            },
            "Reciprocal lattice points per bin; only the reciprocal method fills it.")
        .def_readonly("frames", &an::StructureFactor::frames)
        .def_readonly("method", &an::StructureFactor::method)
        .def("__len__", &an::StructureFactor::size)
        .def("__repr__", &an::StructureFactor::to_string);

    m.def(
        "structure_factor",
        [](const std::vector<Atoms>& frames, const std::string& method, double qmin,
           double qmax, int points, bool lorch, double rmax) {
            if (method == "reciprocal") {
                return an::structure_factor_reciprocal(frames, qmax, points);
            }
            if (method == "debye") {
                return an::structure_factor_debye(
                    frames, sq_options(qmin, qmax, points, lorch, rmax));
            }
            throw an::AnalysisError("'" + method +
                                    "' is not a method; use 'debye' or 'reciprocal', or call "
                                    "structure_factor_from_rdf");
        },
        py::arg("trajectory"), py::kw_only(), py::arg("method") = "reciprocal",
        py::arg("qmin") = 0.5, py::arg("qmax") = 15.0, py::arg("points") = 300,
        py::arg("lorch") = true, py::arg("rmax") = 0.0,
        R"doc(
The static structure factor of a trajectory.

    sq = fa.analysis.structure_factor(frames)                    # reciprocal lattice
    sq = fa.analysis.structure_factor(frames, method="debye")

`"reciprocal"` sums over the wavevectors the periodic cell actually supports.
It is exact - no cutoff, no window, no truncation ripple - and is the one to
trust for bulk MD; `points` is the number of magnitude bins. `"debye"` uses
the Debye scattering equation, which is exact for a finite cluster and needs a
cutoff for a periodic one.
)doc");

    m.def(
        "structure_factor_from_rdf",
        [](const an::RadialDistribution& rdf, double qmin, double qmax, int points, bool lorch,
           double rmax) {
            return an::structure_factor_from_rdf(rdf, sq_options(qmin, qmax, points, lorch,
                                                                 rmax));
        },
        py::arg("rdf"), py::kw_only(), py::arg("qmin") = 0.5, py::arg("qmax") = 15.0,
        py::arg("points") = 300, py::arg("lorch") = true, py::arg("rmax") = 0.0,
        "S(q) by Fourier transforming a g(r) that has already been computed.");

    // ------------------------------------------------------- surface tension
    py::class_<an::SurfaceTension>(m, "SurfaceTension", "Interfacial tension from a trajectory.")
        .def_readonly("gamma", &an::SurfaceTension::gamma, "eV per square angstrom.")
        .def_readonly("gamma_mN_per_m", &an::SurfaceTension::gamma_mN_per_m,
                      "The same, in millinewtons per metre.")
        .def_readonly("standard_error", &an::SurfaceTension::standard_error)
        .def_readonly("standard_error_mN_per_m", &an::SurfaceTension::standard_error_mN_per_m)
        .def_property_readonly(
            "per_frame",
            [](py::object self) {
                return view_of(self, self.cast<const an::SurfaceTension&>().per_frame);
            },
            "The per-frame values, so the run can be blocked or plotted.")
        .def_readonly("mean_pressure", &an::SurfaceTension::mean_pressure,
                      "The averaged pressure tensor, in eV per cubic angstrom.")
        .def_readonly("normal", &an::SurfaceTension::normal)
        .def_readonly("frames", &an::SurfaceTension::frames)
        .def("__repr__", &an::SurfaceTension::to_string);

    m.def(
        "surface_tension",
        [](const py::object& trajectory, int normal) {
            std::vector<Atoms> frames = py::isinstance<Atoms>(trajectory)
                                            ? std::vector<Atoms>{trajectory.cast<Atoms>()}
                                            : trajectory.cast<std::vector<Atoms>>();
            return an::surface_tension(frames, normal);
        },
        py::arg("trajectory"), py::kw_only(), py::arg("normal") = 2,
        R"doc(
Interfacial tension from a trajectory.

    gamma = fa.analysis.surface_tension(frames)
    print(gamma.gamma_mN_per_m, "+/-", gamma.standard_error_mN_per_m)

`gamma = (L_n / 2) [P_nn - (P_t1t1 + P_t2t2) / 2]`, averaged over frames. The
half is there because a periodic slab has *two* interfaces, not as a
normalisation of the average. Each frame needs a calculator that can produce a
stress; the kinetic term is included when the frame carries momenta.
)doc");

    m.def(
        "surface_tension_from_pressures",
        [](const std::vector<Mat3>& pressures, const std::vector<double>& lengths, int normal) {
            return an::surface_tension_from_pressures(pressures, lengths, normal);
        },
        py::arg("pressures"), py::arg("lengths"), py::kw_only(), py::arg("normal") = 2,
        "Surface tension from pressure tensors computed elsewhere.");

    m.def(
        "pressure_tensor",
        [](Atoms& atoms) { return an::pressure_tensor(atoms); }, py::arg("atoms"),
        "The pressure tensor of one frame, in eV per cubic angstrom.");

    m.attr("mN_per_m") = an::kMilliNewtonPerMetre;
}

}  // namespace fastatoms::bindings
