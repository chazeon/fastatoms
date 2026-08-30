// SPDX-License-Identifier: MIT
///
/// Electronic structure analysis: band paths, band structures and the density
/// of states. Arrays come back as read-only views onto the C++ storage, as in
/// the trajectory analysis module.

#include "bindings.hpp"

#include <string>
#include <vector>

#include "fastatoms/electronic/bandpath.hpp"
#include "fastatoms/electronic/bands.hpp"
#include "fastatoms/electronic/dos.hpp"

namespace fastatoms::bindings {
namespace {

namespace el = fastatoms::electronic;

py::array_t<double> view_of_doubles(const py::object& owner, const std::vector<double>& data) {
    return make_view(const_cast<double*>(data.data()),
                     {static_cast<py::ssize_t>(data.size())}, owner, false);
}

el::Smearing smearing_of(const py::object& value) {
    if (py::isinstance<el::Smearing>(value)) {
        return value.cast<el::Smearing>();
    }
    return el::smearing_from_string(value.cast<std::string>());
}

el::DOSOptions dos_options(double emin, double emax, int points, double width,
                           const py::object& smearing, int order, int spin_degeneracy) {
    el::DOSOptions options;
    options.emin = emin;
    options.emax = emax;
    options.points = points;
    options.width = width;
    options.smearing = smearing_of(smearing);
    options.order = order;
    options.spin_degeneracy = spin_degeneracy;
    return options;
}

}  // namespace

void bind_electronic(py::module_& parent) {
    py::module_ m = parent.def_submodule("electronic", "Band structures and densities of state.");

    // ------------------------------------------------------------- band gaps
    py::class_<el::BandGap>(m, "BandGap", "Where a band gap is and how big.")
        .def_readonly("gap", &el::BandGap::gap,
                      "The fundamental gap in eV; zero for a metal.")
        .def_readonly("direct_gap", &el::BandGap::direct_gap,
                      "The smallest separation at a single k-point.")
        .def_readonly("metallic", &el::BandGap::metallic)
        .def_readonly("direct", &el::BandGap::direct,
                      "True when the fundamental gap is found at one k-point.")
        .def_readonly("valence_maximum", &el::BandGap::valence_maximum)
        .def_readonly("conduction_minimum", &el::BandGap::conduction_minimum)
        .def_readonly("valence_kpoint", &el::BandGap::valence_kpoint)
        .def_readonly("conduction_kpoint", &el::BandGap::conduction_kpoint)
        .def_readonly("direct_kpoint", &el::BandGap::direct_kpoint)
        .def_readonly("valence_band", &el::BandGap::valence_band)
        .def_readonly("conduction_band", &el::BandGap::conduction_band)
        .def("__repr__", &el::BandGap::to_string);

    // -------------------------------------------------------- band structure
    py::class_<el::BandStructure>(m, "BandStructure", R"doc(
Eigenvalues on a set of k-points.

    bands = fa.electronic.BandStructure(kpoints, energies, bands=8, fermi_level=5.2)
    gap = bands.band_gap()
    print(gap.gap, "eV", "direct" if gap.direct else "indirect")

`energies` is laid out ``[spin][kpoint][band]``, which is how every electronic
structure code writes them.
)doc")
        .def(py::init([](const py::object& kpoints, const py::object& energies,
                         std::size_t bands, std::size_t spins, double fermi_level,
                         const py::object& weights) {
                 const auto points = kpoints.cast<std::vector<Vec3>>();
                 const auto values = energies.cast<std::vector<double>>();
                 std::vector<double> w;
                 if (!weights.is_none()) {
                     w = weights.cast<std::vector<double>>();
                 }
                 return el::BandStructure(points, values, bands, spins, fermi_level,
                                          std::move(w));
             }),
             py::arg("kpoints"), py::arg("energies"), py::kw_only(), py::arg("bands"),
             py::arg("spins") = 1, py::arg("fermi_level") = 0.0,
             py::arg("weights") = py::none())
        .def_property_readonly("nkpoints", &el::BandStructure::nkpoints)
        .def_property_readonly("nbands", &el::BandStructure::nbands)
        .def_property_readonly("nspins", &el::BandStructure::nspins)
        .def_property_readonly("kpoints",
                               [](const el::BandStructure& self) { return self.kpoints(); })
        .def_property_readonly(
            "energies",
            [](py::object self) {
                const auto& bands = self.cast<const el::BandStructure&>();
                return make_view(const_cast<double*>(bands.energies().data()),
                                 {static_cast<py::ssize_t>(bands.nspins()),
                                  static_cast<py::ssize_t>(bands.nkpoints()),
                                  static_cast<py::ssize_t>(bands.nbands())},
                                 self, false);
            },
            "A zero-copy [spin][kpoint][band] view of the eigenvalues.")
        .def_property_readonly("weights",
                               [](py::object self) {
                                   return view_of_doubles(
                                       self, self.cast<const el::BandStructure&>().weights());
                               })
        .def_property("fermi_level", &el::BandStructure::fermi_level,
                      &el::BandStructure::set_fermi_level)
        .def("energy", &el::BandStructure::energy, py::arg("spin"), py::arg("kpoint"),
             py::arg("band"), "One eigenvalue.")
        .def("band", &el::BandStructure::band, py::arg("spin"), py::arg("band"),
             "One band across every k-point - a single line of a band plot.")
        .def("minimum", &el::BandStructure::minimum)
        .def("maximum", &el::BandStructure::maximum)
        .def("band_gap", &el::BandStructure::band_gap, "Where the gap is and how big.")
        .def("is_metal", &el::BandStructure::is_metal, py::kw_only(),
             py::arg("tolerance") = 1e-6, "True when a band crosses the Fermi level.")
        .def("shifted", &el::BandStructure::shifted,
             "A copy with every energy measured from the Fermi level.")
        .def_static("from_calculator", &el::BandStructure::from_calculator, py::arg("atoms"),
                    "Read eigenvalues out of a calculator's extra results.")
        .def("__len__", &el::BandStructure::nkpoints)
        .def("__repr__", &el::BandStructure::to_string);

    // ------------------------------------------------------------------- DOS
    py::enum_<el::Smearing>(m, "Smearing", "Which bump to put at each eigenvalue.")
        .value("gaussian", el::Smearing::Gaussian)
        .value("methfessel_paxton", el::Smearing::MethfesselPaxton)
        .value("fermi_dirac", el::Smearing::FermiDirac);

    py::class_<el::DensityOfStates>(m, "DensityOfStates", "A computed density of states.")
        .def_property_readonly("energy",
                               [](py::object self) {
                                   return view_of_doubles(
                                       self, self.cast<const el::DensityOfStates&>().energy);
                               },
                               "The energy grid, in eV.")
        .def_property_readonly("dos",
                               [](py::object self) {
                                   return view_of_doubles(
                                       self, self.cast<const el::DensityOfStates&>().dos);
                               },
                               "States per unit cell per eV.")
        .def_property_readonly("integrated",
                               [](py::object self) {
                                   return view_of_doubles(
                                       self, self.cast<const el::DensityOfStates&>().integrated);
                               },
                               "How many states lie below each energy.")
        .def_property_readonly(
            "spin_dos",
            [](py::object self) {
                const auto& dos = self.cast<const el::DensityOfStates&>();
                if (dos.spin_dos.empty()) {
                    return py::array_t<double>(std::vector<py::ssize_t>{0, 0});
                }
                return make_view(const_cast<double*>(dos.spin_dos.data()),
                                 {2, static_cast<py::ssize_t>(dos.size())}, self, false);
            },
            "Per spin channel, [spin][point]. Empty for one spin.")
        .def_readonly("fermi_level", &el::DensityOfStates::fermi_level)
        .def_readonly("spins", &el::DensityOfStates::spins)
        .def_readonly("method", &el::DensityOfStates::method)
        .def("at_fermi", &el::DensityOfStates::at_fermi,
             "The density of states at the Fermi level, interpolated.")
        .def("__len__", &el::DensityOfStates::size)
        .def("__repr__", &el::DensityOfStates::to_string);

    m.def("monkhorst_pack", &el::monkhorst_pack, py::arg("n1"), py::arg("n2"), py::arg("n3"),
          py::kw_only(), py::arg("gamma_centred") = true,
          "The k-points of a regular grid, in the order the tetrahedron method expects.");

    m.def(
        "density_of_states",
        [](const el::BandStructure& bands, double emin, double emax, int points, double width,
           const py::object& smearing, int order, int spin_degeneracy) {
            return el::density_of_states(
                bands, dos_options(emin, emax, points, width, smearing, order, spin_degeneracy));
        },
        py::arg("bands"), py::kw_only(), py::arg("emin") = 0.0, py::arg("emax") = 0.0,
        py::arg("points") = 600, py::arg("width") = 0.1, py::arg("smearing") = "gaussian",
        py::arg("order") = 1, py::arg("spin_degeneracy") = 0,
        R"doc(
Density of states by smearing.

    dos = fa.electronic.density_of_states(bands, width=0.1)
    dos = fa.electronic.density_of_states(bands, smearing="methfessel-paxton", order=1)

Works on any set of k-points. Methfessel-Paxton converges faster in the width
than a Gaussian does, at the cost of going slightly negative - which is
correct, not a bug.
)doc");

    m.def(
        "density_of_states_tetrahedron",
        [](const el::BandStructure& bands, const std::array<int, 3>& grid, double emin,
           double emax, int points, int spin_degeneracy, const py::object& reciprocal) {
            el::DOSOptions options =
                dos_options(emin, emax, points, 0.1, py::str("gaussian"), 0, spin_degeneracy);
            Mat3 basis = Mat3::identity();
            if (!reciprocal.is_none()) {
                basis = reciprocal.cast<Mat3>();
            }
            return el::density_of_states_tetrahedron(bands, grid, options, basis);
        },
        py::arg("bands"), py::arg("grid"), py::kw_only(), py::arg("emin") = 0.0,
        py::arg("emax") = 0.0, py::arg("points") = 600, py::arg("spin_degeneracy") = 0,
        py::arg("reciprocal") = py::none(),
        R"doc(
Density of states by the linear tetrahedron method.

    kpoints = fa.electronic.monkhorst_pack(12, 12, 12)
    dos = fa.electronic.density_of_states_tetrahedron(bands, (12, 12, 12))

The bands must be on the grid ``monkhorst_pack`` returns, in that order. Each
microcell is cut into six tetrahedra along its shortest body diagonal and the
isosurface integral is done in closed form, so van Hove singularities come out
sharp rather than smeared. The error falls as one over the grid squared, with
no width to converge separately.
)doc");

    // Both take the enum or its name, as `density_of_states` does; having one
    // of the three accept only the enum would be a trap.
    m.def(
        "smearing_delta",
        [](double x, const py::object& smearing, int order) {
            return el::smearing_delta(x, smearing_of(smearing), order);
        },
        py::arg("x"), py::kw_only(), py::arg("smearing") = "gaussian", py::arg("order") = 1,
        "The smearing function, normalised to integrate to one.");
    m.def(
        "smearing_occupation",
        [](double x, const py::object& smearing, int order) {
            return el::smearing_occupation(x, smearing_of(smearing), order);
        },
        py::arg("x"), py::kw_only(), py::arg("smearing") = "gaussian", py::arg("order") = 1,
        "The occupation a smearing implies: one below the Fermi level, zero above.");

    // ------------------------------------------------------------ band paths
    py::enum_<el::BravaisLattice>(m, "BravaisLattice", "One of the fourteen Bravais lattices.")
        .value("CUB", el::BravaisLattice::CUB).value("FCC", el::BravaisLattice::FCC)
        .value("BCC", el::BravaisLattice::BCC).value("TET", el::BravaisLattice::TET)
        .value("BCT", el::BravaisLattice::BCT).value("ORC", el::BravaisLattice::ORC)
        .value("ORCF", el::BravaisLattice::ORCF).value("ORCI", el::BravaisLattice::ORCI)
        .value("ORCC", el::BravaisLattice::ORCC).value("HEX", el::BravaisLattice::HEX)
        .value("RHL", el::BravaisLattice::RHL).value("MCL", el::BravaisLattice::MCL)
        .value("MCLC", el::BravaisLattice::MCLC).value("TRI", el::BravaisLattice::TRI);

    py::class_<el::BandPath>(m, "BandPath", "A sampled path through the Brillouin zone.")
        .def_property_readonly("kpoints",
                               [](const el::BandPath& self) { return self.kpoints; })
        .def_property_readonly("distances",
                               [](py::object self) {
                                   return view_of_doubles(
                                       self, self.cast<const el::BandPath&>().distances);
                               },
                               "Cumulative distance along the path, the axis of a band plot.")
        .def_readonly("labels", &el::BandPath::labels)
        .def_property_readonly("special_distances",
                               [](py::object self) {
                                   return view_of_doubles(
                                       self,
                                       self.cast<const el::BandPath&>().special_distances);
                               },
                               "Where the tick marks go.")
        .def_readonly("special_labels", &el::BandPath::special_labels,
                      "What the tick marks say.")
        .def_readonly("path", &el::BandPath::path)
        .def_readonly("lattice", &el::BandPath::lattice)
        .def("__len__", &el::BandPath::size)
        .def("__repr__", &el::BandPath::to_string);

    m.def(
        "bandpath",
        [](const py::object& source, const std::string& path, int points, double tolerance) {
            const Cell cell = py::isinstance<Atoms>(source)
                                  ? source.cast<const Atoms&>().cell()
                                  : source.cast<Cell>();
            return el::bandpath(cell, path, points, tolerance);
        },
        py::arg("cell"), py::kw_only(), py::arg("path") = "", py::arg("points") = 100,
        py::arg("tolerance") = 1e-5,
        R"doc(
Sample a high-symmetry path through the Brillouin zone.

    path = fa.electronic.bandpath(atoms, path="GXWKGL", points=200)
    path = fa.electronic.bandpath(atoms)          # the conventional path

Takes an `Atoms` or a `Cell`. A comma in the path makes it jump rather than
interpolate. Points are shared between segments in proportion to their length
in reciprocal space, so the sampling is even along the plotted axis.
)doc");

    m.def(
        "bandpath_from_points",
        [](const py::object& source, const std::vector<Vec3>& corners,
           const std::vector<std::string>& labels, int points) {
            const Cell cell = py::isinstance<Atoms>(source)
                                  ? source.cast<const Atoms&>().cell()
                                  : source.cast<Cell>();
            return el::bandpath_from_points(cell, corners, labels, points);
        },
        py::arg("cell"), py::arg("corners"), py::kw_only(),
        py::arg("labels") = std::vector<std::string>{}, py::arg("points") = 100,
        "Sample a path given explicitly, for a lattice with no table or a path of your own.");

    m.def(
        "identify_bravais",
        [](const py::object& source, double tolerance) {
            const Cell cell = py::isinstance<Atoms>(source)
                                  ? source.cast<const Atoms&>().cell()
                                  : source.cast<Cell>();
            return el::identify_bravais(cell, tolerance);
        },
        py::arg("cell"), py::kw_only(), py::arg("tolerance") = 1e-5,
        "Which of the fourteen Bravais lattices a cell is.");

    m.def(
        "special_points",
        [](const py::object& source, double tolerance) {
            const Cell cell = py::isinstance<Atoms>(source)
                                  ? source.cast<const Atoms&>().cell()
                                  : source.cast<Cell>();
            return el::special_points(cell, tolerance);
        },
        py::arg("cell"), py::kw_only(), py::arg("tolerance") = 1e-5,
        "The special points of a lattice, in fractional reciprocal coordinates.");

    m.def(
        "default_path",
        [](const py::object& source, double tolerance) {
            const Cell cell = py::isinstance<Atoms>(source)
                                  ? source.cast<const Atoms&>().cell()
                                  : source.cast<Cell>();
            return el::default_path(cell, tolerance);
        },
        py::arg("cell"), py::kw_only(), py::arg("tolerance") = 1e-5,
        "The conventional path through a lattice's special points.");

    m.def("tabulated_lattices", &el::tabulated_lattices,
          "Which lattices have their special points tabulated in this build.");
}

}  // namespace fastatoms::bindings
