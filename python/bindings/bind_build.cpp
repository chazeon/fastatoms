// SPDX-License-Identifier: MIT
///
/// The structure builders.

#include "bindings.hpp"

#include "fastatoms/build/surface.hpp"

#include "fastatoms/core/elements.hpp"

#include <string>
#include <vector>

#include "fastatoms/build/bulk.hpp"
#include "fastatoms/build/cluster.hpp"
#include "fastatoms/build/nanotube.hpp"
#include "fastatoms/build/twod.hpp"

namespace fastatoms::bindings {
namespace {

/// One or several chemical symbols, however the caller spelled it.
std::vector<std::string> symbol_list(const py::object& value) {
    if (py::isinstance<py::str>(value)) {
        return {value.cast<std::string>()};
    }
    return value.cast<std::vector<std::string>>();
}

build::Lattice lattice_of(const py::object& value) {
    if (py::isinstance<build::Lattice>(value)) {
        return value.cast<build::Lattice>();
    }
    return build::lattice_from_string(value.cast<std::string>());
}

build::Phase phase_of(const py::object& value) {
    if (py::isinstance<build::Phase>(value)) {
        return value.cast<build::Phase>();
    }
    return build::phase_from_string(value.cast<std::string>());
}

/// Surface energies as `{"111": 1.0, "100": 1.15}` or a list of triples.
std::vector<build::SurfaceEnergy> surface_energies(const py::object& value) {
    std::vector<build::SurfaceEnergy> out;

    const auto parse_indices = [](const std::string& text) {
        // "111" and "1,1,1" and "(1 1 1)" all mean the same thing, and a
        // negative index needs the comma form.
        std::vector<int> indices;
        std::string number;
        for (const char c : text) {
            if (std::isdigit(static_cast<unsigned char>(c)) || c == '-') {
                number += c;
                if (text.find(',') == std::string::npos && c != '-') {
                    indices.push_back(std::stoi(number));
                    number.clear();
                }
            } else if (!number.empty()) {
                indices.push_back(std::stoi(number));
                number.clear();
            }
        }
        if (!number.empty()) {
            indices.push_back(std::stoi(number));
        }
        if (indices.size() != 3) {
            throw build::BuildError("'" + text + "' is not a set of three Miller indices");
        }
        return indices;
    };

    if (py::isinstance<py::dict>(value)) {
        for (const auto item : py::reinterpret_borrow<py::dict>(value)) {
            const std::vector<int> indices = parse_indices(py::str(item.first).cast<std::string>());
            out.push_back({indices[0], indices[1], indices[2], item.second.cast<double>()});
        }
        return out;
    }
    for (const py::handle entry : value) {
        const auto row = entry.cast<std::vector<double>>();
        if (row.size() != 4) {
            throw build::BuildError(
                "each surface energy needs h, k, l and the energy - four numbers");
        }
        out.push_back({static_cast<int>(row[0]), static_cast<int>(row[1]),
                       static_cast<int>(row[2]), row[3]});
    }
    return out;
}

}  // namespace

void bind_build(py::module_& parent) {
    py::module_ m = parent.def_submodule("build", "Generators for atomic structures.");

    py::enum_<build::Lattice>(m, "Lattice", "Which lattice a bulk crystal has.")
        .value("sc", build::Lattice::SC)
        .value("fcc", build::Lattice::FCC)
        .value("bcc", build::Lattice::BCC)
        .value("hcp", build::Lattice::HCP)
        .value("diamond", build::Lattice::Diamond)
        .value("rocksalt", build::Lattice::RockSalt)
        .value("zincblende", build::Lattice::ZincBlende)
        .value("cscl", build::Lattice::CsCl);

    py::enum_<build::Phase>(m, "Phase", "Which polymorph a dichalcogenide monolayer has.")
        .value("H", build::Phase::H, "Trigonal prismatic - the 1H monolayer.")
        .value("T", build::Phase::T, "Octahedral - the 1T monolayer.");

    // ---------------------------------------------------------------- bulk
    m.def("sc", &build::sc, py::arg("symbol"), py::arg("a"),
          "Simple cubic, one atom.");
    m.def("fcc", &build::fcc, py::arg("symbol"), py::arg("a"), py::kw_only(),
          py::arg("cubic") = false,
          "Face-centred cubic. The nearest-neighbour distance is a/sqrt(2).");
    m.def("bcc", &build::bcc, py::arg("symbol"), py::arg("a"), py::kw_only(),
          py::arg("cubic") = false,
          "Body-centred cubic. The nearest-neighbour distance is a*sqrt(3)/2.");
    m.def("hcp", &build::hcp, py::arg("symbol"), py::arg("a"), py::kw_only(),
          py::arg("c") = 0.0,
          "Hexagonal close packed. `c` defaults to the ideal a*sqrt(8/3).");
    m.def("diamond", &build::diamond, py::arg("symbol"), py::arg("a"), py::kw_only(),
          py::arg("cubic") = false, "Diamond cubic. The bond length is a*sqrt(3)/4.");
    m.def("rocksalt", &build::rocksalt, py::arg("first"), py::arg("second"), py::arg("a"),
          py::kw_only(), py::arg("cubic") = false, "Rock salt (NaCl).");
    m.def("zincblende", &build::zincblende, py::arg("first"), py::arg("second"), py::arg("a"),
          py::kw_only(), py::arg("cubic") = false, "Zinc blende (GaAs).");
    m.def("cscl", &build::cscl, py::arg("first"), py::arg("second"), py::arg("a"),
          "Caesium chloride.");

    m.def(
        "bulk",
        [](const py::object& symbols, const py::object& lattice, double a, double c, bool cubic,
           bool orthorhombic) {
            // One symbol and no lattice named is the "what does this element
            // crystallise as" question, which the periodic table answers.
            const bool named = !lattice.is_none();
            if (py::isinstance<py::str>(symbols) && (!named || py::isinstance<py::str>(lattice))) {
                return build::bulk(symbols.cast<std::string>(),
                                   named ? lattice.cast<std::string>() : std::string{}, a, c,
                                   cubic, orthorhombic);
            }
            if (!named) {
                throw build::BuildError("bulk(): several symbols need a lattice; the periodic "
                                        "table only knows what the elements do on their own");
            }
            if (orthorhombic) {
                throw build::BuildError("bulk(): orthorhombic= applies to a single element's own "
                                        "hexagonal lattice");
            }
            return build::bulk(symbol_list(symbols), lattice_of(lattice), a, c, cubic);
        },
        py::arg("symbols"), py::arg("lattice") = py::none(), py::arg("a") = 0.0, py::kw_only(),
        py::arg("c") = 0.0, py::arg("cubic") = false, py::arg("orthorhombic") = false,
        R"doc(
A bulk crystal.

    fa.build.bulk("Cu")                       # fcc, a = 3.61, from the table
    fa.build.bulk("Si")                       # diamond, a = 5.43
    fa.build.bulk("Cu", a=3.70)               # the same lattice, strained
    fa.build.bulk("Ti", orthorhombic=True)    # hcp in a rectangular cell
    fa.build.bulk("Si", "diamond", 5.43)      # say it explicitly
    fa.build.bulk(["Na", "Cl"], "rocksalt", 5.64, cubic=True)

With one symbol and nothing else, the element's own ambient structure and
lattice constant are looked up - that is what makes `bulk("Cu")` mean
something. Naming a lattice overrides the table, which is how a hypothetical
phase gets built, and `a` or `c` override just that constant.

An element whose ambient solid is molecular (N, O, the halogens) or has a
complex cell (Mn, white P) raises rather than returning a plausible-looking
wrong crystal; name a structure to build an approximation.

`a` is the conventional cubic lattice constant, not the bond length. The
tabulated values are experimental and at room temperature: a relaxation will
move them by a per cent or two.

Lattices: sc, fcc, bcc, hcp, diamond, rocksalt, zincblende, cscl, fluorite,
wurtzite, hexagonal, tetragonal, orthorhombic, rhombohedral.
)doc");

    m.def(
        "reference_state",
        [](const py::object& element) {
            const int z = py::isinstance<py::str>(element)
                              ? number_from_symbol(element.cast<std::string>())
                              : element.cast<int>();
            const ReferenceState& state = reference_state(z);
            py::dict out;
            out["symmetry"] = std::string(state.symmetry);
            if (state.a > 0.0) {
                out["a"] = state.a;
            }
            if (state.covera > 0.0) {
                out["covera"] = state.covera;
            }
            if (state.alpha > 0.0) {
                out["alpha"] = state.alpha;
            }
            if (state.bovera > 0.0) {
                out["bovera"] = state.bovera;
            }
            return out;
        },
        py::arg("element"),
        R"doc(
What an element crystallises as at room temperature and pressure.

    fa.build.reference_state("Cu")   # {'symmetry': 'fcc', 'a': 3.61}
    fa.build.reference_state("Ti")   # {'symmetry': 'hcp', 'a': 2.95, 'covera': 1.588}

An empty `symmetry` means nothing is tabulated. These are the values
:func:`bulk` uses when given only a symbol.
)doc");


    // ------------------------------------------------------------- surfaces
    m.def(
        "surface",
        [](const Atoms& crystal, const std::array<int, 3>& indices, int layers, double vacuum,
           bool periodic) { return build::surface(crystal, indices, layers, vacuum, periodic); },
        py::arg("crystal"), py::arg("indices"), py::arg("layers"), py::kw_only(),
        py::arg("vacuum") = 0.0, py::arg("periodic") = false,
        R"doc(
Cut a slab along a Miller plane.

    slab = fa.build.surface(fa.build.bulk("Cu"), (2, 1, 1), layers=9, vacuum=10.0)

Works for any crystal and any plane, which is what the named faces below are
special cases of. The result has its first cell vector along x and the surface
normal along z, with `pbc = (True, True, False)`.

`vacuum` is added on each side. Leaving it at zero keeps the slab periodic
along the normal, which is what an interface calculation wants.
)doc");

    const auto face = [&m](const char* name, auto function, const char* doc) {
        m.def(
            name,
            [function](const std::string& symbol, const std::array<int, 3>& size, double a,
                       double vacuum, bool orthogonal) {
                return function(symbol, size, a, vacuum, orthogonal);
            },
            py::arg("symbol"), py::arg("size"), py::arg("a") = 0.0, py::kw_only(),
            py::arg("vacuum") = 0.0, py::arg("orthogonal") = false, doc);
    };

    face("fcc100", &build::fcc100, "The (100) face of an fcc metal. `size` is (nx, ny, layers).");
    face("fcc110", &build::fcc110, "The (110) face of an fcc metal. `size` is (nx, ny, layers).");
    face("fcc111", &build::fcc111, R"doc(
The close-packed (111) face of an fcc metal.

    slab = fa.build.fcc111("Pt", (4, 4, 5), vacuum=12.0)
    fa.build.add_adsorbate(slab, oxygen, 1.9, "fcc")

`size` is (nx, ny, layers) and `a` of zero looks the lattice constant up. The
slab records where its adsorption sites are, which is what lets
:func:`add_adsorbate` take a site by name.
)doc");
    face("fcc211", &build::fcc211, "The stepped (211) face of an fcc metal.");
    face("bcc100", &build::bcc100, "The (100) face of a bcc metal.");
    face("bcc110", &build::bcc110, "The (110) face of a bcc metal.");
    face("bcc111", &build::bcc111, "The (111) face of a bcc metal.");

    m.def(
        "hcp0001",
        [](const std::string& symbol, const std::array<int, 3>& size, double a, double c,
           double vacuum, bool orthogonal) {
            return build::hcp0001(symbol, size, a, c, vacuum, orthogonal);
        },
        py::arg("symbol"), py::arg("size"), py::arg("a") = 0.0, py::kw_only(), py::arg("c") = 0.0,
        py::arg("vacuum") = 0.0, py::arg("orthogonal") = false,
        "The basal (0001) face of an hcp metal.");

    m.def(
        "diamond100",
        [](const std::string& symbol, const std::array<int, 3>& size, double a, double vacuum) {
            return build::diamond100(symbol, size, a, vacuum);
        },
        py::arg("symbol"), py::arg("size"), py::arg("a") = 0.0, py::kw_only(),
        py::arg("vacuum") = 0.0, "The (100) face of a diamond-structure crystal.");

    m.def(
        "diamond111",
        [](const std::string& symbol, const std::array<int, 3>& size, double a, double vacuum) {
            return build::diamond111(symbol, size, a, vacuum);
        },
        py::arg("symbol"), py::arg("size"), py::arg("a") = 0.0, py::kw_only(),
        py::arg("vacuum") = 0.0, "The (111) face of a diamond-structure crystal.");

    m.def(
        "add_vacuum",
        [](Atoms& atoms, double vacuum) { build::add_vacuum(atoms, vacuum); },
        py::arg("atoms"), py::arg("vacuum"),
        "Add vacuum along the third cell vector, on top of whatever is there. "
        "Use :func:`set_vacuum` to set the total instead.");

    m.def(
        "add_adsorbate",
        [](Atoms& slab, const py::object& adsorbate, double height, const py::object& position,
           const std::array<double, 2>& offset, std::size_t mol_index) {
            // An adsorbate may be a structure or the name of one.
            const Atoms molecule = py::isinstance<py::str>(adsorbate)
                                       ? adsorption::molecule(adsorbate.cast<std::string>())
                                       : adsorbate.cast<Atoms>();
            if (py::isinstance<py::str>(position)) {
                build::add_adsorbate(slab, molecule, height,
                                     build::site_from_string(position.cast<std::string>()), offset,
                                     mol_index);
                return;
            }
            const auto xy = position.cast<std::array<double, 2>>();
            build::add_adsorbate(slab, molecule, height,
                                 std::array<double, 2>{xy[0] + offset[0], xy[1] + offset[1]},
                                 mol_index);
        },
        py::arg("slab"), py::arg("adsorbate"), py::arg("height"), py::arg("position") = "ontop",
        py::kw_only(), py::arg("offset") = std::array<double, 2>{0.0, 0.0},
        py::arg("mol_index") = 0,
        R"doc(
Put an adsorbate on a slab.

    fa.build.add_adsorbate(slab, "CO", 1.9, "ontop")
    fa.build.add_adsorbate(slab, oh, 2.0, "fcc", mol_index=0)   # bound through atom 0

`position` is a site name (ontop, bridge, hollow, fcc, hcp) or an explicit
(x, y) in angstrom. `height` is measured from the topmost slab atom, and
`mol_index` picks which atom of the adsorbate sits at that height - the rest
follow rigidly, so an OH binds through whichever atom is named.

Site names need a slab that knows where its sites are, which the named face
builders record. For an arbitrary structure use :class:`AdsorptionFinder`,
which works the sites out.
)doc");

    // ------------------------------------------------------------ 2D sheets
    m.def(
        "honeycomb",
        [](const py::object& symbols, double a, double vacuum, double buckling) {
            return build::honeycomb(symbol_list(symbols), a, vacuum, buckling);
        },
        py::arg("symbols"), py::arg("a"), py::kw_only(), py::arg("vacuum") = 15.0,
        py::arg("buckling") = 0.0,
        R"doc(
A honeycomb monolayer of one or two species.

    fa.build.honeycomb("C", 2.46)                    # graphene
    fa.build.honeycomb(["B", "N"], 2.51)             # hexagonal boron nitride
    fa.build.honeycomb("Si", 3.87, buckling=0.44)    # silicene, which is not flat
)doc");

    m.def("graphene", &build::graphene, py::kw_only(), py::arg("a") = 2.46,
          py::arg("vacuum") = 15.0, py::arg("nx") = 1, py::arg("ny") = 1,
          "A graphene sheet; the default lattice constant gives the usual 1.42 A bond.");
    m.def("graphene_rectangular", &build::graphene_rectangular, py::kw_only(),
          py::arg("a") = 2.46, py::arg("vacuum") = 15.0, py::arg("nx") = 1, py::arg("ny") = 1,
          "A rectangular four-atom graphene cell: armchair along x, zigzag along y.");

    m.def(
        "tmd",
        [](const std::string& metal, const std::string& chalcogen, const py::object& phase,
           double a, double thickness, double vacuum) {
            return build::tmd(metal, chalcogen, phase_of(phase), a, thickness, vacuum);
        },
        py::arg("metal"), py::arg("chalcogen"), py::kw_only(), py::arg("phase") = "1H",
        py::arg("a") = 0.0, py::arg("thickness") = 0.0, py::arg("vacuum") = 15.0,
        R"doc(
A transition-metal dichalcogenide monolayer, MX2.

    fa.build.tmd("Mo", "S")                # measured values from the table
    fa.build.tmd("W", "Se", phase="1T")
    fa.build.tmd("Mo", "S", a=3.18, thickness=3.13)

`a` and `thickness` of zero look the pair up; `known_tmds()` lists what is
tabulated.
)doc");

    m.def("tmd_parameters", &build::tmd_parameters, py::arg("metal"), py::arg("chalcogen"),
          "The measured (a, thickness) of a known MX2, in angstrom.");
    m.def("known_tmds", &build::known_tmds, "The dichalcogenides this build has numbers for.");
    m.def("set_vacuum", &build::set_vacuum, py::arg("atoms"), py::arg("vacuum"),
          "Set the vacuum gap of an existing sheet or slab along z, recentring it.");

    // ------------------------------------------------------------ nanotubes
    py::class_<build::TubeGeometry>(m, "TubeGeometry",
                                    "What the (n, m) indices imply, before any atoms.")
        .def_readonly("radius", &build::TubeGeometry::radius, "Angstrom.")
        .def_readonly("circumference", &build::TubeGeometry::circumference)
        .def_readonly("period", &build::TubeGeometry::period,
                      "Length of one repeat along the axis.")
        .def_readonly("chiral_angle", &build::TubeGeometry::chiral_angle,
                      "Degrees: 0 for zigzag, 30 for armchair.")
        .def_readonly("cells", &build::TubeGeometry::cells, "Sheet cells in one repeat.")
        .def_readonly("t1", &build::TubeGeometry::t1)
        .def_readonly("t2", &build::TubeGeometry::t2)
        .def("__repr__", &build::TubeGeometry::to_string);

    m.def("tube_geometry", &build::tube_geometry, py::arg("n"), py::arg("m"),
          py::arg("a") = 2.46, "The geometry of an (n, m) tube, without building it.");
    m.def("roll", &build::roll, py::arg("sheet"), py::arg("n"), py::arg("m"), py::kw_only(),
          py::arg("length") = 1, py::arg("vacuum") = 15.0,
          R"doc(
Roll any hexagonal monolayer into an (n, m) tube.

    fa.build.roll(fa.build.tmd("Mo", "S"), 10, 0)

A sheet with thickness rolls into a tube with an inner and an outer wall,
because each atom keeps its height above the mid-plane as a radial offset.
)doc");
    m.def("nanotube", &build::nanotube, py::arg("n"), py::arg("m"), py::kw_only(),
          py::arg("length") = 1, py::arg("a") = 2.46, py::arg("symbol") = "C",
          py::arg("vacuum") = 15.0, "A carbon nanotube.");

    // ------------------------------------------------------------- clusters
    m.def("icosahedron", &build::icosahedron, py::arg("symbol"), py::arg("shells"),
          py::arg("d"), py::kw_only(), py::arg("vacuum") = 10.0,
          R"doc(
A Mackay icosahedron. Closed shells at 13, 55, 147, 309, 561 atoms.

`d` is the spacing within a shell. The shells themselves come out 1.0515 times
closer together - twenty tetrahedra cannot meet at a point without straining,
and that strain is why icosahedra stop winning above a few hundred atoms.
)doc");
    m.def("cuboctahedron", &build::cuboctahedron, py::arg("symbol"), py::arg("shells"),
          py::arg("a"), py::kw_only(), py::arg("vacuum") = 10.0,
          "An fcc cuboctahedron, with {111} and {100} facets. Same magic numbers.");
    m.def("decahedron", &build::decahedron, py::arg("symbol"), py::arg("p"), py::arg("q"),
          py::kw_only(), py::arg("r") = 0, py::arg("a"), py::arg("vacuum") = 10.0,
          R"doc(
A decahedron: five fcc tetrahedra twinned about a common five-fold axis.

`p` sets the {100} side facets, `q` the extent along the axis, and `r` the
depth of the Marks re-entrance - zero gives an Ino decahedron.
)doc");

    py::class_<build::SurfaceEnergy>(m, "SurfaceEnergy", "One facet family and its cost.")
        .def(py::init([](int h, int k, int l, double energy) {
                 return build::SurfaceEnergy{h, k, l, energy};
             }),
             py::arg("h"), py::arg("k"), py::arg("l"), py::arg("energy"))
        .def_readwrite("h", &build::SurfaceEnergy::h)
        .def_readwrite("k", &build::SurfaceEnergy::k)
        .def_readwrite("l", &build::SurfaceEnergy::l)
        .def_readwrite("energy", &build::SurfaceEnergy::energy);

    m.def(
        "wulff",
        [](const std::string& symbol, const py::object& lattice, double a,
           const py::object& energies, int atoms, double vacuum) {
            return build::wulff(symbol, lattice_of(lattice), a, surface_energies(energies), atoms,
                                vacuum);
        },
        py::arg("symbol"), py::arg("lattice"), py::arg("a"), py::arg("energies"),
        py::arg("atoms"), py::kw_only(), py::arg("vacuum") = 10.0,
        R"doc(
The Wulff construction: the equilibrium shape of a crystal.

    fa.build.wulff("Ag", "fcc", 4.09, {"111": 1.0, "100": 1.15}, atoms=500)

Each facet sits at a distance proportional to its surface energy, so cheap
facets grow and dear ones shrink. Only the ratios matter; the overall scale is
set by the atom count you ask for, and the nearest achievable size is used
since the count jumps as whole facets appear.
)doc");

    m.def("equivalent_directions", &build::equivalent_directions, py::arg("h"), py::arg("k"),
          py::arg("l"),
          "Every symmetry-equivalent direction under cubic symmetry: {100} gives six.");
}

}  // namespace fastatoms::bindings
