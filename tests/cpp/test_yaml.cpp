// SPDX-License-Identifier: MIT
#include <cmath>
#include <string>

#include "fastatoms/io/yaml_input.hpp"
#include "testing.hpp"

using namespace fastatoms;

FA_TEST(reads_the_documented_layout) {
    const std::string text = R"(
structure:
  cell: [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
  pbc: [true, true, false]
  atoms:
    - {symbol: O, position: [0.0, 0.0, 0.0], charge: -0.8}
    - {symbol: H, position: [0.96, 0.0, 0.0], charge: 0.4}
    - {symbol: H, position: [-0.24, 0.93, 0.0], charge: 0.4}

simulation:
  task: md
  calculator:
    name: lennard_jones
    parameters: {epsilon: 0.0103, sigma: 3.4, cutoff: 10.0}
  md:
    ensemble: nvt
    timestep: 0.5
    steps: 1000
    temperature: 300.0
    thermostat: langevin
  output: {trajectory: traj.yaml, interval: 10}
)";

    const io::InputDeck deck = io::read_yaml_string(text);
    FA_CHECK(deck.atoms.size() == 3);
    FA_CHECK(deck.atoms.formula() == "H2O");
    FA_CHECK_CLOSE(deck.atoms.volume(), 1000.0, 1e-9);
    FA_CHECK(deck.atoms.pbc()[0] && !deck.atoms.pbc()[2]);
    FA_CHECK(deck.atoms.has_charges());
    FA_CHECK_CLOSE(deck.atoms.charges()[0], -0.8, 1e-12);

    FA_CHECK(deck.simulation.task == Task::MolecularDynamics);
    FA_CHECK(deck.simulation.calculator.name == "lennard_jones");
    FA_CHECK_CLOSE(deck.simulation.calculator.parameters["sigma"].as_double(), 3.4, 1e-12);
    FA_CHECK(deck.simulation.md.ensemble == "nvt");
    FA_CHECK_CLOSE(deck.simulation.md.timestep_fs, 0.5, 1e-12);
    FA_CHECK(deck.simulation.md.steps == 1000);
    FA_CHECK_CLOSE(deck.simulation.md.temperature_K, 300.0, 1e-12);
    FA_CHECK(deck.simulation.output.trajectory == "traj.yaml");
    FA_CHECK(deck.simulation.output.interval == 10);

    // The unparsed document is available for keys FastAtoms does not know.
    FA_CHECK(deck.document["simulation.md.thermostat"].as_string() == "langevin");
}

FA_TEST(accepts_the_columnar_form_and_scaled_coordinates) {
    const std::string text = R"(
structure:
  cell: {a: 5.43, b: 5.43, c: 5.43}
  coordinates: fractional
  symbols: [Si, Si]
  positions:
    - [0.0, 0.0, 0.0]
    - [0.25, 0.25, 0.25]
  tags: [0, 1]
)";

    const Atoms atoms = io::read_yaml_string(text).atoms;
    FA_CHECK(atoms.size() == 2);
    FA_CHECK_CLOSE(atoms.position(1).x, 0.25 * 5.43, 1e-12);
    FA_CHECK(atoms.pbc()[0]);
    FA_CHECK(atoms.has_tags());
    FA_CHECK(atoms.tags()[1] == 1);
}

FA_TEST(accepts_the_shorthand_cell_and_pbc_spellings) {
    FA_CHECK_CLOSE(io::read_yaml_string("structure: {cell: 8.0, symbols: [Ar], "
                                        "positions: [[0, 0, 0]]}")
                       .atoms.volume(),
                   512.0, 1e-9);

    const Atoms slab =
        io::read_yaml_string("structure: {cell: [4.0, 4.0, 20.0], pbc: xy, symbols: [Al], "
                             "positions: [[0, 0, 0]]}")
            .atoms;
    FA_CHECK(slab.pbc()[0] && slab.pbc()[1] && !slab.pbc()[2]);

    const Atoms compact =
        io::read_yaml_string("structure: {atoms: [[Na, 0.0, 0.0, 0.0], [Cl, 2.8, 0.0, 0.0]]}")
            .atoms;
    FA_CHECK(compact.formula() == "ClNa");
    FA_CHECK_CLOSE(compact.distance(0, 1), 2.8, 1e-12);

    // A bare structure document, with no `structure:` wrapper.
    const Atoms bare =
        io::read_yaml_string("symbols: [He]\npositions: [[1.0, 2.0, 3.0]]").atoms;
    FA_CHECK(bare.size() == 1);
    FA_CHECK_CLOSE(bare.position(0).z, 3.0, 1e-12);
}

FA_TEST(reads_velocities_and_extra_arrays) {
    const std::string text = R"(
structure:
  symbols: [H, H]
  positions: [[0, 0, 0], [0.74, 0, 0]]
  velocities: [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]
  arrays:
    forces: [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    weights: [0.5, 1.5]
)";
    const Atoms atoms = io::read_yaml_string(text).atoms;
    FA_CHECK(atoms.has_momenta());
    FA_CHECK_CLOSE(atoms.velocities()[0], 0.1, 1e-12);
    FA_CHECK(atoms.array("forces").ncols == 3);
    FA_CHECK_CLOSE(atoms.array("forces").data[5], -1.0, 1e-12);
    FA_CHECK(atoms.array("weights").ncols == 1);
}

FA_TEST(reports_useful_errors) {
    FA_CHECK_THROWS(io::read_yaml_string("structure: {symbols: [Zz], positions: [[0,0,0]]}"),
                    InputError);
    FA_CHECK_THROWS(io::read_yaml_string("structure: {symbols: [H], positions: [[0,0]]}"),
                    InputError);
    FA_CHECK_THROWS(io::read_yaml_string("structure: {atoms: [{symbol: H}]}"), InputError);
    FA_CHECK_THROWS(io::read_yaml_string("structure: {cell: [1, 2], symbols: [H], "
                                         "positions: [[0,0,0]]}"),
                    InputError);
    FA_CHECK_THROWS(io::read_yaml_string("structure: {symbols: [H], positions: [[0,0,0]], "
                                         "coordinates: polar}"),
                    InputError);
    FA_CHECK_THROWS(io::read_yaml_string("structure: {symbols: [H], positions: [[0,0,0]]}\n"
                                         "simulation: {md: {timestep: -1.0}}"),
                    InputError);
    FA_CHECK_THROWS(io::read_yaml_file("/nonexistent/path/to/input.yaml"), InputError);
    FA_CHECK_THROWS(io::read_yaml_string("structure: [this, is, not, a, mapping]"), InputError);
}

FA_TEST(structures_round_trip_through_yaml) {
    Atoms atoms(std::vector<std::string>{"Si", "O"},
                {0.1, 0.2, 0.3, 1.4, 1.5, 1.6},
                Cell::from_parameters(5.0, 6.0, 7.0, 80.0, 85.0, 95.0));
    atoms.set_charges({1.2, -0.6});
    atoms.set_tags({3, 4});
    atoms.set_velocities({0.01, 0.0, 0.0, 0.0, 0.02, 0.0});
    atoms.set_array("forces", {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 3);

    const Atoms restored = io::read_yaml_string(io::write_yaml_string(atoms)).atoms;
    FA_CHECK(restored.size() == atoms.size());
    FA_CHECK(restored.formula() == atoms.formula());
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        FA_CHECK_CLOSE(restored.position(i).x, atoms.position(i).x, 1e-12);
        FA_CHECK_CLOSE(restored.position(i).z, atoms.position(i).z, 1e-12);
    }
    FA_CHECK_CLOSE(restored.volume(), atoms.volume(), 1e-10);
    FA_CHECK_CLOSE(restored.charges()[0], 1.2, 1e-12);
    FA_CHECK(restored.tags()[1] == 4);
    FA_CHECK_CLOSE(restored.momenta()[0], atoms.momenta()[0], 1e-12);
    FA_CHECK_CLOSE(restored.array("forces").data[4], 5.0, 1e-12);

    // The same, written in fractional coordinates.
    const Atoms scaled_round_trip =
        io::read_yaml_string(io::write_yaml_string(atoms, true)).atoms;
    FA_CHECK_CLOSE(scaled_round_trip.position(1).y, atoms.position(1).y, 1e-10);
}

FA_TEST(value_tree_navigation) {
    const Value document = io::parse_yaml_string(R"(
a:
  b:
    c: 42
    d: [1, 2.5, "three"]
    e: true
)");
    FA_CHECK(document["a.b.c"].as_int() == 42);
    FA_CHECK_CLOSE(document["a.b.d"].at(1).as_double(), 2.5, 1e-12);
    FA_CHECK(document["a.b.d"].at(2).as_string() == "three");
    FA_CHECK(document["a.b.e"].as_bool());
    FA_CHECK(document["a.b.missing"].is_null());
    FA_CHECK(document["nowhere.at.all"].value_or(7) == 7);
    FA_CHECK_THROWS(document["a.b.c"].as_string(), InputError);
    FA_CHECK_THROWS(document.at("nope"), InputError);
}

FA_TEST_MAIN()
