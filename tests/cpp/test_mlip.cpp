// SPDX-License-Identifier: MIT
//
// The machine-learning potential layer: the graph an Atoms object is turned
// into, the calculator that wraps a backend, backend selection, and the two
// backends themselves.
//
// Nothing here needs a trained model. The plumbing is checked against a stub
// backend that computes Lennard-Jones from the graph it is handed, which makes
// every claim about units, conventions and caching testable with no download
// and no Python. The tests that do need a model are guarded by environment
// variables and skip themselves otherwise:
//
//   FASTATOMS_TEST_MLIP_MODEL   an exported TorchScript archive
//   FASTATOMS_TEST_MLIP_PYTHON  an interpreter with a potential installed
//   FASTATOMS_TEST_MLIP_FAMILY  which family that interpreter should load

#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "fastatoms/calculators/mlip.hpp"
#include "fastatoms/calculators/mlip_models.hpp"
#include "fastatoms/calculators/mlip_worker.hpp"
#include "fastatoms/calculators/registry.hpp"
#include "fastatoms/core/atoms.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::mlip;

namespace {

std::string environment(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr ? std::string(value) : std::string();
}

Atoms argon_crystal(double a = 5.30, int repeat = 2) {
    Atoms atoms;
    const double basis[4][3] = {{0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5},
                                {0.5, 0.5, 0.0}};
    for (const auto& site : basis) {
        atoms.append(18, Vec3{site[0] * a, site[1] * a, site[2] * a});
    }
    atoms.set_cell(Mat3::diagonal(a, a, a));
    atoms.set_pbc({true, true, true});
    return repeat > 1 ? atoms.repeat(repeat, repeat, repeat) : atoms;
}

Atoms water_molecule() {
    Atoms atoms;
    atoms.append(8, Vec3{0.0, 0.0, 0.0});
    atoms.append(1, Vec3{0.96, 0.0, 0.0});
    atoms.append(1, Vec3{-0.24, 0.93, 0.0});
    atoms.set_cell(Mat3::diagonal(12.0, 12.0, 12.0));
    atoms.set_pbc({true, true, true});
    return atoms;
}

/// A displacement along one axis.
///
/// Written out rather than indexing `&vec.x`, which is out-of-bounds on a
/// scalar and lets the optimiser assume y and z were never written - a bug
/// that silently turns two thirds of a finite-difference check into 0 == 0.
Vec3 axis_shift(int axis, double amount) {
    return axis == 0   ? Vec3{amount, 0.0, 0.0}
           : axis == 1 ? Vec3{0.0, amount, 0.0}
                       : Vec3{0.0, 0.0, amount};
}

/// A backend that needs no model: cut-and-shifted Lennard-Jones evaluated on
/// exactly the graph the real backends receive.
///
/// Because it is analytic, every number it produces is known in advance, so it
/// pins down what the calculator does with a backend's output: the sign of the
/// forces, the Voigt order and sign of the stress, and which properties are
/// filled in.
class StubBackend final : public Backend {
public:
    explicit StubBackend(double epsilon = 0.0103, double sigma = 3.40, double cutoff = 8.5)
        : epsilon_(epsilon), sigma_(sigma) {
        info_.family = "stub";
        info_.path = "analytic";
        info_.backend = "stub";
        info_.cutoff = cutoff;
        info_.species = {18, 1, 8};
        info_.provides_forces = true;
        info_.provides_stress = true;
    }

    [[nodiscard]] const ModelInfo& info() const override { return info_; }

    [[nodiscard]] std::size_t evaluations() const noexcept { return evaluations_; }

    [[nodiscard]] Prediction evaluate(const Graph& graph, bool need_forces,
                                      bool need_stress) override {
        ++evaluations_;
        Prediction prediction;
        prediction.forces.assign(3 * graph.natoms, 0.0);

        const double rc = info_.cutoff;
        const double sr6_c = std::pow(sigma_ / rc, 6.0);
        const double shift = 4.0 * epsilon_ * (sr6_c * sr6_c - sr6_c);

        double virial[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        const std::size_t m = graph.nedges();
        for (std::size_t e = 0; e < m; ++e) {
            const auto i = static_cast<std::size_t>(graph.receivers()[e]);
            const auto j = static_cast<std::size_t>(graph.senders()[e]);
            double d[3];
            for (int axis = 0; axis < 3; ++axis) {
                double shift_component = 0.0;
                for (int k = 0; k < 3; ++k) {
                    shift_component += graph.unit_shifts[3 * e + static_cast<std::size_t>(k)] *
                                       graph.cell.m[static_cast<std::size_t>(3 * k + axis)];
                }
                d[axis] = graph.positions[3 * j + static_cast<std::size_t>(axis)] +
                          shift_component -
                          graph.positions[3 * i + static_cast<std::size_t>(axis)];
            }
            const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if (r2 >= rc * rc || r2 == 0.0) {
                continue;
            }
            const double inverse2 = sigma_ * sigma_ / r2;
            const double inverse6 = inverse2 * inverse2 * inverse2;
            // Every pair appears twice in a full list, so each edge carries half.
            prediction.energy += 0.5 * (4.0 * epsilon_ * (inverse6 * inverse6 - inverse6) - shift);
            const double magnitude = 24.0 * epsilon_ * (2.0 * inverse6 * inverse6 - inverse6) / r2;
            for (int axis = 0; axis < 3; ++axis) {
                // dE/dr_i for this half-edge; the other direction supplies the rest.
                prediction.forces[3 * i + static_cast<std::size_t>(axis)] -= magnitude * d[axis];
                for (int beta = 0; beta < 3; ++beta) {
                    virial[3 * axis + beta] += 0.5 * magnitude * d[axis] * d[beta];
                }
            }
        }

        const double volume = std::abs(graph.cell.determinant());
        if (need_stress && volume > 0.0) {
            prediction.stress = {-virial[0] / volume,
                                 -virial[4] / volume,
                                 -virial[8] / volume,
                                 -0.5 * (virial[5] + virial[7]) / volume,
                                 -0.5 * (virial[2] + virial[6]) / volume,
                                 -0.5 * (virial[1] + virial[3]) / volume};
            prediction.has_stress = true;
        }
        prediction.has_forces = need_forces;
        return prediction;
    }

private:
    ModelInfo info_{};
    double epsilon_;
    double sigma_;
    std::size_t evaluations_{0};
};

}  // namespace

// ---------------------------------------------------------------------------
// The graph
// ---------------------------------------------------------------------------
FA_TEST(graph_carries_the_structure) {
    const Atoms atoms = argon_crystal(5.30, 2);
    const Graph graph = build_graph(atoms, 6.0);

    FA_CHECK(graph.natoms == atoms.size());
    FA_CHECK(graph.positions.size() == 3 * atoms.size());
    FA_CHECK(graph.numbers.size() == atoms.size());
    FA_CHECK(graph.numbers[0] == 18);
    FA_CHECK(graph.cutoff == 6.0);
    FA_CHECK(graph.nedges() > 0);
    FA_CHECK(graph.edge_index.size() == 2 * graph.nedges());
    FA_CHECK(graph.unit_shifts.size() == 3 * graph.nedges());
    for (int axis = 0; axis < 3; ++axis) {
        FA_CHECK(graph.pbc[static_cast<std::size_t>(axis)]);
        FA_CHECK_CLOSE(graph.cell.m[static_cast<std::size_t>(4 * axis)], 10.6, 1e-12);
    }
}

FA_TEST(graph_edges_obey_the_shift_convention) {
    // r_ij = r[sender] + unit_shifts * cell - r[receiver], and every edge is
    // inside the cutoff. Checking it directly is what makes the convention a
    // fact rather than a comment.
    const Atoms atoms = argon_crystal(5.30, 1);
    const double cutoff = 6.0;
    const Graph graph = build_graph(atoms, cutoff);

    std::size_t checked = 0;
    for (std::size_t e = 0; e < graph.nedges(); ++e) {
        const auto i = static_cast<std::size_t>(graph.receivers()[e]);
        const auto j = static_cast<std::size_t>(graph.senders()[e]);
        FA_CHECK(i < graph.natoms && j < graph.natoms);
        double d[3];
        for (int axis = 0; axis < 3; ++axis) {
            double offset = 0.0;
            for (int k = 0; k < 3; ++k) {
                offset += graph.unit_shifts[3 * e + static_cast<std::size_t>(k)] *
                          graph.cell.m[static_cast<std::size_t>(3 * k + axis)];
            }
            d[axis] = graph.positions[3 * j + static_cast<std::size_t>(axis)] + offset -
                      graph.positions[3 * i + static_cast<std::size_t>(axis)];
        }
        const double r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        FA_CHECK(r > 1e-8);
        FA_CHECK(r < cutoff + 1e-9);
        ++checked;
    }
    FA_CHECK(checked == graph.nedges());

    // A full list: (i, j) appears exactly as often as (j, i).
    std::size_t forward = 0;
    std::size_t backward = 0;
    for (std::size_t e = 0; e < graph.nedges(); ++e) {
        if (graph.receivers()[e] == 0) {
            ++forward;
        }
        if (graph.senders()[e] == 0) {
            ++backward;
        }
    }
    FA_CHECK(forward == backward);
    FA_CHECK(forward > 0);
}

FA_TEST(graph_without_edges_skips_the_neighbour_list) {
    const Atoms atoms = argon_crystal(5.30, 2);
    const Graph graph = build_graph(atoms, 0.0, /*with_edges=*/false);

    FA_CHECK(graph.natoms == atoms.size());
    FA_CHECK(graph.positions.size() == 3 * atoms.size());
    FA_CHECK(graph.nedges() == 0);
    FA_CHECK(graph.cutoff == 0.0);
}

FA_TEST(graph_rejects_nonsense) {
    const Atoms empty;
    FA_CHECK_THROWS(build_graph(empty, 5.0), MLIPError);
    FA_CHECK_THROWS(build_graph(argon_crystal(), 0.0), MLIPError);
    FA_CHECK_THROWS(build_graph(argon_crystal(), -1.0), MLIPError);
}

// ---------------------------------------------------------------------------
// ModelInfo
// ---------------------------------------------------------------------------
FA_TEST(model_info_species) {
    ModelInfo info;
    FA_CHECK(info.supports(92));  // no declared list means no restriction

    info.species = {1, 6, 8};
    FA_CHECK(info.supports(1));
    FA_CHECK(info.supports(8));
    FA_CHECK(!info.supports(18));

    info.family = "mace";
    info.path = "model.pt";
    info.backend = "torchscript";
    info.cutoff = 5.0;
    FA_CHECK(info.to_string().find("mace") != std::string::npos);
    FA_CHECK(info.to_string().find("torchscript") != std::string::npos);
}

// ---------------------------------------------------------------------------
// The calculator, against a backend whose answers are known
// ---------------------------------------------------------------------------
FA_TEST(calculator_reports_what_the_backend_returns) {
    Atoms atoms = argon_crystal(5.30, 2);
    MLIPCalculator calculator;
    auto stub = std::make_shared<StubBackend>();
    calculator.set_backend(stub);

    FA_CHECK(calculator.loaded());
    FA_CHECK(calculator.model().backend == "stub");
    FA_CHECK_CLOSE(calculator.cutoff(), 8.5, 1e-12);

    const double energy = calculator.get_potential_energy(atoms);
    FA_CHECK(energy < 0.0);  // argon near its equilibrium volume is bound

    const std::vector<double> forces = calculator.get_forces(atoms);
    FA_CHECK(forces.size() == 3 * atoms.size());
    // A perfect lattice: every atom sits at a symmetry point, so no net force.
    for (const double component : forces) {
        FA_CHECK_CLOSE(component, 0.0, 1e-9);
    }

    const std::array<double, 6> stress = calculator.get_stress(atoms);
    FA_CHECK_CLOSE(stress[0], stress[1], 1e-9);
    FA_CHECK_CLOSE(stress[1], stress[2], 1e-9);
    for (std::size_t k = 3; k < 6; ++k) {
        FA_CHECK_CLOSE(stress[k], 0.0, 1e-9);
    }
}

FA_TEST(calculator_forces_match_finite_differences) {
    // The stub is analytic, so this checks the whole path - graph, backend,
    // results - rather than the potential.
    Atoms atoms = argon_crystal(5.30, 1);
    atoms.set_position(0, atoms.position(0) + Vec3{0.13, -0.07, 0.05});

    MLIPCalculator calculator;
    calculator.set_backend(std::make_shared<StubBackend>());

    const std::vector<double> forces = calculator.get_forces(atoms);
    const double step = 1e-5;
    for (std::size_t atom = 0; atom < atoms.size(); ++atom) {
        for (int axis = 0; axis < 3; ++axis) {
            Atoms plus = atoms;
            Atoms minus = atoms;
            const Vec3 shift = axis_shift(axis, step);
            plus.set_position(atom, plus.position(atom) + shift);
            minus.set_position(atom, minus.position(atom) - shift);

            MLIPCalculator probe;
            probe.set_backend(std::make_shared<StubBackend>());
            const double e_plus = probe.get_potential_energy(plus);
            probe.set_backend(std::make_shared<StubBackend>());
            const double e_minus = probe.get_potential_energy(minus);

            const double numerical = -(e_plus - e_minus) / (2.0 * step);
            FA_CHECK_CLOSE(forces[3 * atom + static_cast<std::size_t>(axis)], numerical, 1e-5);
        }
    }
}

FA_TEST(calculator_caches_between_requests) {
    Atoms atoms = argon_crystal(5.30, 1);
    MLIPCalculator calculator;
    auto stub = std::make_shared<StubBackend>();
    calculator.set_backend(stub);

    // Energy, forces and stress come out of one model call, so asking for all
    // three must not run the network three times. This is what makes an MD
    // step cost one evaluation rather than two.
    calculator.get_potential_energy(atoms);
    calculator.get_forces(atoms);
    calculator.get_stress(atoms);
    FA_CHECK(stub->evaluations() == 1);

    atoms.set_position(0, atoms.position(0) + Vec3{0.01, 0.0, 0.0});
    calculator.get_potential_energy(atoms);
    FA_CHECK(stub->evaluations() == 2);  // moving an atom invalidates it
}

FA_TEST(calculator_refuses_species_the_model_does_not_know) {
    Atoms atoms;
    atoms.append(92, Vec3{0.0, 0.0, 0.0});  // uranium: not in the stub's list
    atoms.append(92, Vec3{2.0, 0.0, 0.0});
    atoms.set_cell(Mat3::diagonal(10.0, 10.0, 10.0));
    atoms.set_pbc({true, true, true});

    MLIPCalculator calculator;
    calculator.set_backend(std::make_shared<StubBackend>());
    FA_CHECK_THROWS(calculator.get_potential_energy(atoms), CalculatorError);
}

FA_TEST(calculator_leaves_stress_alone_for_a_molecule) {
    Atoms molecule = water_molecule();
    molecule.set_pbc({false, false, false});

    MLIPCalculator calculator;
    calculator.set_backend(std::make_shared<StubBackend>());
    calculator.calculate(molecule, {Property::Energy, Property::Forces});

    FA_CHECK(calculator.results().energy.has_value());
    FA_CHECK(calculator.results().forces.size() == 3 * molecule.size());
    // There is no strain to differentiate against without a cell, so no stress
    // is produced - and asking for one anyway is an error, not a zero.
    FA_CHECK(!calculator.results().stress.has_value());
    FA_CHECK_THROWS(calculator.get_stress(molecule), CalculatorError);
}

FA_TEST(calculator_records_where_the_numbers_came_from) {
    Atoms atoms = argon_crystal(5.30, 1);
    MLIPCalculator calculator;
    calculator.set_backend(std::make_shared<StubBackend>());
    calculator.get_potential_energy(atoms);

    const Value& extra = calculator.results().extra;
    FA_CHECK(extra.is_map());
    FA_CHECK(extra.find("backend") != nullptr);
    FA_CHECK(extra.find("backend")->as_string() == "stub");
    FA_CHECK(extra.find("edges") != nullptr);
    FA_CHECK(extra.find("edges")->as_int() > 0);
    FA_CHECK_CLOSE(extra.find("cutoff")->as_double(), 8.5, 1e-12);
}

FA_TEST(unloading_drops_the_model) {
    MLIPCalculator calculator;
    calculator.set_backend(std::make_shared<StubBackend>());
    FA_CHECK(calculator.loaded());
    calculator.unload();
    FA_CHECK(!calculator.loaded());
}

// ---------------------------------------------------------------------------
// Configuration and selection
// ---------------------------------------------------------------------------
FA_TEST(families_are_registered) {
    const std::vector<std::string> names = registered_calculators();
    for (const char* expected : {"mace", "mattersim", "chgnet", "nequip", "allegro", "mlip"}) {
        FA_CHECK(std::find(names.begin(), names.end(), expected) != names.end());
    }

    FA_CHECK(make_calculator("mace")->name() == "mace");
    FA_CHECK(make_calculator("MatterSim")->name() == "mattersim");
    FA_CHECK(make_calculator("matter-sim")->name() == "mattersim");
    FA_CHECK(make_calculator("mace_mp")->name() == "mace");
    FA_CHECK(make_calculator("allegro")->name() == "allegro");
    FA_CHECK(make_calculator("nequip")->name() == "nequip");
}

FA_TEST(a_calculator_without_a_model_says_so) {
    MLIPCalculator calculator;
    Atoms atoms = argon_crystal();
    // The generic family has no default model, so loading has to fail loudly.
    FA_CHECK_THROWS(calculator.get_potential_energy(atoms), CalculatorError);
}

FA_TEST(unknown_backend_names_are_rejected) {
    Value parameters(Value::Map{});
    parameters.insert("backend", Value(std::string("magic")));
    FA_CHECK_THROWS(make_backend("mace", "model.pt", parameters), MLIPError);
}

FA_TEST(torchscript_backend_reports_whether_it_exists) {
    // Whichever way this build went, the answer must be self-consistent.
    if (torch_available()) {
        FA_CHECK(!torch_version().empty());
    } else {
        Value parameters(Value::Map{});
        parameters.insert("backend", Value(std::string("torchscript")));
        FA_CHECK_THROWS(make_backend("mace", "model.pt", parameters), MLIPError);
    }
}

FA_TEST(conda_environments_are_resolved_or_reported) {
    FA_CHECK(conda_python("").empty());
    FA_CHECK(conda_python("an-environment-that-does-not-exist-42").empty());

    // The worker script ships with the library, so it must be findable.
    FA_CHECK(!worker_script_path().empty());
}

FA_TEST(a_missing_conda_environment_is_an_error_not_a_crash) {
    Value parameters(Value::Map{});
    parameters.insert("backend", Value(std::string("worker")));
    parameters.insert("conda_env", Value(std::string("no-such-environment-42")));
    FA_CHECK_THROWS(make_backend("mace", "whatever", parameters), MLIPError);
}

FA_TEST(mace_resolves_short_names_to_cached_checkpoints) {
    // Only asserts the shape of the answer: on a machine with no MACE cache
    // the name comes back unchanged, which is also correct.
    auto calculator = std::make_shared<mlip::MACECalculator>();
    FA_CHECK(calculator->name() == "mace");
    FA_CHECK(default_environment_for("mace") == "mace_env" ||
             !default_environment_for("mace").empty());
    FA_CHECK(default_environment_for("mattersim") == "mlip_env" ||
             !default_environment_for("mattersim").empty());
}

// ---------------------------------------------------------------------------
// Live models, when the environment points at one
// ---------------------------------------------------------------------------
FA_TEST(an_exported_archive_runs_in_process) {
    const std::string archive = environment("FASTATOMS_TEST_MLIP_MODEL");
    if (archive.empty() || !torch_available()) {
        std::printf("    (skipped: set FASTATOMS_TEST_MLIP_MODEL and build with LibTorch)\n");
        return;
    }

    Value parameters(Value::Map{});
    parameters.insert("model", Value(archive));
    parameters.insert("backend", Value(std::string("torchscript")));
    mlip::MACECalculator calculator(parameters);

    const ModelInfo& info = calculator.model();
    FA_CHECK(info.backend == "torchscript");
    FA_CHECK(info.cutoff > 0.0);
    std::printf("    %s\n", info.to_string().c_str());

    Atoms atoms = water_molecule();
    const double energy = calculator.get_potential_energy(atoms);
    FA_CHECK(std::isfinite(energy));

    const std::vector<double> forces = calculator.get_forces(atoms);
    FA_CHECK(forces.size() == 3 * atoms.size());

    // The strongest check available without a reference number: the forces the
    // model reports must be the gradient of the energy it reports.
    const double step = 1e-4;
    for (std::size_t atom = 0; atom < atoms.size(); ++atom) {
        for (int axis = 0; axis < 3; ++axis) {
            Atoms plus = atoms;
            Atoms minus = atoms;
            const Vec3 shift = axis_shift(axis, step);
            plus.set_position(atom, plus.position(atom) + shift);
            minus.set_position(atom, minus.position(atom) - shift);

            const double e_plus = calculator.get_potential_energy(plus);
            const double e_minus = calculator.get_potential_energy(minus);
            const double numerical = -(e_plus - e_minus) / (2.0 * step);
            FA_CHECK_CLOSE(forces[3 * atom + static_cast<std::size_t>(axis)], numerical, 2e-4);
        }
    }

    // Repeated requests for the same structure must not re-run the model. The
    // loop above left the cache primed on a displaced copy, so prime it again.
    calculator.get_potential_energy(atoms);
    const std::size_t before = calculator.call_count();
    calculator.get_potential_energy(atoms);
    calculator.get_forces(atoms);
    calculator.get_stress(atoms);
    FA_CHECK(calculator.call_count() == before);
}

FA_TEST(a_worker_evaluates_the_same_structure) {
    const std::string interpreter = environment("FASTATOMS_TEST_MLIP_PYTHON");
    if (interpreter.empty()) {
        std::printf("    (skipped: set FASTATOMS_TEST_MLIP_PYTHON to an interpreter)\n");
        return;
    }
    const std::string family = environment("FASTATOMS_TEST_MLIP_FAMILY");
    const std::string model = environment("FASTATOMS_TEST_MLIP_WORKER_MODEL");

    Value parameters(Value::Map{});
    parameters.insert("python", Value(interpreter));
    parameters.insert("backend", Value(std::string("worker")));
    if (!model.empty()) {
        parameters.insert("model", Value(model));
    }

    auto calculator = make_calculator(family.empty() ? "mace" : family, parameters);
    Atoms atoms = water_molecule();

    const double energy = calculator->get_potential_energy(atoms);
    FA_CHECK(std::isfinite(energy));
    const std::vector<double> forces = calculator->get_forces(atoms);
    FA_CHECK(forces.size() == 3 * atoms.size());
    std::printf("    %s: E = %.6f eV\n", calculator->to_string().c_str(), energy);

    // The point of a persistent worker: many evaluations, one process.
    for (int step = 0; step < 5; ++step) {
        atoms.set_position(1, atoms.position(1) + Vec3{0.005, 0.0, 0.0});
        FA_CHECK(std::isfinite(calculator->get_potential_energy(atoms)));
    }
    FA_CHECK(calculator->call_count() == 6);
}

FA_TEST(both_backends_agree) {
    // The whole design rests on this: a model exported to TorchScript and run
    // in C++ on a graph FastAtoms built must produce the same numbers as the
    // package's own ASE calculator running in Python on a graph it built
    // itself. If they disagree, something in the conversion is wrong.
    const std::string archive = environment("FASTATOMS_TEST_MLIP_MODEL");
    const std::string interpreter = environment("FASTATOMS_TEST_MLIP_PYTHON");
    const std::string checkpoint = environment("FASTATOMS_TEST_MLIP_WORKER_MODEL");
    if (archive.empty() || interpreter.empty() || checkpoint.empty() || !torch_available()) {
        std::printf("    (skipped: needs both an archive and an interpreter)\n");
        return;
    }

    Value in_process(Value::Map{});
    in_process.insert("model", Value(archive));
    in_process.insert("backend", Value(std::string("torchscript")));
    mlip::MACECalculator compiled(in_process);

    Value out_of_process(Value::Map{});
    out_of_process.insert("model", Value(checkpoint));
    out_of_process.insert("backend", Value(std::string("worker")));
    out_of_process.insert("python", Value(interpreter));
    mlip::MACECalculator worker(out_of_process);

    // Something asymmetric, so that agreement is not a coincidence of symmetry.
    Atoms atoms = water_molecule();
    atoms.set_position(1, atoms.position(1) + Vec3{0.07, -0.03, 0.02});
    atoms.set_position(2, atoms.position(2) + Vec3{-0.02, 0.05, 0.01});

    const double e_compiled = compiled.get_potential_energy(atoms);
    const double e_worker = worker.get_potential_energy(atoms);
    const std::vector<double> f_compiled = compiled.get_forces(atoms);
    const std::vector<double> f_worker = worker.get_forces(atoms);

    double worst_force = 0.0;
    for (std::size_t k = 0; k < f_compiled.size(); ++k) {
        worst_force = std::max(worst_force, std::abs(f_compiled[k] - f_worker[k]));
    }
    std::printf("    energy  %.9f (in process) vs %.9f (worker), difference %.2e eV\n",
                e_compiled, e_worker, std::abs(e_compiled - e_worker));
    std::printf("    largest force difference %.2e eV/A\n", worst_force);

    FA_CHECK(std::abs(e_compiled - e_worker) < 1e-7);
    FA_CHECK(worst_force < 1e-7);
}

FA_TEST_MAIN()
