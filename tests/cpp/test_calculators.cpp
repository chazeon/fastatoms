// SPDX-License-Identifier: MIT
//
// The calculator base class, the analytic reference calculator, the registry
// and the process runner. The DFT interfaces are exercised in
// test_dft_calculators.cpp.

#include <cmath>
#include <filesystem>
#include <random>
#include <string>

#include "fastatoms/calculators/lennard_jones.hpp"
#include "fastatoms/calculators/registry.hpp"
#include "fastatoms/core/atoms.hpp"
#include "fastatoms/util/process.hpp"
#include "fastatoms/util/text.hpp"
#include "testing.hpp"

using namespace fastatoms;

namespace {

/// A scratch directory under the build tree, removed when it goes out of scope.
class ScratchDirectory {
public:
    explicit ScratchDirectory(const std::string& stem) {
        static std::mt19937_64 rng(20260829);
        path_ = std::filesystem::current_path() /
                (stem + "-" + std::to_string(rng() % 1000000000ull));
        std::filesystem::create_directories(path_);
    }
    ~ScratchDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    ScratchDirectory(const ScratchDirectory&) = delete;
    ScratchDirectory& operator=(const ScratchDirectory&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

Value parameters(std::initializer_list<std::pair<const char*, Value>> entries) {
    Value map{Value::Map{}};
    for (const auto& [key, value] : entries) {
        map.insert(key, value);
    }
    return map;
}

/// A jittered argon lattice: dense enough to have real forces and a real
/// stress, small enough for finite differences to stay cheap.
Atoms argon_cluster() {
    std::mt19937 rng(4242);
    std::uniform_real_distribution<double> jitter(-0.25, 0.25);

    std::vector<int> numbers;
    std::vector<double> positions;
    const double a = 5.26;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                numbers.push_back(18);
                positions.push_back(a * (i + 0.25) + jitter(rng));
                positions.push_back(a * (j + 0.25) + jitter(rng));
                positions.push_back(a * (k + 0.25) + jitter(rng));
            }
        }
    }
    return Atoms(std::move(numbers), std::move(positions), Cell::cubic(2 * a));
}

}  // namespace

// ---------------------------------------------------------------------------
// Text and process utilities
// ---------------------------------------------------------------------------

FA_TEST(number_scanner_recovers_glued_columns) {
    // Fortran fixed-width output runs large values together; the scanner has
    // to split them anyway.
    const std::vector<double> values = util::scan_numbers("  in kB  -1234.56789-2345.67891 3.5");
    FA_CHECK(values.size() == 3);
    FA_CHECK_CLOSE(values[0], -1234.56789, 1e-9);
    FA_CHECK_CLOSE(values[1], -2345.67891, 1e-9);
    FA_CHECK_CLOSE(values[2], 3.5, 1e-9);

    FA_CHECK(util::scan_numbers("no numbers at all").empty());
    FA_CHECK(util::scan_numbers("atom    1 type  1   force =  1.0e-3 -2.0E+2").size() == 4);
}

FA_TEST(process_runner_reports_exit_status) {
    ScratchDirectory scratch("process");

    util::ProcessOptions options;
    options.working_directory = scratch.path();

    const util::ProcessResult ok = util::run_command("echo hello from the shell", options);
    FA_CHECK(ok.ok());
    FA_CHECK(ok.exit_code == 0);
    FA_CHECK(ok.log_tail.find("hello from the shell") != std::string::npos);

    const util::ProcessResult failed = util::run_command("echo oops >&2; exit 3", options);
    FA_CHECK(!failed.ok());
    FA_CHECK(failed.exit_code == 3);
    // stderr is merged into the log, which is what makes error reports useful.
    FA_CHECK(failed.log_tail.find("oops") != std::string::npos);
}

FA_TEST(process_runner_uses_the_working_directory_and_environment) {
    ScratchDirectory scratch("process-env");

    util::ProcessOptions options;
    options.working_directory = scratch.path();
    options.environment = {{"FASTATOMS_TEST_VARIABLE", "spam"}};
    options.log_path = scratch.path() / "run.log";

    const util::ProcessResult result =
        util::run_command("pwd; echo \"value=$FASTATOMS_TEST_VARIABLE\"", options);
    FA_CHECK(result.ok());
    FA_CHECK(result.log_tail.find("value=spam") != std::string::npos);
    FA_CHECK(result.log_tail.find(scratch.path().filename().string()) != std::string::npos);
    // A named log survives the call.
    FA_CHECK(std::filesystem::is_regular_file(scratch.path() / "run.log"));
}

FA_TEST(process_runner_enforces_a_timeout) {
    ScratchDirectory scratch("process-timeout");

    util::ProcessOptions options;
    options.working_directory = scratch.path();
    options.timeout_seconds = 0.3;

    const util::ProcessResult result = util::run_command("sleep 30", options);
    FA_CHECK(result.timed_out);
    FA_CHECK(!result.ok());
    FA_CHECK(result.status_string() == "timed out");
}

// ---------------------------------------------------------------------------
// Lennard-Jones: the reference implementation of the interface
// ---------------------------------------------------------------------------

FA_TEST(lennard_jones_reproduces_the_analytic_dimer) {
    const double epsilon = 0.01;
    const double sigma = 3.0;

    const auto dimer_at = [&](double r) {
        Atoms dimer(std::vector<std::string>{"Ar", "Ar"}, {0.0, 0.0, 0.0, r, 0.0, 0.0});
        dimer.set_calculator(std::make_shared<LennardJonesCalculator>(parameters(
            {{"epsilon", Value(epsilon)}, {"sigma", Value(sigma)}, {"cutoff", Value(20.0)},
             {"shift", Value(false)}})));
        return dimer;
    };

    // The minimum sits at 2^(1/6) sigma = 3.367 A for these parameters.
    const double minimum = std::pow(2.0, 1.0 / 6.0) * sigma;

    for (const double r : {3.30, 4.00}) {
        Atoms dimer = dimer_at(r);
        const double ratio6 = std::pow(sigma / r, 6);
        const double expected = 4.0 * epsilon * (ratio6 * ratio6 - ratio6);
        FA_CHECK_CLOSE(dimer.get_potential_energy(), expected, 1e-12);

        const std::vector<double> forces = dimer.get_forces();
        // Inside the minimum the pair repels (atom 0 pushed to -x); outside it
        // attracts.
        FA_CHECK((r < minimum) ? (forces[0] < 0.0) : (forces[0] > 0.0));
        FA_CHECK_CLOSE(forces[0], -forces[3], 1e-12);
        FA_CHECK_CLOSE(forces[1], 0.0, 1e-12);
    }

    // The energy is stationary at the minimum.
    Atoms at_minimum = dimer_at(minimum);
    FA_CHECK_CLOSE(at_minimum.get_forces()[0], 0.0, 1e-9);
    FA_CHECK_CLOSE(at_minimum.get_potential_energy(), -epsilon, 1e-12);
}

FA_TEST(lennard_jones_forces_match_finite_differences) {
    Atoms atoms = argon_cluster();
    auto calculator = std::make_shared<LennardJonesCalculator>();
    atoms.set_calculator(calculator);

    const std::vector<double> analytic = atoms.get_forces();

    const double step = 1e-5;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        for (std::size_t a = 0; a < 3; ++a) {
            Atoms probe = atoms;
            probe.set_calculator(std::make_shared<LennardJonesCalculator>());

            std::vector<double> positions = probe.positions();
            positions[3 * i + a] += step;
            probe.set_positions(positions);
            const double plus = probe.get_potential_energy();

            positions[3 * i + a] -= 2.0 * step;
            probe.set_positions(positions);
            const double minus = probe.get_potential_energy();

            // F = -dE/dx
            FA_CHECK_CLOSE(analytic[3 * i + a], -(plus - minus) / (2.0 * step), 1e-5);
        }
    }
}

FA_TEST(lennard_jones_stress_matches_the_strain_derivative) {
    // This pins down the sign and the Voigt ordering of the stress at the same
    // time: sigma_ab = (1/V) dE/d(strain_ab).
    Atoms atoms = argon_cluster();
    auto calculator = std::make_shared<LennardJonesCalculator>();
    atoms.set_calculator(calculator);

    const std::array<double, 6> analytic = atoms.get_stress();
    const double volume = atoms.volume();
    const double step = 1e-6;

    // Voigt index -> the (row, column) pair of the strain tensor it drives.
    const std::array<std::pair<int, int>, 6> components{
        {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}}};

    for (std::size_t v = 0; v < 6; ++v) {
        // Plain copies rather than structured bindings: capturing the latter in
        // a lambda is only standard from C++20 on.
        const int a = components[v].first;
        const int b = components[v].second;

        const auto strained_energy = [&, a, b](double amount) {
            Mat3 strain = Mat3::identity();
            if (a == b) {
                strain(a, b) += amount;
            } else {
                // A symmetric shear: both off-diagonal entries get half, so
                // the derivative is with respect to the engineering-free
                // strain component used by the Voigt convention.
                strain(a, b) += 0.5 * amount;
                strain(b, a) += 0.5 * amount;
            }
            Atoms probe = atoms;
            probe.set_calculator(std::make_shared<LennardJonesCalculator>());
            probe.set_cell(probe.cell().matrix() * strain, true);
            return probe.get_potential_energy();
        };

        const double derivative = (strained_energy(step) - strained_energy(-step)) / (2.0 * step);
        FA_CHECK_CLOSE(analytic[v], derivative / volume, 2e-4);
    }
}

FA_TEST(results_are_cached_until_the_structure_changes) {
    Atoms atoms = argon_cluster();
    auto calculator = std::make_shared<LennardJonesCalculator>();
    atoms.set_calculator(calculator);

    FA_CHECK(calculator->call_count() == 0);
    (void)atoms.get_potential_energy();
    FA_CHECK(calculator->call_count() == 1);

    // Energy, forces and stress all came from that one evaluation.
    (void)atoms.get_forces();
    (void)atoms.get_stress();
    (void)atoms.get_potential_energy();
    FA_CHECK(calculator->call_count() == 1);

    // Moving an atom invalidates them.
    std::vector<double> positions = atoms.positions();
    positions[0] += 0.01;
    atoms.set_positions(positions);
    (void)atoms.get_potential_energy();
    FA_CHECK(calculator->call_count() == 2);

    // So does changing the cell, and so does changing a parameter.
    atoms.set_cell(atoms.cell().matrix() * 1.01, true);
    (void)atoms.get_potential_energy();
    FA_CHECK(calculator->call_count() == 3);

    calculator->set_parameter("epsilon", Value(0.02));
    (void)atoms.get_potential_energy();
    FA_CHECK(calculator->call_count() == 4);
}

FA_TEST(forces_are_stored_on_the_structure) {
    Atoms atoms = argon_cluster();
    atoms.set_calculator(std::make_shared<LennardJonesCalculator>());

    FA_CHECK(!atoms.has_array("forces"));
    const std::vector<double> forces = atoms.get_forces();
    FA_CHECK(atoms.has_array("forces"));
    FA_CHECK(atoms.array("forces").ncols == 3);
    FA_CHECK(atoms.array("forces").data == forces);

    // Per-atom results follow the structure through slicing.
    const Atoms half = atoms.subset({0, 1});
    FA_CHECK(half.array("forces").data.size() == 6);
    FA_CHECK_CLOSE(half.array("forces").data[0], forces[0], 1e-15);
}

FA_TEST(a_structure_without_a_calculator_says_so) {
    Atoms atoms = argon_cluster();
    FA_CHECK(!atoms.has_calculator());
    FA_CHECK_THROWS(atoms.get_potential_energy(), CalculatorError);
    FA_CHECK_THROWS(atoms.get_forces(), CalculatorError);
    FA_CHECK_THROWS(atoms.get_stress(), CalculatorError);
}

FA_TEST(unimplemented_properties_are_rejected) {
    Atoms atoms = argon_cluster();
    auto calculator = std::make_shared<LennardJonesCalculator>();
    FA_CHECK(!calculator->implements(Property::Charges));
    FA_CHECK_THROWS(calculator->calculate(atoms, {Property::Charges}), CalculatorError);
    FA_CHECK_THROWS(calculator->get_magnetic_moment(atoms), CalculatorError);
}

FA_TEST(stress_helpers_round_trip) {
    const std::array<double, 6> voigt{1.0, 2.0, 3.0, 0.4, 0.5, 0.6};
    const Mat3 matrix = voigt_to_matrix(voigt);
    FA_CHECK_CLOSE(matrix(0, 0), 1.0, 1e-15);
    FA_CHECK_CLOSE(matrix(1, 2), 0.4, 1e-15);
    FA_CHECK_CLOSE(matrix(2, 1), 0.4, 1e-15);
    FA_CHECK_CLOSE(matrix(0, 2), 0.5, 1e-15);
    FA_CHECK_CLOSE(matrix(0, 1), 0.6, 1e-15);

    const std::array<double, 6> back = matrix_to_voigt(matrix);
    for (std::size_t k = 0; k < 6; ++k) {
        FA_CHECK_CLOSE(back[k], voigt[k], 1e-15);
    }
    FA_CHECK_CLOSE(pressure_from_stress(voigt), -2.0, 1e-15);
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

FA_TEST(registry_builds_the_built_in_calculators) {
    FA_CHECK(make_calculator("vasp")->name() == "vasp");
    FA_CHECK(make_calculator("siesta")->name() == "siesta");
    FA_CHECK(make_calculator("espresso")->name() == "espresso");
    FA_CHECK(make_calculator("lennard_jones")->name() == "lennard_jones");

    // Names are matched ignoring case, hyphens and underscores.
    FA_CHECK(make_calculator("Quantum-Espresso")->name() == "espresso");
    FA_CHECK(make_calculator("QE")->name() == "espresso");
    FA_CHECK(make_calculator("pw")->name() == "espresso");
    FA_CHECK(make_calculator("LJ")->name() == "lennard_jones");
    FA_CHECK(make_calculator("VASP")->name() == "vasp");

    FA_CHECK_THROWS(make_calculator("gaussian"), CalculatorError);

    const std::vector<std::string> names = registered_calculators();
    FA_CHECK(names.size() == 4);
}

FA_TEST(registry_passes_parameters_through) {
    auto calculator = make_calculator("lj", parameters({{"epsilon", Value(0.5)}}));
    FA_CHECK_CLOSE(calculator->parameters().at("epsilon").as_double(), 0.5, 1e-15);

    CalculatorParameters spec;
    spec.name = "lennard_jones";
    spec.parameters = parameters({{"sigma", Value(4.0)}});
    auto from_spec = make_calculator(spec);
    FA_CHECK_CLOSE(from_spec->parameters().at("sigma").as_double(), 4.0, 1e-15);

    CalculatorParameters nameless;
    FA_CHECK_THROWS(make_calculator(nameless), CalculatorError);
}

FA_TEST(custom_calculators_can_be_registered) {
    register_calculator("test_dummy", [](const Value& p) {
        return std::static_pointer_cast<Calculator>(
            std::make_shared<LennardJonesCalculator>(p));
    });
    FA_CHECK(make_calculator("test_dummy")->name() == "lennard_jones");
    FA_CHECK(make_calculator("Test-Dummy") != nullptr);
}

FA_TEST_MAIN()
