// SPDX-License-Identifier: MIT
//
// Thermostats, barostats and the loop machinery: does NVT hold a temperature,
// does NPT hold a pressure, are the extended-system energies conserved, and
// does the trajectory land on disk in a readable form.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <vector>

#include "fastatoms/calculators/lennard_jones.hpp"
#include "fastatoms/core/units.hpp"
#include "fastatoms/md/registry.hpp"
#include "fastatoms/md/velocity.hpp"
#include "fastatoms/util/text.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::md;

namespace {

Value settings(std::initializer_list<std::pair<const char*, Value>> entries) {
    Value map{Value::Map{}};
    for (const auto& [key, value] : entries) {
        map.insert(key, value);
    }
    return map;
}

class ScratchDirectory {
public:
    explicit ScratchDirectory(const std::string& stem) {
        static std::mt19937_64 rng(20260831);
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

Atoms fcc_argon(double a = 5.30, int repeat = 2) {
    const double basis[4][3] = {
        {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
    std::vector<double> scaled;
    for (const auto& site : basis) {
        scaled.insert(scaled.end(), {site[0], site[1], site[2]});
    }
    Atoms cell(std::vector<int>(4, 18), std::vector<double>(12, 0.0), Cell::cubic(a));
    cell.set_scaled_positions(scaled);
    Atoms atoms = repeat > 1 ? cell.repeat(repeat, repeat, repeat) : cell;
    atoms.set_calculator(std::make_shared<LennardJonesCalculator>(settings(
        {{"epsilon", Value(0.0103)}, {"sigma", Value(3.40)}, {"cutoff", Value(8.5)}})));
    return atoms;
}

/// Mean of a quantity over the second half of a run, once the thermostat has
/// had time to work.
double late_average(const std::vector<DynamicsStep>& history,
                    double DynamicsStep::*field) {
    const std::size_t start = history.size() / 2;
    double sum = 0.0;
    for (std::size_t k = start; k < history.size(); ++k) {
        sum += history[k].*field;
    }
    return sum / static_cast<double>(history.size() - start);
}

}  // namespace

// ---------------------------------------------------------------------------
// Thermostats
// ---------------------------------------------------------------------------

FA_TEST(every_thermostat_reaches_the_target_temperature) {
    for (const char* name : {"berendsen", "langevin", "nose", "nose_hoover"}) {
        Atoms atoms = fcc_argon(5.30, 2);
        // Start cold: the thermostat has to do real work to get there.
        set_maxwell_boltzmann_velocities(atoms, 10.0, 4242);

        Dynamics dynamics(settings({{"timestep", Value(5.0)},
                                    {"temperature", Value(80.0)},
                                    {"thermostat", Value(name)},
                                    {"tau", Value(100.0)},
                                    {"seed", Value(7)},
                                    {"initialize_velocities", Value(false)},
                                    {"interval", Value(5)}}));
        const DynamicsResult result = dynamics.run(atoms, 1500);

        FA_CHECK(dynamics.ensemble() == "nvt");
        FA_CHECK(result.reason == DynamicsStopReason::Completed);
        const double mean = late_average(result.history, &DynamicsStep::temperature);
        FA_CHECK(std::abs(mean - 80.0) < 20.0);
        // One force evaluation per step, plus one for the initial state: a
        // thermostat only rescales velocities, so it costs nothing.
        FA_CHECK(result.evaluations == 1501);
    }
}

FA_TEST(nose_hoover_conserves_its_extended_energy) {
    Atoms atoms = fcc_argon(5.30, 2);
    Dynamics dynamics(settings({{"timestep", Value(4.0)},
                                {"temperature", Value(60.0)},
                                {"thermostat", Value("nose_hoover")},
                                {"tau", Value(100.0)},
                                {"seed", Value(11)},
                                {"interval", Value(10)}}));
    const DynamicsResult result = dynamics.run(atoms, 1000);

    // The physical energy is free to move; the extended-system Hamiltonian is
    // not. That is the whole point of the Nose-Hoover formulation.
    const double physical_range = [&] {
        double lowest = result.history.front().total_energy;
        double highest = lowest;
        for (const DynamicsStep& frame : result.history) {
            lowest = std::min(lowest, frame.total_energy);
            highest = std::max(highest, frame.total_energy);
        }
        return highest - lowest;
    }();
    FA_CHECK(physical_range > 1e-4);
    FA_CHECK(std::abs(result.energy_drift_per_atom(atoms.size())) < 1e-4);
}

FA_TEST(nose_conserves_its_own_hamiltonian) {
    Atoms atoms = fcc_argon(5.30, 2);
    Dynamics dynamics(settings({{"timestep", Value(4.0)},
                                {"temperature", Value(60.0)},
                                {"thermostat", Value("nose")},
                                {"tau", Value(100.0)},
                                {"seed", Value(13)},
                                {"interval", Value(10)}}));
    const DynamicsResult result = dynamics.run(atoms, 1000);
    FA_CHECK(std::abs(result.energy_drift_per_atom(atoms.size())) < 1e-4);

    // Nose counts one more degree of freedom than Hoover's real-time version,
    // so the two are genuinely different dynamics, not aliases.
    auto nose = make_thermostat("nose");
    auto hoover = make_thermostat("nose_hoover");
    DynamicsContext context;
    context.degrees_of_freedom = 30;
    context.temperature = 100.0;
    context.timestep = 1.0;
    FA_CHECK(nose->name() == "nose");
    FA_CHECK(hoover->name() == "nose_hoover");
    FA_CHECK_CLOSE(dynamic_cast<NoseThermostat&>(*nose).mass(context) /
                       dynamic_cast<NoseHooverThermostat&>(*hoover).masses(context)[0],
                   31.0 / 30.0, 1e-12);
}

FA_TEST(nose_hoover_chains_are_supported) {
    Atoms atoms = fcc_argon(5.30, 2);
    Dynamics dynamics(settings({{"timestep", Value(4.0)},
                                {"temperature", Value(60.0)},
                                {"thermostat", Value(Value::Map{{"name", Value("nose_hoover")},
                                                                {"tau", Value(80.0)},
                                                                {"chain_length", Value(4)}})},
                                {"seed", Value(17)},
                                {"interval", Value(10)}}));
    const DynamicsResult result = dynamics.run(atoms, 800);

    auto* chain = dynamic_cast<NoseHooverThermostat*>(dynamics.thermostat().get());
    FA_CHECK(chain != nullptr);
    FA_CHECK(chain->chain_length() == 4);
    FA_CHECK(chain->friction().size() == 4);
    // The whole chain, not just its first link, must stay bounded.
    FA_CHECK(std::abs(result.energy_drift_per_atom(atoms.size())) < 1e-4);
    FA_CHECK(std::abs(late_average(result.history, &DynamicsStep::temperature) - 60.0) < 25.0);
}

FA_TEST(a_thermostat_keeps_leapfrog_in_step) {
    // Leapfrog holds its own copy of the velocities, so a thermostat that
    // rescales the structure's must be echoed into it or the rescaling is
    // silently undone on the next step.
    Atoms atoms = fcc_argon(5.30, 2);
    Dynamics dynamics(settings({{"timestep", Value(5.0)},
                                {"temperature", Value(100.0)},
                                {"thermostat", Value("berendsen")},
                                {"integrator", Value("leapfrog")},
                                {"tau", Value(50.0)},
                                {"seed", Value(3)},
                                {"initialize_velocities", Value(true)},
                                {"interval", Value(10)}}));
    // Start well below the target and check it climbs.
    set_maxwell_boltzmann_velocities(atoms, 100.0, 3);
    const DynamicsResult result = dynamics.run(atoms, 800);
    FA_CHECK(std::abs(late_average(result.history, &DynamicsStep::temperature) - 100.0) < 20.0);
}

FA_TEST(langevin_samples_the_canonical_distribution) {
    // In the canonical ensemble the temperature fluctuates with a variance
    // fixed by the number of degrees of freedom: sigma(T)/T = sqrt(2/N_f).
    // A thermostat that samples correctly reproduces that width; one that only
    // holds the mean does not.
    Atoms atoms = fcc_argon(5.30, 2);
    const int dof = momentum_degrees_of_freedom(atoms, true);
    const double target = 80.0;
    const double expected = target * std::sqrt(2.0 / static_cast<double>(dof));

    const auto width_for = [&](const char* name) {
        Atoms probe = fcc_argon(5.30, 2);
        Dynamics dynamics(settings({{"timestep", Value(5.0)},
                                    {"temperature", Value(target)},
                                    {"thermostat", Value(name)},
                                    {"tau", Value(100.0)},
                                    {"seed", Value(4242)},
                                    {"interval", Value(5)}}));
        const DynamicsResult result = dynamics.run(probe, 3000);
        // Discard the first half: the thermostat needs time to equilibrate.
        const std::size_t start = result.history.size() / 2;
        double sum = 0.0;
        double sum_squares = 0.0;
        std::size_t count = 0;
        for (std::size_t k = start; k < result.history.size(); ++k) {
            const double value = result.history[k].temperature;
            sum += value;
            sum_squares += value * value;
            ++count;
        }
        const double mean = sum / static_cast<double>(count);
        return std::sqrt(sum_squares / static_cast<double>(count) - mean * mean);
    };

    const double langevin = width_for("langevin");
    const double berendsen = width_for("berendsen");

    // Within a third of the canonical width is as much as a run this short can
    // resolve, but it is far more than enough to tell the two apart.
    FA_CHECK(std::abs(langevin - expected) < 0.35 * expected);
    // Berendsen deliberately damps the fluctuations, which is the whole reason
    // it is an equilibration tool rather than a sampling one.
    FA_CHECK(berendsen < 0.75 * expected);
}

FA_TEST(langevin_accounts_for_the_heat_it_moves) {
    // A stochastic thermostat conserves nothing of its own, so the check is
    // that the energy it takes out of the system is the energy it books into
    // the bath: potential + kinetic + heat has to stay flat.
    Atoms atoms = fcc_argon(5.30, 2);
    Dynamics dynamics(settings({{"timestep", Value(4.0)},
                                {"temperature", Value(60.0)},
                                {"thermostat", Value("langevin")},
                                {"friction", Value(0.01)},
                                {"seed", Value(11)},
                                {"interval", Value(10)}}));
    const DynamicsResult result = dynamics.run(atoms, 1500);

    auto* langevin = dynamic_cast<LangevinThermostat*>(dynamics.thermostat().get());
    FA_CHECK(langevin != nullptr);
    FA_CHECK_CLOSE(langevin->friction(), 0.01, 1e-12);
    // Starting from rest at a 60 K target, the bath must have supplied energy.
    FA_CHECK(langevin->heat() < 0.0);

    // The physical energy wanders; the books balance.
    const double physical_range = [&] {
        double lowest = result.history.front().total_energy;
        double highest = lowest;
        for (const DynamicsStep& frame : result.history) {
            lowest = std::min(lowest, frame.total_energy);
            highest = std::max(highest, frame.total_energy);
        }
        return highest - lowest;
    }();
    FA_CHECK(physical_range > 1e-3);
    FA_CHECK(std::abs(result.energy_drift_per_atom(atoms.size())) < 1e-4);
}

FA_TEST(langevin_noise_is_reproducible) {
    const auto trajectory = [](std::int64_t seed) {
        Atoms atoms = fcc_argon(5.30, 1);
        Dynamics dynamics(settings({{"timestep", Value(4.0)},
                                    {"temperature", Value(100.0)},
                                    {"thermostat", Value("langevin")},
                                    {"friction", Value(0.02)},
                                    {"seed", Value(seed)}}));
        (void)dynamics.run(atoms, 50);
        return atoms.positions();
    };

    FA_CHECK(trajectory(1234) == trajectory(1234));
    FA_CHECK(trajectory(1234) != trajectory(4321));
}

FA_TEST(langevin_reaches_leapfrog_half_step_velocities) {
    // Langevin's update is affine, not a rescaling: leapfrog has to receive
    // the exact noise that was drawn, not just the damping factor. If it does
    // not, the thermostat is fighting stale velocities every step and the
    // temperature never settles.
    Atoms atoms = fcc_argon(5.30, 2);
    set_maxwell_boltzmann_velocities(atoms, 10.0, 3);

    Dynamics dynamics(settings({{"timestep", Value(5.0)},
                                {"temperature", Value(100.0)},
                                {"thermostat", Value("langevin")},
                                {"integrator", Value("leapfrog")},
                                {"friction", Value(0.01)},
                                {"seed", Value(5)},
                                {"initialize_velocities", Value(false)},
                                {"interval", Value(10)}}));
    const DynamicsResult result = dynamics.run(atoms, 1500);
    FA_CHECK(std::abs(late_average(result.history, &DynamicsStep::temperature) - 100.0) < 20.0);
}

FA_TEST(thermostat_names_and_errors) {
    FA_CHECK(make_thermostat("") == nullptr);
    FA_CHECK(make_thermostat("none") == nullptr);
    FA_CHECK(make_thermostat("Nose-Hoover")->name() == "nose_hoover");
    FA_CHECK(make_thermostat("NHC")->name() == "nose_hoover");
    FA_CHECK(make_thermostat("berendsen")->name() == "berendsen");
    FA_CHECK(make_thermostat("Langevin")->name() == "langevin");
    FA_CHECK(make_thermostat("langevin")->samples_canonical());
    FA_CHECK_THROWS(make_thermostat("andersen"), DynamicsError);
    FA_CHECK(thermostat_names().size() == 4);
    FA_CHECK(!make_thermostat("berendsen")->samples_canonical());
    FA_CHECK(make_thermostat("nose_hoover")->samples_canonical());
}

// ---------------------------------------------------------------------------
// Barostats
// ---------------------------------------------------------------------------

FA_TEST(every_barostat_reaches_the_target_pressure) {
    for (const char* name : {"berendsen", "parrinello_rahman"}) {
        // Start compressed, so there is a real pressure to relieve.
        Atoms atoms = fcc_argon(5.00, 2);
        Dynamics dynamics(settings({{"timestep", Value(5.0)},
                                    {"temperature", Value(30.0)},
                                    {"thermostat", Value("nose_hoover")},
                                    {"barostat", Value(name)},
                                    {"pressure", Value(0.0)},
                                    {"tau", Value(100.0)},
                                    {"seed", Value(7)},
                                    {"interval", Value(20)}}));
        const DynamicsResult result = dynamics.run(atoms, 2000);

        FA_CHECK(dynamics.ensemble() == "npt");
        FA_CHECK(result.reason == DynamicsStopReason::Completed);
        // The cell had to expand to relieve the compression.
        FA_CHECK(atoms.volume() > 4.0 * 5.00 * 5.00 * 5.00);
        const double mean = late_average(result.history, &DynamicsStep::pressure);
        FA_CHECK(std::abs(mean) < 5.0e-4);  // eV/A^3, about 0.08 GPa
        // Deforming the cell in front of the integrator costs one extra force
        // evaluation per step, and no more than that. The barostat's kicks and
        // the sampling both land on cached forces.
        FA_CHECK(result.evaluations == 2 * 2000 + 1);
    }
}

FA_TEST(a_barostat_holds_a_finite_target_pressure) {
    const auto relaxed_volume = [](double pressure_gpa) {
        Atoms atoms = fcc_argon(5.30, 2);
        Dynamics dynamics(settings({{"timestep", Value(5.0)},
                                    {"temperature", Value(30.0)},
                                    {"thermostat", Value("nose_hoover")},
                                    {"barostat", Value("berendsen")},
                                    {"pressure", Value(pressure_gpa)},
                                    {"compressibility", Value(1.0)},
                                    {"tau", Value(200.0)},
                                    {"seed", Value(19)},
                                    {"interval", Value(50)}}));
        const DynamicsResult result = dynamics.run(atoms, 2000);
        const double mean = late_average(result.history, &DynamicsStep::pressure);
        return std::make_pair(atoms.volume(), mean);
    };

    const auto [free_volume, free_pressure] = relaxed_volume(0.0);
    const auto [squeezed_volume, squeezed_pressure] = relaxed_volume(1.0);

    FA_CHECK(squeezed_volume < free_volume);
    FA_CHECK(std::abs(free_pressure) < 5e-4);
    // The internal pressure has to sit near the applied one.
    FA_CHECK(std::abs(squeezed_pressure - 1.0 * units::GPa) < 2.0 * units::GPa);
}

FA_TEST(an_isotropic_barostat_keeps_the_cell_cubic) {
    Atoms atoms = fcc_argon(5.00, 2);
    Dynamics dynamics(settings({{"timestep", Value(5.0)},
                                {"temperature", Value(30.0)},
                                {"thermostat", Value("nose_hoover")},
                                {"barostat", Value(Value::Map{
                                                 {"name", Value("parrinello_rahman")},
                                                 {"tau", Value(500.0)},
                                                 {"isotropic", Value(true)}})},
                                {"seed", Value(23)},
                                {"interval", Value(50)}}));
    (void)dynamics.run(atoms, 600);

    const Vec3 lengths = atoms.cell().lengths();
    FA_CHECK_CLOSE(lengths.x, lengths.y, 1e-9);
    FA_CHECK_CLOSE(lengths.y, lengths.z, 1e-9);
    FA_CHECK(atoms.cell().is_orthorhombic(1e-9));

    auto* barostat = dynamic_cast<ParrinelloRahmanBarostat*>(dynamics.barostat().get());
    FA_CHECK(barostat != nullptr);
    // The strain rate stays diagonal and isotropic.
    const Mat3& rate = barostat->strain_rate();
    FA_CHECK_CLOSE(rate(0, 1), 0.0, 1e-15);
    FA_CHECK_CLOSE(rate(0, 0), rate(2, 2), 1e-12);
}

FA_TEST(barostat_names_and_errors) {
    FA_CHECK(make_barostat("") == nullptr);
    FA_CHECK(make_barostat("Parrinello-Rahman")->name() == "parrinello_rahman");
    FA_CHECK(make_barostat("PR")->name() == "parrinello_rahman");
    FA_CHECK(make_barostat("berendsen")->name() == "berendsen");
    FA_CHECK_THROWS(make_barostat("andersen"), DynamicsError);
    FA_CHECK(barostat_names().size() == 2);
    FA_CHECK(!make_barostat("berendsen")->samples_isobaric());

    // A barostat needs something it can actually deform.
    Atoms molecule(std::vector<std::string>{"Ar", "Ar"}, {0.0, 0.0, 0.0, 4.0, 0.0, 0.0});
    molecule.set_calculator(std::make_shared<LennardJonesCalculator>());
    Dynamics dynamics(settings({{"temperature", Value(50.0)},
                                {"thermostat", Value("nose_hoover")},
                                {"barostat", Value("berendsen")}}));
    FA_CHECK_THROWS(dynamics.run(molecule, 1), DynamicsError);
}

// ---------------------------------------------------------------------------
// The loop, sampling and output
// ---------------------------------------------------------------------------

FA_TEST(sampling_happens_at_the_requested_interval) {
    Atoms atoms = fcc_argon(5.30, 1);
    Dynamics dynamics(settings({{"timestep", Value(2.0)}, {"interval", Value(10)}}));
    const DynamicsResult result = dynamics.run(atoms, 100);

    // Step zero, then every tenth.
    FA_CHECK(result.history.size() == 11);
    FA_CHECK(result.history.front().step == 0);
    FA_CHECK(result.history.back().step == 100);
    FA_CHECK_CLOSE(result.history.back().time, 200.0, 1e-9);
    FA_CHECK_CLOSE(result.time, 200.0, 1e-9);
    for (std::size_t k = 1; k < result.history.size(); ++k) {
        FA_CHECK(result.history[k].step - result.history[k - 1].step == 10);
    }
}

FA_TEST(observers_see_every_frame_and_can_stop_the_run) {
    Atoms atoms = fcc_argon(5.30, 1);
    Dynamics dynamics(settings({{"timestep", Value(2.0)}}));

    std::vector<int> seen;
    dynamics.attach(
        [&seen](const Atoms&, const DynamicsStep& frame) {
            seen.push_back(frame.step);
            return frame.step < 5;
        },
        1);

    const DynamicsResult result = dynamics.run(atoms, 100);
    FA_CHECK(result.reason == DynamicsStopReason::Interrupted);
    FA_CHECK(result.steps == 5);
    FA_CHECK(seen.size() == 6);
    FA_CHECK(seen.front() == 0 && seen.back() == 5);
}

FA_TEST(a_trajectory_lands_on_disk_in_extended_xyz) {
    ScratchDirectory scratch("md-trajectory");
    const std::filesystem::path path = scratch.path() / "run.xyz";

    Atoms atoms = fcc_argon(5.30, 1);
    Dynamics dynamics(settings({{"timestep", Value(2.0)},
                                {"temperature", Value(50.0)},
                                {"thermostat", Value("nose_hoover")},
                                {"seed", Value(5)},
                                {"trajectory", Value(path.string())},
                                {"trajectory_interval", Value(10)}}));
    const DynamicsResult result = dynamics.run(atoms, 50);
    FA_CHECK(result.steps == 50);

    const std::vector<std::string> lines = util::split_lines(util::read_text_file(path));
    // Six frames of 4 atoms: the header, the comment, and one line each.
    const std::size_t frame_lines = 2 + atoms.size();
    FA_CHECK(lines.size() == 6 * frame_lines);

    FA_CHECK(util::trim(lines[0]) == "4");
    const std::string& comment = lines[1];
    FA_CHECK(comment.find("Lattice=\"") != std::string::npos);
    FA_CHECK(comment.find("Properties=species:S:1:pos:R:3:velo:R:3:forces:R:3") !=
             std::string::npos);
    FA_CHECK(comment.find("pbc=\"T T T\"") != std::string::npos);
    FA_CHECK(comment.find("temperature=") != std::string::npos);
    FA_CHECK(comment.find("conserved_energy=") != std::string::npos);

    // symbol plus position, velocity and force triples.
    const std::vector<std::string> fields = util::split_whitespace(lines[2]);
    FA_CHECK(fields.size() == 10);
    FA_CHECK(fields[0] == "Ar");

    // The last frame's positions must be the ones the structure ended on.
    const std::vector<double> last = util::scan_numbers(lines[5 * frame_lines + 2]);
    FA_CHECK_CLOSE(last[0], atoms.position(0).x, 1e-6);
}

FA_TEST(a_log_file_records_the_scalars) {
    ScratchDirectory scratch("md-log");
    const std::filesystem::path path = scratch.path() / "run.log";

    Atoms atoms = fcc_argon(5.30, 1);
    Dynamics dynamics(settings({{"timestep", Value(2.0)},
                                {"log", Value(path.string())},
                                {"log_interval", Value(25)}}));
    (void)dynamics.run(atoms, 100);

    const std::vector<std::string> lines = util::split_lines(util::read_text_file(path));
    FA_CHECK(lines.size() == 6);  // header plus five records
    FA_CHECK(lines[0].find("E_cons[eV]") != std::string::npos);
    FA_CHECK(util::scan_numbers(lines[1]).size() == 9);
}

FA_TEST(velocities_are_only_drawn_when_they_are_wanted) {
    // No momenta and a target temperature: draw them.
    Atoms fresh = fcc_argon(5.30, 1);
    FA_CHECK(!fresh.has_momenta());
    Dynamics warm(settings({{"temperature", Value(120.0)},
                            {"thermostat", Value("nose_hoover")},
                            {"seed", Value(2)}}));
    (void)warm.run(fresh, 0);
    FA_CHECK(fresh.has_momenta());
    FA_CHECK_CLOSE(fresh.temperature(warm.context().degrees_of_freedom), 120.0, 1e-6);

    // Velocities already set, and none asked for: leave them alone. The drift
    // removal still runs, so compare within rounding rather than bit for bit.
    Atoms moving = fcc_argon(5.30, 1);
    set_maxwell_boltzmann_velocities(moving, 42.0, 8);
    const std::vector<double> before = moving.momenta();
    Dynamics keep(settings({{"timestep", Value(1.0)}}));
    (void)keep.run(moving, 0);
    const std::vector<double> after = moving.momenta();
    FA_CHECK(after.size() == before.size());
    for (std::size_t k = 0; k < before.size(); ++k) {
        FA_CHECK_CLOSE(after[k], before[k], 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Wiring from an input file
// ---------------------------------------------------------------------------

FA_TEST(a_run_can_be_built_from_the_input_file) {
    MolecularDynamicsParameters spec;
    spec.ensemble = "nvt";
    spec.timestep_fs = 2.5;
    spec.steps = 42;
    spec.temperature_K = 250.0;
    spec.thermostat = "nose_hoover";
    spec.friction = 0.02;  // 1/fs, so tau = 50 fs
    spec.seed = 99;

    auto dynamics = make_dynamics(spec);
    FA_CHECK_CLOSE(dynamics->timestep_fs(), 2.5, 1e-12);
    FA_CHECK(dynamics->configured_steps() == 42);
    FA_CHECK_CLOSE(dynamics->temperature(), 250.0, 1e-12);

    Atoms atoms = fcc_argon(5.30, 1);
    const DynamicsResult result = dynamics->run(atoms);
    FA_CHECK(result.steps == 42);
    FA_CHECK(dynamics->ensemble() == "nvt");
    FA_CHECK_CLOSE(dynamics->thermostat()->time_constant_fs(), 50.0, 1e-12);

    // Settings the typed struct does not know about come through the subtree.
    MolecularDynamicsParameters rich;
    rich.temperature_K = 300.0;
    rich.parameters = settings({{"integrator", Value("leapfrog")},
                                {"thermostat", Value("nose")},
                                {"tau", Value(30.0)}});
    auto configured = make_dynamics(rich);
    Atoms probe = fcc_argon(5.30, 1);
    (void)configured->run(probe, 1);
    FA_CHECK(configured->integrator()->name() == "leapfrog");
    FA_CHECK(configured->thermostat()->name() == "nose");
    FA_CHECK_CLOSE(configured->thermostat()->time_constant_fs(), 30.0, 1e-12);
}

FA_TEST(the_output_section_feeds_the_writers) {
    ScratchDirectory scratch("md-deck");
    SimulationParameters simulation;
    simulation.md.temperature_K = 100.0;
    simulation.md.timestep_fs = 2.0;
    simulation.md.steps = 20;
    simulation.md.thermostat = "berendsen";
    simulation.output.trajectory = (scratch.path() / "traj.xyz").string();
    simulation.output.log = (scratch.path() / "run.log").string();
    simulation.output.interval = 5;

    auto dynamics = make_dynamics(simulation);
    Atoms atoms = fcc_argon(5.30, 1);
    const DynamicsResult result = dynamics->run(atoms);

    FA_CHECK(result.steps == 20);
    FA_CHECK(std::filesystem::is_regular_file(scratch.path() / "traj.xyz"));
    FA_CHECK(std::filesystem::is_regular_file(scratch.path() / "run.log"));
    FA_CHECK(result.history.size() == 5);  // step 0 and every fifth
}

FA_TEST(the_pieces_are_resolved_before_the_first_run) {
    // The description has to be accurate as soon as the object exists, and an
    // explicit choice has to outrank the parameters.
    Dynamics dynamics(settings({{"timestep", Value(5.0)},
                                {"temperature", Value(60.0)},
                                {"thermostat", Value("nose_hoover")}}));
    FA_CHECK(dynamics.ensemble() == "nvt");
    FA_CHECK(dynamics.thermostat()->name() == "nose_hoover");
    FA_CHECK(dynamics.integrator()->name() == "velocity_verlet");
    FA_CHECK(dynamics.to_string().find("nvt") != std::string::npos);

    dynamics.set_thermostat(nullptr);
    FA_CHECK(dynamics.ensemble() == "nve");

    Atoms atoms = fcc_argon(5.30, 1);
    set_maxwell_boltzmann_velocities(atoms, 50.0, 1);
    (void)dynamics.run(atoms, 2);
    FA_CHECK(dynamics.ensemble() == "nve");
    FA_CHECK(dynamics.thermostat() == nullptr);

    // A bad name is caught up front, not on the first step.
    FA_CHECK_THROWS(Dynamics(settings({{"thermostat", Value("andersen")}})), DynamicsError);
}

FA_TEST(the_ensemble_shorthand_fills_in_the_pieces) {
    Atoms atoms = fcc_argon(5.30, 1);

    Dynamics nve(settings({{"ensemble", Value("nve")}}));
    (void)nve.run(atoms, 1);
    FA_CHECK(nve.ensemble() == "nve");
    FA_CHECK(nve.thermostat() == nullptr);

    Atoms warm = fcc_argon(5.30, 1);
    Dynamics nvt(settings({{"ensemble", Value("nvt")}, {"temperature", Value(50.0)}}));
    (void)nvt.run(warm, 1);
    FA_CHECK(nvt.ensemble() == "nvt");
    FA_CHECK(nvt.thermostat()->name() == "nose_hoover");

    Atoms squeezed = fcc_argon(5.30, 1);
    Dynamics npt(settings({{"ensemble", Value("npt")}, {"temperature", Value(50.0)}}));
    (void)npt.run(squeezed, 1);
    FA_CHECK(npt.ensemble() == "npt");
    FA_CHECK(npt.barostat()->name() == "parrinello_rahman");
}

FA_TEST_MAIN()
