# SPDX-License-Identifier: MIT
"""Tests for the molecular dynamics module."""

from __future__ import annotations

import numpy as np
import pytest

import fastatoms as fa

INTEGRATORS = ["euler", "velocity_verlet", "leapfrog"]
THERMOSTATS = ["berendsen", "langevin", "nose", "nose_hoover"]
BAROSTATS = ["berendsen", "parrinello_rahman"]


def fcc_argon(a: float = 5.30, repeat: int = 2) -> fa.Atoms:
    basis = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    atoms = fa.Atoms("Ar4", scaled_positions=basis, cell=a, pbc=True)
    if repeat > 1:
        atoms = atoms.repeat(repeat, repeat, repeat)
    atoms.calc = fa.LennardJones(epsilon=0.0103, sigma=3.40, cutoff=8.5)
    return atoms


def late_mean(history, field: str) -> float:
    values = [getattr(step, field) for step in history[len(history) // 2 :]]
    return float(np.mean(values))


# --------------------------------------------------------------------------
# Velocities
# --------------------------------------------------------------------------
def test_maxwell_boltzmann():
    atoms = fcc_argon()
    dof = fa.momentum_degrees_of_freedom(atoms)
    assert dof == 3 * len(atoms) - 3

    fa.maxwell_boltzmann(atoms, 300.0, seed=1234)
    assert atoms.get_temperature(dof) == pytest.approx(300.0)
    assert np.linalg.norm(atoms.get_total_momentum()) == pytest.approx(0.0, abs=1e-10)

    same = fcc_argon()
    fa.maxwell_boltzmann(same, 300.0, seed=1234)
    assert same.get_momenta() == pytest.approx(atoms.get_momenta())

    other = fcc_argon()
    fa.maxwell_boltzmann(other, 300.0, seed=4321)
    assert not np.allclose(other.get_momenta(), atoms.get_momenta())


def test_velocity_helpers():
    atoms = fcc_argon()
    fa.maxwell_boltzmann(atoms, 0.0, seed=1)
    assert atoms.get_kinetic_energy() == pytest.approx(0.0)

    with pytest.raises(fa.DynamicsError):
        fa.maxwell_boltzmann(atoms, -1.0)

    fa.maxwell_boltzmann(atoms, 100.0, seed=2, exact=False)
    dof = fa.momentum_degrees_of_freedom(atoms)
    fa.rescale_to_temperature(atoms, 250.0, dof)
    assert atoms.get_temperature(dof) == pytest.approx(250.0)


# --------------------------------------------------------------------------
# Integrators
# --------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", INTEGRATORS)
def test_free_particle_travels_in_a_straight_line(scheme):
    atoms = fa.Atoms("H", [[0.0, 0.0, 0.0]], cell=50.0, pbc=False)
    # A cutoff far smaller than the box: nothing to interact with.
    atoms.calc = fa.LennardJones(cutoff=1.0)
    atoms.set_velocities([[0.1, -0.2, 0.05]])

    fa.run_md(atoms, 100, timestep=1.0, integrator=scheme, fix_center_of_mass=False)

    elapsed = 100.0 * fa.units.fs
    assert atoms.positions[0] == pytest.approx([0.1, -0.2, 0.05] * np.array(elapsed), rel=1e-9)


def test_leapfrog_matches_velocity_verlet():
    def trajectory(scheme):
        atoms = fcc_argon()
        fa.maxwell_boltzmann(atoms, 60.0, seed=77)
        fa.run_md(atoms, 60, timestep=4.0, integrator=scheme)
        return atoms.get_positions()

    # The two schemes are algebraically identical; only the instant at which
    # the velocities are defined differs.
    assert trajectory("leapfrog") == pytest.approx(trajectory("velocity_verlet"), abs=1e-10)


def test_euler_drifts_and_verlet_does_not():
    def drift(scheme):
        atoms = fcc_argon()
        fa.maxwell_boltzmann(atoms, 40.0, seed=2024)
        result = fa.run_md(atoms, 400, timestep=2.0, integrator=scheme, interval=10)
        assert result.completed
        return abs(result.energy_drift_per_atom(len(atoms)))

    assert drift("velocity_verlet") < 1e-5
    assert drift("leapfrog") < 1e-5
    assert drift("euler") > 50 * drift("velocity_verlet")


def test_nve_conserves_energy_and_momentum():
    atoms = fcc_argon()
    fa.maxwell_boltzmann(atoms, 50.0, seed=31337)
    result = fa.run_md(atoms, 300, timestep=5.0, interval=10)

    assert result.completed
    assert abs(result.energy_drift_per_atom(len(atoms))) < 1e-5
    assert np.linalg.norm(atoms.get_total_momentum()) == pytest.approx(0.0, abs=1e-9)
    # An NVE run has nothing but the physical energy to conserve.
    first = result.history[0]
    assert first.conserved_energy == pytest.approx(first.total_energy)
    # One force call per step, plus the initial state.
    assert result.evaluations == 301


def test_integrator_objects_and_errors():
    assert fa.make_integrator("").name == "velocity_verlet"
    assert fa.make_integrator("VV").name == "velocity_verlet"
    assert fa.make_integrator("Leap-Frog").name == "leapfrog"
    assert not fa.make_integrator("euler").is_symplectic
    assert fa.make_integrator("leapfrog").is_symplectic
    with pytest.raises(fa.DynamicsError):
        fa.make_integrator("runge_kutta")
    assert set(fa.integrator_names()) == {"euler", "leapfrog", "velocity_verlet"}

    # The pieces can also be assigned as objects.
    md = fa.Dynamics(timestep=2.0)
    md.integrator = fa.Leapfrog()
    assert md.integrator.name == "leapfrog"
    md.integrator = "euler"
    assert md.integrator.name == "euler"


def test_a_reckless_timestep_destroys_energy_conservation():
    # With a truncated pair potential a blown-up run does not produce NaNs -
    # the atoms simply fly apart at large but finite speed - so the thing to
    # watch is the drift of the conserved quantity, which is enormous here.
    def drift(timestep):
        atoms = fcc_argon()
        fa.maxwell_boltzmann(atoms, 40.0, seed=5)
        result = fa.run_md(atoms, 200, timestep=timestep, interval=10)
        return abs(result.energy_drift_per_atom(len(atoms)))

    assert drift(2.0) < 1e-5
    assert drift(200.0) > 1e-3


# --------------------------------------------------------------------------
# Thermostats
# --------------------------------------------------------------------------
@pytest.mark.parametrize("thermostat", THERMOSTATS)
def test_thermostats_reach_the_target(thermostat):
    atoms = fcc_argon()
    fa.maxwell_boltzmann(atoms, 10.0, seed=4242)  # start cold

    result = fa.run_md(
        atoms,
        1500,
        timestep=5.0,
        temperature=80.0,
        thermostat=thermostat,
        tau=100.0,
        seed=7,
        initialize_velocities=False,
        interval=5,
    )
    assert result.completed
    assert abs(late_mean(result.history, "temperature") - 80.0) < 20.0
    # A thermostat only rescales velocities, so it costs nothing.
    assert result.evaluations == 1501


@pytest.mark.parametrize("thermostat", ["langevin", "nose", "nose_hoover"])
def test_extended_system_energy_is_conserved(thermostat):
    atoms = fcc_argon()
    result = fa.run_md(
        atoms,
        1000,
        timestep=4.0,
        temperature=60.0,
        thermostat=thermostat,
        tau=100.0,
        seed=11,
        interval=10,
    )
    # The physical energy moves; the extended-system quantity does not - for
    # Langevin that is the heat booked into the bath.
    physical = [s.total_energy for s in result.history]
    assert max(physical) - min(physical) > 1e-4
    assert abs(result.energy_drift_per_atom(len(atoms))) < 1e-4


def test_nose_hoover_chains():
    atoms = fcc_argon()
    md = fa.Dynamics(timestep=4.0, temperature=60.0, seed=17, interval=10)
    md.thermostat = fa.NoseHoover(tau=80.0, chain_length=4)
    result = md.run(atoms, 800)

    assert md.thermostat.chain_length == 4
    assert md.thermostat.friction.shape == (4,)
    assert abs(result.energy_drift_per_atom(len(atoms))) < 1e-4
    assert abs(late_mean(result.history, "temperature") - 60.0) < 25.0


def test_thermostat_names():
    assert set(fa.thermostat_names()) == {"berendsen", "langevin", "nose", "nose_hoover"}
    assert fa.make_thermostat("nose").name == "nose"
    assert fa.make_thermostat("Nose-Hoover").name == "nose_hoover"
    assert fa.make_thermostat("NHC").name == "nose_hoover"
    assert fa.make_thermostat("Langevin").name == "langevin"
    assert fa.make_thermostat("") is None
    assert not fa.make_thermostat("berendsen").samples_canonical
    assert fa.make_thermostat("nose_hoover").samples_canonical
    assert fa.make_thermostat("langevin").samples_canonical
    with pytest.raises(fa.DynamicsError):
        fa.make_thermostat("andersen")


def test_langevin_samples_the_canonical_distribution():
    # sigma(T)/T = sqrt(2/N_f) in the canonical ensemble. A thermostat that
    # samples correctly reproduces that width; one that only holds the mean
    # does not.
    atoms = fcc_argon()
    dof = fa.momentum_degrees_of_freedom(atoms)
    target = 80.0
    expected = target * np.sqrt(2.0 / dof)

    def width(name):
        result = fa.run_md(
            fcc_argon(), 3000, timestep=5.0, temperature=target, thermostat=name,
            tau=100.0, seed=4242, interval=5,
        )
        late = [s.temperature for s in result.history[len(result.history) // 2 :]]
        return float(np.std(late))

    assert abs(width("langevin") - expected) < 0.35 * expected
    assert width("berendsen") < 0.75 * expected


def test_langevin_accounts_for_the_heat_it_moves():
    atoms = fcc_argon()
    md = fa.Dynamics(
        timestep=4.0, temperature=60.0, friction=0.01, seed=11, interval=10
    )
    md.thermostat = fa.Langevin(friction=0.01, seed=11)
    result = md.run(atoms, 1500)

    assert md.thermostat.friction == pytest.approx(0.01)
    # Starting from rest at a 60 K target, the bath must have supplied energy.
    assert md.thermostat.heat < 0.0
    # The physical energy wanders; potential + kinetic + heat does not.
    physical = [s.total_energy for s in result.history]
    assert max(physical) - min(physical) > 1e-3
    assert abs(result.energy_drift_per_atom(len(atoms))) < 1e-4


def test_langevin_noise_is_reproducible():
    def trajectory(seed):
        atoms = fcc_argon(repeat=1)
        fa.run_md(
            atoms, 50, timestep=4.0, temperature=100.0, thermostat="langevin",
            friction=0.02, seed=seed,
        )
        return atoms.get_positions()

    assert trajectory(1234) == pytest.approx(trajectory(1234))
    assert not np.allclose(trajectory(1234), trajectory(4321))


def test_langevin_reaches_leapfrog_half_step_velocities():
    # Langevin's update is affine, not a rescaling, so leapfrog has to receive
    # the exact noise that was drawn rather than just the damping factor.
    atoms = fcc_argon()
    fa.maxwell_boltzmann(atoms, 10.0, seed=3)
    result = fa.run_md(
        atoms, 1500, timestep=5.0, temperature=100.0, thermostat="langevin",
        integrator="leapfrog", friction=0.01, seed=5,
        initialize_velocities=False, interval=10,
    )
    assert abs(late_mean(result.history, "temperature") - 100.0) < 20.0


def test_a_thermostat_needs_a_target():
    atoms = fcc_argon(repeat=1)
    with pytest.raises(fa.DynamicsError):
        fa.run_md(atoms, 1, thermostat="nose_hoover")


def test_thermostat_with_leapfrog_still_controls_temperature():
    # Leapfrog holds its own velocities, so the rescaling has to reach them.
    atoms = fcc_argon()
    fa.maxwell_boltzmann(atoms, 10.0, seed=3)
    result = fa.run_md(
        atoms,
        800,
        timestep=5.0,
        temperature=100.0,
        thermostat="berendsen",
        integrator="leapfrog",
        tau=50.0,
        initialize_velocities=False,
        interval=10,
    )
    assert abs(late_mean(result.history, "temperature") - 100.0) < 20.0


# --------------------------------------------------------------------------
# Barostats
# --------------------------------------------------------------------------
@pytest.mark.parametrize("barostat", BAROSTATS)
def test_barostats_relieve_a_compressed_cell(barostat):
    atoms = fcc_argon(a=5.00)
    initial_volume = atoms.get_volume()

    result = fa.run_md(
        atoms,
        2000,
        timestep=5.0,
        temperature=30.0,
        thermostat="nose_hoover",
        barostat=barostat,
        pressure=0.0,
        tau=100.0,
        seed=7,
        interval=20,
    )
    assert result.completed
    assert atoms.get_volume() > initial_volume
    assert abs(late_mean(result.history, "pressure")) < 5e-4
    # Deforming the cell in front of the integrator costs one extra force call.
    assert result.evaluations == 2 * 2000 + 1


def test_a_target_pressure_is_held():
    def relaxed(pressure_gpa):
        atoms = fcc_argon()
        result = fa.run_md(
            atoms,
            2000,
            timestep=5.0,
            temperature=30.0,
            thermostat="nose_hoover",
            barostat="berendsen",
            pressure=pressure_gpa,
            compressibility=1.0,
            tau=200.0,
            seed=19,
            interval=50,
        )
        return atoms.get_volume(), late_mean(result.history, "pressure")

    free_volume, free_pressure = relaxed(0.0)
    squeezed_volume, squeezed_pressure = relaxed(1.0)

    assert squeezed_volume < free_volume
    assert abs(free_pressure) < 5e-4
    assert squeezed_pressure > free_pressure


def test_isotropic_barostat_keeps_the_cell_cubic():
    atoms = fcc_argon(a=5.00)
    md = fa.Dynamics(timestep=5.0, temperature=30.0, seed=23, interval=50)
    md.thermostat = fa.NoseHoover(tau=100.0)
    md.barostat = fa.ParrinelloRahman(tau=500.0, isotropic=True)
    md.run(atoms, 600)

    lengths = atoms.cell.lengths
    assert lengths[0] == pytest.approx(lengths[1], abs=1e-9)
    assert lengths[1] == pytest.approx(lengths[2], abs=1e-9)
    assert md.barostat.is_isotropic
    assert md.barostat.strain_rate.shape == (3, 3)


def test_barostat_names_and_errors():
    assert fa.make_barostat("") is None
    assert fa.make_barostat("PR").name == "parrinello_rahman"
    assert fa.make_barostat("Parrinello-Rahman").name == "parrinello_rahman"
    assert not fa.make_barostat("berendsen").samples_isobaric
    with pytest.raises(fa.DynamicsError):
        fa.make_barostat("andersen")
    assert set(fa.barostat_names()) == {"berendsen", "parrinello_rahman"}

    # A molecule has no box to squeeze.
    molecule = fa.Atoms("Ar2", [[0, 0, 0], [4.0, 0, 0]])
    molecule.calc = fa.LennardJones()
    with pytest.raises(fa.DynamicsError):
        fa.run_md(
            molecule, 1, temperature=50.0, thermostat="nose_hoover", barostat="berendsen"
        )


def test_pressure_observables():
    atoms = fcc_argon()
    fa.maxwell_boltzmann(atoms, 100.0, seed=3)

    kinetic = fa.kinetic_stress(atoms)
    total = fa.total_stress(atoms)
    assert kinetic.shape == (3, 3)
    # The kinetic term is a positive pressure, so a negative stress trace.
    assert np.trace(kinetic) < 0.0
    assert total == pytest.approx(fa.voigt_to_matrix(atoms.get_stress()) + kinetic)
    assert fa.instantaneous_pressure(atoms) == pytest.approx(
        fa.pressure_from_stress(fa.matrix_to_voigt(total))
    )


# --------------------------------------------------------------------------
# The loop, sampling and output
# --------------------------------------------------------------------------
def test_sampling_interval_and_timing():
    atoms = fcc_argon(repeat=1)
    result = fa.run_md(atoms, 100, timestep=2.0, interval=10)

    assert len(result.history) == 11
    assert result.history[0].step == 0
    assert result.history[-1].step == 100
    assert result.history[-1].time == pytest.approx(200.0)
    assert result.time == pytest.approx(200.0)
    assert "DynamicsStep(" in repr(result.history[0])
    assert "DynamicsResult(" in repr(result)


def test_observers_can_stop_a_run():
    atoms = fcc_argon(repeat=1)
    md = fa.Dynamics(timestep=2.0)

    seen = []
    md.attach(lambda atoms, step: seen.append(step.step) or step.step < 5)
    result = md.run(atoms, 100)

    assert result.reason == fa.DynamicsStopReason.interrupted
    assert result.steps == 5
    assert seen == [0, 1, 2, 3, 4, 5]


def test_a_callback_that_returns_nothing_keeps_going():
    atoms = fcc_argon(repeat=1)
    md = fa.Dynamics(timestep=2.0, interval=5)
    collected = []
    md.attach(lambda atoms, step: collected.append(step.temperature), interval=5)
    md.run(atoms, 50)
    assert len(collected) == 11


def test_trajectory_is_written_as_extended_xyz(tmp_path):
    path = tmp_path / "run.xyz"
    atoms = fcc_argon(repeat=1)
    result = fa.run_md(
        atoms,
        50,
        timestep=2.0,
        temperature=50.0,
        thermostat="nose_hoover",
        seed=5,
        trajectory=str(path),
        trajectory_interval=10,
    )
    assert result.steps == 50

    lines = path.read_text().splitlines()
    frame_lines = 2 + len(atoms)
    assert len(lines) == 6 * frame_lines
    assert lines[0].strip() == "4"

    comment = lines[1]
    assert 'Lattice="' in comment
    assert "Properties=species:S:1:pos:R:3:velo:R:3:forces:R:3" in comment
    assert 'pbc="T T T"' in comment
    assert "temperature=" in comment and "conserved_energy=" in comment

    fields = lines[2].split()
    assert len(fields) == 10  # symbol plus three vectors
    assert fields[0] == "Ar"

    # The last frame holds the geometry the run ended on.
    last = [float(x) for x in lines[5 * frame_lines + 2].split()[1:4]]
    assert last == pytest.approx(atoms.positions[0], abs=1e-6)


def test_trajectory_writer_can_be_used_directly(tmp_path):
    path = tmp_path / "manual.xyz"
    atoms = fcc_argon(repeat=1)
    md = fa.Dynamics(timestep=2.0)
    md.prepare(atoms)

    writer = fa.TrajectoryWriter(str(path), velocities=False, forces=False)
    for _ in range(3):
        md.step(atoms)
        writer.write(atoms, md.sample(atoms))

    assert writer.frames_written == 3
    lines = path.read_text().splitlines()
    assert len(lines) == 3 * (2 + len(atoms))
    assert len(lines[2].split()) == 4  # symbol plus position only


def test_log_file(tmp_path):
    path = tmp_path / "run.log"
    atoms = fcc_argon(repeat=1)
    fa.run_md(atoms, 100, timestep=2.0, log=str(path), log_interval=25)

    lines = path.read_text().splitlines()
    assert len(lines) == 6  # header plus five records
    assert "E_cons[eV]" in lines[0]
    assert len(lines[1].split()) == 9


def test_ensemble_detection_and_repr():
    atoms = fcc_argon(repeat=1)

    nve = fa.Dynamics(timestep=1.0)
    nve.run(atoms, 1)
    assert nve.ensemble == "nve"
    assert nve.thermostat is None

    nvt = fa.Dynamics(ensemble="nvt", temperature=50.0)
    nvt.run(fcc_argon(repeat=1), 1)
    assert nvt.ensemble == "nvt"
    assert nvt.thermostat.name == "nose_hoover"

    npt = fa.Dynamics(ensemble="npt", temperature=50.0)
    npt.run(fcc_argon(repeat=1), 1)
    assert npt.ensemble == "npt"
    assert npt.barostat.name == "parrinello_rahman"
    assert "npt" in repr(npt)


def test_pieces_are_resolved_before_the_first_run():
    # The repr and the ensemble have to describe what will actually happen,
    # not what has been built so far.
    md = fa.Dynamics(timestep=5.0, temperature=60.0, thermostat="nose_hoover")
    assert md.ensemble == "nvt"
    assert md.thermostat.name == "nose_hoover"
    assert md.integrator.name == "velocity_verlet"
    assert "nvt" in repr(md)

    # An explicit assignment wins over the parameters and survives a run.
    md.thermostat = None
    assert md.ensemble == "nve"
    atoms = fcc_argon(repeat=1)
    fa.maxwell_boltzmann(atoms, 50.0, seed=1)
    md.run(atoms, 2)
    assert md.ensemble == "nve"
    assert md.thermostat is None

    # An unknown name is caught when the object is built, not on the first run.
    with pytest.raises(fa.DynamicsError):
        fa.Dynamics(thermostat="andersen")


def test_missing_calculator_and_bad_settings():
    atoms = fcc_argon(repeat=1)
    atoms.calc = None
    with pytest.raises(fa.DynamicsError):
        fa.Dynamics().run(atoms, 1)
    assert issubclass(fa.DynamicsError, RuntimeError)

    good = fcc_argon(repeat=1)
    with pytest.raises(fa.DynamicsError):
        fa.run_md(good, 1, timestep=-1.0)


def test_stepping_by_hand():
    atoms = fcc_argon(repeat=1)
    fa.maxwell_boltzmann(atoms, 50.0, seed=4)
    md = fa.Dynamics(timestep=2.0)
    md.prepare(atoms)

    first = md.sample(atoms)
    for _ in range(10):
        md.step(atoms)
    later = md.sample(atoms)

    assert later.step == 10
    assert later.total_energy == pytest.approx(first.total_energy, abs=1e-4)


# --------------------------------------------------------------------------
# The YAML path
# --------------------------------------------------------------------------
MD_INPUT = """
structure:
  cell: {a: 5.3, b: 5.3, c: 5.3}
  coordinates: fractional
  symbols: [Ar, Ar, Ar, Ar]
  positions: [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]

simulation:
  task: md
  calculator:
    name: lennard_jones
    parameters: {epsilon: 0.0103, sigma: 3.40, cutoff: 8.5}
  md:
    ensemble: nvt
    integrator: leapfrog
    thermostat: nose_hoover
    timestep: 4.0
    steps: 50
    temperature: 60.0
    friction: 0.02
    seed: 99
    chain_length: 3
  output:
    interval: 10
"""


def test_dynamics_comes_from_the_input_file(tmp_path):
    deck = fa.read_string(MD_INPUT)
    assert deck.simulation.md.ensemble == "nvt"
    assert deck.simulation.md.timestep_fs == 4.0
    assert deck.simulation.md.steps == 50

    md = deck.make_dynamics()
    assert md.timestep == 4.0
    assert md.temperature == 60.0

    atoms = deck.atoms
    atoms.calc = deck.make_calculator()
    result = md.run(atoms)

    assert result.steps == 50
    assert md.ensemble == "nvt"
    assert md.integrator.name == "leapfrog"
    assert md.thermostat.name == "nose_hoover"
    assert md.thermostat.chain_length == 3
    # friction 0.02 /fs means a 50 fs coupling time.
    assert md.thermostat.time_constant == pytest.approx(50.0)
    assert len(result.history) == 6  # step 0 and every tenth

    # Keyword arguments override the file.
    assert deck.make_dynamics(timestep=1.0).timestep == 1.0


def test_output_section_feeds_the_writers(tmp_path):
    text = MD_INPUT.replace(
        "  output:\n    interval: 10",
        f"  output:\n    interval: 10\n    trajectory: {tmp_path / 'traj.xyz'}\n"
        f"    log: {tmp_path / 'run.log'}",
    )
    deck = fa.read_string(text)
    atoms = deck.atoms
    atoms.calc = deck.make_calculator()
    deck.make_dynamics().run(atoms)

    assert (tmp_path / "traj.xyz").is_file()
    assert (tmp_path / "run.log").is_file()


def test_atoms_run_md_shortcut():
    atoms = fcc_argon(repeat=1)
    result = atoms.run_md(20, timestep=2.0, temperature=50.0, thermostat="berendsen", seed=1)
    assert result.completed
    assert result.steps == 20
