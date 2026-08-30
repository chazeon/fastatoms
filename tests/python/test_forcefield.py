# SPDX-License-Identifier: MIT
"""Tests for the classical force field: SPC/SPC/E water, Ewald, SHAKE/RATTLE.

The C++ suite carries the heavy validation - the Madelung constant, the
independence of the Ewald splitting, finite-difference forces and stress. What
is checked here is that all of it is reachable and correct from Python, and
that the pieces compose: a water box, a rigid MD run, and a temperature that
counts the right degrees of freedom.
"""

from __future__ import annotations

import numpy as np
import pytest

import fastatoms as fa


def box(model: str = "spce", molecules: int = 32) -> fa.Atoms:
    atoms = fa.water_box(model, molecules)
    atoms.calc = fa.Forcefield(model=model)
    return atoms


# --------------------------------------------------------------------------
# The models
# --------------------------------------------------------------------------
def test_published_parameters():
    spc, spce = fa.water_model("spc"), fa.water_model("spce")

    assert spc.q_oxygen == pytest.approx(-0.82)
    assert spc.q_hydrogen == pytest.approx(0.41)
    assert spce.q_oxygen == pytest.approx(-0.8476)
    assert spce.q_hydrogen == pytest.approx(0.4238)

    for model in (spc, spce):
        assert model.q_oxygen + 2 * model.q_hydrogen == pytest.approx(0.0)
        assert model.bond_length == pytest.approx(1.0)
        assert model.angle == pytest.approx(109.47)
        assert model.sigma_oxygen == pytest.approx(3.166)

    # The dipole moments the papers quote, which the parameters have to imply.
    assert spc.dipole_moment == pytest.approx(2.27, abs=5e-3)
    assert spce.dipole_moment == pytest.approx(2.35, abs=5e-3)

    # SPC/E's polarisation self-energy is the term SPC is missing: 5.22 kcal/mol.
    assert spc.self_energy == 0.0
    assert spce.self_energy == pytest.approx(5.22 * fa.units.kcal_per_mol)

    # The H-H distance that makes the angle rigid.
    assert spce.hydrogen_distance == pytest.approx(2 * np.sin(np.radians(109.47) / 2))


def test_model_lookup_is_forgiving():
    for name in ("spce", "SPC/E", "spc_e", "SPC-E"):
        assert fa.water_model(name).name == "spce"
    assert set(fa.water_model_names()) >= {"spc", "spce", "tip3p"}
    with pytest.raises(Exception):
        fa.water_model("dihydrogen monoxide")


def test_water_box_has_the_requested_density():
    atoms = fa.water_box("spce", 32)
    assert len(atoms) == 96
    mass = sum(atoms.get_masses())
    density = mass / atoms.get_volume() * 1.66053906660     # amu/A^3 -> g/cm^3
    assert density == pytest.approx(0.997, rel=1e-6)

    # The count is rounded up to fill the lattice, so molecules never overlap.
    assert len(fa.water_box("spce", 5)) == 3 * 32


def test_water_molecule_is_at_the_model_geometry():
    model = fa.water_model("spc")
    atoms = fa.water_molecule("spc", box=20.0)
    assert atoms.get_chemical_symbols() == ["O", "H", "H"]
    positions = atoms.positions
    for hydrogen in (1, 2):
        assert np.linalg.norm(positions[hydrogen] - positions[0]) == pytest.approx(
            model.bond_length)
    assert np.linalg.norm(positions[1] - positions[2]) == pytest.approx(
        model.hydrogen_distance)


# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------
def test_topology_describes_the_box():
    atoms = fa.water_box("spce", 32)
    model = fa.water_model("spce")
    topology = fa.water_topology(atoms, "spce")

    assert len(topology) == 96
    assert len(topology.molecules) == 32
    assert len(topology.constraints) == 96          # three per molecule
    assert len(topology.excluded_pairs()) == 96
    assert topology.net_charge == pytest.approx(0.0, abs=1e-12)

    numbers = atoms.get_atomic_numbers()
    assert topology.charges[numbers == 8] == pytest.approx(model.q_oxygen)
    assert topology.charges[numbers == 1] == pytest.approx(model.q_hydrogen)
    # Only the oxygen carries dispersion in a three-site model.
    assert np.all(topology.epsilons[numbers == 1] == 0.0)
    assert topology.epsilons[numbers == 8] == pytest.approx(model.epsilon_oxygen)

    # Every intramolecular pair is excluded, and nothing else is.
    first = topology.molecules[0].atoms
    assert topology.excluded(first[0], first[1])
    assert not topology.excluded(first[0], topology.molecules[1].atoms[0])


def test_molecule_detection():
    atoms = fa.water_box("spce", 32)
    molecules = fa.detect_molecules(atoms)
    assert len(molecules) == 32
    assert all(len(molecule) == 3 for molecule in molecules)


def test_a_non_water_structure_is_rejected():
    atoms = fa.Atoms("OH", [[0, 0, 0], [1, 0, 0]], cell=15.0, pbc=True)
    with pytest.raises(Exception, match="one oxygen and two hydrogens"):
        fa.water_topology(atoms, "spce")


# --------------------------------------------------------------------------
# Ewald
# --------------------------------------------------------------------------
def test_ewald_reproduces_the_madelung_constant():
    """Rock salt, whose lattice sum is known in closed form."""
    a = 5.64
    cations = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    anions = [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]]
    atoms = fa.Atoms("Na4Cl4", scaled_positions=cations + anions, cell=a, pbc=True)

    topology = fa.Topology(len(atoms))
    for i in range(len(atoms)):
        topology.set_charge(i, 1.0 if atoms.get_atomic_numbers()[i] == 11 else -1.0)

    ewald = fa.Ewald(accuracy=1e-10)
    ewald.prepare(atoms)
    result = ewald.compute(atoms, topology)

    madelung = -result["energy"].total * (a / 2) / (4 * fa.units.coulomb)
    assert madelung == pytest.approx(1.7475645946, abs=1e-8)

    # The electrostatic forces sum to zero, as any internal force must.
    assert result["forces"].sum(axis=0) == pytest.approx([0, 0, 0], abs=1e-10)


def test_ewald_pieces_and_settings():
    atoms = fa.water_box("spce", 32)
    topology = fa.water_topology(atoms, "spce")
    ewald = fa.Ewald(accuracy=1e-8, cutoff=4.5)
    ewald.prepare(atoms)

    assert ewald.prepared
    assert ewald.cutoff == pytest.approx(4.5)
    assert ewald.alpha > 0
    assert ewald.kvector_count > 0
    assert ewald.estimated_error(topology, atoms.get_volume()) >= 0

    energy = ewald.compute(atoms, topology)["energy"]
    assert energy.self < 0                      # always negative
    assert energy.reciprocal >= 0               # a sum of squares
    assert energy.excluded > 0                  # undoing negative intramolecular terms
    assert energy.total == pytest.approx(
        energy.real + energy.reciprocal + energy.self + energy.excluded)


def test_ewald_needs_a_periodic_cell():
    atoms = fa.Atoms("OH2", [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(Exception):
        fa.Ewald().prepare(atoms)


# --------------------------------------------------------------------------
# The calculator
# --------------------------------------------------------------------------
def test_energy_terms_add_up():
    atoms = box()
    energy = atoms.get_potential_energy()
    breakdown = atoms.calc.breakdown

    assert breakdown.total == pytest.approx(energy)
    assert breakdown.total == pytest.approx(
        breakdown.lennard_jones + breakdown.lennard_jones_tail
        + breakdown.coulomb.total + breakdown.self)

    # SPC/E's constant, once per molecule.
    assert breakdown.self == pytest.approx(32 * fa.water_model("spce").self_energy)
    assert atoms.calc.model.name == "spce"
    assert atoms.calc.cutoff > 0
    assert len(atoms.calc.constraints) == 96

    # The charges travel with the structure, as from a DFT calculator.
    assert atoms.get_charges()[0] == pytest.approx(fa.water_model("spce").q_oxygen)


def test_spc_and_spce_differ_by_more_than_the_constant():
    spc = box("spc")
    spce = box("spce")
    spc.get_potential_energy()
    spce.get_potential_energy()

    # Same geometry, different charges: the electrostatics must differ.
    assert spc.calc.breakdown.coulomb.total != pytest.approx(
        spce.calc.breakdown.coulomb.total, rel=1e-6)
    # SPC has no polarisation correction; SPC/E does.
    assert spc.calc.breakdown.self == 0.0
    assert spce.calc.breakdown.self > 0.0


def test_forces_match_finite_differences():
    atoms = box(molecules=32)
    atoms.calc = fa.Forcefield(model="spce", accuracy=1e-12, cutoff=4.5)
    atoms.positions[0] += [0.11, -0.07, 0.05]

    forces = atoms.get_forces()
    step = 1e-6
    for atom in (0, 1, 40):
        for axis in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.calc = fa.Forcefield(model="spce", accuracy=1e-12, cutoff=4.5)
            minus.calc = fa.Forcefield(model="spce", accuracy=1e-12, cutoff=4.5)
            plus.positions[atom, axis] += step
            minus.positions[atom, axis] -= step
            numerical = -(plus.get_potential_energy()
                          - minus.get_potential_energy()) / (2 * step)
            assert forces[atom, axis] == pytest.approx(numerical, abs=1e-5)


def test_a_hand_built_topology_can_replace_the_model():
    """Two sodium ions in a box: not water, so the model cannot describe it."""
    atoms = fa.Atoms("Na2", [[0, 0, 0], [4, 0, 0]], cell=12.0, pbc=True)
    topology = fa.Topology(2)
    topology.set_charge(0, 1.0)
    topology.set_charge(1, -1.0)
    topology.set_lennard_jones(0, 0.01, 2.5)
    topology.set_lennard_jones(1, 0.01, 2.5)

    calc = fa.Forcefield(cutoff=5.0)
    calc.set_topology(topology)
    atoms.calc = calc
    assert np.isfinite(atoms.get_potential_energy())
    assert calc.model is None
    assert len(calc.constraints) == 0


def test_a_cluster_uses_the_direct_sum():
    """Ewald exists because a lattice sum is conditionally convergent. A
    molecule in vacuum has no such problem, so it gets the sum itself - and the
    two must agree once the box is big enough for the images not to matter."""
    single = fa.water_molecule("spce", box=0.0)
    positions = np.vstack([single.positions, single.positions + [3.0, 2.2, 0.0]])
    symbols = "".join(single.get_chemical_symbols() * 2)

    cluster = fa.Atoms(symbols, positions)              # no cell at all
    cluster.calc = fa.Forcefield(model="spce", self_energy=False, tail=False)
    direct = cluster.get_potential_energy()
    assert "direct sum" in repr(cluster.calc)

    box_length = 60.0
    periodic = fa.Atoms(symbols, positions + box_length / 2, cell=box_length, pbc=True)
    periodic.calc = fa.Forcefield(model="spce", self_energy=False, tail=False,
                                  accuracy=1e-10)
    assert direct == pytest.approx(periodic.get_potential_energy(), abs=1e-3)
    assert cluster.get_forces() == pytest.approx(periodic.get_forces(), abs=1e-3)

    # No cell means no stress to report.
    with pytest.raises(fa.CalculatorError):
        cluster.get_stress()


def test_named_models_are_registered_calculators():
    # `spce` on its own must mean SPC/E water, not a force field waiting to be
    # told which model to use.
    for name in ("spc", "spce", "tip3p"):
        calc = fa.make_calculator(name)
        atoms = fa.water_box(name, 32)
        atoms.calc = calc
        atoms.calc.prepare(atoms)
        assert atoms.calc.model.name == name
    # An explicit model still wins over the alias.
    assert fa.make_calculator("spc", model="spce").parameters["model"] == "spce"


def test_a_model_is_required():
    atoms = fa.water_box("spce", 32)
    atoms.calc = fa.Forcefield()
    with pytest.raises(Exception, match="set 'model'"):
        atoms.get_potential_energy()


# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------
def test_shake_and_rattle():
    atoms = fa.water_box("spce", 32)
    constraints = fa.Constraints(fa.water_topology(atoms, "spce").constraints)

    assert len(constraints) == 96
    assert constraints.removed_degrees_of_freedom == 96
    assert constraints.max_deviation(atoms) < 1e-12

    # Break the geometry, then put it back.
    rng = np.random.default_rng(0)
    atoms.positions += rng.normal(0, 0.01, atoms.positions.shape)
    assert constraints.max_deviation(atoms) > 1e-4
    report = constraints.enforce(atoms)
    assert report.converged
    assert constraints.max_deviation(atoms) < 1e-9

    # RATTLE takes the velocity along each bond away.
    fa.maxwell_boltzmann(atoms, 300.0, seed=1)
    velocities = atoms.get_velocities()
    assert constraints.max_velocity_deviation(atoms, velocities) > 1e-4


def test_dynamics_takes_the_constraints_from_the_calculator():
    atoms = box()
    dynamics = fa.make_dynamics(timestep=2.0, temperature=300.0,
                                initialize_velocities=True, seed=1)
    dynamics.prepare(atoms)

    assert dynamics.constraints is not None
    assert len(dynamics.constraints) == 96
    # 288 momentum components, less 3 for the centre of mass, less 96 for the
    # constraints: six per molecule, which is what a rigid body has.
    assert dynamics.constraints.removed_degrees_of_freedom == 96


def test_constraints_can_be_turned_off():
    atoms = box()
    dynamics = fa.make_dynamics(timestep=0.5, temperature=300.0,
                                initialize_velocities=True, seed=1, constraints=False)
    dynamics.prepare(atoms)
    assert dynamics.constraints is None


def test_rigid_water_stays_rigid_through_a_run():
    atoms = box()
    dynamics = fa.make_dynamics(timestep=2.0, temperature=300.0,
                                initialize_velocities=True, seed=11)
    dynamics.prepare(atoms)
    constraints = dynamics.constraints

    result = dynamics.run(atoms, 50)
    assert result.completed
    # Every O-H bond and H-O-H angle is still exactly what the model says.
    assert constraints.max_deviation(atoms) < 1e-8
    assert constraints.max_velocity_deviation(atoms, atoms.get_velocities()) < 1e-8


@pytest.mark.parametrize("integrator", ["velocity_verlet", "leapfrog"])
def test_every_integrator_enforces_them(integrator):
    atoms = box()
    dynamics = fa.make_dynamics(timestep=1.0, temperature=250.0, integrator=integrator,
                                initialize_velocities=True, seed=3)
    dynamics.run(atoms, 20)
    assert dynamics.constraints.max_deviation(atoms) < 1e-8


# --------------------------------------------------------------------------
# Input files
# --------------------------------------------------------------------------
def test_yaml_selects_the_model():
    deck = fa.read_input_string("""
structure:
  cell: {a: 9.8654, b: 9.8654, c: 9.8654}
  pbc: true
  atoms:
    - {symbol: O, position: [0.0, 0.0, 0.0]}
    - {symbol: H, position: [0.5774, 0.8165, 0.0]}
    - {symbol: H, position: [0.5774, -0.8165, 0.0]}

simulation:
  task: md
  calculator:
    name: forcefield
    parameters: {model: spce, accuracy: 1.0e-6}
  md: {ensemble: nve, timestep: 2.0, steps: 5}
""")
    assert deck.simulation.calculator.name == "forcefield"
    atoms = deck.atoms
    atoms.calc = deck.make_calculator()
    assert atoms.calc.name == "forcefield"
    assert np.isfinite(atoms.get_potential_energy())
    assert len(atoms.calc.constraints) == 3      # one molecule
