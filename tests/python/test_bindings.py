# SPDX-License-Identifier: MIT
"""Tests for the Python bindings.

Run with ``ctest`` or directly::

    PYTHONPATH=build/python pytest tests/python
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import fastatoms as fa


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------
def water() -> fa.Atoms:
    return fa.Atoms("H2O", [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def test_formula_string_expands_to_atoms():
    atoms = water()
    assert len(atoms) == 3
    assert atoms.get_chemical_symbols() == ["H", "H", "O"]
    assert atoms.get_chemical_formula() == "H2O"


def test_construction_from_symbol_list_and_numbers():
    from_symbols = fa.Atoms(["Si", "O"], [[0, 0, 0], [1, 1, 1]])
    from_numbers = fa.Atoms(numbers=[14, 8], positions=[[0, 0, 0], [1, 1, 1]])
    assert from_symbols.get_atomic_numbers() == from_numbers.get_atomic_numbers() == [14, 8]


def test_cell_accepts_several_spellings():
    assert fa.Atoms("Ar", [[0, 0, 0]], cell=5.0).get_volume() == pytest.approx(125.0)
    assert fa.Atoms("Ar", [[0, 0, 0]], cell=[2.0, 3.0, 4.0]).get_volume() == pytest.approx(24.0)
    matrix = fa.Atoms("Ar", [[0, 0, 0]], cell=np.diag([2.0, 3.0, 4.0]))
    assert matrix.get_volume() == pytest.approx(24.0)


def test_scaled_positions_constructor():
    atoms = fa.Atoms("Si", scaled_positions=[[0.5, 0.5, 0.5]], cell=4.0, pbc=True)
    assert atoms.get_positions()[0] == pytest.approx([2.0, 2.0, 2.0])


def test_pbc_spellings():
    assert list(fa.Atoms("H", [[0, 0, 0]], cell=5.0, pbc=True).pbc) == [True, True, True]
    assert list(fa.Atoms("H", [[0, 0, 0]], cell=5.0, pbc="xy").pbc) == [True, True, False]
    assert list(fa.Atoms("H", [[0, 0, 0]], cell=5.0, pbc=[1, 0, 1]).pbc) == [True, False, True]


def test_bad_input_raises():
    with pytest.raises(ValueError):
        fa.Atoms("H2O", [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        fa.Atoms("Zz", [[0.0, 0.0, 0.0]])


# --------------------------------------------------------------------------
# Zero-copy views
# --------------------------------------------------------------------------
def test_positions_is_a_writable_view():
    atoms = water()
    positions = atoms.positions
    assert positions.shape == (3, 3)
    assert positions.dtype == np.float64

    positions[0, 2] = 5.0
    assert atoms.get_positions()[0, 2] == 5.0
    # The view aliases the C++ buffer rather than copying it.
    assert atoms.positions.base is not None


def test_get_positions_returns_a_copy():
    atoms = water()
    copy = atoms.get_positions()
    copy[0, 0] = 99.0
    assert atoms.positions[0, 0] == 0.0


def test_numbers_view_round_trips():
    atoms = water()
    assert atoms.numbers.tolist() == [1, 1, 8]
    atoms.numbers[0] = 6
    assert atoms.get_chemical_symbols()[0] == "C"


def test_setting_positions_validates_shape():
    atoms = water()
    with pytest.raises(ValueError):
        atoms.positions = np.zeros((2, 3))
    atoms.positions = np.ones((3, 3))
    assert atoms.positions.sum() == pytest.approx(9.0)


# --------------------------------------------------------------------------
# Cell and geometry
# --------------------------------------------------------------------------
def test_cell_is_live():
    atoms = fa.Atoms("Si", [[0, 0, 0]], cell=5.0, pbc=True)
    atoms.cell.pbc = "xy"
    assert list(atoms.pbc) == [True, True, False]
    assert np.asarray(atoms.cell) == pytest.approx(np.diag([5.0, 5.0, 5.0]))


def test_cell_from_parameters_round_trip():
    cell = fa.Cell.from_parameters(4.0, 5.0, 6.0, 70.0, 80.0, 95.0)
    assert cell.lengths == pytest.approx([4.0, 5.0, 6.0])
    assert cell.angles == pytest.approx([70.0, 80.0, 95.0])
    assert cell.volume > 0.0


def test_scaled_positions_round_trip():
    atoms = fa.Atoms("Si2", [[0, 0, 0], [1.3575] * 3], cell=5.43, pbc=True)
    scaled = atoms.get_scaled_positions()
    assert scaled[1] == pytest.approx([0.25, 0.25, 0.25])
    atoms.set_scaled_positions(scaled)
    assert atoms.positions[1] == pytest.approx([1.3575] * 3)


def test_wrap_and_minimum_image_distance():
    atoms = fa.Atoms("H2", [[0, 0, 0], [9.0, 0, 0]], cell=10.0, pbc=True)
    assert atoms.get_distance(0, 1) == pytest.approx(9.0)
    assert atoms.get_distance(0, 1, mic=True) == pytest.approx(1.0)

    atoms.positions[1] = [11.0, -1.0, 0.0]
    atoms.wrap()
    assert atoms.positions[1] == pytest.approx([1.0, 9.0, 0.0])


def test_all_distances_is_symmetric():
    atoms = water()
    matrix = atoms.get_all_distances()
    assert matrix.shape == (3, 3)
    assert matrix == pytest.approx(matrix.T)
    assert np.diag(matrix) == pytest.approx(np.zeros(3))


def test_center_adds_vacuum():
    atoms = fa.Atoms("H2", [[0, 0, 0], [2.0, 0, 0]])
    atoms.center(vacuum=5.0)
    assert atoms.cell.lengths[0] == pytest.approx(12.0)
    assert atoms.positions[0, 0] == pytest.approx(5.0)


def test_rotate_uses_degrees_by_default():
    atoms = fa.Atoms("H2", [[1.0, 0, 0], [-1.0, 0, 0]])
    atoms.rotate(90.0, [0, 0, 1], center=[0, 0, 0])
    assert atoms.positions[0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)


def test_center_of_mass():
    atoms = fa.Atoms(["H", "O"], [[0, 0, 0], [2.0, 0, 0]])
    com = atoms.get_center_of_mass()
    assert com[0] > 1.5  # the oxygen dominates


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------
def test_repeat_and_multiplication():
    atoms = fa.Atoms("Si", [[0, 0, 0]], cell=5.0, pbc=True)
    assert len(atoms.repeat(2, 3, 4)) == 24
    assert len(atoms * 2) == 8
    assert len(atoms * (1, 2, 3)) == 6
    assert (atoms * 2).get_volume() == pytest.approx(1000.0)


def test_indexing_slicing_and_masks():
    atoms = water()
    assert atoms[0].symbol == "H"
    assert atoms[-1].symbol == "O"
    assert len(atoms[:2]) == 2
    assert atoms[[2]].get_chemical_formula() == "O"

    mask = np.array([True, False, True])
    assert atoms[mask].get_chemical_formula() == "HO"

    with pytest.raises(IndexError):
        _ = atoms[7]


def test_atom_proxy_writes_through():
    atoms = water()
    atom = atoms[0]
    atom.position[1] = 3.0
    assert atoms.positions[0, 1] == 3.0

    atom.symbol = "C"
    assert atoms.numbers[0] == 6

    atom.charge = -0.5
    assert atoms.get_charges()[0] == pytest.approx(-0.5)


def test_iteration_yields_atoms():
    atoms = water()
    assert [a.symbol for a in atoms] == ["H", "H", "O"]


def test_append_extend_and_delete():
    atoms = water()
    atoms.append("Na", [3.0, 0.0, 0.0])
    assert len(atoms) == 4
    assert atoms.get_chemical_symbols()[-1] == "Na"

    combined = atoms + water()
    assert len(combined) == 7

    del combined[[0, 1]]
    assert len(combined) == 5


def test_copy_is_independent():
    atoms = water()
    clone = atoms.copy()
    clone.positions[0, 0] = 42.0
    assert atoms.positions[0, 0] == 0.0


# --------------------------------------------------------------------------
# Optional arrays and dynamics
# --------------------------------------------------------------------------
def test_masses_default_to_atomic_weights():
    atoms = water()
    assert atoms.get_masses() == pytest.approx([1.008, 1.008, 15.999])
    atoms.set_masses([2.0, 2.0, 16.0])
    assert atoms.get_masses()[0] == pytest.approx(2.0)


def test_velocities_momenta_and_temperature():
    atoms = fa.Atoms("H", [[0, 0, 0]])
    assert atoms.get_kinetic_energy() == 0.0

    atoms.set_velocities([[1.0, 0.0, 0.0]])
    mass = atoms.get_masses()[0]
    assert atoms.get_momenta()[0, 0] == pytest.approx(mass)
    assert atoms.get_kinetic_energy() == pytest.approx(0.5 * mass)
    expected = 2.0 * 0.5 * mass / (3.0 * fa.units.kB)
    assert atoms.get_temperature() == pytest.approx(expected)


def test_custom_arrays_follow_the_atoms():
    atoms = water()
    atoms.set_array("forces", np.arange(9, dtype=float).reshape(3, 3))
    assert atoms.has_array("forces")
    assert atoms.get_array("forces").shape == (3, 3)
    assert atoms.array_names() == ["forces"]

    subset = atoms[[0, 2]]
    assert subset.get_array("forces").shape == (2, 3)
    assert subset.get_array("forces")[1] == pytest.approx([6.0, 7.0, 8.0])

    assert "forces" in atoms.arrays
    with pytest.raises(ValueError):
        atoms.set_array("positions", np.zeros((3, 3)))


# --------------------------------------------------------------------------
# Neighbour list
# --------------------------------------------------------------------------
def test_neighbor_list_on_simple_cubic():
    crystal = fa.Atoms("Cu", [[0, 0, 0]], cell=3.0, pbc=True).repeat(4, 4, 4)
    nl = fa.neighbor_list(crystal, 3.1)

    assert len(nl) == 6 * len(crystal)
    assert nl.i.shape == (len(nl),)
    assert nl.shifts.shape == (len(nl), 3)
    assert nl.distances == pytest.approx(np.full(len(nl), 3.0))
    assert nl.coordination_numbers().tolist() == [6] * len(crystal)

    neighbours, offsets = nl.get_neighbors(0)
    assert len(neighbours) == 6
    assert offsets.shape == (6, 3)


def test_neighbor_list_vectors_match_the_shift_convention():
    crystal = fa.Atoms("Cu", [[0, 0, 0]], cell=3.0, pbc=True).repeat(3, 3, 3)
    nl = fa.neighbor_list(crystal, 3.1)

    positions = crystal.get_positions()
    cell = np.asarray(crystal.cell)
    expected = positions[nl.j] + nl.shifts @ cell - positions[nl.i]
    assert nl.vectors == pytest.approx(expected)
    assert nl.distances == pytest.approx(np.linalg.norm(expected, axis=1))


def test_half_list_is_half_the_size():
    crystal = fa.Atoms("Cu", [[0, 0, 0]], cell=3.0, pbc=True).repeat(3, 3, 3)
    full = fa.neighbor_list(crystal, 3.1)
    half = fa.neighbor_list(crystal, 3.1, full=False)
    assert 2 * len(half) == len(full)


def test_neighbor_list_with_natural_cutoffs():
    # O first, so that both O-H bonds are present; the 1.52 A H...H separation
    # must not be counted.
    molecule = fa.Atoms(
        ["O", "H", "H"], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
    )
    cutoffs = fa.natural_cutoffs(molecule, mult=1.2)
    bonds = fa.neighbor_list(molecule, per_atom_cutoffs=cutoffs, full=False)
    assert len(bonds) == 2
    assert fa.neighbor_list(
        molecule, per_atom_cutoffs=cutoffs
    ).coordination_numbers().tolist() == [2, 1, 1]

    with pytest.raises(ValueError):
        fa.neighbor_list(molecule)


def test_neighbor_list_arrays_are_read_only():
    crystal = fa.Atoms("Cu", [[0, 0, 0]], cell=3.0, pbc=True).repeat(2, 2, 2)
    nl = fa.neighbor_list(crystal, 3.1)
    with pytest.raises(ValueError):
        nl.i[0] = 0


# --------------------------------------------------------------------------
# YAML input
# --------------------------------------------------------------------------
INPUT = """
structure:
  cell: {a: 5.43, b: 5.43, c: 5.43}
  coordinates: fractional
  symbols: [Si, Si]
  positions: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]

simulation:
  task: md
  calculator:
    name: lennard_jones
    parameters: {epsilon: 0.0103, sigma: 3.4}
  md: {ensemble: nvt, timestep: 0.5, steps: 1000, temperature: 300.0}
  output: {trajectory: traj.yaml, interval: 10}
"""


def test_read_string():
    deck = fa.read_string(INPUT)
    assert len(deck.atoms) == 2
    assert deck.atoms.get_positions()[1] == pytest.approx([1.3575] * 3)

    simulation = deck.simulation
    assert simulation.task == fa.Task.md
    assert simulation.calculator.name == "lennard_jones"
    assert simulation.calculator.parameters == {"epsilon": 0.0103, "sigma": 3.4}
    assert simulation.md.ensemble == "nvt"
    assert simulation.md.steps == 1000
    assert simulation.md.timestep_fs == pytest.approx(0.5)
    assert simulation.md.timestep == pytest.approx(0.5 * fa.units.fs)
    assert simulation.output.interval == 10


def test_document_is_a_plain_dict():
    deck = fa.read_string(INPUT)
    assert isinstance(deck.document, dict)
    assert deck.document["simulation"]["md"]["temperature"] == 300.0


def test_read_and_write_files(tmp_path):
    source = tmp_path / "input.yaml"
    source.write_text(INPUT)

    deck = fa.read(str(source))
    assert len(deck.atoms) == 2

    out = tmp_path / "out.yaml"
    fa.write(str(out), deck.atoms)
    restored = fa.read_atoms(str(out))
    assert restored.get_positions() == pytest.approx(deck.atoms.get_positions())
    assert restored.get_volume() == pytest.approx(deck.atoms.get_volume())


def test_input_errors_are_reported():
    with pytest.raises(fa.InputError):
        fa.read_string("structure: {symbols: [Zz], positions: [[0, 0, 0]]}")
    with pytest.raises(fa.InputError):
        fa.read_string("structure: {symbols: [H], positions: [[0, 0]]}")
    with pytest.raises(fa.InputError):
        fa.read("/nonexistent/input.yaml")
    # InputError is a ValueError, so ordinary handlers catch it too.
    assert issubclass(fa.InputError, ValueError)


def test_atoms_from_dict():
    atoms = fa.atoms_from_dict(
        {"symbols": ["Na", "Cl"], "positions": [[0, 0, 0], [2.8, 0, 0]], "cell": 10.0}
    )
    assert atoms.get_chemical_formula() == "ClNa"
    assert atoms.get_distance(0, 1) == pytest.approx(2.8)


def test_parse_yaml_gives_plain_data():
    data = fa.parse_yaml("a: {b: [1, 2.5, true, null, text]}")
    assert data == {"a": {"b": [1, 2.5, True, None, "text"]}}


def test_to_yaml_round_trip_preserves_arrays():
    atoms = water()
    atoms.set_charges([0.4, 0.4, -0.8])
    atoms.set_tags([1, 1, 2])
    text = fa.to_yaml(atoms)
    restored = fa.read_string(text).atoms
    assert restored.get_charges() == pytest.approx(atoms.get_charges())
    assert restored.get_tags().tolist() == [1, 1, 2]


# --------------------------------------------------------------------------
# Reference data and units
# --------------------------------------------------------------------------
def test_element_helpers():
    assert fa.number_from_symbol("Si") == 14
    assert fa.number_from_symbol("si") == 14
    assert fa.symbol_from_number(8) == "O"
    assert fa.atomic_mass(1) == pytest.approx(1.008)
    assert fa.covalent_radius(6) == pytest.approx(0.76)
    with pytest.raises(ValueError):
        fa.number_from_symbol("Zz")


def test_units_match_ase_conventions():
    assert fa.units.kB == pytest.approx(8.617333262145e-5)
    assert fa.units.fs == pytest.approx(0.09822694788464063)
    assert math.isclose(fa.units.ps, 1000 * fa.units.fs)


def test_version_is_exposed():
    assert fa.__version__.count(".") >= 2


def test_package_docstring_examples():
    """The quick-start examples in the package docstring have to still work."""
    import doctest

    result = doctest.testmod(fa, extraglobs={"fa": fa}, verbose=False)
    assert result.attempted > 0
    assert result.failed == 0
