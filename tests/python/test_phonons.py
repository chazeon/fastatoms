# SPDX-License-Identifier: MIT
"""Tests for the phonon module."""

from __future__ import annotations

import numpy as np
import pytest

import fastatoms as fa

LJ = dict(epsilon=0.0103, sigma=3.40, cutoff=8.0)


def fcc(a: float = 5.26, symbols: str = "Ar", calculator: bool = True) -> fa.Atoms:
    """Primitive fcc argon, near the Lennard-Jones equilibrium volume."""
    cell = [[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]]
    atoms = fa.Atoms(symbols, [[0.0, 0.0, 0.0]], cell=cell, pbc=True)
    if calculator:
        atoms.calc = fa.LennardJones(**LJ)
    return atoms


def diamond(a: float = 5.43) -> fa.Atoms:
    atoms = fa.Atoms("Si2", scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
                     cell=[[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]],
                     pbc=True)
    atoms.calc = fa.LennardJones(**LJ)
    return atoms


def simple_cubic(a: float = 3.0) -> fa.Atoms:
    atoms = fa.Atoms("Ar", [[0.0, 0.0, 0.0]], cell=a, pbc=True)
    atoms.calc = fa.LennardJones(**LJ)
    return atoms


# --------------------------------------------------------------------------
# Supercells
# --------------------------------------------------------------------------
def test_diagonal_supercell():
    unit = fcc()
    mapping = fa.make_supercell(unit, [2, 2, 3])

    assert mapping.cell_count == 12
    assert mapping.unit_cell_size == 1
    assert len(mapping) == 12
    assert len(mapping.atoms) == 12
    assert mapping.atoms.get_volume() == pytest.approx(12 * unit.get_volume())

    # The first `unit_cell_size` atoms are the original cell at the origin.
    assert mapping.lattice_points[0].tolist() == [0, 0, 0]
    assert mapping.atoms.positions[0] == pytest.approx(unit.positions[0])
    assert mapping.unit_cell_index.tolist() == [0] * 12
    assert np.unique(mapping.lattice_points, axis=0).shape[0] == 12

    # An integer is shorthand for a diagonal matrix.
    assert fa.make_supercell(unit, 2).cell_count == 8


def test_non_diagonal_supercell():
    """fcc primitive -> conventional cubic, the textbook case."""
    mapping = fa.make_supercell(fcc(a=5.26), [[-1, 1, 1], [1, -1, 1], [1, 1, -1]])

    assert mapping.cell_count == 4
    assert len(mapping.atoms) == 4
    assert mapping.atoms.cell.lengths == pytest.approx([5.26, 5.26, 5.26])
    assert mapping.atoms.cell.angles == pytest.approx([90.0, 90.0, 90.0])

    scaled = np.sort(mapping.atoms.get_scaled_positions(), axis=0)
    expected = np.sort([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]], axis=0)
    assert scaled == pytest.approx(expected, abs=1e-12)


def test_supercell_rejects_singular_transformations():
    with pytest.raises(fa.PhononError):
        fa.make_supercell(fcc(), [[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    with pytest.raises(fa.PhononError):
        fa.make_supercell(fcc(), [0, 1, 1])


# --------------------------------------------------------------------------
# Symmetry
# --------------------------------------------------------------------------
def test_symmetry_of_a_cubic_crystal():
    symmetry = fa.find_symmetry(simple_cubic())

    assert len(symmetry) == 48
    assert len(symmetry.operations) == 48
    assert len(symmetry.point_group) == 48
    assert symmetry.independent_atoms.tolist() == [0]
    assert symmetry.equivalent_atoms.tolist() == [0]
    assert len(fa.site_symmetry(symmetry, 0)) == 48

    for operation in symmetry.operations:
        rotation = operation.cartesian_rotation
        assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-10)
        assert abs(np.linalg.det(operation.rotation)) == pytest.approx(1.0)
        assert operation.atom_map.tolist() == [0]


def test_symmetry_of_diamond_silicon():
    symmetry = fa.find_symmetry(diamond())

    assert len(symmetry) == 48
    assert symmetry.independent_atoms.tolist() == [0]      # the two sites are equivalent
    assert symmetry.equivalent_atoms.tolist() == [0, 0]
    assert len(fa.site_symmetry(symmetry, 0)) == 24        # tetrahedral site

    # Half the operations swap the two atoms, half leave them alone.
    swaps = sum(1 for op in symmetry.operations if op.atom_map.tolist() == [1, 0])
    assert swaps == 24


def test_lower_symmetry_from_a_second_species():
    """Zincblende: the same lattice, two species, half the operations."""
    silicon = diamond()
    zincblende = fa.Atoms("GaAs", scaled_positions=silicon.get_scaled_positions(),
                          cell=silicon.cell.array, pbc=True)
    symmetry = fa.find_symmetry(zincblende)

    assert len(symmetry) == 24
    assert symmetry.independent_atoms.tolist() == [0, 1]
    assert all(op.atom_map.tolist() == [0, 1] for op in symmetry.operations)


def test_lattice_point_group():
    assert len(fa.lattice_point_group(np.diag([3.0, 3.0, 3.0]))) == 48
    assert len(fa.lattice_point_group(np.diag([3.0, 3.0, 5.0]))) == 16
    assert len(fa.lattice_point_group(np.diag([3.0, 4.0, 5.0]))) == 8
    assert len(fa.lattice_point_group(fcc().cell.array)) == 48


def test_supercell_symmetry_includes_pure_translations():
    symmetry = fa.find_symmetry(fa.make_supercell(simple_cubic(), 2).atoms)

    assert len(symmetry) == 48 * 8
    translations = [op for op in symmetry.operations if op.is_pure_translation()]
    assert len(translations) == 8


# --------------------------------------------------------------------------
# Displacements
# --------------------------------------------------------------------------
def test_symmetry_reduces_the_displacement_count():
    ph = fa.Phonons(simple_cubic(), supercell=[2, 2, 2], displacement=0.02)

    assert ph.calculation_count == 2                      # +x and -x
    assert ph.calculation_count_without_symmetry == 6     # 3 directions, both signs
    assert len(ph.displacements) == 2
    assert ph.displacement_distance == 0.02
    assert ph.uses_symmetry
    assert ph.central_differences

    for displacement in ph.displacements:
        assert displacement.atom == 0
        assert np.linalg.norm(displacement.vector) == pytest.approx(0.02)

    plain = fa.Phonons(simple_cubic(), supercell=[2, 2, 2], symmetry=False)
    assert plain.calculation_count == 6
    assert not plain.uses_symmetry


def test_one_sided_differences_halve_the_count():
    ph = fa.Phonons(simple_cubic(), supercell=[2, 2, 2], central=False)

    assert ph.calculation_count == 1
    assert not ph.central_differences
    assert ph.calculation_count_without_symmetry == 3


def test_displaced_supercells_carry_the_displacement():
    ph = fa.Phonons(fcc(), supercell=[2, 2, 2])
    cells = ph.displaced_supercells()

    assert len(cells) == len(ph.displacements)
    reference = ph.supercell.atoms.positions
    for cell, displacement in zip(cells, ph.displacements):
        assert len(cell) == len(reference)
        delta = cell.positions - reference
        assert delta[displacement.atom] == pytest.approx(displacement.vector)
        assert np.delete(delta, displacement.atom, axis=0) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Force constants and frequencies
# --------------------------------------------------------------------------
def test_frequencies_of_argon():
    ph = fa.Phonons(fcc(), supercell=[3, 3, 3])
    assert not ph.has_force_constants
    ph.run()
    assert ph.has_force_constants

    # The equilibrium structure really is one.
    assert ph.residual_force < 1e-10
    assert ph.force_constants.acoustic_sum_violation() < 1e-12

    # Three acoustic branches, all going to zero at the zone centre.
    gamma = ph.frequencies([0.0, 0.0, 0.0])
    assert gamma.shape == (3,)
    assert gamma == pytest.approx(0.0, abs=1e-6)

    # Away from it the crystal is stable and the branches are ordered.
    for q in ([0.5, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.5, 0.5]):
        frequencies = ph.frequencies(q)
        assert np.all(frequencies > 0.1)
        assert np.all(np.diff(frequencies) >= 0.0)


def test_force_constants_shape_and_sum_rule():
    ph = fa.Phonons(fcc(), supercell=[2, 2, 2])
    ph.run()
    constants = ph.force_constants

    assert constants.rows == 1                   # atoms in the unit cell
    assert constants.columns == 8                # atoms in the supercell
    assert constants.array.shape == (3, 24)

    # Each row of 3x3 blocks sums to zero: a rigid translation costs nothing.
    blocks = constants.array.reshape(1, 3, 8, 3).sum(axis=2)
    assert blocks == pytest.approx(0.0, abs=1e-12)


def test_symmetry_reduction_matches_the_full_calculation():
    """The whole point of the module, tested directly."""
    reduced = fa.Phonons(fcc(), supercell=[2, 2, 2])
    reduced.run()
    full = fa.Phonons(fcc(), supercell=[2, 2, 2], symmetry=False)
    full.run()

    assert reduced.calculation_count < full.calculation_count
    assert reduced.force_constants.array == pytest.approx(full.force_constants.array, abs=1e-8)
    for q in ([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.3, 0.1, 0.7]):
        assert reduced.frequencies(q) == pytest.approx(full.frequencies(q), abs=1e-6)


def test_dynamical_matrix_is_hermitian_and_periodic():
    ph = fa.Phonons(fcc(), supercell=[2, 2, 2])
    ph.run()

    matrix = ph.dynamical_matrix([0.3, 0.1, 0.0])
    assert matrix.shape == (3, 3)
    assert matrix.dtype == np.complex128
    assert matrix == pytest.approx(matrix.conj().T)

    # D(q + G) = D(q), and time reversal gives D(-q) = D(q)*.
    shifted = ph.dynamical_matrix([1.3, -0.9, 1.0])
    assert shifted == pytest.approx(matrix, abs=1e-12)
    assert ph.dynamical_matrix([-0.3, -0.1, 0.0]) == pytest.approx(matrix.conj(), abs=1e-12)


def test_modes_are_orthonormal():
    ph = fa.Phonons(diamond(), supercell=[2, 2, 2])
    ph.run()

    modes = ph.modes([0.25, 0.0, 0.0])
    assert len(modes.frequencies) == 6           # 3 branches per atom
    assert modes.eigenvectors.shape == (6, 6)
    assert modes.eigenvectors @ modes.eigenvectors.conj().T == pytest.approx(np.eye(6), abs=1e-10)
    assert modes.frequencies == pytest.approx(np.sort(modes.frequencies))

    # Frequencies are the signed square roots of the eigenvalues, in THz. The
    # eigenvalues are in eV/(A^2 amu), whose square root is an angular
    # frequency in the internal time unit.
    to_thz = fa.units.fs * 1000.0 / (2.0 * np.pi)
    signed = np.sign(modes.squared_frequencies) * np.sqrt(abs(modes.squared_frequencies))
    assert modes.frequencies == pytest.approx(signed * to_thz, abs=1e-9)


def test_unstable_structure_gives_imaginary_frequencies():
    """Frequencies of unstable modes come back negative, as is conventional."""
    ph = fa.Phonons(fcc(a=7.0), supercell=[2, 2, 2])
    ph.run()

    frequencies = ph.frequencies([0.5, 0.5, 0.5])
    assert np.any(frequencies < -0.05)


def test_band_structure_and_mesh():
    ph = fa.Phonons(fcc(), supercell=[2, 2, 2])
    ph.run()

    path = [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]
    bands = ph.band_structure(path)
    assert bands.shape == (3, 3)
    assert bands[0] == pytest.approx(0.0, abs=1e-6)
    assert bands[2] == pytest.approx(ph.frequencies(path[2]))

    # The mesh is Gamma-centred and uniform, so it is exactly what asking for
    # every one of its q-points by hand would give - including the three zeros
    # the zone centre contributes.
    mesh = ph.mesh_frequencies(4, 4, 4)
    assert mesh.shape == (4 * 4 * 4 * 3,)
    assert np.count_nonzero(abs(mesh) < 1e-6) == 3

    by_hand = np.concatenate([ph.frequencies([i / 4, j / 4, k / 4])
                              for i in range(4) for j in range(4) for k in range(4)])
    assert np.sort(mesh) == pytest.approx(np.sort(by_hand))


def test_unit_conversions():
    assert fa.Phonons.THZ_TO_WAVENUMBER == pytest.approx(33.35641, rel=1e-6)
    assert fa.Phonons.THZ_TO_MEV == pytest.approx(4.135667696, rel=1e-9)


# --------------------------------------------------------------------------
# Supplying forces from elsewhere
# --------------------------------------------------------------------------
def test_set_forces_reproduces_run():
    reference = fa.Phonons(fcc(), supercell=[2, 2, 2])
    reference.run()

    offline = fa.Phonons(fcc(calculator=False), supercell=[2, 2, 2])
    calculator = fa.LennardJones(**LJ)
    forces = [calculator.get_forces(cell) for cell in offline.displaced_supercells()]
    offline.set_forces(forces)

    assert offline.force_constants.array == pytest.approx(reference.force_constants.array)
    assert offline.frequencies([0.5, 0.0, 0.0]) == pytest.approx(
        reference.frequencies([0.5, 0.0, 0.0]))


def test_set_forces_checks_the_count():
    ph = fa.Phonons(fcc(calculator=False), supercell=[2, 2, 2])
    with pytest.raises(fa.PhononError):
        ph.set_forces([np.zeros((8, 3))])          # one array short
    with pytest.raises(ValueError):
        ph.set_forces([np.zeros((5, 3)), np.zeros((5, 3))])   # wrong shape


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
def test_errors():
    with pytest.raises(fa.PhononError):
        fa.Phonons(fa.Atoms("Ar", [[0.0, 0.0, 0.0]]))          # no cell

    with pytest.raises(fa.PhononError):
        fa.Phonons(fcc(), supercell=[0, 1, 1])                  # singular

    ph = fa.Phonons(fcc(calculator=False), supercell=[2, 2, 2])
    with pytest.raises(fa.PhononError):
        ph.run()                                                # no calculator
    with pytest.raises(fa.PhononError):
        ph.frequencies([0.0, 0.0, 0.0])                         # nothing solved yet


# --------------------------------------------------------------------------
# Input files
# --------------------------------------------------------------------------
PHONON_INPUT = """
structure:
  cell:
    - [0.0, 2.63, 2.63]
    - [2.63, 0.0, 2.63]
    - [2.63, 2.63, 0.0]
  pbc: true
  coordinates: fractional
  atoms:
    - {symbol: Ar, position: [0.0, 0.0, 0.0]}

simulation:
  task: phonon
  calculator:
    name: lennard_jones
    parameters: {epsilon: 0.0103, sigma: 3.40, cutoff: 8.0}
  phonons:
    supercell: [2, 2, 2]
    displacement: 0.02
    symmetry: true
"""


def test_yaml_section():
    deck = fa.read_input_string(PHONON_INPUT)

    assert deck.simulation.task == fa.Task.phonon
    assert deck.simulation.phonon.supercell == [2.0, 2.0, 2.0]
    assert deck.simulation.phonon.displacement == 0.02
    assert deck.simulation.phonon.symmetry

    atoms = deck.atoms
    atoms.calc = deck.make_calculator()
    ph = deck.make_phonons(atoms)

    assert ph.supercell.cell_count == 8
    assert ph.displacement_distance == 0.02
    ph.run()
    assert ph.frequencies([0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-6)

    # Keyword arguments override the file.
    assert deck.make_phonons(atoms, displacement=0.05).displacement_distance == 0.05


def test_make_phonons_from_a_dict():
    atoms = fcc()
    ph = fa.make_phonons(atoms, {"supercell": 2, "displacement": 0.015})

    assert ph.supercell.cell_count == 8
    assert ph.displacement_distance == 0.015
    assert fa.make_phonons(atoms, supercell=[1, 1, 1]).supercell.cell_count == 1
