# SPDX-License-Identifier: MIT
"""Motion constraints from Python.

The C++ suite checks the projections themselves and that every consumer
honours them. What is checked here is the Python surface - the spellings a
constraint can be written in, and that attaching one to an ``Atoms`` is enough
for the optimizers, the dynamics and the band to respect it.
"""

from __future__ import annotations

import numpy as np
import pytest

import fastatoms as fa


def cluster() -> fa.Atoms:
    """Four argon atoms at no symmetric point, so every force component is
    non-zero and a projection that does nothing would show."""
    atoms = fa.Atoms("Ar4", [[0, 0, 0], [3.6, 0, 0], [0, 3.6, 0], [1.8, 1.8, 2.6]],
                     cell=20.0, pbc=False)
    atoms.calc = fa.LennardJones(epsilon=0.0103, sigma=3.40, cutoff=9.0)
    return atoms


# --------------------------------------------------------------------------
# Attaching them
# --------------------------------------------------------------------------
def test_constraints_round_trip():
    atoms = cluster()
    assert atoms.constraints == []

    atoms.constraints = [fa.FixAtoms([0, 1])]
    assert len(atoms.constraints) == 1
    assert atoms.constraints[0].name == "FixAtoms"
    assert list(atoms.constraints[0].indices) == [0, 1]

    atoms.add_constraint(fa.FixPlane([3], "z"))
    assert len(atoms.constraints) == 2

    atoms.clear_constraints()
    assert atoms.constraints == []


def test_a_single_constraint_need_not_be_wrapped_in_a_list():
    atoms = cluster()
    atoms.constraints = fa.FixAtoms([0])
    assert len(atoms.constraints) == 1

    atoms.constraints = None
    assert atoms.constraints == []


def test_indices_can_be_written_several_ways():
    assert list(fa.FixAtoms([0, 2]).indices) == [0, 2]
    assert list(fa.FixAtoms(range(3)).indices) == [0, 1, 2]
    assert list(fa.FixAtoms(mask=[True, False, True, False]).indices) == [0, 2]
    # Sorted and deduplicated, so the degrees of freedom cannot be double-counted.
    assert list(fa.FixAtoms([2, 0, 2]).indices) == [0, 2]


def test_a_direction_can_be_an_axis_name_or_a_vector():
    assert fa.FixLine([0], "z").direction == pytest.approx([0, 0, 1])
    assert fa.FixPlane([0], "x").normal == pytest.approx([1, 0, 0])
    # And it is normalised.
    assert fa.FixLine([0], [0, 3, 4]).direction == pytest.approx([0, 0.6, 0.8])


def test_bad_constraints_are_refused():
    with pytest.raises(fa.ConstraintError):
        fa.FixLine([0], [0, 0, 0])                     # no direction
    with pytest.raises(fa.ConstraintError):
        fa.FixAtoms([0], mask=[True])                  # both spellings at once
    with pytest.raises(fa.ConstraintError):
        fa.FixAtoms()                                  # neither
    with pytest.raises(fa.ConstraintError):
        cluster().constraints = [fa.FixAtoms([9])]     # no such atom


# --------------------------------------------------------------------------
# The projections
# --------------------------------------------------------------------------
def test_fix_atoms_zeroes_the_forces_it_names():
    atoms = cluster()
    raw = atoms.get_forces().copy()
    atoms.constraints = [fa.FixAtoms([0, 2])]
    held = atoms.get_forces()

    assert held[[0, 2]] == pytest.approx(0.0)
    assert held[[1, 3]] == pytest.approx(raw[[1, 3]])
    assert np.linalg.norm(raw[0]) > 1e-6          # there was something to remove


def test_fix_line_and_fix_plane_are_complementary():
    atoms = cluster()
    raw = atoms.get_forces().copy()

    line = cluster()
    line.constraints = [fa.FixLine([3], "z")]
    along = line.get_forces()[3]

    plane = cluster()
    plane.constraints = [fa.FixPlane([3], "z")]
    within = plane.get_forces()[3]

    # One keeps exactly what the other throws away.
    assert along + within == pytest.approx(raw[3])
    assert along[:2] == pytest.approx(0.0)
    assert within[2] == pytest.approx(0.0)


def test_the_calculators_own_results_stay_unconstrained():
    """The projection describes how atoms may move, not what the model said."""
    atoms = cluster()
    raw = atoms.get_forces().copy()
    atoms.constraints = [fa.FixAtoms([0])]

    assert atoms.get_forces()[0] == pytest.approx(0.0)
    cached = np.asarray(atoms.calc.results.forces).reshape(-1, 3)
    assert cached[0] == pytest.approx(raw[0])


def test_a_plane_passes_through_the_atom_not_the_origin():
    atoms = cluster()
    start = atoms.positions.copy()
    atoms.constraints = [fa.FixPlane([3], "z")]

    moved = start.copy()
    moved[3] += [0.5, 0.0, 0.5]
    atoms.move_to(moved)

    assert atoms.positions[3][0] == pytest.approx(start[3][0] + 0.5)
    assert atoms.positions[3][2] == pytest.approx(start[3][2])     # 2.6, not 0


# --------------------------------------------------------------------------
# Degrees of freedom
# --------------------------------------------------------------------------
@pytest.mark.parametrize("constraint, removed", [
    (lambda: fa.FixAtoms([0, 1]), 6),
    (lambda: fa.FixLine([0], "z"), 2),
    (lambda: fa.FixPlane([0], "z"), 1),
])
def test_degrees_of_freedom_removed(constraint, removed):
    atoms = cluster()
    atoms.constraints = [constraint()]
    assert atoms.constrained_degrees_of_freedom() == removed


def test_overlapping_constraints_are_not_counted_twice():
    atoms = cluster()
    # A plane inside a frozen atom removes nothing further: three, not four.
    atoms.constraints = [fa.FixAtoms([0]), fa.FixPlane([0], "z")]
    assert atoms.constrained_degrees_of_freedom() == 3
    # Two perpendicular planes leave a line.
    atoms.constraints = [fa.FixPlane([0], "z"), fa.FixPlane([0], "x")]
    assert atoms.constrained_degrees_of_freedom() == 2


# --------------------------------------------------------------------------
# Everything respects them
# --------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ["sd", "cg", "bfgs", "lbfgs", "fire"])
def test_every_optimizer_respects_them(algorithm):
    atoms = cluster()
    start = atoms.positions.copy()
    atoms.constraints = [fa.FixAtoms([0, 1]), fa.FixPlane([3], "z")]

    result = fa.make_optimizer(algorithm, fmax=1e-4, max_steps=3000).run(atoms)

    assert result.converged
    assert atoms.positions[[0, 1]] == pytest.approx(start[[0, 1]])
    assert atoms.positions[3][2] == pytest.approx(start[3][2])
    assert np.linalg.norm(atoms.positions[2] - start[2]) > 1e-3     # something moved


def test_molecular_dynamics_respects_them():
    atoms = cluster()
    start = atoms.positions.copy()
    atoms.constraints = [fa.FixAtoms([0, 1]), fa.FixLine([3], "z")]

    fa.maxwell_boltzmann(atoms, 300.0, seed=1)
    # A frozen atom handed a Maxwell-Boltzmann velocity would wander off.
    velocities = atoms.get_velocities()
    assert velocities[[0, 1]] == pytest.approx(0.0)
    assert velocities[3][:2] == pytest.approx(0.0)

    fa.run_md(atoms, 300, timestep=1.0)

    assert atoms.positions[[0, 1]] == pytest.approx(start[[0, 1]])
    assert atoms.positions[3][:2] == pytest.approx(start[3][:2])
    assert abs(atoms.positions[3][2] - start[3][2]) > 1e-6


def test_the_band_respects_them():
    """The same barrier as ``NEB(fix=...)``, reached through the constraint."""
    cell = [[20.0, 0.0, 0.0], [0.0, 4.5, 0.0], [0.0, 0.0, 4.5]]
    wall = [0.0, 2.25, 2.25]
    initial = fa.Atoms("Ar2", [[-5.0, 0, 0], wall], cell=cell, pbc=[False, True, True])
    final = fa.Atoms("Ar2", [[5.0, 0, 0], wall], cell=cell, pbc=[False, True, True])
    for atoms in (initial, final):
        atoms.calc = fa.LennardJones(epsilon=0.0103, sigma=3.40, cutoff=9.0)
        atoms.constraints = [fa.FixAtoms([1])]

    band = fa.NEB(initial, final, images=7, k=0.5, climb=True)
    for index in range(1, len(band) - 1):
        band.image(index).positions[0] += [0.45 * (-1) ** index, 0.35, 0.2]

    result = band.run("fire", fmax=1e-5, max_steps=8000)
    assert result.converged
    assert band.profile().forward_barrier == pytest.approx(0.122725, abs=1e-4)
    for index in range(len(band)):
        assert band.image(index).positions[1] == pytest.approx(wall)
