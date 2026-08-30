# SPDX-License-Identifier: MIT
"""Tests for the optional ASE bridge.

Skipped in full when ASE is not installed, which is why they live in their own
module: a module-level ``importorskip`` would otherwise take the rest of the
suite with it.
"""

from __future__ import annotations

import numpy as np
import pytest

import fastatoms as fa

pytest.importorskip("ase", reason="the ASE bridge is optional")


def diamond_silicon() -> fa.Atoms:
    basis = [
        [0.00, 0.00, 0.00], [0.00, 0.50, 0.50], [0.50, 0.00, 0.50], [0.50, 0.50, 0.00],
        [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
    ]
    return fa.Atoms("Si8", scaled_positions=basis, cell=5.43, pbc=True)


def test_ase_round_trip_preserves_everything():
    atoms = diamond_silicon()
    atoms.set_charges([0.1] * 8)
    atoms.set_tags(list(range(8)))
    atoms.set_velocities(np.full((8, 3), 0.01))
    atoms.set_array("forces", np.arange(24, dtype=float).reshape(8, 3))

    back = fa.from_ase(fa.to_ase(atoms))
    assert back.get_positions() == pytest.approx(atoms.get_positions())
    assert np.asarray(back.cell) == pytest.approx(np.asarray(atoms.cell))
    assert back.get_momenta() == pytest.approx(atoms.get_momenta())
    assert back.get_charges() == pytest.approx(atoms.get_charges())
    assert back.get_tags().tolist() == atoms.get_tags().tolist()


def test_agrees_with_ase_on_distances():
    atoms = diamond_silicon()
    reference = fa.to_ase(atoms)
    assert atoms.get_all_distances(mic=True) == pytest.approx(
        reference.get_all_distances(mic=True), abs=1e-12
    )


def test_agrees_with_ase_on_the_neighbour_list():
    atoms = diamond_silicon().repeat(2, 2, 2)
    reference = fa.to_ase(atoms)

    from ase.neighborlist import neighbor_list as ase_neighbor_list

    ase_i, ase_j, ase_d = ase_neighbor_list("ijd", reference, 4.0)
    mine = fa.neighbor_list(atoms, 4.0)

    assert len(mine) == len(ase_i)
    # Compare as multisets: the two implementations order pairs differently.
    assert np.allclose(np.sort(mine.distances), np.sort(ase_d))
    assert np.array_equal(np.sort(mine.i), np.sort(ase_i))
    assert np.array_equal(
        np.sort(mine.coordination_numbers()), np.sort(np.bincount(ase_i))
    )
