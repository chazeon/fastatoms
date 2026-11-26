"""Neighbour pairs against ase.neighborlist and vesin, as sets of (i, j, S, d)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

import fastatoms
from _util import (
    assert_same_pairs,
    random_positions,
    random_triclinic_cell,
    require_neighbors,
    to_ase_structure,
)

ase_nl = pytest.importorskip("ase.neighborlist")
vesin = pytest.importorskip("vesin")

pytestmark = pytest.mark.usefixtures("neighbors_available")


@pytest.fixture
def neighbors_available() -> Any:
    require_neighbors()


def check_against_references(atoms: fastatoms.Atoms, cutoff: float) -> None:
    """fastatoms must match ASE and vesin on the same structure."""
    ours = fastatoms.neighbor_list("ijSd", atoms, cutoff)
    ref = to_ase_structure(atoms)
    assert_same_pairs(ours, ase_nl.neighbor_list("ijSd", ref, cutoff))
    assert_same_pairs(ours, vesin.ase_neighbor_list("ijSd", ref, cutoff))


@pytest.mark.parametrize("seed", range(6))
def test_skewed_triclinic(seed: int) -> None:
    rng = np.random.default_rng(1000 + seed)
    cell = random_triclinic_cell(rng, length=(4.0, 7.0), skew=0.9)
    atoms = fastatoms.Atoms(numbers=[6] * 40, positions=random_positions(rng, cell, 40), cell=cell, pbc=True)
    check_against_references(atoms, cutoff=3.5 + 0.2 * seed)


@pytest.mark.parametrize("seed", range(3))
def test_cutoff_larger_than_cell(seed: int) -> None:
    rng = np.random.default_rng(2000 + seed)
    # The cell is much smaller than the cutoff, so several images of each
    # neighbour appear with different shifts S.
    cell = random_triclinic_cell(rng, length=(2.0, 2.5), skew=0.5)
    atoms = fastatoms.Atoms(numbers=[1, 8, 8], positions=random_positions(rng, cell, 3), cell=cell, pbc=True)
    check_against_references(atoms, cutoff=5.5)


@pytest.mark.parametrize("pbc", [(True, False, True), (False, True, False), (True, True, False)])
def test_mixed_pbc(pbc: Any) -> None:
    rng = np.random.default_rng(3000)
    cell = random_triclinic_cell(rng, length=(4.0, 6.0), skew=0.6)
    atoms = fastatoms.Atoms(numbers=[26] * 30, positions=random_positions(rng, cell, 30), cell=cell, pbc=pbc)
    check_against_references(atoms, cutoff=4.0)


def test_molecule_no_pbc_zero_cell() -> None:
    rng = np.random.default_rng(4000)
    positions = rng.uniform(-3.0, 3.0, size=(20, 3))
    atoms = fastatoms.Atoms(numbers=[6, 1] * 10, positions=positions)
    assert not atoms.pbc.any()
    assert np.all(np.asarray(atoms.cell) == 0.0)
    check_against_references(atoms, cutoff=2.5)


def test_self_images_are_returned_only_with_shift() -> None:
    # One atom in a 2 A cubic cell, cutoff 2.5: the only pairs are its six
    # nearest images (i == j, S != 0). The zero-shift self pair is excluded.
    atoms = fastatoms.Atoms(numbers=[1], positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 2.0, pbc=True)
    i, j, S, d = fastatoms.neighbor_list("ijSd", atoms, 2.5)
    assert len(i) == 6
    assert np.all(i == 0) and np.all(j == 0)
    assert np.all(np.abs(S).sum(axis=1) == 1)
    np.testing.assert_allclose(d, 2.0)


def test_tuple_order_follows_request() -> None:
    rng = np.random.default_rng(5000)
    cell = random_triclinic_cell(rng)
    atoms = fastatoms.Atoms(numbers=[6] * 10, positions=random_positions(rng, cell, 10), cell=cell, pbc=True)
    out = fastatoms.neighbor_list("dijS", atoms, 3.0)
    assert len(out) == 4
    i, j, S, d = fastatoms.neighbor_list("ijSd", atoms, 3.0)
    np.testing.assert_array_equal(out[0], d)
    np.testing.assert_array_equal(out[1], i)
    np.testing.assert_array_equal(out[2], j)
    np.testing.assert_array_equal(out[3], S)


def test_random_box_2000_atoms_timing() -> None:
    rng = np.random.default_rng(6000)
    n = 2000
    box = 40.0
    cell = np.eye(3) * box
    atoms = fastatoms.Atoms(numbers=[14] * n, positions=rng.uniform(0.0, box, size=(n, 3)), cell=cell, pbc=True)
    cutoff = 3.0

    start = time.perf_counter()
    ours = fastatoms.neighbor_list("ijSd", atoms, cutoff)
    t_fast = time.perf_counter() - start

    ref = to_ase_structure(atoms)
    start = time.perf_counter()
    expected = ase_nl.neighbor_list("ijSd", ref, cutoff)
    t_ase = time.perf_counter() - start

    start = time.perf_counter()
    vesin_out = vesin.ase_neighbor_list("ijSd", ref, cutoff)
    t_vesin = time.perf_counter() - start

    print(
        f"\n2000-atom box: {len(ours[0])} pairs | fastatoms {t_fast * 1e3:.1f} ms, "
        f"ase {t_ase * 1e3:.1f} ms, vesin {t_vesin * 1e3:.1f} ms"
    )
    assert_same_pairs(ours, expected)
    assert_same_pairs(ours, vesin_out)
