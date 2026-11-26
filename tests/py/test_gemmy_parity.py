"""gemmy stage-1 parity: repeat each example (2, 2, 2), then neighbour pairs.

The fastatoms result must match ASE's on the same repeated structure, as sorted
sets of (i, j, S, d). Labels must equal the POSCAR species tokens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import fastatoms
from _util import (
    assert_same_pairs,
    assert_same_structure,
    gemmy_poscars,
    require_format,
    require_neighbors,
    to_ase_structure,
)

ase_io = pytest.importorskip("ase.io")
ase_nl = pytest.importorskip("ase.neighborlist")

CUTOFF = 3.0


def poscar_species(text: str, n_atoms: int) -> list[str]:
    """Per-atom species tokens from a VASP 5 POSCAR (line 6 names, line 7 counts)."""
    lines = text.splitlines()
    names = lines[5].split()
    if all(tok.lstrip("+-").isdigit() for tok in names):
        pytest.skip("VASP 4 POSCAR without species names")
    counts = [int(tok) for tok in lines[6].split()]
    labels = [name for name, count in zip(names, counts, strict=False) for _ in range(count)]
    assert len(labels) == n_atoms
    return labels


_POSCARS: list[Any] = gemmy_poscars() or [pytest.param(None, marks=pytest.mark.skip(reason="no gemmy examples"))]


@pytest.mark.parametrize("poscar", _POSCARS, ids=lambda p: getattr(p, "parent", None) and p.parent.name or "none")
def test_labels_equal_poscar_tokens(poscar: Path) -> None:
    require_format("vasp")
    text = poscar.read_text()
    atoms = fastatoms.read_string(text, "vasp")
    assert atoms.labels == poscar_species(text, len(atoms))


@pytest.mark.parametrize("poscar", _POSCARS, ids=lambda p: getattr(p, "parent", None) and p.parent.name or "none")
def test_repeat_neighbor_list_matches_ase(poscar: Path) -> None:
    require_format("vasp")
    require_neighbors()
    atoms = fastatoms.read_string(poscar.read_text(), "vasp")
    try:
        reference = ase_io.read(poscar, format="vasp")
    except Exception:  # noqa: BLE001 - e.g. species labels such as H1 that ASE cannot map
        # The POSCAR is beyond ASE's reader, so compare the neighbour lists on an
        # ase.Atoms holding the same structure. The label test above still checks the file.
        reference = to_ase_structure(atoms)

    repeated = atoms.repeat((2, 2, 2))
    reference_rep = reference.repeat((2, 2, 2))
    assert len(repeated) == len(reference_rep) == 8 * len(atoms)
    assert_same_structure(repeated, reference_rep, pos_atol=1e-6)

    ours = fastatoms.neighbor_list("ijSd", repeated, CUTOFF)
    expected = ase_nl.neighbor_list("ijSd", reference_rep, CUTOFF)
    assert_same_pairs(ours, expected)
