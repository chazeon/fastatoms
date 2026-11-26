"""Shared helpers for the ASE-parity suite (imported by the test modules)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import fastatoms

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "tests" / "data"

# gemmy example structures. Set FASTATOMS_GEMMY_DIR to a gemmy checkout
# (the directory holding examples/); the default is a checkout next to this
# repository (../gemmy). Tests that need it skip when it is absent.
GEMMY_DIR = Path(os.environ.get("FASTATOMS_GEMMY_DIR", str(REPO.parent / "gemmy")))


def ase_read_or_xfail(path: Path, **kwargs: Any) -> Any:
    """ase.io.read, or an xfail when ASE rejects the file (e.g. labels such as H1)."""
    import ase.io

    try:
        return ase.io.read(path, **kwargs)
    except Exception as exc:  # noqa: BLE001 - ASE rejects some files on purpose
        pytest.xfail(f"beyond ASE: ase.io.read raised {type(exc).__name__}: {exc}")


def have_format(name: str) -> bool:
    return name in fastatoms.available_formats()


def require_format(name: str) -> None:
    if not have_format(name):
        pytest.skip(f"format '{name}' is not compiled into this build")


def require_neighbors() -> None:
    if not hasattr(fastatoms, "neighbor_list"):
        pytest.skip("neighbour lists not compiled (FASTATOMS_WITH_NEIGHBORS=OFF)")


def fixture_format(folder: str, path: Path) -> tuple[str, str] | None:
    """(fastatoms format, ASE format) for a fixture file, or None to ignore it.

    tests/data holds more than structure files (e.g. neighbour reference text
    and ASE's own ``.ase`` dumps), so the file name decides, not only the folder.
    """
    name, suffix = path.name, path.suffix.lower()
    if folder == "vasp" and not name.startswith("."):
        if name.startswith("XDATCAR"):
            return "xdatcar", "vasp-xdatcar"
        return "vasp", "vasp"
    if folder in ("xyz", "extxyz") and suffix in (".xyz", ".extxyz"):
        return "extxyz", "extxyz"
    if folder == "cif" and suffix == ".cif":
        return "cif", "cif"
    if folder in ("lammps", "lammps_data", "lammps-data") and suffix == ".data":
        return "lammps-data", "lammps-data"
    if folder in ("lammps", "lammps_dump", "lammps-dump") and suffix in (".lammpstrj", ".dump"):
        return "lammps-dump", "lammps-dump-text"
    return None


def fixture_files() -> list[tuple[str, str, str, Path]]:
    """(folder, fastatoms format, ASE format, path) for every structure fixture."""
    found: list[tuple[str, str, str, Path]] = []
    if not DATA.is_dir():
        return found
    for folder in sorted(p for p in DATA.iterdir() if p.is_dir()):
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            rule = fixture_format(folder.name, path)
            if rule is not None:
                found.append((folder.name, *rule, path))
    return found


def gemmy_poscars() -> list[Path]:
    return sorted((GEMMY_DIR / "examples").glob("*/POSCAR"))


def gemmy_extxyz_small(limit: int = 1_000_000) -> list[Path]:
    return sorted(p for p in (GEMMY_DIR / "examples").glob("*/*.extxyz") if p.stat().st_size < limit)


def random_triclinic_cell(rng: np.random.Generator, length: Any = (4.0, 8.0), skew: Any = 0.9) -> np.ndarray:
    """A right-handed, strongly sheared lower-triangular cell (rows = lattice vectors)."""
    lengths = rng.uniform(*length, size=3)
    cell = np.zeros((3, 3))
    cell[0, 0] = lengths[0]
    cell[1, 0] = rng.uniform(-skew, skew) * lengths[0]
    cell[1, 1] = lengths[1]
    cell[2, 0] = rng.uniform(-skew, skew) * lengths[0]
    cell[2, 1] = rng.uniform(-skew, skew) * lengths[1]
    cell[2, 2] = lengths[2]
    return cell


def random_positions(rng: np.random.Generator, cell: np.ndarray, n: int) -> np.ndarray:
    """Fractional coordinates in [-0.5, 1.5) so that wrapping is exercised too."""
    frac = rng.uniform(-0.5, 1.5, size=(n, 3))
    return np.asarray(frac @ cell, dtype=float)


def canonical_pairs(i: Any, j: Any, S: Any, d: Any) -> Any:
    """Sort (i, j, S) pairs so two lists can be compared as sets."""
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    S = np.asarray(S, dtype=np.int64).reshape(-1, 3)
    d = np.asarray(d, dtype=float)
    order = np.lexsort((S[:, 2], S[:, 1], S[:, 0], j, i))
    return i[order], j[order], S[order], d[order]


def assert_same_pairs(actual: Any, expected: Any, atol: float = 1e-8) -> None:
    """Assert two (i, j, S, d) neighbour lists hold the same pairs as sets."""
    ai, aj, aS, ad = canonical_pairs(*actual)
    ei, ej, eS, ed = canonical_pairs(*expected)
    assert len(ai) == len(ei), f"pair count {len(ai)} != reference {len(ei)}"
    np.testing.assert_array_equal(ai, ei, err_msg="i indices differ")
    np.testing.assert_array_equal(aj, ej, err_msg="j indices differ")
    np.testing.assert_array_equal(aS, eS, err_msg="cell shifts differ")
    np.testing.assert_allclose(ad, ed, atol=atol, err_msg="distances differ")


def assert_same_structure(actual: Any, expected: Any, pos_atol: float = 1e-6) -> None:
    """Compare numbers, positions, cell and pbc of an Atoms and an ase.Atoms."""
    np.testing.assert_array_equal(actual.numbers, expected.numbers)
    np.testing.assert_allclose(actual.positions, expected.positions, atol=pos_atol)
    np.testing.assert_allclose(np.asarray(actual.cell), np.asarray(expected.cell), atol=pos_atol)
    np.testing.assert_array_equal(actual.pbc, expected.pbc)


def to_ase_structure(atoms: fastatoms.Atoms) -> Any:
    """An ase.Atoms with the same numbers, positions, cell and pbc."""
    import ase

    return ase.Atoms(
        numbers=atoms.numbers,
        positions=atoms.positions,
        cell=np.asarray(atoms.cell),
        pbc=atoms.pbc,
    )
