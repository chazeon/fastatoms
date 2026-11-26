"""fastatoms.symmetry (spglib in C++) against the spglib Python package.

Both sides get the same cell: the atoms' lattice, fractional positions and
numbers. Differences can then only come from the symmetry code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import fastatoms
from _util import DATA, GEMMY_DIR

spglib = pytest.importorskip("spglib")

if not hasattr(fastatoms, "symmetry"):
    pytest.skip("symmetry not compiled (FASTATOMS_WITH_SYMMETRY=OFF)", allow_module_level=True)

sym = fastatoms.symmetry
SYMPREC = 1e-5
POS_ATOL = 1e-5


def _fcc() -> fastatoms.Atoms:
    a = 3.6
    cell = np.eye(3) * a
    frac = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    return fastatoms.Atoms(numbers=[29] * 4, positions=frac @ cell, cell=cell, pbc=True)


def _stishovite() -> fastatoms.Atoms | None:
    path = GEMMY_DIR / "examples" / "stishovite" / "POSCAR"
    if not path.is_file():
        return None
    return fastatoms.io.read(path, format="vasp")


def _cases() -> Any:
    cases = [
        pytest.param(_fcc(), id="fcc"),
        pytest.param(fastatoms.io.read(DATA / "cif" / "rutile.cif"), id="rutile"),
        pytest.param(fastatoms.io.read(DATA / "cif" / "wurtzite.cif"), id="wurtzite"),
        pytest.param(fastatoms.io.read(DATA / "cif" / "monoclinic.cif"), id="monoclinic"),
    ]
    stish = _stishovite()
    if stish is None:
        cases.append(pytest.param(None, id="stishovite", marks=pytest.mark.skip(reason="no gemmy POSCAR")))
    else:
        cases.append(pytest.param(stish, id="stishovite"))
    return cases


CASES = _cases()


def _spglib_cell(atoms: fastatoms.Atoms) -> Any:
    cell = np.asarray(atoms.cell, dtype=float)
    return cell, np.asarray(atoms.scaled_positions(), dtype=float), np.asarray(atoms.numbers)


def _wrapped_diff(a: Any, b: Any) -> np.ndarray:
    """Coordinate differences modulo one lattice vector, so 0.0 and 1.0 agree."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return np.abs(d - np.round(d))


def _ops_as_sets(ours: Any, ref: Any) -> None:
    """Rotations and translations hold the same operations, as unordered sets."""
    ours_r = np.asarray(ours.rotations)
    ours_t = np.asarray(ours.translations)
    ref_r = np.asarray(ref.rotations)
    ref_t = np.asarray(ref.translations)
    assert len(ours_r) == len(ref_r), "operation count differs"
    matched = [False] * len(ref_r)
    for r, t in zip(ours_r, ours_t, strict=False):
        found = False
        for k, (rr, rt) in enumerate(zip(ref_r, ref_t, strict=False)):
            if matched[k] or not np.array_equal(r, rr):
                continue
            if np.all(_wrapped_diff(t, rt) < 1e-4):
                matched[k] = True
                found = True
                break
        assert found, f"operation missing from spglib:\nR=\n{r}\nt={t}"


@pytest.mark.parametrize("atoms", CASES)
def test_dataset_matches_spglib(atoms: fastatoms.Atoms) -> None:
    if atoms is None:
        pytest.skip("no gemmy POSCAR")
    ref = spglib.get_symmetry_dataset(_spglib_cell(atoms), symprec=SYMPREC)
    ours = sym.get_symmetry_dataset(atoms, symprec=SYMPREC)

    assert ours.number == ref.number
    assert ours.international == ref.international
    assert ours.hall_number == ref.hall_number
    assert ours.hall == ref.hall
    assert ours.pointgroup == ref.pointgroup
    assert list(ours.wyckoffs) == list(ref.wyckoffs)
    np.testing.assert_array_equal(ours.equivalent_atoms, ref.equivalent_atoms)
    np.testing.assert_array_equal(ours.crystallographic_orbits, ref.crystallographic_orbits)
    _ops_as_sets(ours, ref)


@pytest.mark.parametrize("atoms", CASES)
def test_standardize_cell_matches_spglib(atoms: fastatoms.Atoms) -> None:
    if atoms is None:
        pytest.skip("no gemmy POSCAR")
    lattice, positions, numbers = spglib.standardize_cell(_spglib_cell(atoms), symprec=SYMPREC)
    ours = sym.standardize_cell(atoms, symprec=SYMPREC)

    assert isinstance(ours, fastatoms.Atoms)
    np.testing.assert_allclose(np.asarray(ours.cell), lattice, atol=POS_ATOL)
    np.testing.assert_array_equal(ours.numbers, numbers)
    assert np.all(_wrapped_diff(ours.scaled_positions(), positions) < POS_ATOL)


@pytest.mark.parametrize("atoms", CASES)
def test_find_primitive_matches_spglib(atoms: fastatoms.Atoms) -> None:
    if atoms is None:
        pytest.skip("no gemmy POSCAR")
    lattice, _positions, numbers = spglib.find_primitive(_spglib_cell(atoms), symprec=SYMPREC)
    ours = sym.find_primitive(atoms, symprec=SYMPREC)
    assert len(ours) == len(numbers)
    assert ours.volume == pytest.approx(abs(np.linalg.det(lattice)), rel=1e-6)
    # standardize_cell(to_primitive=True) is the same cell
    again = sym.standardize_cell(atoms, to_primitive=True, symprec=SYMPREC)
    assert len(again) == len(ours)


@pytest.mark.parametrize("atoms", CASES)
def test_niggli_keeps_volume_and_atoms(atoms: fastatoms.Atoms) -> None:
    if atoms is None:
        pytest.skip("no gemmy POSCAR")
    reduced = sym.niggli_reduce(atoms, symprec=SYMPREC)
    assert len(reduced) == len(atoms)
    assert reduced.volume == pytest.approx(atoms.volume, rel=1e-8)


@pytest.mark.parametrize("atoms", CASES)
def test_analyse_and_number_helpers_agree(atoms: fastatoms.Atoms) -> None:
    if atoms is None:
        pytest.skip("no gemmy POSCAR")
    ref = spglib.get_symmetry_dataset(_spglib_cell(atoms), symprec=SYMPREC)
    group = sym.analyse(atoms, symprec=SYMPREC)
    assert group.number == ref.number
    assert group.international == ref.international
    assert sym.space_group_number(ref.international) == ref.number
    assert sym.space_group_symbol(ref.number) == ref.international


def test_space_group_table_round_trips() -> None:
    for number in range(1, 231):
        assert sym.space_group_number(sym.space_group_symbol(number)) == number


def test_space_group_symbol_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        sym.space_group_symbol(231)


def test_unknown_symbol_gives_zero() -> None:
    assert sym.space_group_number("P 9 9 9") == 0


def test_non_periodic_structure_raises_symmetry_error() -> None:
    atoms = fastatoms.Atoms(numbers=[8, 8], positions=[[0, 0, 0], [1, 1, 1]])
    with pytest.raises(sym.SymmetryError):
        sym.get_symmetry_dataset(atoms)
