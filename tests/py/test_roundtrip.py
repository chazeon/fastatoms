"""Round trips between fastatoms and ASE, plus the gemmy example structures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import fastatoms
from _util import (
    ase_read_or_xfail,
    assert_same_structure,
    gemmy_extxyz_small,
    gemmy_poscars,
    have_format,
    random_positions,
    random_triclinic_cell,
    require_format,
    to_ase_structure,
)

ase_io = pytest.importorskip("ase.io")

# ASE's CIF reader takes about 13 s on a 3.5k-atom frame, so the CIF round trips
# through ASE are limited to structures of this size or less.
MAX_ASE_CIF_ATOMS = 1000


# Formats that always describe a periodic cell (VASP, CIF and LAMMPS data as
# ASE reads them), so a non-periodic molecule cannot round-trip through them.
PERIODIC_ONLY = {"vasp", "cif", "lammps-data"}


def _skip_unless_comparable(fmt: str, atoms: fastatoms.Atoms) -> None:
    if fmt == "cif" and len(atoms) > MAX_ASE_CIF_ATOMS:
        pytest.skip(f"ASE CIF I/O is too slow for {len(atoms)} atoms (cap {MAX_ASE_CIF_ATOMS})")
    if fmt in PERIODIC_ONLY and not np.asarray(atoms.pbc).all():
        pytest.skip(f"'{fmt}' is always periodic; this structure has pbc={atoms.pbc.tolist()}")


# fastatoms format -> (ASE format, file suffix, extra ASE read kwargs)
ROUNDTRIP_FORMATS = {
    "vasp": ("vasp", ".vasp", {}),
    "extxyz": ("extxyz", ".extxyz", {}),
    "cif": ("cif", ".cif", {}),
    "lammps-data": ("lammps-data", ".data", {"style": "atomic"}),
}


def _synthetic() -> Any:
    rng = np.random.default_rng(7000)
    cell = random_triclinic_cell(rng, length=(5.0, 6.0), skew=0.4)
    return fastatoms.Atoms(
        numbers=[8, 1, 1, 26, 26, 6],
        positions=(random_positions(rng, cell, 6) @ np.linalg.inv(cell) % 1.0) @ cell,
        cell=cell,
        pbc=True,
    )


def _structures() -> Any:
    out = [pytest.param(_synthetic(), id="synthetic")]
    if have_format("vasp"):
        for poscar in gemmy_poscars():
            atoms = fastatoms.read_string(poscar.read_text(), "vasp")
            out.append(pytest.param(atoms, id=f"gemmy-{poscar.parent.name}"))
    if have_format("extxyz"):
        for path in gemmy_extxyz_small():
            frames = fastatoms.read_all(path.read_text(), "extxyz")
            out.append(pytest.param(frames[0], id=f"{path.stem}-first"))
            out.append(pytest.param(frames[-1], id=f"{path.stem}-last"))
    return out


STRUCTURES = _structures()


@pytest.mark.parametrize("fmt", sorted(ROUNDTRIP_FORMATS))
@pytest.mark.parametrize("atoms", STRUCTURES)
def test_fastatoms_write_ase_read(tmp_path: Path, fmt: str, atoms: fastatoms.Atoms) -> None:
    require_format(fmt)
    if not fastatoms.can_write(fmt):
        pytest.skip(f"format '{fmt}' has no writer in this build")
    _skip_unless_comparable(fmt, atoms)
    ase_fmt, suffix, kwargs = ROUNDTRIP_FORMATS[fmt]
    path = tmp_path / f"out{suffix}"
    path.write_text(fastatoms.write_string(atoms, fmt))

    back = ase_read_or_xfail(path, format=ase_fmt, **kwargs)
    assert_same_structure(atoms, back, pos_atol=1e-5)


@pytest.mark.parametrize("fmt", sorted(ROUNDTRIP_FORMATS))
@pytest.mark.parametrize("atoms", STRUCTURES)
def test_ase_write_fastatoms_read(tmp_path: Path, fmt: str, atoms: fastatoms.Atoms) -> None:
    require_format(fmt)
    _skip_unless_comparable(fmt, atoms)
    ase_fmt, suffix, _ = ROUNDTRIP_FORMATS[fmt]
    reference = to_ase_structure(atoms)
    path = tmp_path / f"in{suffix}"
    write_kwargs: dict[str, Any] = {}
    read_kwargs: dict[str, Any] = {}
    if fmt == "lammps-data":
        # ASE writes no Masses section and numbers types alphabetically by symbol.
        write_kwargs["atom_style"] = "atomic"
        read_kwargs["types"] = sorted(set(reference.get_chemical_symbols()))
    ase_io.write(path, reference, format=ase_fmt, **write_kwargs)

    ours = fastatoms.read_string(path.read_text(), fmt, **read_kwargs)
    assert_same_structure(ours, reference, pos_atol=1e-5)


@pytest.mark.parametrize(
    "path",
    gemmy_extxyz_small() or [pytest.param(None, marks=pytest.mark.skip(reason="no gemmy extxyz files"))],
)
def test_gemmy_extxyz_frames_round_trip(tmp_path: Path, path: Path) -> None:
    require_format("extxyz")
    frames = fastatoms.read_all(path.read_text(), "extxyz")
    out = tmp_path / "round.extxyz"
    out.write_text(fastatoms.write_string(frames[-1], "extxyz"))
    ase_back = ase_io.read(out, format="extxyz")
    assert_same_structure(frames[-1], ase_back, pos_atol=1e-5)
