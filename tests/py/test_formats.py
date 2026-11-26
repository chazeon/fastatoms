"""Every fixture under tests/data/<format>/ against ase.io.read."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import fastatoms
from _util import (
    assert_same_structure,
    fixture_files,
    require_format,
)

ase_io = pytest.importorskip("ase.io")

# Fixtures where ASE itself stops early, so ASE is the side that is wrong.
# Each is an xfail with its reason; if ASE starts to agree, the test fails so
# the entry is removed.
ASE_LIMITATIONS = {
    "xyz/multi.xyz": (
        "ASE's extxyz reader stops after the first frame when frames are separated "
        "by a blank line; fastatoms reads every frame"
    ),
    "vasp/XDATCAR_variable": (
        "ASE's vasp-xdatcar reader returns only the first frame of a variable-cell "
        "XDATCAR; fastatoms reads both configurations"
    ),
}

# CIF symmetry expansion: both readers generate the same sites, but the file
# gives fractions to 6 decimals and each reader rounds differently, so sites
# agree to about 4e-6 A here. That is far below any chemical distance.
CIF_POSITION_ATOL = 1e-5


# ---------------------------------------------------------------------------
# Deliberate differences from ASE.
#
# Each entry asserts fastatoms' documented behaviour (not merely an xfail), and
# the reason says why ASE differs. The checker runs instead of the generic
# comparison for that fixture.
# ---------------------------------------------------------------------------


def _check_cif_disorder(text: str, path: Path, fmt: str, ase_fmt: str) -> None:
    # keep_all (the default) keeps every site with its partial occupancy as data.
    sites = fastatoms.read_all(text, "cif")[0]
    assert list(sites.numbers) == [26, 27, 28, 29], "keep_all must keep all four sites"
    # max_occupancy keeps one site per position; on the 0.5/0.5 tie at (0.5, 0.5, 0.5)
    # the first listed site (Ni, Z=28) wins. ASE keeps the last (Cu, Z=29).
    picked = fastatoms.read_all(text, "cif", disorder="max_occupancy")[0]
    assert list(picked.numbers) == [26, 28], "max_occupancy must keep Fe and the first tied site"


def _check_cif_no_atoms(text: str, path: Path, fmt: str, ase_fmt: str) -> None:
    with pytest.raises(fastatoms.FormatError, match="no atoms"):
        fastatoms.read_all(text, fmt)


def _check_cif_no_cell(text: str, path: Path, fmt: str, ase_fmt: str) -> None:
    with pytest.raises(fastatoms.FormatError, match="no cell parameters"):
        fastatoms.read_all(text, fmt)


def _check_general_triclinic_data(text: str, path: Path, fmt: str, ase_fmt: str) -> None:
    """LAMMPS general triclinic: fastatoms subtracts the 'abc origin'; ASE does not.

    ASE records the origin in get_celldisp() and leaves positions unshifted. The
    data reader keeps positions as written (image flags are not applied by
    default), so the view adds image @ cell as in the generic comparison.
    """
    ours = fastatoms.read_all(text, fmt)[0]
    ref = ase_io.read(path, index=-1, format=ase_fmt)
    origin = ref.get_celldisp().reshape(3)
    view = ours.positions + np.asarray(ours.arrays["image"]) @ np.asarray(ours.cell)
    np.testing.assert_array_equal(ours.numbers, ref.numbers)
    np.testing.assert_allclose(view, ref.positions - origin, atol=1e-6)


def _check_dump_type_without_element(text: str, path: Path, fmt: str, ase_fmt: str, *, unscaled: bool) -> None:
    """A dump with a type column and no element column, and no `types` map.

    fastatoms gives element X (Z=0) with the label set to the type number as a
    string. ASE uses Z = type. Positions still agree: unscaled x/y/z columns are
    shifted by the box origin and, by default, unwrapped by the image flags, while
    scaled columns are not shifted at all.
    """
    frames = fastatoms.read_all(text, fmt)
    refs = ase_io.read(path, index=":", format=ase_fmt)
    if not isinstance(refs, list):
        refs = [refs]
    assert len(frames) == len(refs)
    for ours, ref in zip(frames, refs, strict=False):
        types = [int(t) for t in np.asarray(ours.arrays["type"])]
        assert list(ours.numbers) == [0] * len(ours), "no element column and no types: Z=0"
        assert ours.labels == [str(t) for t in types], "label must be the type number"
        assert list(ref.numbers) == types, "ASE takes Z = type"

        if unscaled:
            expected = ref.positions - ref.get_celldisp().reshape(3)
            if "image" in ours.arrays:
                expected = expected + np.asarray(ours.arrays["image"]) @ np.asarray(ours.cell)
        else:
            expected = ref.positions
        np.testing.assert_allclose(ours.positions, expected, atol=1e-6)


DELIBERATE_DIFFERENCES: dict[str, tuple[str, Callable[..., None]]] = {
    "cif/disorder.cif": (
        (
            "keep_all (the default) keeps every site with its occupancy as data; ASE drops "
            "some sites. On a 0.5/0.5 tie max_occupancy keeps the first site; ASE keeps the last"
        ),
        _check_cif_disorder,
    ),
    "cif/no_atoms.cif": (
        "a CIF with no atoms raises FormatError; ASE returns an empty result",
        _check_cif_no_atoms,
    ),
    "cif/no_cell.cif": (
        "a CIF block with no cell parameters raises FormatError; ASE returns a zero cell",
        _check_cif_no_cell,
    ),
    "lammps/general_triclinic.data": (
        (
            "general triclinic 'abc origin' is subtracted from positions; ASE keeps the origin "
            "as celldisp and leaves positions unshifted"
        ),
        _check_general_triclinic_data,
    ),
    "lammps/general_rotated.data": (
        (
            "general triclinic data: image flags are kept in the 'image' array and applied only "
            "with unwrap=true (default false); ASE always applies them. The 'abc origin' is "
            "subtracted, as for general_triclinic.data"
        ),
        _check_general_triclinic_data,
    ),
    "lammps/orthogonal_scaled.lammpstrj": (
        (
            "a dump with a type column but no element column and no 'types' map gives element "
            "X (Z=0) with the type number as label; ASE uses Z = type"
        ),
        partial(_check_dump_type_without_element, unscaled=False),
    ),
    "lammps/unsorted_ids.lammpstrj": (
        (
            "a dump with a type column but no element column and no 'types' map gives element "
            "X (Z=0) with the type number as label; ASE uses Z = type"
        ),
        partial(_check_dump_type_without_element, unscaled=True),
    ),
    "lammps/triclinic_image.lammpstrj": (
        (
            "a tilted dump: the box origin is subtracted (ASE keeps it as celldisp), image flags "
            "are applied by default (unwrap=true; ASE ignores ix/iy/iz for x/y/z columns), and "
            "the element is X (Z=0) with the type number as label; ASE uses Z = type"
        ),
        partial(_check_dump_type_without_element, unscaled=True),
    ),
}


_FIXTURES = fixture_files()
_PARAMS = (
    [pytest.param(d, f, a, p, id=f"{d}/{p.name}") for d, f, a, p in _FIXTURES]
    if _FIXTURES
    else [
        pytest.param(
            None,
            None,
            None,
            None,
            marks=pytest.mark.skip(reason="no structure fixtures under tests/data"),
            id="no-fixtures",
        )
    ]
)


def _ase_view(frame: Any, fmt: str, atol: float) -> Any:
    """The frame in ASE's convention, and the tolerance to compare at.

    The LAMMPS data reader keeps positions as written and records the image flags in
    arrays["image"]; ASE adds image @ cell to the positions. Applying the same
    shift here makes the comparison about the structure, not the convention. The dump
    reader already unwraps by default, so dumps are compared as they are.
    """
    if fmt == "lammps-data" and "image" in frame.arrays:
        view = frame.copy()
        view.positions = frame.positions + np.asarray(frame.arrays["image"]) @ np.asarray(frame.cell)
        return view, atol
    return frame, atol


def _compare(frames: Any, ase_frames: Any, fmt: str, atol: float) -> None:
    assert len(frames) == len(ase_frames), "frame count differs"
    for ours, theirs in zip(frames, ase_frames, strict=False):
        view, tol = _ase_view(ours, fmt, atol)
        assert_same_structure(view, theirs, pos_atol=tol)


@pytest.mark.parametrize("folder,fmt,ase_fmt,path", _PARAMS)
def test_fixture_matches_ase(folder: str, fmt: str, ase_fmt: str, path: Path) -> None:
    require_format(fmt)
    text = path.read_text()
    atol = CIF_POSITION_ATOL if fmt == "cif" else 1e-6
    key = f"{folder}/{path.name}"

    if key in DELIBERATE_DIFFERENCES:
        _reason, check = DELIBERATE_DIFFERENCES[key]
        check(text, path, fmt, ase_fmt)
        return

    try:
        ase_frames = ase_io.read(path, index=":", format=ase_fmt)
    except Exception as exc:  # noqa: BLE001 - ASE rejects some fixtures on purpose
        pytest.xfail(f"beyond ASE: ase.io.read raised {type(exc).__name__}: {exc}")
    if not isinstance(ase_frames, list):
        ase_frames = [ase_frames]

    frames = fastatoms.read_all(text, fmt)
    assert frames, "fastatoms read no frames"

    if key in ASE_LIMITATIONS:
        try:
            _compare(frames, ase_frames, fmt, atol)
        except AssertionError:
            pytest.xfail(ASE_LIMITATIONS[key])
        pytest.fail(f"{key} now matches ASE; remove it from ASE_LIMITATIONS")

    _compare(frames, ase_frames, fmt, atol)
    # read() defaults to the last frame, like ASE; index selects explicitly.
    assert_same_structure(_ase_view(fastatoms.read_string(text, fmt), fmt, atol)[0], ase_frames[-1], pos_atol=atol)
    assert_same_structure(
        _ase_view(fastatoms.read_string(text, fmt, index=0), fmt, atol)[0],
        ase_frames[0],
        pos_atol=atol,
    )


@pytest.mark.parametrize(
    "filename,fmt",
    [
        ("POSCAR", "vasp"),
        ("CONTCAR", "vasp"),
        ("structure.vasp", "vasp"),
        ("structure.extxyz", "extxyz"),
        ("structure.cif", "cif"),
        ("structure.data", "lammps-data"),
        ("structure.lammpstrj", "lammps-dump"),
    ],
)
def test_guess_format_from_name(filename: str, fmt: str) -> None:
    require_format(fmt)
    assert fastatoms.guess_format(filename) == fmt


def test_available_formats_and_can_write_are_consistent() -> None:
    names = fastatoms.available_formats()
    assert isinstance(names, list)
    for name in names:
        assert isinstance(fastatoms.can_write(name), bool)
    assert not fastatoms.can_write("no-such-format")


def test_unknown_format_is_rejected() -> None:
    with pytest.raises(fastatoms.FormatError):
        fastatoms.read_all("", "no-such-format")
