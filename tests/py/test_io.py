"""The ASE-style ``fastatoms.io`` API: input kinds, index forms, gzip, sniffing, write rules."""

from __future__ import annotations

import gzip
import io as stdio
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import fastatoms
import fastatoms.io as fio
from _util import DATA, assert_same_structure, fixture_files, require_format
from test_formats import ASE_LIMITATIONS, DELIBERATE_DIFFERENCES

ase_io = pytest.importorskip("ase.io")

CIF_POSITION_ATOL = 1e-5


def _multi_extxyz(n: int) -> str:
    """n two-atom frames; the first H's z coordinate is the frame number."""
    frames = []
    for i in range(n):
        frames.append(
            f'2\nLattice="4 0 0 0 4 0 0 0 4" Properties=species:S:1:pos:R:3 pbc="T T T"\nH 0 0 {i}\nO 1 1 1\n'
        )
    return "".join(frames)


@pytest.fixture
def traj(tmp_path: Path) -> Path:
    path = tmp_path / "traj.extxyz"
    path.write_text(_multi_extxyz(6))
    return path


def _zs(frames: Any) -> list[float]:
    return [float(frame.positions[0, 2]) for frame in frames]


# --- string2index --------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("-1", -1),
        ("0", 0),
        ("3", 3),
        (":", slice(None, None, None)),
        ("::2", slice(None, None, 2)),
        ("-3:", slice(-3, None, None)),
        ("1:5", slice(1, 5, None)),
        ("1:5:2", slice(1, 5, 2)),
    ],
)
def test_string2index_forms(text: str, expected: Any) -> None:
    assert fio.string2index(text) == expected


@pytest.mark.parametrize("text", ["1:2:3:4", "x", "::0", "1.5"])
def test_string2index_rejects_bad_text(text: str) -> None:
    with pytest.raises(ValueError):
        fio.string2index(text)


# --- input kinds ---------------------------------------------------------------


def test_path_str_and_pathlib_read_the_same(traj: Path) -> None:
    by_str = fastatoms.io.read(str(traj), index=":")
    by_path = fastatoms.io.read(traj, index=":")
    assert _zs(by_str) == _zs(by_path) == [float(i) for i in range(6)]


def test_text_file_object(traj: Path) -> None:
    with open(traj, encoding="utf-8") as handle:
        frames = fastatoms.io.read(handle, index=":")
    assert _zs(frames) == [float(i) for i in range(6)]


def test_binary_file_object_is_decoded_as_utf8(traj: Path) -> None:
    with open(traj, "rb") as handle:
        frames = fastatoms.io.read(handle, index=":")
    assert len(frames) == 6


def test_binary_input_with_bom_is_read(tmp_path: Path) -> None:
    path = tmp_path / "bom.extxyz"
    path.write_bytes(b"\xef\xbb\xbf" + _multi_extxyz(2).encode("utf-8"))
    assert len(fastatoms.io.read(path, index=":")) == 2


def test_bytesio_with_explicit_format(traj: Path) -> None:
    raw = stdio.BytesIO(traj.read_bytes())
    assert len(fastatoms.io.read(raw, format="extxyz", index=":")) == 6


def test_text_string_form_is_read_string() -> None:
    atoms = fastatoms.io.read_string(_multi_extxyz(1), "extxyz")
    assert len(atoms) == 2


# --- index forms ---------------------------------------------------------------


def test_default_index_is_the_last_frame(traj: Path) -> None:
    atoms = fastatoms.io.read(traj)
    assert isinstance(atoms, fastatoms.Atoms)
    assert atoms.positions[0, 2] == pytest.approx(5.0)


@pytest.mark.parametrize("index,expected", [(0, 0), (2, 2), (-1, 5), (-6, 0), ("0", 0), ("-1", 5)])
def test_int_index_gives_one_atoms(traj: Path, index: Any, expected: Any) -> None:
    atoms = fastatoms.io.read(traj, index=index)
    assert isinstance(atoms, fastatoms.Atoms)
    assert atoms.positions[0, 2] == pytest.approx(float(expected))


@pytest.mark.parametrize(
    "index,expected",
    [
        (":", [0, 1, 2, 3, 4, 5]),
        ("::2", [0, 2, 4]),
        ("-3:", [3, 4, 5]),
        ("1:5", [1, 2, 3, 4]),
        ("1:5:2", [1, 3]),
        (slice(1, None), [1, 2, 3, 4, 5]),
        (slice(None, None, -1), [5, 4, 3, 2, 1, 0]),
    ],
)
def test_slice_index_gives_a_list(traj: Path, index: Any, expected: Any) -> None:
    frames = fastatoms.io.read(traj, index=index)
    assert isinstance(frames, list)
    assert _zs(frames) == [float(i) for i in expected]


def test_out_of_range_int_raises(traj: Path) -> None:
    with pytest.raises(IndexError):
        fastatoms.io.read(traj, index=10)


# --- keyword options -----------------------------------------------------------


def test_format_options_pass_through_to_the_core() -> None:
    text_path = DATA / "cif" / "disorder.cif"
    kept = fastatoms.io.read(text_path, index=0)  # default keep_all keeps all four sites
    assert len(kept) == 4
    picked = fastatoms.io.read(text_path, index=0, disorder="max_occupancy")
    assert list(picked.numbers) == [26, 28]


def test_types_option_passes_through() -> None:
    path = DATA / "lammps" / "unsorted_ids.lammpstrj"
    atoms = fastatoms.io.read(path, format="lammps-dump", types={"1": "Fe", "2": "O"})
    assert list(atoms.numbers) == [26, 8, 8]


# --- formats and gzip ----------------------------------------------------------


def test_gzip_path_is_read_transparently(tmp_path: Path) -> None:
    path = tmp_path / "traj.extxyz.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(_multi_extxyz(3))
    frames = fastatoms.io.read(path, index=":")
    assert _zs(frames) == [0.0, 1.0, 2.0]


def test_gzip_name_gives_the_format(tmp_path: Path) -> None:
    path = tmp_path / "POSCAR.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write((DATA / "vasp" / "POSCAR_pv").read_text())
    atoms = fastatoms.io.read(path)
    assert len(atoms) > 0


def test_format_guessed_from_content_when_name_gives_nothing(tmp_path: Path) -> None:
    # The core sniffs the first bytes for CIF, LAMMPS dump and LAMMPS data headers.
    path = tmp_path / "structure.txt"
    path.write_text((DATA / "cif" / "nacl.cif").read_text())
    atoms = fastatoms.io.read(path)
    assert len(atoms) > 0 and atoms.cell is not None


def test_format_guessed_from_content_for_nameless_file_object() -> None:
    raw = stdio.BytesIO((DATA / "lammps" / "unsorted_ids.lammpstrj").read_bytes())
    atoms = fastatoms.io.read(raw)
    assert list(atoms.arrays["id"]) == [1, 2, 3]


def test_unnamed_xyz_stream_is_guessed_from_content() -> None:
    atoms = fastatoms.io.read(stdio.StringIO("1\n\nH 0 0 0\n"))
    assert list(atoms.symbols) == ["H"]
    raw = stdio.BytesIO(b"1\n\nH 0 0 0\n")
    assert list(fastatoms.io.read(raw).symbols) == ["H"]


def test_content_signature_beats_xyz_extension(tmp_path: Path) -> None:
    text = (DATA / "lammps" / "unsorted_ids.lammpstrj").read_text()
    path = tmp_path / "traj.xyz"
    path.write_text(text)
    assert fastatoms._core.guess_format("traj.xyz", text[:4096]) == "lammps-dump"
    atoms = fastatoms.io.read(path)
    assert "id" in atoms.arrays


# --- Latin-1 CIF -----------------------------------------------------------------


def test_latin1_cif_bytes_are_read() -> None:
    body = "# café\n" + (DATA / "cif" / "nacl.cif").read_text(encoding="utf-8")
    raw = body.encode("latin-1")
    atoms = fastatoms.io.read(stdio.BytesIO(raw), format="cif")
    assert len(atoms) > 0
    assert len(fastatoms.io.read_string(raw, "cif")) == len(atoms)


def test_unknown_format_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "mystery.abc"
    path.write_text("hello world\n")
    with pytest.raises(ValueError, match="mystery.abc"):
        fastatoms.io.read(path)


# --- iread ---------------------------------------------------------------------


def test_iread_yields_the_selected_frames(traj: Path) -> None:
    frames = list(fastatoms.io.iread(traj, index="::2"))
    assert _zs(frames) == [0.0, 2.0, 4.0]
    single = list(fastatoms.io.iread(traj, index=-1))
    assert len(single) == 1 and single[0].positions[0, 2] == pytest.approx(5.0)


def test_iread_without_index_yields_every_frame(traj: Path) -> None:
    assert _zs(list(fastatoms.io.iread(traj))) == [float(i) for i in range(6)]
    assert len(list(ase_io.iread(traj))) == 6


def test_iread_two_frame_stream_yields_both_frames() -> None:
    frames = list(fastatoms.io.iread(stdio.StringIO(_multi_extxyz(2)), format="extxyz"))
    assert _zs(frames) == [0.0, 1.0]


# --- filename index suffix (traj.xyz@::2) ----------------------------------------


def test_at_sign_slice_suffix_selects_frames(traj: Path) -> None:
    frames = fastatoms.io.read(f"{traj}@::2")
    assert _zs(frames) == [0.0, 2.0, 4.0]
    assert _zs(list(fastatoms.io.iread(f"{traj}@::2"))) == [0.0, 2.0, 4.0]


def test_at_sign_int_suffix_selects_one_frame(traj: Path) -> None:
    atoms = fastatoms.io.read(f"{traj}@2")
    assert isinstance(atoms, fastatoms.Atoms)
    assert atoms.positions[0, 2] == pytest.approx(2.0)


def test_at_sign_explicit_slice_wins_over_suffix(traj: Path) -> None:
    assert _zs(fastatoms.io.read(f"{traj}@0", index=":2")) == [0.0, 1.0]


def test_do_not_split_by_at_sign_keeps_the_whole_name(traj: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fastatoms.io.read(f"{traj}@::2", do_not_split_by_at_sign=True)


def test_at_sign_that_is_not_an_index_stays_in_the_name(tmp_path: Path) -> None:
    path = tmp_path / "run@v2.extxyz"
    path.write_text(_multi_extxyz(1))
    atoms = fastatoms.io.read(path)
    assert isinstance(atoms, fastatoms.Atoms) and len(atoms) == 2


# --- write ---------------------------------------------------------------------


def _cell_atoms() -> Any:
    return fastatoms.Atoms(
        numbers=[26, 8],
        positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )


def test_write_single_atoms_to_extxyz_path(tmp_path: Path) -> None:
    path = tmp_path / "one.extxyz"
    fastatoms.io.write(path, _cell_atoms())
    back = fastatoms.io.read(path)
    assert list(back.numbers) == [26, 8]


def test_write_list_concatenates_frames_for_extxyz(tmp_path: Path) -> None:
    path = tmp_path / "many.xyz"
    fastatoms.io.write(path, [_cell_atoms(), _cell_atoms(), _cell_atoms()])
    assert len(fastatoms.io.read(path, index=":")) == 3


def test_write_list_to_single_frame_format_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="1 Atoms"):
        fastatoms.io.write(tmp_path / "POSCAR", [_cell_atoms(), _cell_atoms()], format="vasp")


def _shifted_atoms(dz: float) -> Any:
    atoms = _cell_atoms()
    atoms.positions = atoms.positions + np.array([0.0, 0.0, dz])
    return atoms


def test_write_two_cif_frames_reads_back_and_matches_ase(tmp_path: Path) -> None:
    path = tmp_path / "two.cif"
    fastatoms.io.write(path, [_cell_atoms(), _shifted_atoms(0.5)])
    assert path.read_text().count("data_") == 2

    frames = fastatoms.io.read(path, index=slice(None))
    assert len(frames) == 2
    assert frames[1].positions[1, 2] == pytest.approx(1.5, abs=1e-5)

    ase_frames = ase_io.read(path, index=":", format="cif")
    assert len(ase_frames) == 2


def test_write_string_list_for_cif_has_one_block_per_frame() -> None:
    text = fastatoms.io.write_string([_cell_atoms(), _shifted_atoms(0.5)], "cif")
    assert text.count("data_") == 2
    assert len(fastatoms.io.read_string(text, "cif", index=":")) == 2


def test_write_cif_list_of_one_is_a_single_block(tmp_path: Path) -> None:
    text = fastatoms.io.write_string([_cell_atoms()], "cif")
    assert text.count("data_") == 1


def test_write_to_text_file_object_uses_the_given_format() -> None:
    buffer = stdio.StringIO()
    fastatoms.io.write(buffer, _cell_atoms(), format="extxyz")
    assert buffer.getvalue().startswith("2\n")


def test_write_string_needs_a_format() -> None:
    with pytest.raises(ValueError, match="format"):
        # None on purpose: the guard must catch it.
        fastatoms.io.write_string(_cell_atoms(), None)  # type: ignore[arg-type]


def test_write_gzip_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "out.extxyz.gz"
    fastatoms.io.write(path, [_cell_atoms(), _cell_atoms()])
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert handle.read().count("Lattice=") == 2
    assert len(fastatoms.io.read(path, index=":")) == 2


def test_write_string_list_rule() -> None:
    text = fastatoms.io.write_string([_cell_atoms(), _cell_atoms()], "extxyz")
    assert text.count("Lattice=") == 2
    with pytest.raises(ValueError):
        fastatoms.io.write_string([_cell_atoms(), _cell_atoms()], "vasp")


def test_public_aliases_point_at_the_io_module() -> None:
    assert fastatoms.io is fio
    assert fastatoms.read is fio.read
    assert fastatoms.write is fio.write
    assert fastatoms.read_string is fio.read_string
    assert fastatoms.write_string is fio.write_string


def test_read_string_and_write_string_round_trip() -> None:
    require_format("vasp")
    text = (DATA / "vasp" / "POSCAR_pv").read_text()
    atoms = fastatoms.read_string(text, "vasp")
    again = fastatoms.read_string(fastatoms.write_string(atoms, "vasp"), "vasp")
    assert_same_structure(atoms, again, pos_atol=1e-6)


# --- parity with ase.io.read on multi-frame fixtures ----------------------------


def _multi_frame_fixtures() -> Any:
    excluded = set(ASE_LIMITATIONS) | set(DELIBERATE_DIFFERENCES)
    out = []
    for folder, fmt, ase_fmt, path in fixture_files():
        if f"{folder}/{path.name}" in excluded:
            continue
        out.append(pytest.param(fmt, ase_fmt, path, id=f"{folder}/{path.name}"))
    return out


@pytest.mark.parametrize("fmt,ase_fmt,path", _multi_frame_fixtures())
def test_read_all_frames_matches_ase(fmt: str, ase_fmt: str, path: Path) -> None:
    require_format(fmt)
    try:
        ase_frames = ase_io.read(path, index=":", format=ase_fmt)
    except Exception as exc:  # noqa: BLE001 - ASE rejects some fixtures
        pytest.skip(f"ASE cannot read {path.name}: {exc}")
    if not isinstance(ase_frames, list):
        pytest.skip("single structure: no multi-frame parity to check")
    if len(ase_frames) < 2:
        pytest.skip("single structure: no multi-frame parity to check")

    ours = fastatoms.io.read(path, index=slice(None), format=fmt)
    assert len(ours) == len(ase_frames)
    atol = CIF_POSITION_ATOL if fmt == "cif" else 1e-6
    for mine, theirs in zip(ours, ase_frames, strict=False):
        assert_same_structure(mine, theirs, pos_atol=atol)


def test_gemmy_extxyz_frames_match_ase(tmp_path: Path) -> None:
    from _util import gemmy_extxyz_small

    require_format("extxyz")
    paths = gemmy_extxyz_small()
    if not paths:
        pytest.skip("no small gemmy extxyz files")
    path = paths[0]
    ours = fastatoms.io.read(path, index=slice(None))
    theirs = ase_io.read(path, index=":")
    assert len(ours) == len(theirs)
    for mine, ref in zip(ours, theirs, strict=False):
        assert_same_structure(mine, ref, pos_atol=1e-5)
