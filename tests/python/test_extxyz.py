# SPDX-License-Identifier: MIT
"""Extended XYZ, the default format, from Python.

The C++ suite carries the parsing detail. What is checked here is that the
format is reachable and that `read`/`write` mean what they now mean: a
structure, in extxyz unless the extension says otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

import fastatoms as fa


def water() -> fa.Atoms:
    atoms = fa.Atoms("OH2", [[0.0, 0.0, 0.0], [0.958, 0.0, 0.0], [-0.24, 0.927, 0.0]],
                     cell=[10.0, 11.0, 12.0], pbc=True)
    return atoms


FRAME = """3
Lattice="10 0 0 0 11 0 0 0 12" Properties=species:S:1:pos:R:3:forces:R:3 pbc="T T F" energy=-1.5
O  0.0   0.0   0.0    0.1 -0.2  0.3
H  0.958 0.0   0.0   -0.05 0.1  0.0
H -0.24  0.927 0.0   -0.05 0.1  0.0
"""


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def test_a_frame_carries_everything_the_comment_line_says():
    atoms = fa.parse_extxyz(FRAME)
    assert len(atoms) == 3
    assert atoms.get_chemical_symbols() == ["O", "H", "H"]
    assert atoms.cell.lengths == pytest.approx([10.0, 11.0, 12.0])
    assert atoms.pbc.tolist() == [True, True, False]
    assert atoms.info["energy"] == pytest.approx(-1.5)
    assert atoms.arrays["forces"][0] == pytest.approx([0.1, -0.2, 0.3])


def test_a_plain_xyz_file_still_reads():
    atoms = fa.parse_extxyz("2\nfree-form comment\nH 0 0 0\nH 0 0 0.74\n")
    assert len(atoms) == 2
    assert atoms.positions[1][2] == pytest.approx(0.74)


def test_reading_a_bad_frame_says_where():
    with pytest.raises(Exception, match="line"):
        fa.parse_extxyz("2\nProperties=species:S:1:pos:R:3\nH 0 0 0\nH 0 0 x\n")


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def test_round_trip_is_exact(tmp_path):
    atoms = water()
    atoms.set_array("forces", [[1e-17, 2.5e8, -0.5]] * 3)
    atoms.info["energy"] = -1.2345678901234567

    path = tmp_path / "s.extxyz"
    fa.write(str(path), atoms)
    back = fa.read(str(path))

    # The default writes the shortest text that reads back as the same double,
    # so this is equality and not approximation.
    assert back.positions.tolist() == atoms.positions.tolist()
    assert back.arrays["forces"].tolist() == atoms.arrays["forces"].tolist()
    assert back.info["energy"] == atoms.info["energy"]


def test_precision_can_be_traded_for_size(tmp_path):
    # Coordinates that actually need the digits: water's 0.958 is already
    # shorter than four significant figures, so it could not show a difference.
    rng = np.random.default_rng(0)
    atoms = fa.Atoms("Ar64", positions=rng.random((64, 3)) * 10.0, cell=10.0, pbc=True)

    exact = tmp_path / "exact.extxyz"
    small = tmp_path / "small.extxyz"
    fa.write(str(exact), atoms)
    fa.write(str(small), atoms, precision=4)

    assert small.stat().st_size < exact.stat().st_size
    # Exact by default, approximate when asked.
    assert fa.read(str(exact)).positions.tolist() == atoms.positions.tolist()
    assert fa.read(str(small)).positions == pytest.approx(atoms.positions, abs=1e-2)


def test_append_builds_a_trajectory(tmp_path):
    path = str(tmp_path / "md.extxyz")
    for step in range(4):
        atoms = water()
        atoms.positions[0] += [0.1 * step, 0, 0]
        atoms.info["step"] = step
        fa.write(path, atoms, append=(step > 0))

    frames = fa.read_frames(path)
    assert len(frames) == 4
    assert [f.info["step"] for f in frames] == [0, 1, 2, 3]
    # A negative index counts from the end, and the default read is the last.
    assert fa.read(path).info["step"] == 3
    assert fa.read(path, index=0).info["step"] == 0
    assert fa.read(path, index=-2).info["step"] == 2


def test_append_helper(tmp_path):
    path = str(tmp_path / "t.extxyz")
    fa.write(path, water())
    fa.append(path, water())
    assert len(fa.read_frames(path)) == 2


def test_a_list_of_frames_writes_in_one_call(tmp_path):
    path = str(tmp_path / "all.extxyz")
    fa.write(path, [water(), water(), water()])
    assert len(fa.read_frames(path)) == 3


def test_choosing_what_to_write(tmp_path):
    atoms = water()
    atoms.set_array("forces", [[1.0, 0.0, 0.0]] * 3)
    atoms.set_velocities([[0.5, 0.0, 0.0]] * 3)

    text = fa.to_extxyz(atoms)
    assert "forces:R:3" in text and "velo:R:3" in text
    assert text.index("velo:R:3") < text.index("forces:R:3")

    bare = fa.to_extxyz(atoms, forces=False, velocities=False)
    assert "forces" not in bare
    assert "Properties=species:S:1:pos:R:3" in bare

    with pytest.raises(Exception, match="unknown option"):
        fa.to_extxyz(atoms, nonsense=True)


# --------------------------------------------------------------------------
# The default format
# --------------------------------------------------------------------------
def test_the_extension_decides_and_extxyz_is_the_default(tmp_path):
    atoms = water()

    xyz = tmp_path / "s.extxyz"
    fa.write(str(xyz), atoms)
    assert xyz.read_text().splitlines()[0].strip() == "3"

    yaml = tmp_path / "s.yaml"
    fa.write(str(yaml), atoms)
    assert "structure" in yaml.read_text()
    assert len(fa.read(str(yaml))) == 3

    # A file with no recognised extension is extended XYZ, which is what
    # "the default format" has to mean to be worth saying.
    plain = tmp_path / "dump"
    fa.write(str(plain), atoms)
    assert len(fa.read(str(plain))) == 3

    # YAML holds one structure, so a trajectory is an error rather than a
    # silent loss of every frame but one.
    with pytest.raises(Exception, match="cannot be written"):
        fa.write(str(yaml), [atoms, atoms])


def test_read_input_still_loads_a_run(tmp_path):
    """extxyz has nowhere to say which calculator to attach, so a deck is YAML."""
    deck = fa.read_input_string("""
structure:
  cell: {a: 5.3, b: 5.3, c: 5.3}
  pbc: true
  atoms:
    - {symbol: Ar, position: [0.0, 0.0, 0.0]}
simulation:
  task: md
  calculator: {name: lennard_jones}
  md: {steps: 5, timestep: 5.0}
""")
    assert deck.simulation.task == fa.Task.md
    assert len(deck.atoms) == 1
    assert deck.make_calculator().name == "lennard_jones"


def test_md_writes_a_readable_trajectory(tmp_path):
    """The point of the format: a run leaves something that reads straight back."""
    atoms = fa.Atoms("Ar4", scaled_positions=[[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]],
                     cell=5.3, pbc=True)
    atoms.calc = fa.LennardJones()
    path = str(tmp_path / "run.extxyz")
    fa.run_md(atoms, 20, timestep=5.0, temperature=60.0, initialize_velocities=True,
              seed=1, trajectory=path, interval=5)

    frames = fa.read_frames(path)
    assert len(frames) == 5                       # step 0 and every fifth
    assert all(len(frame) == 4 for frame in frames)
    for frame in frames:
        assert "energy" in frame.info
        assert "temperature" in frame.info
        assert frame.has_array("forces")
        assert frame.get_velocities().shape == (4, 3)
    assert [f.info["step"] for f in frames] == [0, 5, 10, 15, 20]
