"""VASP's own formats: POSCAR, CONTCAR, XDATCAR, OUTCAR.

Most plane-wave work arrives as one of these, so reading them is what lets a
finished calculation be picked up without a conversion step. The OUTCAR case
carries results as well as structures, and those come back attached the same
way an extended XYZ frame's do.
"""

import numpy as np
import pytest

import fastatoms as fa


def alloy():
    atoms = fa.build.fcc("Cu", 3.61, cubic=True)
    atoms.symbols = ["Cu", "Cu", "Ni", "Ni"]
    return atoms


def test_poscar_round_trip(tmp_path):
    atoms = alloy()
    path = tmp_path / "POSCAR"
    fa.write(str(path), atoms)
    back = fa.read(str(path))

    assert list(back.symbols) == list(atoms.symbols)
    assert np.allclose(back.cell, atoms.cell)
    assert np.allclose(np.sort(back.positions, axis=0), np.sort(atoms.positions, axis=0))
    assert list(back.pbc) == [True, True, True]


def test_poscar_groups_by_species(tmp_path):
    # VASP counts atoms per species, so the file has to be grouped even when
    # the structure in memory is not.
    atoms = fa.build.fcc("Cu", 3.61, cubic=True)
    atoms.symbols = ["Cu", "Ni", "Cu", "Ni"]
    path = tmp_path / "POSCAR"
    fa.write(str(path), atoms)

    lines = path.read_text().splitlines()
    assert lines[5].split() == ["Cu", "Ni"]
    assert lines[6].split() == ["2", "2"]
    assert list(fa.read(str(path)).symbols) == ["Cu", "Cu", "Ni", "Ni"]


def test_selective_dynamics_survives(tmp_path):
    atoms = alloy()
    atoms.add_constraint(fa.FixAtoms([0, 1]))
    path = tmp_path / "POSCAR"
    fa.write(str(path), atoms)
    assert "Selective dynamics" in path.read_text()

    back = fa.read(str(path))
    assert [c.name for c in back.constraints] == ["FixAtoms"]
    assert list(back.constraints[0].indices) == [0, 1]


def test_cartesian_coordinates_are_read(tmp_path):
    path = tmp_path / "POSCAR"
    path.write_text(
        "Cu\n1.0\n"
        "3.61 0.0 0.0\n0.0 3.61 0.0\n0.0 0.0 3.61\n"
        "Cu\n1\nCartesian\n"
        "1.805 1.805 0.0\n"
    )
    atoms = fa.read(str(path))
    assert np.allclose(atoms.positions, [[1.805, 1.805, 0.0]])


def test_a_negative_scale_is_a_target_volume(tmp_path):
    # VASP reads a negative scale as the volume the cell should end up with.
    path = tmp_path / "POSCAR"
    path.write_text(
        "Cu\n-8.0\n"
        "1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n"
        "Cu\n1\nDirect\n0.0 0.0 0.0\n"
    )
    atoms = fa.read(str(path))
    assert atoms.get_volume() == pytest.approx(8.0)


def test_a_vasp4_file_without_species_says_what_is_wrong(tmp_path):
    path = tmp_path / "POSCAR"
    path.write_text(
        "some run\n1.0\n"
        "3.61 0.0 0.0\n0.0 3.61 0.0\n0.0 0.0 3.61\n"
        "1\nDirect\n0.0 0.0 0.0\n"
    )
    with pytest.raises(fa.ExtxyzError, match="POTCAR"):
        fa.read(str(path))


def test_a_vasp4_file_naming_species_in_its_comment_is_read(tmp_path):
    path = tmp_path / "POSCAR"
    path.write_text(
        "Cu Ni\n1.0\n"
        "3.61 0.0 0.0\n0.0 3.61 0.0\n0.0 0.0 3.61\n"
        "1 1\nDirect\n0.0 0.0 0.0\n0.5 0.5 0.5\n"
    )
    assert list(fa.read(str(path)).symbols) == ["Cu", "Ni"]


XDATCAR = """unknown system
           1
     3.610000     0.000000     0.000000
     0.000000     3.610000     0.000000
     0.000000     0.000000     3.610000
   Cu
     2
Direct configuration=     1
  0.00000000  0.00000000  0.00000000
  0.50000000  0.50000000  0.00000000
Direct configuration=     2
  0.01000000  0.00000000  0.00000000
  0.50000000  0.50000000  0.00000000
Direct configuration=     3
  0.02000000  0.00000000  0.00000000
  0.50000000  0.50000000  0.00000000
"""


def test_xdatcar_is_a_trajectory(tmp_path):
    path = tmp_path / "XDATCAR"
    path.write_text(XDATCAR)

    frames = fa.read_frames(str(path))
    assert len(frames) == 3
    assert all(list(frame.symbols) == ["Cu", "Cu"] for frame in frames)
    assert frames[2].positions[0][0] == pytest.approx(0.02 * 3.61)

    # A bare read gives the last frame, as it does for any trajectory.
    assert np.allclose(fa.read(str(path)).positions, frames[-1].positions)
    assert np.allclose(fa.read(str(path), index=0).positions, frames[0].positions)


OUTCAR = """ vasp.6.3.0
   VRHFIN =Cu: d10 p1
   ions per type =               2
 direct lattice vectors                 reciprocal lattice vectors
     3.610000000  0.000000000  0.000000000     0.277008310  0.000000000  0.000000000
     0.000000000  3.610000000  0.000000000     0.000000000  0.277008310  0.000000000
     0.000000000  0.000000000  3.610000000     0.000000000  0.000000000  0.277008310

  in kB      -10.00000    -10.00000    -10.00000      0.00000      0.00000      0.00000

 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000      0.00000      0.00000         0.100000      0.000000      0.000000
      1.80500      1.80500      0.00000        -0.100000      0.000000      0.000000
 -----------------------------------------------------------------------------------

  free  energy   TOTEN  =        -7.12345678 eV
  energy  without entropy=       -7.10000000  energy(sigma->0) =       -7.11000000

  in kB       -5.00000     -5.00000     -5.00000      0.00000      0.00000      0.00000

 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.01000      0.00000      0.00000         0.010000      0.000000      0.000000
      1.80500      1.80500      0.00000        -0.010000      0.000000      0.000000
 -----------------------------------------------------------------------------------

  free  energy   TOTEN  =        -7.20000000 eV
  energy  without entropy=       -7.19000000  energy(sigma->0) =       -7.19500000
"""


def test_outcar_carries_structures_and_their_results(tmp_path):
    path = tmp_path / "OUTCAR"
    path.write_text(OUTCAR)

    frames = fa.read_frames(str(path))
    assert len(frames) == 2
    assert list(frames[0].symbols) == ["Cu", "Cu"]
    assert np.allclose(frames[0].cell, np.eye(3) * 3.61)

    assert frames[0].get_potential_energy() == pytest.approx(-7.11)
    assert frames[1].get_potential_energy() == pytest.approx(-7.195)
    assert np.allclose(frames[0].get_forces()[0], [0.1, 0.0, 0.0])
    assert np.allclose(frames[1].positions[0], [0.01, 0.0, 0.0])


def test_outcar_stress_is_converted_to_this_sign_and_unit(tmp_path):
    path = tmp_path / "OUTCAR"
    path.write_text(OUTCAR)
    frames = fa.read_frames(str(path))

    # VASP reports kilobar with the opposite sign to the one used here, so
    # -10 kB becomes +1 GPa in eV/A^3.
    assert np.allclose(frames[0].get_stress()[:3], 1.0 * fa.units.GPa)
    assert np.allclose(frames[0].get_stress()[3:], 0.0)
    assert np.allclose(frames[1].get_stress()[:3], 0.5 * fa.units.GPa)


def test_a_vasp_output_is_not_something_to_write(tmp_path):
    with pytest.raises(fa.ExtxyzError, match="not input"):
        fa.write(str(tmp_path / "OUTCAR"), alloy())
