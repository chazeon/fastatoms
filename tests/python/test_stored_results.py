"""Results computed elsewhere: written to a file, read back, asked for again.

A structure that has been through a calculation carries an energy, forces and
often a stress. Losing them on the way to disk turns a trajectory of results
into a trajectory of geometries, and answering with them after the structure
has moved is worse - a number that is no longer true but looks like one that
is. Both halves are checked here.
"""

import numpy as np
import pytest

import fastatoms as fa


def relaxed_copper():
    atoms = fa.build.fcc("Cu", 3.61, cubic=True)
    atoms.calc = fa.LennardJones(epsilon=0.0103, sigma=3.4, cutoff=8.0)
    atoms.get_potential_energy()
    return atoms


def test_results_survive_a_round_trip(tmp_path):
    atoms = relaxed_copper()
    energy = atoms.get_potential_energy()
    forces = np.asarray(atoms.get_forces())
    stress = np.asarray(atoms.get_stress())

    path = tmp_path / "relaxed.extxyz"
    fa.write(str(path), atoms)
    back = fa.read(str(path))

    assert back.calc is not None
    assert back.get_potential_energy() == pytest.approx(energy)
    assert np.allclose(back.get_forces(), forces)
    assert np.allclose(back.get_stress(), stress, atol=1e-12)


def test_the_file_says_what_it_holds(tmp_path):
    path = tmp_path / "relaxed.extxyz"
    fa.write(str(path), relaxed_copper())
    comment = path.read_text().splitlines()[1]
    assert "energy=" in comment
    assert "stress=" in comment


def test_moving_an_atom_invalidates_stored_results(tmp_path):
    path = tmp_path / "relaxed.extxyz"
    fa.write(str(path), relaxed_copper())
    back = fa.read(str(path))
    back.positions[0] += 0.1
    with pytest.raises(fa.CalculatorError):
        back.get_potential_energy()


def test_reading_can_opt_out(tmp_path):
    path = tmp_path / "relaxed.extxyz"
    fa.write(str(path), relaxed_copper())
    assert fa.read(str(path), results=False).calc is None


def test_every_frame_of_a_trajectory_carries_its_own(tmp_path):
    path = tmp_path / "run.extxyz"
    energies = []
    for shift in (0.0, 0.05, 0.10):
        atoms = relaxed_copper()
        atoms.positions[0] += shift
        energies.append(atoms.get_potential_energy())
        fa.write(str(path), atoms, append=True)

    frames = fa.read_frames(str(path))
    assert len(frames) == 3
    for frame, energy in zip(frames, energies):
        assert frame.get_potential_energy() == pytest.approx(energy)


def test_constructed_by_hand():
    atoms = fa.build.fcc("Cu", 3.61, cubic=True)
    forces = np.zeros((len(atoms), 3))
    atoms.calc = fa.SinglePointCalculator(atoms, energy=-4.2, forces=forces)
    assert atoms.get_potential_energy() == pytest.approx(-4.2)
    assert np.allclose(atoms.get_forces(), 0.0)
    with pytest.raises(fa.PropertyNotImplementedError):
        atoms.get_stress()


def test_a_structure_with_nothing_to_attach_gets_nothing():
    atoms = fa.build.fcc("Cu", 3.61, cubic=True)
    assert fa.attach_stored_results(atoms) is False
    assert atoms.calc is None


# --------------------------------------------------------------------------
# Writing a file whose name promises a format this build does not produce.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name", ["protein.pdb", "run.traj", "atoms.json", "density.cube", "cell.xsf",
             "system.lmp", "box.gro", "CHGCAR"],
)
def test_an_unsupported_format_is_refused_not_silently_mislabelled(tmp_path, name):
    atoms = fa.build.fcc("Cu", 3.61, cubic=True)
    with pytest.raises(fa.ExtxyzError, match="cannot write"):
        fa.write(str(tmp_path / name), atoms)
    assert not (tmp_path / name).exists()


@pytest.mark.parametrize("name", ["out.extxyz", "out.xyz", "dump", "run.out"])
def test_an_unclaimed_name_still_gets_extended_xyz(tmp_path, name):
    atoms = fa.build.fcc("Cu", 3.61, cubic=True)
    fa.write(str(tmp_path / name), atoms)
    assert len(fa.read(str(tmp_path / name))) == len(atoms)


@pytest.mark.parametrize("name", ["structure.cif", "POSCAR", "CONTCAR", "cell.vasp"])
def test_a_supported_format_is_written_as_itself(tmp_path, name):
    # These were refused while nothing here could write them; now that they
    # can be, refusing would be the bug.
    atoms = fa.build.fcc("Cu", 3.61, cubic=True)
    fa.write(str(tmp_path / name), atoms)
    assert len(fa.read(str(tmp_path / name))) == len(atoms)
