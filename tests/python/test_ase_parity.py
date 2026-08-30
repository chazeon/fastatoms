"""Cross-validation against ASE, where a shared file format lets us compare.

These are not tests that FastAtoms behaves like ASE - it deliberately does not
in several places. They are tests that a file written by one is read correctly
by the other, because a format is only worth having if it interoperates.

Skipped when ASE is not installed.
"""

import numpy as np
import pytest

ase_io = pytest.importorskip("ase.io")

import fastatoms as fa


def sorted_rows(values):
    return np.sort(np.asarray(values), axis=0)


@pytest.fixture
def alloy():
    atoms = fa.build.fcc("Cu", 3.61, cubic=True)
    atoms.symbols = ["Cu", "Cu", "Ni", "Ni"]
    return atoms


def test_ase_reads_the_poscar_we_write(tmp_path, alloy):
    path = str(tmp_path / "POSCAR")
    fa.write(path, alloy)
    theirs = ase_io.read(path)

    assert list(theirs.symbols) == list(alloy.symbols)
    assert np.allclose(theirs.cell, alloy.cell)
    assert np.allclose(sorted_rows(theirs.positions), sorted_rows(alloy.positions))


def test_we_read_the_poscar_ase_writes(tmp_path, alloy):
    path = str(tmp_path / "POSCAR")
    ase_io.write(path, ase_io.read(_ase_source(tmp_path, alloy)), format="vasp")
    ours = fa.read(path)

    assert sorted(ours.symbols) == sorted(alloy.symbols)
    assert np.allclose(sorted_rows(ours.positions), sorted_rows(alloy.positions), atol=1e-6)


def _ase_source(tmp_path, atoms):
    path = str(tmp_path / "source.extxyz")
    fa.write(path, atoms)
    return path


def test_ase_reads_the_cif_we_write(tmp_path, alloy):
    path = str(tmp_path / "ours.cif")
    fa.write(path, alloy)
    theirs = ase_io.read(path)

    assert len(theirs) == len(alloy)
    assert np.allclose(sorted_rows(theirs.get_scaled_positions()),
                       sorted_rows(alloy.get_scaled_positions()), atol=1e-6)


def test_we_read_the_cif_ase_writes(tmp_path, alloy):
    path = str(tmp_path / "theirs.cif")
    ase_io.write(path, ase_io.read(_ase_source(tmp_path, alloy)), format="cif")
    ours = fa.read(path)

    assert len(ours) == len(alloy)
    assert sorted(ours.symbols) == sorted(alloy.symbols)


NACL = """data_NaCl
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'F m -3 m'
loop_
_space_group_symop_operation_xyz
 'x, y, z'
 '-x, -y, z'
 '-x, y, -z'
 'x, -y, -z'
 'x, y+1/2, z+1/2'
 '-x, -y+1/2, z+1/2'
 '-x, y+1/2, -z+1/2'
 'x, -y+1/2, -z+1/2'
 'x+1/2, y, z+1/2'
 '-x+1/2, -y, z+1/2'
 'x+1/2, y+1/2, z'
 '-x+1/2, -y+1/2, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
 Na1 Na 0.0 0.0 0.0
 Cl1 Cl 0.5 0.0 0.0
"""


def test_symmetry_expansion_agrees_with_ase(tmp_path):
    # The case that matters: two sites in the asymmetric unit have to become
    # the eight atoms of the conventional cell, on the same sites.
    path = str(tmp_path / "nacl.cif")
    (tmp_path / "nacl.cif").write_text(NACL)

    ours = fa.read(path)
    theirs = ase_io.read(path)

    assert len(ours) == len(theirs) == 8
    assert sorted(ours.symbols) == sorted(theirs.get_chemical_symbols())
    assert np.allclose(sorted_rows(ours.get_scaled_positions()),
                       sorted_rows(theirs.get_scaled_positions()), atol=1e-6)


def test_extxyz_results_survive_a_trip_through_ase(tmp_path):
    atoms = fa.build.fcc("Cu", 3.61, cubic=True)
    atoms.calc = fa.LennardJones(epsilon=0.0103, sigma=3.4, cutoff=8.0)
    energy = atoms.get_potential_energy()

    path = str(tmp_path / "results.extxyz")
    fa.write(path, atoms)
    theirs = ase_io.read(path)
    assert theirs.get_potential_energy() == pytest.approx(energy)
