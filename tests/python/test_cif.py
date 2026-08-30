"""CIF: an asymmetric unit plus the symmetry that generates the rest.

The expansion is the whole job. A CIF read as a plain list of sites gives a
structure missing most of its atoms, and that is a wrong answer that looks
like a right one - so the cases here are mostly about atom counts.
"""

from collections import Counter

import numpy as np
import pytest

import fastatoms as fa

# NaCl in Fm-3m. Two sites in the asymmetric unit; the face centrings and the
# mirrors take them to the eight atoms of the conventional cell.
NACL = """data_NaCl
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
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
_atom_site_occupancy
  Na1 Na 0.0 0.0 0.0 1.0
  Cl1 Cl 0.5 0.0 0.0 1.0
"""


def write(tmp_path, text, name="structure.cif"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_symmetry_is_expanded(tmp_path):
    atoms = fa.read(write(tmp_path, NACL))
    assert len(atoms) == 8
    assert Counter(atoms.symbols) == {"Na": 4, "Cl": 4}
    assert np.allclose(np.asarray(atoms.cell).diagonal(), 5.64)
    assert list(atoms.pbc) == [True, True, True]


def test_atoms_generated_onto_the_same_site_are_merged(tmp_path):
    # Every operation maps the origin to itself, so without merging the single
    # Na site would come back twelve times over.
    atoms = fa.read(write(tmp_path, NACL))
    scaled = np.asarray(atoms.get_scaled_positions())
    separations = np.linalg.norm(scaled[:, None, :] - scaled[None, :, :], axis=-1)
    np.fill_diagonal(separations, 1.0)
    assert separations.min() > 1e-3


def test_round_trip_through_our_own_writer(tmp_path):
    atoms = fa.read(write(tmp_path, NACL))
    path = write(tmp_path, "", "out.cif")
    fa.write(path, atoms)
    back = fa.read(path)

    assert Counter(back.symbols) == Counter(atoms.symbols)
    assert np.allclose(np.sort(back.get_scaled_positions(), axis=0),
                       np.sort(atoms.get_scaled_positions(), axis=0), atol=1e-6)
    assert np.allclose(back.cell, atoms.cell)


def test_we_write_p1_and_say_so(tmp_path):
    atoms = fa.read(write(tmp_path, NACL))
    path = write(tmp_path, "", "out.cif")
    fa.write(path, atoms)
    text = (tmp_path / "out.cif").read_text()
    assert "'P 1'" in text
    assert text.count("_atom_site_label") == 1
    # Every atom listed, since P1 generates none of them.
    assert len([line for line in text.splitlines() if line.strip().startswith(("Na", "Cl"))]) == 8


def test_a_file_with_no_operations_is_refused_not_under_reported(tmp_path):
    text = NACL.replace(
        "loop_\n_space_group_symop_operation_xyz", "_space_group_name_H-M_alt 'F m -3 m'\nloop_\n_unused"
    )
    with pytest.raises(fa.ExtxyzError, match="drop every atom"):
        fa.read(write(tmp_path, text))


def test_p1_needs_no_operation_list(tmp_path):
    text = """data_one
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
  Cu1 0.0 0.0 0.0
  Cu2 0.5 0.5 0.5
"""
    atoms = fa.read(write(tmp_path, text))
    assert list(atoms.symbols) == ["Cu", "Cu"]


def test_standard_uncertainties_are_stripped(tmp_path):
    # Published CIFs write measured numbers as "5.6402(3)".
    text = NACL.replace("_cell_length_a 5.64", "_cell_length_a 5.6402(3)")
    atoms = fa.read(write(tmp_path, text))
    assert np.asarray(atoms.cell)[0][0] == pytest.approx(5.6402)


def test_a_two_letter_element_is_not_read_as_one(tmp_path):
    text = NACL.replace("  Cl1 Cl 0.5", "  Cl1 Cl 0.5")
    atoms = fa.read(write(tmp_path, text))
    assert "Cl" in set(atoms.symbols)
    assert "C" not in set(atoms.symbols)


def test_the_element_can_come_from_the_label_alone(tmp_path):
    text = NACL.replace("_atom_site_type_symbol\n", "")
    text = text.replace("  Na1 Na 0.0", "  Na1 0.0").replace("  Cl1 Cl 0.5", "  Cl1 0.5")
    atoms = fa.read(write(tmp_path, text))
    assert Counter(atoms.symbols) == {"Na": 4, "Cl": 4}


def test_a_triclinic_cell_comes_back_with_its_angles(tmp_path):
    text = NACL.replace("_cell_angle_alpha 90", "_cell_angle_alpha 80")
    text = text.replace("_cell_angle_beta 90", "_cell_angle_beta 85")
    text = text.replace("_cell_angle_gamma 90", "_cell_angle_gamma 95")
    atoms = fa.read(write(tmp_path, text))
    assert np.allclose(atoms.get_cell().angles, [80.0, 85.0, 95.0], atol=1e-6)


def test_several_data_blocks_are_several_structures(tmp_path):
    doubled = NACL + "\n" + NACL.replace("data_NaCl", "data_NaCl_again")
    frames = fa.read_frames(write(tmp_path, doubled))
    assert len(frames) == 2
    assert all(len(frame) == 8 for frame in frames)
