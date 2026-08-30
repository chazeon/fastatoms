"""Bulk from an element name, slabs from Miller indices, adsorbates on sites.

The failure mode these guard against is a slab that looks plausible and has
the wrong termination or the wrong spacing - nothing downstream detects that,
so the checks here are on interlayer spacings and layer counts rather than on
"it returned an Atoms".
"""

import numpy as np
import pytest

import fastatoms as fa


def planes(atoms, tolerance=0.1):
    """How many distinct atomic layers a slab has."""
    z = np.sort(np.asarray(atoms.positions)[:, 2])
    count = 1
    for previous, current in zip(z, z[1:]):
        if current - previous > tolerance:
            count += 1
    return count


# ---------------------------------------------------------------- bulk
def test_an_element_knows_its_own_structure():
    assert fa.build.reference_state("Cu")["symmetry"] == "fcc"
    assert fa.build.reference_state("Cu")["a"] == pytest.approx(3.61)
    assert fa.build.reference_state("Ti")["symmetry"] == "hcp"
    assert fa.build.reference_state("Ti")["covera"] == pytest.approx(1.588)


@pytest.mark.parametrize("symbol,structure,natoms", [
    ("Cu", "fcc", 1), ("Fe", "bcc", 1), ("Si", "diamond", 2),
    ("Ti", "hcp", 2), ("Po", "sc", 1), ("Bi", "rhombohedral", 1),
])
def test_bulk_from_a_symbol_alone(symbol, structure, natoms):
    atoms = fa.build.bulk(symbol)
    assert len(atoms) == natoms
    assert list(atoms.pbc) == [True, True, True]
    assert atoms.get_volume() > 0.0
    assert fa.build.reference_state(symbol)["symmetry"] == structure


def test_the_fcc_primitive_volume_is_a_cubed_over_four():
    assert fa.build.bulk("Cu").get_volume() == pytest.approx(3.61**3 / 4)
    assert len(fa.build.bulk("Cu", cubic=True)) == 4


def test_a_lattice_constant_can_be_overridden_on_its_own():
    strained = fa.build.bulk("Cu", a=3.70)
    assert strained.get_volume() == pytest.approx(3.70**3 / 4)


def test_naming_a_structure_overrides_the_table():
    # Copper in a phase copper is not in: a real question, so it is allowed.
    assert len(fa.build.bulk("Cu", "bcc", 2.87)) == 1


def test_a_hexagonal_cell_can_be_asked_for_rectangular():
    hexagonal = fa.build.bulk("Ti")
    rectangular = fa.build.bulk("Ti", orthorhombic=True)
    assert len(rectangular) == 2 * len(hexagonal)
    assert np.allclose(rectangular.get_cell().angles, 90.0)


@pytest.mark.parametrize("symbol,reason", [("N", "molecular"), ("O", "molecular"),
                                           ("Mn", "complex"), ("P", "complex")])
def test_an_unbuildable_element_says_why(symbol, reason):
    with pytest.raises(fa.BuildError, match=reason):
        fa.build.bulk(symbol)


def test_the_binary_lattices_still_take_two_symbols():
    assert len(fa.build.bulk(["Ca", "F"], "fluorite", 5.46, cubic=True)) == 12
    assert len(fa.build.bulk(["Ga", "N"], "wurtzite", 3.19)) == 4
    assert len(fa.build.bulk(["Na", "Cl"], "rocksalt", 5.64, cubic=True)) == 8


# ---------------------------------------------------------------- surfaces
def test_a_general_cut_puts_the_normal_along_z():
    slab = fa.build.surface(fa.build.bulk("Cu"), (2, 1, 1), layers=6, vacuum=8.0)
    cell = np.asarray(slab.cell)
    assert cell[0][1] == pytest.approx(0.0) and cell[0][2] == pytest.approx(0.0)
    assert cell[2][0] == pytest.approx(0.0) and cell[2][1] == pytest.approx(0.0)
    assert list(slab.pbc) == [True, True, False]


def test_vacuum_is_added_on_both_sides():
    slab = fa.build.surface(fa.build.bulk("Cu"), (1, 1, 1), layers=4, vacuum=10.0)
    z = np.asarray(slab.positions)[:, 2]
    thickness = np.asarray(slab.cell)[2][2]
    assert z.min() == pytest.approx(10.0)
    assert thickness - z.max() == pytest.approx(10.0)


def test_no_vacuum_leaves_a_periodic_slab():
    slab = fa.build.surface(fa.build.bulk("Cu"), (1, 1, 1), layers=4, periodic=True)
    assert list(slab.pbc) == [True, True, True]


@pytest.mark.parametrize("indices", [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1), (3, 2, 1)])
def test_a_cut_of_any_plane_keeps_the_bulk_density(indices):
    crystal = fa.build.bulk("Cu")
    slab = fa.build.surface(crystal, indices, layers=4, periodic=True)
    density = len(slab) / slab.get_volume()
    assert density == pytest.approx(len(crystal) / crystal.get_volume(), rel=1e-9)


@pytest.mark.parametrize("face,symbol,spacing", [
    # The interlayer spacing is what says the right plane was cut.
    ("fcc111", "Pt", 3.92 / np.sqrt(3)),
    ("fcc100", "Pt", 3.92 / 2),
    ("fcc110", "Pt", 3.92 / (2 * np.sqrt(2))),
    ("bcc100", "Fe", 2.87 / 2),
    ("bcc110", "Fe", 2.87 / np.sqrt(2)),
])
def test_the_named_faces_have_the_right_interlayer_spacing(face, symbol, spacing):
    slab = getattr(fa.build, face)(symbol, (2, 2, 6), vacuum=10.0)
    z = np.unique(np.round(np.asarray(slab.positions)[:, 2], 6))
    assert np.diff(z).mean() == pytest.approx(spacing, rel=1e-6)


@pytest.mark.parametrize("face,symbol", [("fcc111", "Pt"), ("fcc100", "Pt"), ("bcc110", "Fe"),
                                         ("hcp0001", "Ti"), ("diamond111", "Si")])
@pytest.mark.parametrize("layers", [3, 4, 5])
def test_layers_counts_atomic_planes_on_every_face(face, symbol, layers):
    # A cell that is two planes thick must not make "four layers" mean eight.
    slab = getattr(fa.build, face)(symbol, (2, 2, layers), vacuum=10.0)
    assert planes(slab) == layers
    assert len(slab) == 4 * layers


def test_a_face_the_element_is_not_in_says_so():
    with pytest.raises(fa.BuildError, match="not fcc"):
        fa.build.fcc111("Fe", (2, 2, 3))
    # ...but naming the lattice constant asks the question anyway.
    assert len(fa.build.fcc111("Fe", (2, 2, 3), a=3.6)) == 12


def test_orthogonal_squares_off_a_hexagonal_cell():
    natural = fa.build.fcc111("Pt", (2, 2, 4), vacuum=10.0)
    square = fa.build.fcc111("Pt", (2, 2, 4), orthogonal=True, vacuum=10.0)
    assert np.allclose(square.get_cell().angles, 90.0)
    assert len(square) == 2 * len(natural)


def test_orthogonal_is_a_no_op_on_a_cell_that_is_already_square():
    # fcc(110) reduces to a rectangular surface cell on its own, so asking for
    # a rectangular one should cost nothing.
    slab = fa.build.fcc110("Pt", (2, 2, 4), vacuum=10.0)
    assert np.allclose(slab.get_cell().angles, 90.0)
    assert len(fa.build.fcc110("Pt", (2, 2, 4), orthogonal=True, vacuum=10.0)) == len(slab)


def test_the_surface_cell_comes_back_reduced():
    # The cut can hand back a badly skewed basis for the same lattice; every
    # later repeat and adsorbate placement inherits it, so it is reduced first.
    for face, symbol in (("fcc100", "Pt"), ("fcc110", "Pt")):
        slab = getattr(fa.build, face)(symbol, (1, 1, 3), vacuum=8.0)
        assert np.allclose(slab.get_cell().angles, 90.0)


def test_add_vacuum_adds_rather_than_sets():
    slab = fa.build.fcc111("Pt", (2, 2, 4), vacuum=5.0)
    before = np.asarray(slab.cell)[2][2]
    fa.build.add_vacuum(slab, 4.0)
    assert np.asarray(slab.cell)[2][2] == pytest.approx(before + 4.0)


# ---------------------------------------------------------------- adsorbates
def test_an_adsorbate_sits_at_the_height_asked_for():
    slab = fa.build.fcc111("Pt", (3, 3, 4), vacuum=10.0)
    count = len(slab)
    top = np.asarray(slab.positions)[:, 2].max()

    fa.build.add_adsorbate(slab, fa.Atoms("O", [[0, 0, 0]]), 1.5, "fcc")
    assert len(slab) == count + 1
    assert np.asarray(slab.positions)[-1][2] - top == pytest.approx(1.5)


def test_a_second_adsorbate_is_measured_from_the_slab_not_the_first():
    slab = fa.build.fcc111("Pt", (3, 3, 4), vacuum=10.0)
    top = np.asarray(slab.positions)[:, 2].max()
    fa.build.add_adsorbate(slab, fa.Atoms("O", [[0, 0, 0]]), 3.0, "fcc")
    fa.build.add_adsorbate(slab, fa.Atoms("H", [[0, 0, 0]]), 1.0, "ontop")
    assert np.asarray(slab.positions)[-1][2] - top == pytest.approx(1.0)


def test_the_sites_of_a_close_packed_face_are_distinct():
    positions = {}
    for site in ("ontop", "bridge", "fcc", "hcp"):
        slab = fa.build.fcc111("Pt", (3, 3, 4), vacuum=10.0)
        fa.build.add_adsorbate(slab, fa.Atoms("O", [[0, 0, 0]]), 1.5, site)
        positions[site] = np.asarray(slab.positions)[-1][:2]
    for one in positions:
        for other in positions:
            if one != other:
                assert not np.allclose(positions[one], positions[other])


def test_a_molecule_binds_through_the_atom_named():
    oh = fa.Atoms("OH", [[0, 0, 0], [0, 0, 0.97]])
    slab = fa.build.fcc111("Pt", (3, 3, 4), vacuum=10.0)
    top = np.asarray(slab.positions)[:, 2].max()

    fa.build.add_adsorbate(slab, oh, 2.0, "hcp", mol_index=0)
    z = np.asarray(slab.positions)[:, 2]
    assert z[-2] - top == pytest.approx(2.0)          # the oxygen is at the height
    assert z[-1] - top == pytest.approx(2.97)         # the hydrogen points away


def test_an_adsorbate_can_be_named():
    slab = fa.build.fcc111("Pt", (3, 3, 4), vacuum=10.0)
    count = len(slab)
    fa.build.add_adsorbate(slab, "CO", 1.9, "ontop")
    assert len(slab) == count + 2
    assert set(list(slab.symbols)[count:]) == {"C", "O"}


def test_a_slab_that_does_not_know_its_sites_says_so():
    plain = fa.build.surface(fa.build.bulk("Cu"), (1, 1, 1), layers=4, vacuum=10.0)
    with pytest.raises(fa.BuildError, match="does not know where its sites are"):
        fa.build.add_adsorbate(plain, "CO", 1.9, "ontop")
    # An explicit position always works.
    fa.build.add_adsorbate(plain, "CO", 1.9, (0.0, 0.0))
    assert len(plain) == 6
