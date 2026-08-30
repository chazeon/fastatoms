# SPDX-License-Identifier: MIT
"""Crystallographic symmetry from Python.

The C++ suite checks the mathematics. What is checked here is the Python
surface and the two promises that matter: a structure's space group does not
depend on the cell it was handed in, and reducing then rebuilding a cell gets
back where it started.
"""

from __future__ import annotations

import numpy as np
import pytest

import fastatoms as fa

build = fa.build
symmetry = fa.symmetry


#: Structures whose space group is in every textbook, so nothing here is
#: checking this code against itself.
KNOWN = [
    ("Pm-3m", 221, build.sc("Po", 3.35)),
    ("Fm-3m", 225, build.fcc("Cu", 3.61)),
    ("Im-3m", 229, build.bcc("Fe", 2.87)),
    ("Fd-3m", 227, build.diamond("Si", 5.43)),
    ("Fm-3m", 225, build.rocksalt("Na", "Cl", 5.64)),
    ("F-43m", 216, build.zincblende("Ga", "As", 5.65)),
    ("Pm-3m", 221, build.cscl("Cs", "Cl", 4.12)),
    ("P6_3/mmc", 194, build.hcp("Mg", 3.21)),
    ("P6/mmm", 191, build.graphene()),
    ("P-6m2", 187, build.tmd("Mo", "S")),
    ("P-3m1", 164, build.tmd("Mo", "S", phase="1T")),
    ("P-6m2", 187, build.honeycomb(["B", "N"], 2.51)),
]


@pytest.mark.parametrize("symbol, number, atoms", KNOWN,
                         ids=[f"{s}-{n}" for s, n, _ in KNOWN])
def test_the_known_space_groups(symbol, number, atoms):
    group = symmetry.analyse(atoms)
    assert group.hermann_mauguin == symbol
    assert group.number == number


def test_the_answer_does_not_depend_on_the_setting():
    """Primitive, conventional and supercell are the same crystal."""
    primitive = build.diamond("Si", 5.43)
    for description in (primitive,
                        build.diamond("Si", 5.43, cubic=True),
                        primitive.repeat(2, 1, 1),
                        primitive.repeat(2, 2, 2)):
        group = symmetry.analyse(description)
        assert group.number == 227
        assert group.primitive_atoms == 2
        assert group.conventional_atoms == 8


def test_atoms_can_be_asked_directly():
    conventional = build.diamond("Si", 5.43, cubic=True)
    assert conventional.get_space_group().hermann_mauguin == "Fd-3m"
    assert len(conventional.get_primitive_cell()) == 2
    assert len(conventional.get_primitive_cell().get_conventional_cell()) == 8


def test_a_space_group_reports_itself():
    group = symmetry.analyse(build.fcc("Cu", 3.61))
    assert group.point_group.symbol == "m-3m"
    assert group.point_group.schoenflies == "Oh"
    assert group.point_group.order == 48
    assert group.point_group.centrosymmetric
    assert group.centring == symmetry.Centring.F
    assert group.system == symmetry.CrystalSystem.cubic
    assert group.order == 48
    assert len(group) == 48
    assert group.schoenflies == "Oh-5"
    assert group.hermann_mauguin_full == "F 4/m -3 2/m"
    assert "Fm-3m" in group.summary()
    assert len(group.operations) == 48


def test_screw_axes_and_glides_are_what_separate_these_groups():
    """The point group cannot tell hcp from graphene, or diamond from copper."""
    hcp = symmetry.analyse(build.hcp("Mg", 3.21))
    graphene = symmetry.analyse(build.graphene())
    assert hcp.point_group.symbol == graphene.point_group.symbol == "6/mmm"
    assert hcp.hermann_mauguin == "P6_3/mmc"      # the 6_3 screw
    assert graphene.hermann_mauguin == "P6/mmm"

    diamond = symmetry.analyse(build.diamond("Si", 5.43))
    copper = symmetry.analyse(build.fcc("Cu", 3.61))
    assert diamond.point_group.symbol == copper.point_group.symbol == "m-3m"
    assert diamond.hermann_mauguin == "Fd-3m"     # the d glide
    assert copper.hermann_mauguin == "Fm-3m"


# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------
@pytest.mark.parametrize("atoms, primitive, conventional", [
    (build.fcc("Al", 4.05, cubic=True), 1, 4),
    (build.bcc("Fe", 2.87, cubic=True), 1, 2),
    (build.diamond("Si", 5.43, cubic=True), 2, 8),
    (build.rocksalt("Na", "Cl", 5.64, cubic=True), 2, 8),
    (build.sc("Po", 3.35), 1, 1),
])
def test_reducing_and_rebuilding(atoms, primitive, conventional):
    reduced = symmetry.primitive_cell(atoms)
    assert len(reduced) == primitive
    # Density is what cannot change when the same crystal is redescribed.
    assert len(reduced) / reduced.get_volume() == pytest.approx(
        len(atoms) / atoms.get_volume())

    grown = symmetry.conventional_cell(reduced)
    assert len(grown) == conventional
    assert len(grown) / grown.get_volume() == pytest.approx(
        len(reduced) / reduced.get_volume())


def test_a_supercell_reduces_to_what_it_repeats():
    unit = build.diamond("Si", 5.43)
    reduced = symmetry.primitive_cell(unit.repeat(2, 2, 2))
    assert len(reduced) == 2
    assert reduced.get_volume() == pytest.approx(unit.get_volume())


def test_the_cell_analysis_says_what_it_found():
    analysis = symmetry.analyse_cell(build.fcc("Al", 4.05, cubic=True))
    assert analysis.input_multiplicity == 4
    assert analysis.primitive_atoms == 1
    assert analysis.conventional_atoms == 4
    assert analysis.centring == symmetry.Centring.F
    assert analysis.point_group.symbol == "m-3m"


def test_niggli_reduction_undoes_a_skew():
    atoms = build.sc("Po", 3.0)
    skew = np.array([[1, 0, 0], [3, 1, 0], [-5, 4, 1]], dtype=float)
    atoms.cell = skew @ np.asarray(atoms.cell)

    reduced = symmetry.niggli_reduce(atoms)
    assert list(reduced.cell.lengths) == pytest.approx([3.0, 3.0, 3.0])
    assert reduced.get_volume() == pytest.approx(27.0)


# --------------------------------------------------------------------------
# Wyckoff positions and point groups
# --------------------------------------------------------------------------
def test_wyckoff_orbits():
    salt = symmetry.analyse(build.rocksalt("Na", "Cl", 5.64))
    assert len(salt.wyckoff) == 2
    assert [w.letter for w in salt.wyckoff] == ["a", "b"]
    for position in salt.wyckoff:
        assert position.multiplicity == 4
        assert position.site_symmetry == "m-3m"

    # The two atoms of the diamond primitive cell are one orbit: symmetry
    # relates them even though no translation does.
    diamond = symmetry.analyse(build.diamond("Si", 5.43))
    assert len(diamond.wyckoff) == 1
    assert diamond.wyckoff[0].multiplicity == 8
    assert diamond.wyckoff[0].site_symmetry == "-43m"
    assert len(diamond.wyckoff[0].atoms) == 2


def test_the_thirty_two_point_groups():
    groups = symmetry.point_groups()
    assert len(groups) == 32
    assert len({g.symbol for g in groups}) == 32
    assert symmetry.point_group(build.fcc("Cu", 3.61)).symbol == "m-3m"
    assert symmetry.point_group(build.tmd("Mo", "S")).symbol == "-6m2"


def test_the_table_round_trips():
    table = symmetry.space_group_table()
    assert len(table) == 230
    assert table[0] == (1, "P1")
    assert table[229] == (230, "Ia-3d")
    for number, symbol in table:
        assert symmetry.space_group_symbol(number) == symbol
        assert symmetry.space_group_number(symbol) == number
    assert symmetry.space_group_number("not a group") == 0


def test_bad_input_is_refused():
    with pytest.raises(fa.SymmetryError):
        symmetry.analyse(fa.Atoms())
    with pytest.raises(fa.SymmetryError):
        symmetry.space_group_symbol(231)
