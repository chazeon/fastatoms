# SPDX-License-Identifier: MIT
"""The structure builders from Python.

The C++ suite checks the geometry - magic numbers, coordination, five-fold
symmetry. What is checked here is the Python surface, and the one promise that
matters on every generator: what comes back has a cell and boundary conditions
that make it ready to use.
"""

from __future__ import annotations

import numpy as np
import pytest

import fastatoms as fa

build = fa.build


def shortest_bond(atoms: fa.Atoms) -> float:
    positions = atoms.positions
    best = np.inf
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            best = min(best, float(np.linalg.norm(positions[i] - positions[j])))
    return best


# --------------------------------------------------------------------------
# Everything is ready to use
# --------------------------------------------------------------------------
def periodic_structures():
    return {
        "fcc": build.fcc("Al", 4.05),
        "bcc": build.bcc("Fe", 2.87),
        "hcp": build.hcp("Mg", 3.21),
        "diamond": build.diamond("Si", 5.43),
        "rocksalt": build.rocksalt("Na", "Cl", 5.64),
        "graphene": build.graphene(),
        "MoS2": build.tmd("Mo", "S"),
        "nanotube": build.nanotube(6, 3),
    }


@pytest.mark.parametrize("name", sorted(periodic_structures()))
def test_everything_comes_back_ready_to_use(name):
    """A cell, sensible periodicity, and no atoms on top of each other."""
    atoms = periodic_structures()[name]
    assert len(atoms) > 0
    assert atoms.get_volume() > 0.0
    assert any(atoms.pbc)
    assert shortest_bond(atoms) > 0.5 if len(atoms) > 1 else True


def test_clusters_are_finite_and_boxed():
    for atoms in (build.icosahedron("Ag", 2, 2.89),
                  build.cuboctahedron("Ag", 2, 4.09),
                  build.decahedron("Ag", 2, 3, a=4.09),
                  build.wulff("Ag", "fcc", 4.09, {"111": 1.0}, atoms=100)):
        assert not any(atoms.pbc)          # a particle is not periodic
        assert atoms.get_volume() > 0.0    # but it still has a box around it
        # And it sits inside that box rather than straddling the boundary.
        assert (atoms.positions > 0).all()
        assert (atoms.positions < np.diag(np.asarray(atoms.cell))).all()


# --------------------------------------------------------------------------
# Bulk
# --------------------------------------------------------------------------
def test_bulk_by_name_matches_the_direct_call():
    assert len(build.bulk("Si", "diamond", 5.43)) == len(build.diamond("Si", 5.43))
    assert len(build.bulk("Si", "diamond", 5.43, cubic=True)) == 8
    assert len(build.bulk(["Na", "Cl"], "rocksalt", 5.64, cubic=True)) == 8
    # The enum and the string are interchangeable.
    assert len(build.bulk("Al", build.Lattice.fcc, 4.05, cubic=True)) == 4


def test_the_lattices_have_the_spacings_they_should():
    assert shortest_bond(build.fcc("Al", 4.05, cubic=True)) == pytest.approx(4.05 / np.sqrt(2))
    assert shortest_bond(build.bcc("Fe", 2.87, cubic=True)) == pytest.approx(
        2.87 * np.sqrt(3) / 2)
    assert shortest_bond(build.diamond("Si", 5.43, cubic=True)) == pytest.approx(
        5.43 * np.sqrt(3) / 4)
    # hcp defaults to the ideal axial ratio.
    assert build.hcp("Mg", 3.21).cell.lengths[2] / 3.21 == pytest.approx(np.sqrt(8 / 3))


# --------------------------------------------------------------------------
# Two dimensions
# --------------------------------------------------------------------------
def test_graphene_and_its_rectangular_cell_agree():
    hexagonal = build.graphene()
    rectangular = build.graphene_rectangular()
    assert len(hexagonal) == 2 and len(rectangular) == 4
    assert shortest_bond(hexagonal) == pytest.approx(2.46 / np.sqrt(3))
    assert shortest_bond(rectangular) == pytest.approx(2.46 / np.sqrt(3))


def test_the_vacuum_is_the_gap_not_the_cell():
    sheet = build.tmd("Mo", "S", vacuum=18.0)
    height = sheet.positions[:, 2]
    assert sheet.cell.lengths[2] - (height.max() - height.min()) == pytest.approx(18.0)


def test_a_tmd_can_be_looked_up_or_given():
    a, thickness = build.tmd_parameters("Mo", "S")
    assert a == pytest.approx(3.160)
    looked_up = build.tmd("Mo", "S")
    explicit = build.tmd("Mo", "S", a=a, thickness=thickness)
    assert looked_up.positions == pytest.approx(explicit.positions)
    assert "MoS2" in build.known_tmds()

    # The two phases put the lower chalcogen in different places.
    h = build.tmd("Mo", "S", phase="1H")
    t = build.tmd("Mo", "S", phase="1T")
    assert h.positions[1][:2] == pytest.approx(h.positions[2][:2])
    assert np.linalg.norm(t.positions[1][:2] - t.positions[2][:2]) > 1.0


# --------------------------------------------------------------------------
# Nanotubes
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n, m, atoms, diameter", [
    (5, 5, 20, 6.78), (10, 0, 40, 7.83), (6, 3, 84, 6.21), (9, 0, 36, 7.05),
])
def test_the_tubes_have_the_published_geometry(n, m, atoms, diameter):
    """Counts and diameters as they are quoted in any review of the subject.

    The tolerance is a hundredth of an angstrom because that is the precision
    the published values carry, not because the construction is uncertain.
    """
    geometry = build.tube_geometry(n, m)
    assert len(build.nanotube(n, m)) == atoms
    assert 2 * geometry.radius == pytest.approx(diameter, abs=0.01)


def test_any_sheet_can_be_rolled():
    """A layer with thickness becomes a tube with two walls."""
    tube = build.roll(build.tmd("Mo", "S"), 10, 0)
    middle = tube.positions.mean(axis=0)
    radii = np.linalg.norm(tube.positions[:, :2] - middle[:2], axis=1)
    sulphur = radii[[s == "S" for s in tube.get_chemical_symbols()]]
    assert sulphur.max() - sulphur.min() == pytest.approx(3.130, abs=1e-6)


# --------------------------------------------------------------------------
# Clusters and Wulff
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shells, magic", [(1, 13), (2, 55), (3, 147), (4, 309)])
def test_the_closed_shell_sizes(shells, magic):
    assert len(build.icosahedron("Ag", shells, 2.89)) == magic
    assert len(build.cuboctahedron("Ag", shells, 4.09)) == magic


def test_a_decahedron_is_five_fold_symmetric():
    cluster = build.decahedron("Ag", 2, 3, a=4.09)
    axis = cluster.positions.mean(axis=0)
    relative = cluster.positions - axis

    turn = 2 * np.pi / 5
    rotation = np.array([[np.cos(turn), -np.sin(turn), 0],
                         [np.sin(turn), np.cos(turn), 0],
                         [0, 0, 1]])
    turned = relative @ rotation.T
    for point in turned:
        assert np.linalg.norm(relative - point, axis=1).min() < 1e-6


def test_wulff_responds_to_the_energies_it_is_given():
    # Dear {111} pushes atoms out of the corners and into the cube faces.
    def reach(atoms, direction):
        relative = atoms.positions - atoms.positions.mean(axis=0)
        return (relative @ np.asarray(direction)).max()

    diagonal = np.array([1, 1, 1]) / np.sqrt(3)
    cheap = build.wulff("Ag", "fcc", 4.09, {"100": 1.0, "111": 0.80}, atoms=400)
    dear = build.wulff("Ag", "fcc", 4.09, {"100": 1.0, "111": 1.30}, atoms=400)

    assert (reach(dear, diagonal) / reach(dear, [1, 0, 0]) >
            reach(cheap, diagonal) / reach(cheap, [1, 0, 0]))


def test_surface_energies_can_be_written_either_way():
    from_dict = build.wulff("Ag", "fcc", 4.09, {"111": 1.0, "100": 1.15}, atoms=200)
    from_list = build.wulff("Ag", "fcc", 4.09, [[1, 1, 1, 1.0], [1, 0, 0, 1.15]], atoms=200)
    assert len(from_dict) == len(from_list)


def test_the_cubic_multiplicities():
    assert len(build.equivalent_directions(1, 0, 0)) == 6
    assert len(build.equivalent_directions(1, 1, 1)) == 8
    assert len(build.equivalent_directions(1, 1, 0)) == 12


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------
@pytest.mark.parametrize("call", [
    lambda: build.fcc("Al", -1.0),
    lambda: build.bulk("Na", "rocksalt", 5.64),           # needs two species
    lambda: build.bulk(["Na", "Cl"], "fcc", 5.64),        # needs one
    lambda: build.bulk("Al", "hexagonal-ish", 4.05),      # no such lattice
    lambda: build.tmd("Xx", "S"),                         # not tabulated
    lambda: build.tube_geometry(0, 0),
    lambda: build.icosahedron("Ag", 0, 2.89),
    lambda: build.decahedron("Ag", 2, 3, r=-1, a=4.09),
    lambda: build.wulff("Ag", "fcc", 4.09, {}, atoms=100),
    lambda: build.wulff("Ag", "fcc", 4.09, {"111": -1.0}, atoms=100),
])
def test_bad_input_is_refused(call):
    with pytest.raises(fa.BuildError):
        call()
