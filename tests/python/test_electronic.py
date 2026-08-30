# SPDX-License-Identifier: MIT
"""Electronic structure analysis from Python.

The C++ suite checks the integration mathematics against the free electron
gas. What is checked here is the Python surface, and that the arrays come back
as views onto the C++ storage rather than copies.
"""

from __future__ import annotations

import numpy as np
import pytest

import fastatoms as fa

el = fa.electronic
build = fa.build


def free_electrons(n: int) -> el.BandStructure:
    """E(k) = k^2 on a grid, whose density of states goes as sqrt(E)."""
    kpoints = np.asarray(el.monkhorst_pack(n, n, n))
    folded = kpoints - np.round(kpoints)
    return el.BandStructure(kpoints, (folded ** 2).sum(axis=1), bands=1)


# --------------------------------------------------------------------------
# Band paths
# --------------------------------------------------------------------------
@pytest.mark.parametrize("atoms, lattice, path", [
    (build.sc("Po", 3.35), "CUB", "GXMGRX,MR"),
    (build.fcc("Al", 4.05), "FCC", "GXWKGLUWLK,UX"),
    (build.bcc("Fe", 2.87), "BCC", "GHNGPH,PN"),
    (build.graphene(), "HEX", "GMKGALHA,LM,KH"),
])
def test_the_lattice_and_its_conventional_path(atoms, lattice, path):
    assert str(el.identify_bravais(atoms)).split(".")[-1] == lattice
    assert el.default_path(atoms) == path


def test_a_path_hits_its_special_points():
    atoms = build.fcc("Al", 4.05)
    path = el.bandpath(atoms, path="GXWK", points=80)

    assert path.lattice == el.BravaisLattice.FCC
    assert path.special_labels == ["G", "X", "W", "K"]
    assert len(path) > 70

    points = el.special_points(atoms)
    assert path.kpoints[0] == pytest.approx(points["G"])
    assert path.kpoints[-1] == pytest.approx(points["K"])

    # The axis is a real distance and runs forwards.
    assert np.all(np.diff(path.distances) >= -1e-12)
    # Gamma to X of an fcc lattice is 2 pi / a, which is 1.551 for a = 4.05.
    assert path.special_distances[1] == pytest.approx(2 * np.pi / 4.05, abs=1e-6)


def test_a_comma_makes_the_path_jump():
    atoms = build.fcc("Al", 4.05)
    joined = el.bandpath(atoms, path="GXWK", points=80)
    jumped = el.bandpath(atoms, path="GX,WK", points=80)
    assert len(jumped.special_labels) == 4
    assert jumped.distances[-1] < joined.distances[-1]


def test_an_explicit_path_needs_no_table():
    atoms = build.fcc("Al", 4.05)
    corners = [[0, 0, 0], [0.5, 0, 0.5], [0.5, 0.5, 0.5]]
    path = el.bandpath_from_points(atoms, corners, labels=["G", "X", "L"], points=40)
    assert path.special_labels == ["G", "X", "L"]
    assert path.kpoints[-1] == pytest.approx(corners[-1])


def test_an_untabulated_lattice_says_so():
    cell = fa.Cell([[3.0, 0, 0], [0.7, 4.1, 0], [0.4, 1.1, 5.3]])
    assert el.identify_bravais(cell) == el.BravaisLattice.TRI
    with pytest.raises(fa.ElectronicError, match="TRI"):
        el.special_points(cell)
    assert el.BravaisLattice.FCC in el.tabulated_lattices()


# --------------------------------------------------------------------------
# Band structure and gaps
# --------------------------------------------------------------------------
def test_an_indirect_gap():
    bands = el.BandStructure([[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]],
                             [0.0, 3.0, -0.5, 2.5, -1.0, 1.5],
                             bands=2, fermi_level=0.5)
    gap = bands.band_gap()
    assert not gap.metallic and not gap.direct
    assert gap.gap == pytest.approx(1.5)
    assert gap.direct_gap == pytest.approx(2.5)
    assert gap.valence_kpoint == 0 and gap.conduction_kpoint == 2
    assert gap.direct_gap >= gap.gap


def test_a_metal_has_no_gap():
    bands = el.BandStructure([[0, 0, 0], [0.5, 0, 0]], [-1.0, 1.0], bands=1)
    assert bands.is_metal()
    assert bands.band_gap().metallic


def test_shifting_and_accessors():
    bands = el.BandStructure([[0, 0, 0]], [-2.0, 4.0], bands=2, fermi_level=1.0)
    assert bands.nkpoints == 1 and bands.nbands == 2 and bands.nspins == 1
    assert bands.energy(0, 0, 1) == pytest.approx(4.0)
    assert bands.band(0, 0) == pytest.approx([-2.0])
    assert bands.minimum() == pytest.approx(-2.0)

    moved = bands.shifted()
    assert moved.fermi_level == pytest.approx(0.0)
    assert moved.energy(0, 0, 0) == pytest.approx(-3.0)
    assert moved.band_gap().gap == pytest.approx(bands.band_gap().gap)


def test_the_container_refuses_what_does_not_add_up():
    with pytest.raises(fa.ElectronicError):
        el.BandStructure([[0, 0, 0], [0.5, 0, 0]], [1.0, 2.0, 3.0], bands=2)
    with pytest.raises(fa.ElectronicError):
        el.BandStructure([[0, 0, 0]], [1.0], bands=1, spins=3)


# --------------------------------------------------------------------------
# Density of states
# --------------------------------------------------------------------------
def test_the_sum_rule():
    bands = free_electrons(10)
    for dos in (el.density_of_states(bands, points=1500, width=0.02),
                el.density_of_states_tetrahedron(bands, (10, 10, 10), points=1500)):
        # One band, two spins.
        assert dos.integrated[-1] == pytest.approx(2.0, abs=5e-3)


def test_the_tetrahedron_method_recovers_the_square_root():
    bands = free_electrons(20)
    dos = el.density_of_states_tetrahedron(bands, (20, 20, 20),
                                           emin=0.0, emax=0.2, points=300)
    assert dos.method == "tetrahedron"
    mask = (dos.energy > 0.04) & (dos.energy < 0.14)
    exact = 4 * np.pi * np.sqrt(dos.energy[mask])
    assert np.abs(dos.dos[mask] - exact).max() / exact.max() < 0.05


def test_the_smearings():
    # Each replacement for a delta function must have unit area.
    x = np.linspace(-40, 40, 400001)
    for smearing in ("gaussian", "methfessel-paxton", "fermi-dirac"):
        values = np.array([el.smearing_delta(float(v), smearing=smearing, order=1) for v in x[::40]])
        assert np.trapezoid(values, x[::40]) == pytest.approx(1.0, abs=1e-4)
        assert el.smearing_occupation(-30.0, smearing=smearing) == pytest.approx(1.0, abs=1e-9)
        assert el.smearing_occupation(0.0, smearing=smearing) == pytest.approx(0.5, abs=1e-9)

    # Methfessel-Paxton goes negative, which is correct and is the price of its
    # faster convergence.
    lobe = [el.smearing_delta(v, smearing="methfessel-paxton", order=1)
            for v in np.linspace(0, 6, 200)]
    assert min(lobe) < -1e-6


def test_spin_resolved():
    kpoints = el.monkhorst_pack(4, 4, 4)
    energies = [-1.0] * len(kpoints) + [1.0] * len(kpoints)
    bands = el.BandStructure(kpoints, energies, bands=1, spins=2)
    dos = el.density_of_states(bands, emin=-3.0, emax=3.0, points=600, width=0.1)

    assert dos.spins == 2
    assert dos.spin_dos.shape == (2, len(dos))
    assert dos.integrated[-1] == pytest.approx(2.0, abs=1e-6)


def test_the_tetrahedron_method_needs_its_grid():
    bands = free_electrons(4)
    with pytest.raises(fa.ElectronicError, match="regular grid"):
        el.density_of_states_tetrahedron(bands, (5, 5, 5))


# --------------------------------------------------------------------------
# The zero-copy promise
# --------------------------------------------------------------------------
def test_the_arrays_are_views():
    bands = free_electrons(4)
    assert bands.energies.base is bands
    assert not bands.energies.flags.writeable
    assert bands.energies.shape == (1, 64, 1)

    dos = el.density_of_states(bands, points=100)
    for name in ("energy", "dos", "integrated"):
        array = getattr(dos, name)
        assert array.base is dos
        assert not array.flags.writeable

    path = el.bandpath(build.fcc("Al", 4.05), points=40)
    assert path.distances.base is path
