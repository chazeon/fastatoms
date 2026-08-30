# SPDX-License-Identifier: MIT
"""Adsorption sites on a Pt55 nanoparticle, from Python.

Built with the `build` module, as a user would. The site counts are checked
against Euler's formula for the surface polyhedron, which nothing in the site
search knows about and which almost nothing that is wrong leaves true.
"""

from __future__ import annotations

import numpy as np
import pytest

import fastatoms as fa

adsorption = fa.adsorption
build = fa.build

#: Platinum's nearest-neighbour distance, from a 3.92 A lattice constant.
PT_NN = 2.77


@pytest.fixture(scope="module")
def particle() -> fa.Atoms:
    """A Pt55 Mackay icosahedron: a 13-atom core inside a 42-atom skin."""
    return build.icosahedron("Pt", shells=2, d=PT_NN)


@pytest.fixture(scope="module")
def finder(particle) -> adsorption.AdsorptionFinder:
    return adsorption.AdsorptionFinder(particle, height=1.9)


def counts_of(finder) -> dict[str, int]:
    counts: dict[str, int] = {}
    for site in finder.sites:
        counts[site.name] = counts.get(site.name, 0) + 1
    return counts


# --------------------------------------------------------------------------
# Site detection
# --------------------------------------------------------------------------
def test_the_particle_is_what_the_builder_says(particle):
    assert len(particle) == 55
    assert set(particle.get_chemical_symbols()) == {"Pt"}


def test_only_the_skin_is_surface(finder):
    # 55 atoms = a 13-atom core plus a 42-atom skin, and only the skin adsorbs.
    assert len(finder.surface_atoms) == 42


@pytest.mark.parametrize("shells, cuboctahedron", [
    (2, False), (3, False), (2, True), (3, True),
])
def test_eulers_formula_holds(shells, cuboctahedron):
    """Atop sites are vertices, bridges are edges, hollows and squares are
    faces. For a closed particle V - E + F must be two."""
    atoms = (build.cuboctahedron("Pt", shells, 3.92) if cuboctahedron
             else build.icosahedron("Pt", shells, PT_NN))
    found = adsorption.AdsorptionFinder(atoms, height=1.9)
    counts = counts_of(found)

    v = len(found.surface_atoms)
    e = counts.get("bridge", 0)
    f = counts.get("hollow", 0) + counts.get("4-fold", 0)
    assert v - e + f == 2
    assert counts["atop"] == v


def test_an_icosahedron_has_no_square_facets(finder):
    assert finder.sites_of_type("hollow")            # triangles, plenty
    assert finder.sites_of_type("four_fold") == []   # squares, none

    # A cuboctahedron has six of them, and they do carry 4-fold sites.
    cubo = adsorption.AdsorptionFinder(build.cuboctahedron("Pt", 2, 3.92), height=1.9)
    assert len(cubo.sites_of_type("four_fold")) == 24


def test_every_site_sits_outside_the_particle(finder, particle):
    centre = particle.positions.mean(axis=0)
    for site in finder.sites:
        radius = np.linalg.norm(np.asarray(site.position) - centre)
        for atom in site.atoms:
            assert radius > np.linalg.norm(particle.positions[atom] - centre)
        # And the normal points out, not in.
        assert np.dot(np.asarray(site.normal),
                      np.asarray(site.centre) - centre) > 0


# --------------------------------------------------------------------------
# Symmetry reduction
# --------------------------------------------------------------------------
def test_the_unique_sites_match_the_icosahedral_symmetry(finder):
    # A two-shell icosahedron has 12 vertex atoms and 30 edge atoms, so there
    # are exactly two kinds of atop site with those multiplicities.
    atop = [len(finder.equivalent_sites(finder.sites[s].unique))
            for s in finder.sites_of_type("atop", unique_only=True)]
    assert sorted(atop) == [12, 30]

    # And two kinds of hollow: the 20 over the icosahedron's own faces, and 60.
    hollow = [len(finder.equivalent_sites(finder.sites[s].unique))
              for s in finder.sites_of_type("hollow", unique_only=True)]
    assert sorted(hollow) == [20, 60]

    assert len(finder.unique_sites) == 6
    assert len(finder.sites) == 242


def test_the_classes_partition_the_sites(finder):
    seen = set()
    for unique in range(len(finder.unique_sites)):
        for site in finder.equivalent_sites(unique):
            assert site not in seen
            seen.add(site)
            assert finder.sites[site].unique == unique
    assert len(seen) == len(finder.sites)


# --------------------------------------------------------------------------
# Placement: single atoms and molecules
# --------------------------------------------------------------------------
def test_a_hydrogen_lands_on_the_site(finder, particle):
    site = finder.unique_sites[0]
    decorated = finder.place_atom("H", site)

    assert len(decorated) == len(particle) + 1
    assert decorated.get_chemical_symbols()[-1] == "H"
    assert decorated.positions[-1] == pytest.approx(finder.sites[site].position)
    # The particle itself is untouched.
    assert decorated.positions[:len(particle)] == pytest.approx(particle.positions)


def test_a_hydroxyl_binds_through_its_oxygen(finder, particle):
    """The orientation is the point: O towards the metal, H pointing away."""
    centre = particle.positions.mean(axis=0)
    for unique in finder.unique_sites:
        frame = finder.place("OH", unique)
        symbols = frame.get_chemical_symbols()
        assert symbols[len(particle)] == "O"
        assert symbols[len(particle) + 1] == "H"

        oxygen = frame.positions[len(particle)]
        hydrogen = frame.positions[len(particle) + 1]

        # The oxygen is at the site; the hydrogen is further out than it.
        assert oxygen == pytest.approx(finder.sites[unique].position)
        assert np.linalg.norm(hydrogen - centre) > np.linalg.norm(oxygen - centre)

        # The O-H bond lies along the site normal, and kept its length.
        bond = hydrogen - oxygen
        assert np.linalg.norm(bond) == pytest.approx(0.97)
        assert np.dot(bond / np.linalg.norm(bond),
                      np.asarray(finder.sites[unique].normal)) == pytest.approx(1.0)


def test_the_binding_atom_can_be_chosen(finder, particle):
    site = finder.unique_sites[0]
    # Carbon monoxide bound through its oxygen instead of its carbon.
    frame = finder.place("CO", site, binding_atom=1)
    assert frame.get_chemical_symbols()[len(particle) + 1] == "O"
    assert frame.positions[len(particle) + 1] == pytest.approx(finder.sites[site].position)


def test_an_adsorbate_can_be_any_atoms(finder):
    custom = fa.Atoms("NH", [[0, 0, 0], [0, 0, 1.02]])
    frame = finder.place(custom, finder.unique_sites[0])
    assert frame.get_chemical_symbols()[-2:] == ["N", "H"]
    assert "OH" in adsorption.known_molecules()
    assert len(adsorption.molecule("H2O")) == 3


# --------------------------------------------------------------------------
# Trajectories and coverage
# --------------------------------------------------------------------------
def test_a_trajectory_has_one_frame_per_site(finder, particle):
    unique = finder.trajectory("H")
    assert len(unique) == len(finder.unique_sites) == 6

    everywhere = finder.trajectory("H", unique_only=False)
    assert len(everywhere) == len(finder.sites) == 242

    for frame in unique:
        assert len(frame) == len(particle) + 1

    hydroxyls = finder.trajectory("OH")
    assert len(hydroxyls[0]) == len(particle) + 2


def test_a_trajectory_round_trips_through_extxyz(finder, tmp_path):
    frames = finder.trajectory("OH")
    path = tmp_path / "sites.extxyz"
    fa.write(str(path), frames)

    read_back = fa.read_frames(str(path))
    assert len(read_back) == len(frames)
    assert read_back[0].positions == pytest.approx(frames[0].positions)
    assert read_back[0].get_chemical_symbols() == frames[0].get_chemical_symbols()


def test_the_coverage_series_grows_and_spreads(finder, particle):
    hollows = finder.sites_of_type("hollow")
    series = finder.coverage_series("H", candidates=hollows, steps=10)
    fractions = finder.coverage_fractions(candidates=hollows, steps=10)

    assert len(series) == len(fractions) == 10
    for k, frame in enumerate(series):
        assert len(frame) == len(particle) + k + 1
    assert fractions[0] == pytest.approx(1 / 42)
    assert fractions[-1] == pytest.approx(10 / 42)

    # Each new adsorbate goes as far as possible from the others, so the
    # closest pair shrinks monotonically as the coverage grows.
    closest = []
    for frame in series[1:]:
        adsorbed = np.asarray(frame.positions[len(particle):])
        gaps = [np.linalg.norm(adsorbed[i] - adsorbed[j])
                for i in range(len(adsorbed)) for j in range(i + 1, len(adsorbed))]
        closest.append(min(gaps))
    assert all(b <= a + 1e-9 for a, b in zip(closest, closest[1:]))
    assert closest[0] > 10.0      # the first two land on opposite sides


def test_bad_input_is_refused(finder):
    with pytest.raises(fa.AdsorptionError):
        adsorption.AdsorptionFinder(fa.Atoms())
    with pytest.raises(fa.AdsorptionError):
        finder.place_atom("H", 100000)
    with pytest.raises(fa.AdsorptionError):
        finder.place("OH", finder.unique_sites[0], binding_atom=9)
    with pytest.raises(fa.AdsorptionError):
        adsorption.molecule("unobtainium")
    # A bulk crystal has no surface at all.
    with pytest.raises(fa.AdsorptionError):
        adsorption.AdsorptionFinder(build.fcc("Pt", 3.92, cubic=True).repeat(3, 3, 3))
