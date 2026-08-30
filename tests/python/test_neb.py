# SPDX-License-Identifier: MIT
"""The nudged elastic band from Python.

The C++ suite checks the physics against surfaces whose saddles are known in
closed form. What is checked here is that the band is drivable from Python by
the optimizers, and that it works on a system with more than one atom - where
the path is not a straight line and the tangent has to do something.
"""

from __future__ import annotations

import numpy as np
import pytest

import fastatoms as fa


def lennard_jones() -> fa.Calculator:
    return fa.LennardJones(epsilon=0.0103, sigma=3.40, cutoff=9.0)


def exchange_endpoints():
    """Two argon atoms swapping places past a third.

    Not a deep barrier, but a real one on a real potential, and the path is
    curved: the moving atom has to go around rather than through.
    """
    positions = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]]
    initial = fa.Atoms("Ar3", positions, cell=20.0, pbc=False)
    final = fa.Atoms("Ar3", [[0.0, 0.0, 0.0], [4.0, 3.0, 0.0], [8.0, 0.0, 0.0]],
                     cell=20.0, pbc=False)
    for atoms in (initial, final):
        atoms.calc = lennard_jones()
    return initial, final


PERIOD = 4.5


def channel_endpoints():
    """An atom pushed through a periodic wall of atoms - a 0.12 eV barrier.

    The periodicity is what makes this a test. A finite pair of atoms with a
    gap between them looks like a barrier and is not one: the band relaxes
    perpendicular to the path, the moving atom simply goes *around* the pair,
    and the converged barrier comes out negative. Wrapping the wall in y and z
    removes the detour, so the only way from one side to the other is through.

    The wall atom has to be held with ``fix=[1]``. The projected band force is
    not required to sum to zero, so a spectator atom keeps a force the tangent
    knows nothing about and the whole band drifts along the reaction
    coordinate - here to x = +80 A before anything converges.
    """
    cell = [[20.0, 0.0, 0.0], [0.0, PERIOD, 0.0], [0.0, 0.0, PERIOD]]
    wall = [0.0, PERIOD / 2, PERIOD / 2]
    initial = fa.Atoms("Ar2", [[-5.0, 0.0, 0.0], wall], cell=cell, pbc=[False, True, True])
    final = fa.Atoms("Ar2", [[5.0, 0.0, 0.0], wall], cell=cell, pbc=[False, True, True])
    for atoms in (initial, final):
        atoms.calc = lennard_jones()
    return initial, final


#: The barrier of the channel system, from a direct scan of the potential on a
#: fine grid in x. Every band below has to reproduce this number.
CHANNEL_BARRIER = 0.122725


def kicked(band):
    """Displace the interior images, so relaxing the band is real work.

    Without this the test proves very little. The straight-line interpolation
    through the channel is *already* the exact path - the moving atom sits on
    the symmetry axis, its perpendicular forces cancel against the periodic
    images of the wall, and the spacing is uniform - so a band built from it
    converges in zero steps and reports the right barrier no matter what the
    projection does. Pushing the images off the axis and off their even
    spacing means the tangent, the springs and the perpendicular projection
    all have to be right for the barrier to come back.
    """
    for index in range(1, len(band) - 1):
        sign = 1.0 if index % 2 else -1.0
        band.image(index).positions[0] += [0.45 * sign, 0.35 * sign, 0.2]
    return band


# --------------------------------------------------------------------------
# Building a band
# --------------------------------------------------------------------------
def test_interpolation_and_shape():
    initial, final = exchange_endpoints()
    band = fa.NEB(initial, final, images=7)

    assert len(band) == 7
    assert band.movable == 5
    assert band.positions.shape == (5 * len(initial), 3)

    # The endpoints are the ones given, and are never moved.
    assert band.image(0).positions == pytest.approx(initial.positions)
    assert band.image(-1).positions == pytest.approx(final.positions)
    # Interpolated evenly in between.
    middle = band.image(3).positions
    assert middle == pytest.approx(0.5 * (initial.positions + final.positions))


def test_interpolate_returns_a_list_of_images():
    initial, final = exchange_endpoints()
    images = fa.NEB.interpolate(initial, final, images=5)
    assert len(images) == 5
    assert all(len(image) == 3 for image in images)
    # A band can be built from images prepared elsewhere.
    for image in images:
        image.calc = lennard_jones()
    assert len(fa.NEB(images)) == 5


def test_a_band_refuses_what_it_cannot_relax():
    initial, final = exchange_endpoints()
    with pytest.raises(Exception):
        fa.NEB(initial, final, images=2)
    with pytest.raises(Exception, match="unknown option"):
        fa.NEB(initial, final, images=5, nonsense=True)

    bare = fa.Atoms("Ar3", initial.positions, cell=20.0)
    with pytest.raises(Exception, match="calculator"):
        fa.NEB([initial, bare, final])


# --------------------------------------------------------------------------
# Forces and tangents
# --------------------------------------------------------------------------
def test_a_held_atom_never_moves_and_keeps_the_band_from_drifting():
    """Without this the band walks off along the reaction coordinate.

    The projection removes a different component from each atom, so the
    projected forces need not cancel; a spectator atom keeps a force the
    tangent does not describe and drags the whole band with it.
    """
    initial, final = channel_endpoints()
    wall = initial.positions[1].copy()

    drifting = fa.NEB(initial, final, images=6, k=0.5)
    drifting.image(2).positions[0] += [0.6, 0, 0]        # unequal spacing
    drifting.run("fire", fmax=0.01, max_steps=4000)
    wandered = max(abs(drifting.image(k).positions[0][0]) for k in range(len(drifting)))

    held = fa.NEB(initial, final, images=6, k=0.5, fix=[1])
    held.image(2).positions[0] += [0.6, 0, 0]
    held.run("fire", fmax=0.01, max_steps=4000)

    # The held atom is exactly where it started, in every image.
    for index in range(len(held)):
        assert held.image(index).positions[1] == pytest.approx(wall)
    # And the band stayed inside the region the endpoints span.
    stayed = max(abs(held.image(k).positions[0][0]) for k in range(len(held)))
    assert stayed <= 5.0 + 1e-6
    assert wandered > stayed


def test_a_nonexistent_fixed_atom_is_rejected():
    initial, final = channel_endpoints()
    with pytest.raises(Exception, match="does not exist"):
        fa.NEB(initial, final, images=5, fix=[7])


def test_the_tangent_is_a_unit_vector_per_image():
    initial, final = exchange_endpoints()
    band = fa.NEB(initial, final, images=7)
    for index in range(1, len(band) - 1):
        tangent = band.tangent(index)
        assert tangent.shape == (len(initial), 3)
        assert np.linalg.norm(tangent) == pytest.approx(1.0)
    with pytest.raises(Exception):
        band.tangent(0)


def test_forces_have_the_shape_of_the_movable_images():
    initial, final = exchange_endpoints()
    band = fa.NEB(initial, final, images=7)
    assert band.forces.shape == (5 * len(initial), 3)
    assert band.true_forces.shape == (7 * len(initial), 3)
    assert band.fmax > 0.0


# --------------------------------------------------------------------------
# Relaxing
# --------------------------------------------------------------------------
def test_a_band_relaxes_and_reports_a_profile():
    initial, final = channel_endpoints()
    band = kicked(fa.NEB(initial, final, images=7, k=0.5, fix=[1]))
    result = band.run("fire", fmax=0.01, max_steps=2000)

    assert result.converged
    assert band.fmax < 0.01

    profile = band.profile()
    assert len(profile.energies) == 7
    assert len(profile.distances) == 7
    # The reaction coordinate only ever increases.
    assert all(np.diff(profile.distances) > 0)
    # Endpoints are equivalent by symmetry, so the reaction energy vanishes.
    assert profile.reaction_energy == pytest.approx(0.0, abs=1e-6)
    # A real barrier, and one the band could not detour around.
    assert profile.forward_barrier > 0.05
    assert 0 < profile.highest < 6


@pytest.mark.parametrize("optimizer", ["fire", "lbfgs", "sd"])
def test_the_existing_optimizers_can_drive_a_band(optimizer):
    initial, final = channel_endpoints()
    band = kicked(fa.NEB(initial, final, images=5, k=0.5, climb=True, fix=[1]))
    result = band.run(optimizer, fmax=0.02, max_steps=5000)
    assert result.converged
    assert band.fmax < 0.02


def test_a_line_search_is_refused():
    """The band force is not a gradient, so there is nothing to search on."""
    initial, final = channel_endpoints()
    band = fa.NEB(initial, final, images=5, fix=[1])
    searching = fa.LBFGS(line_search="wolfe")
    with pytest.raises(Exception, match="line search"):
        band.run(searching)

    # The same optimizer with the search off is accepted.
    band.run(fa.LBFGS(line_search="none", max_steps=20))


def test_climbing_drives_an_image_to_the_top():
    initial, final = channel_endpoints()

    plain = kicked(fa.NEB(initial, final, images=7, k=0.5, fix=[1]))
    plain.run("fire", fmax=0.005, max_steps=4000)
    assert not plain.climbing

    climbing = kicked(fa.NEB(initial, final, images=7, k=0.5, climb=True, fix=[1]))
    climbing.run("fire", fmax=0.005, max_steps=4000)
    assert climbing.climbing

    # The climbing image sits at the top, so its barrier is at least as high as
    # the plain band's best sample of the same barrier.
    assert (climbing.profile().forward_barrier
            >= plain.profile().forward_barrier - 1e-9)
    # And the force on the climbing image is converged, not just the band's.
    assert climbing.profile().image_fmax[climbing.profile().highest - 1] < 0.005


def test_climb_threshold_delays_the_climb():
    """The gate is on the plain band force, the number ``fmax`` reports.

    That is what makes the threshold predictable: a run converging to a
    tolerance tighter than the threshold is guaranteed to start climbing
    before it stops, whatever else is held or constrained.
    """
    initial, final = channel_endpoints()
    band = kicked(
        fa.NEB(initial, final, images=5, k=0.5, climb=True, climb_threshold=1e-2, fix=[1]))

    assert band.fmax > 1e-2
    assert not band.climbing                       # still above the gate

    band.run("fire", fmax=1e-5, max_steps=8000)
    assert band.climbing                           # and it opened on the way down

    # Waiting changes when the climb starts, not where it ends up.
    eager = kicked(fa.NEB(initial, final, images=5, k=0.5, climb=True, fix=[1]))
    eager.run("fire", fmax=1e-5, max_steps=8000)
    assert band.profile().forward_barrier == pytest.approx(
        eager.profile().forward_barrier, abs=1e-6)


# --------------------------------------------------------------------------
# AutoNEB
# --------------------------------------------------------------------------
def test_autoneb_grows_the_band():
    initial, final = channel_endpoints()
    automatic = fa.AutoNEB(initial, final, initial_images=3, max_images=7,
                           fmax=0.01, coarse_fmax=0.05, max_steps=2000, k=0.5, fix=[1])
    result = automatic.run("fire")

    assert result.converged
    assert len(automatic.band) == 7
    # It grew one image at a time, relaxing between each.
    assert result.history[0] == 3
    assert result.history[-1] == 7
    assert result.history == sorted(result.history)
    assert result.stages == len(result.history)
    assert result.profile.forward_barrier > 0.0


def test_autoneb_matches_a_fixed_band_of_the_same_size():
    """Growing to N images should find the same barrier as starting with N."""
    initial, final = channel_endpoints()

    fixed = kicked(fa.NEB(initial, final, images=7, k=0.5, climb=True, fix=[1]))
    fixed.run("fire", fmax=0.005, max_steps=4000)

    automatic = fa.AutoNEB(initial, final, initial_images=3, max_images=7,
                           fmax=0.005, coarse_fmax=0.05, max_steps=4000, k=0.5, fix=[1])
    result = automatic.run("fire")

    assert result.profile.forward_barrier == pytest.approx(
        fixed.profile().forward_barrier, abs=1e-4)
    # And both agree with a direct scan of the same barrier.
    assert result.profile.forward_barrier == pytest.approx(CHANNEL_BARRIER, abs=1e-4)


def test_the_three_moves_can_be_used_directly():
    initial, final = channel_endpoints()
    automatic = fa.AutoNEB(initial, final, initial_images=7, max_images=9,
                           prune_tolerance=1.0, k=0.5, fix=[1])

    before = len(automatic.band)
    where = automatic.insert_image()
    assert len(automatic.band) == before + 1
    assert 0 < where < len(automatic.band)

    removed = automatic.prune_image()
    assert len(automatic.band) == before
    assert removed > 0

    automatic.redistribute()
    profile = automatic.band.profile()
    spacing = np.diff(profile.distances)
    assert spacing == pytest.approx(spacing[0], rel=1e-6)


def test_autoneb_rejects_impossible_settings():
    initial, final = channel_endpoints()
    with pytest.raises(Exception):
        fa.AutoNEB(initial, final, initial_images=2)
    with pytest.raises(Exception):
        fa.AutoNEB(initial, final, initial_images=7, max_images=5)


# --------------------------------------------------------------------------
# FIRE, which the band needs and which is useful on its own
# --------------------------------------------------------------------------
def test_fire_relaxes_a_structure_like_the_others():
    reference = None
    for algorithm in ("fire", "lbfgs"):
        atoms = fa.Atoms("Ar4", scaled_positions=[[0, 0, 0], [0, .5, .5], [.5, 0, .5],
                                                  [.5, .5, 0]], cell=5.5, pbc=True)
        # A fresh stream each time, so both start from the same geometry.
        atoms.positions += np.random.default_rng(0).normal(0, 0.1, atoms.positions.shape)
        atoms.calc = lennard_jones()
        result = fa.optimize(atoms, algorithm, fmax=1e-4, max_steps=5000)
        assert result.converged
        if reference is None:
            reference = result.energy
        else:
            assert result.energy == pytest.approx(reference, abs=1e-6)


def test_fire_exposes_its_adaptation():
    atoms = fa.Atoms("Ar2", [[0, 0, 0], [4.2, 0, 0]], cell=20.0)
    atoms.calc = lennard_jones()
    optimizer = fa.Fire(fmax=1e-4, max_steps=1000)
    assert optimizer.timestep > 0
    optimizer.run(atoms)
    # The timestep adapts away from its starting value over a run.
    assert optimizer.downhill_steps >= 0


# --------------------------------------------------------------------------
# IDPP
# --------------------------------------------------------------------------
def exchange_pair():
    """Two atoms swapping places past a third: a straight line drives them
    through each other, which is the whole reason IDPP exists."""
    initial = fa.Atoms("Ar3", [[0, 0, 0], [3.8, 0, 0], [1.9, 3.6, 0]], cell=20.0, pbc=False)
    final = fa.Atoms("Ar3", [[3.8, 0, 0], [0, 0, 0], [1.9, 3.6, 0]], cell=20.0, pbc=False)
    return initial, final


def test_a_straight_line_overlaps_the_exchanging_atoms():
    initial, final = exchange_pair()
    band = fa.NEB.interpolate(initial, final, images=7, minimum_image=False)
    worst = min(fa.closest_approach(image, minimum_image=False) for image in band)
    assert worst == pytest.approx(0.0, abs=1e-12)


def test_idpp_routes_them_around_each_other():
    initial, final = exchange_pair()
    band = fa.idpp_interpolate(initial, final, 7, minimum_image=False)

    assert len(band) == 7
    worst = min(fa.closest_approach(image, minimum_image=False) for image in band)
    assert worst > 3.0                       # from nothing at all

    # The endpoints are returned untouched.
    assert band[0].positions == pytest.approx(initial.positions)
    assert band[-1].positions == pytest.approx(final.positions)


def test_idpp_makes_a_band_a_calculator_can_actually_use():
    """The point of the exercise, measured the way a user would feel it."""
    initial, final = exchange_pair()
    for atoms in (initial, final):
        atoms.calc = lennard_jones()

    straight = fa.NEB.interpolate(initial, final, images=7, minimum_image=False)
    placed = fa.idpp_interpolate(initial, final, 7, minimum_image=False)

    def highest_energy(band):
        peak = -1e300
        for image in band:
            image.calc = lennard_jones()
            peak = max(peak, image.get_potential_energy())
        return peak

    # Overlapping atoms put the straight-line band at an absurd energy; the
    # IDPP band sits where the potential is defined.
    assert highest_energy(straight) > 1e3
    assert highest_energy(placed) < 1.0


# --------------------------------------------------------------------------
# The input deck
# --------------------------------------------------------------------------
DECK = """
structure:
  cell: [[20.0, 0.0, 0.0], [0.0, 4.5, 0.0], [0.0, 0.0, 4.5]]
  pbc: [false, true, true]
  atoms:
    - [Ar, -5.0, 0.0, 0.0]
    - [Ar,  0.0, 2.25, 2.25]

simulation:
  task: neb
  calculator:
    name: lennard_jones
    parameters: {epsilon: 0.0103, sigma: 3.40, cutoff: 9.0}
  neb:
    final:
      cell: [[20.0, 0.0, 0.0], [0.0, 4.5, 0.0], [0.0, 0.0, 4.5]]
      pbc: [false, true, true]
      atoms:
        - [Ar, 5.0, 0.0, 0.0]
        - [Ar, 0.0, 2.25, 2.25]
    images: 7
    spring_constant: 0.5
    climb: true
    fix: [1]
  optimizer: {algorithm: fire, fmax: 0.0001, max_steps: 4000}
"""


def test_the_deck_parses_a_neb_task():
    deck = fa.read_input_string(DECK)
    assert deck.simulation.task == fa.Task.neb
    assert deck.simulation.task_name == "neb"

    spec = deck.simulation.neb
    assert spec.images == 7
    assert spec.spring_constant == pytest.approx(0.5)
    assert spec.climbing
    assert list(spec.fixed_atoms) == [1]


def test_a_deck_builds_and_relaxes_the_band_it_describes():
    deck = fa.read_input_string(DECK)
    atoms = deck.atoms
    atoms.calc = fa.make_calculator(deck.simulation.calculator)

    band = fa.make_band(atoms, deck.simulation)
    assert len(band) == 7
    # Configured to climb, and climbing as soon as the forces are asked for -
    # no threshold was set, so there is nothing to wait for.
    band.forces
    assert band.climbing

    result = band.run(deck.simulation.optimizer.algorithm,
                      fmax=deck.simulation.optimizer.fmax,
                      max_steps=deck.simulation.optimizer.max_steps)
    assert result.converged
    # The same barrier the hand-built band finds.
    assert band.profile().forward_barrier == pytest.approx(CHANNEL_BARRIER, abs=1e-4)


def test_a_deck_can_ask_for_autoneb_instead():
    deck = fa.read_input_string(DECK.replace("    images: 7", "    images: 5\n    auto: true"))
    atoms = deck.atoms
    atoms.calc = fa.make_calculator(deck.simulation.calculator)

    automatic = fa.make_autoneb(atoms, deck.simulation)
    result = automatic.run("fire")
    assert result.converged
    assert result.profile.forward_barrier == pytest.approx(CHANNEL_BARRIER, abs=1e-4)


def test_the_deck_rejects_a_band_it_cannot_build():
    stub = "structure: {cell: 20.0, atoms: [[Ar, 0.0, 0.0, 0.0]]}\n"
    with pytest.raises(fa.InputError, match="final"):
        fa.read_input_string(stub + "simulation: {task: neb, neb: {images: 5}}")
    with pytest.raises(fa.InputError, match="at least 3"):
        fa.read_input_string(
            stub + "simulation: {task: neb, neb: {final: f.extxyz, images: 2}}")


def test_fixed_atoms_accept_ranges():
    deck = fa.read_input_string(
        "structure: {cell: 20.0, atoms: [[Ar, 0.0, 0.0, 0.0]]}\n"
        'simulation: {task: neb, neb: {final: f.extxyz, fix: ["0-3", 7]}}')
    assert list(deck.simulation.neb.fixed_atoms) == [0, 1, 2, 3, 7]
