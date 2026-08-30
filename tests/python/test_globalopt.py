# SPDX-License-Identifier: MIT
"""Basin hopping from Python.

Checked against the Lennard-Jones clusters, whose global minima have been
tabulated since the seventies. A search that cannot find those is not a search.
"""

from __future__ import annotations

import numpy as np
import pytest

import fastatoms as fa

#: Published global minima in units of epsilon.
REFERENCE = {7: -16.505384, 13: -44.326801, 19: -72.659782}


def random_cluster(n: int, seed: int) -> fa.Atoms:
    rng = np.random.default_rng(seed)
    half = 0.6 * n ** (1 / 3)
    atoms = fa.Atoms(f"Ar{n}", rng.uniform(-half, half, size=(n, 3)))
    # Unshifted, so the energies compare with the published numbers directly.
    atoms.calc = fa.LennardJones(epsilon=1.0, sigma=1.0, cutoff=100.0, shift=False)
    return atoms


def searching(steps: int, seed: int) -> dict:
    return dict(steps=steps, temperature=8000.0, step_size=0.45, seed=seed,
                fmax=1e-4, max_steps=500, minimum_distance=0.6)


# --------------------------------------------------------------------------
# It finds the answer
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n, steps", [(7, 120), (13, 400), (19, 400)])
def test_the_known_minima_are_found(n, steps):
    cluster = random_cluster(n, 7)
    result = fa.BasinHopping(**searching(steps, 12345)).run(cluster)

    assert result.best_energy == pytest.approx(REFERENCE[n], abs=1e-4)
    # The structure handed back is the best one, not the walker's last position.
    assert cluster.get_potential_energy() == pytest.approx(REFERENCE[n], abs=1e-4)


def test_it_escapes_the_minimum_it_started_in():
    """The point of the algorithm: a local optimizer stops at the first
    minimum it meets, and here that is not the global one."""
    cluster = random_cluster(13, 7)
    result = fa.BasinHopping(**searching(400, 12345)).run(cluster)

    assert result.initial_energy > -44.0
    assert result.best_energy < result.initial_energy - 2.0
    assert result.best_energy == pytest.approx(REFERENCE[13], abs=1e-4)
    assert result.best_step > 0
    # Exactly one improvement: rediscovering the same minimum by a different
    # relaxation path differs only in the last bits and is not a discovery.
    assert result.improved == 1


def test_the_same_seed_reproduces_the_search():
    a = fa.BasinHopping(**searching(60, 999)).run(random_cluster(7, 3))
    b = fa.BasinHopping(**searching(60, 999)).run(random_cluster(7, 3))
    assert a.best_energy == pytest.approx(b.best_energy, abs=1e-15)
    assert a.accepted == b.accepted
    assert [s.energy for s in a.history] == pytest.approx([s.energy for s in b.history])


# --------------------------------------------------------------------------
# The Metropolis rule
# --------------------------------------------------------------------------
def test_at_zero_temperature_the_walk_never_goes_uphill():
    cluster = random_cluster(13, 5)
    options = searching(60, 4242) | dict(temperature=0.0, adapt_step=False)
    result = fa.BasinHopping(**options).run(cluster)

    previous = result.initial_energy
    for step in result.history:
        if step.accepted:
            assert step.energy <= previous + 1e-12
        assert step.current_energy <= previous + 1e-12
        previous = step.current_energy
    assert result.accepted < result.steps


def test_a_hotter_search_accepts_more():
    def acceptance(temperature):
        options = searching(80, 777) | dict(temperature=temperature, adapt_step=False)
        return fa.BasinHopping(**options).run(random_cluster(13, 11)).acceptance_ratio

    assert acceptance(50000.0) > acceptance(10.0)


def test_the_best_is_tracked_apart_from_the_walker():
    """The walk may wander uphill; the answer is the lowest it ever saw."""
    cluster = random_cluster(13, 2)
    result = fa.BasinHopping(**searching(100, 31337)).run(cluster)

    previous = result.initial_energy
    for step in result.history:
        assert step.best_energy <= previous + 1e-12          # can only fall
        assert step.best_energy <= step.current_energy + 1e-12
        previous = step.best_energy
    assert result.history[-1].best_energy == pytest.approx(result.best_energy)


# --------------------------------------------------------------------------
# The moves
# --------------------------------------------------------------------------
@pytest.mark.parametrize("move", ["cartesian", "single", "twist", "mixed"])
def test_every_move_moves_something(move):
    cluster = random_cluster(13, 1)
    before = cluster.positions.copy()
    fa.BasinHopping(move=move, step_size=0.5).perturb(cluster, 4242)
    assert np.abs(cluster.positions - before).max() > 1e-6


def test_a_single_move_moves_one_atom():
    cluster = random_cluster(13, 1)
    before = cluster.positions.copy()
    fa.BasinHopping(move="single", step_size=0.5).perturb(cluster, 99)
    moved = np.linalg.norm(cluster.positions - before, axis=1) > 1e-9
    assert moved.sum() == 1


def test_an_exchange_swaps_species_without_moving_anything():
    alloy = random_cluster(12, 8)
    numbers = alloy.numbers.copy()
    numbers[:6] = 28
    alloy.numbers = numbers

    before = alloy.positions.copy()
    fa.BasinHopping(move="exchange").perturb(alloy, 5)

    # The set of occupied positions is unchanged; only who sits where.
    assert np.sort(alloy.positions, axis=0) == pytest.approx(np.sort(before, axis=0))
    assert (np.linalg.norm(alloy.positions - before, axis=1) > 1e-9).sum() == 2


def test_a_perturbation_respects_constraints():
    cluster = random_cluster(13, 4)
    cluster.constraints = [fa.FixAtoms([0, 1])]
    before = cluster.positions.copy()

    fa.BasinHopping(step_size=0.6).perturb(cluster, 7)
    assert cluster.positions[:2] == pytest.approx(before[:2])
    assert np.linalg.norm(cluster.positions[2] - before[2]) > 1e-9


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------
def test_a_callback_can_watch_and_stop():
    cluster = random_cluster(13, 6)
    search = fa.BasinHopping(**searching(200, 1))

    seen = []
    search.attach(lambda step: (seen.append(step.step), len(seen) < 20)[1])
    result = search.run(cluster)

    assert seen == list(range(1, 21))
    assert result.steps == 20
    assert "callback" in result.message


def test_a_stalled_search_gives_up_but_keeps_the_answer():
    cluster = random_cluster(7, 9)
    options = searching(500, 3) | dict(stall_limit=15)
    result = fa.BasinHopping(**options).run(cluster)

    assert result.steps < 500
    assert "without improvement" in result.message
    assert result.best_energy == pytest.approx(REFERENCE[7], abs=1e-4)


def test_the_step_size_adapts():
    cluster = random_cluster(13, 12)
    options = searching(150, 8) | dict(step_size=0.02, adapt_interval=10)
    result = fa.BasinHopping(**options).run(cluster)
    # A step accepted too often gets bigger.
    assert result.history[-1].step_size > result.history[0].step_size


def test_bad_input_is_refused():
    with pytest.raises(fa.GlobalOptimizationError):
        fa.BasinHopping().run(fa.Atoms())
    with pytest.raises(fa.GlobalOptimizationError):
        fa.BasinHopping().run(fa.Atoms("Ar2", [[0, 0, 0], [3.8, 0, 0]]))
    with pytest.raises(fa.GlobalOptimizationError):
        fa.BasinHopping(step_size=-1.0)
    with pytest.raises(fa.GlobalOptimizationError):
        fa.BasinHopping(temperature=-5.0)
    with pytest.raises(fa.GlobalOptimizationError):
        fa.BasinHopping(move="teleport")
