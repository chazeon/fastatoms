#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Finding the global minimum of a Lennard-Jones cluster.

    python python/examples/basin_hopping.py

A local optimizer answers "what is the nearest minimum". Basin hopping answers
"what is the lowest one", by relaxing every structure before looking at its
energy - which turns the basin around each minimum into a flat plateau and
leaves a Metropolis walk with nothing to do but step between basins.

The Lennard-Jones clusters are the standard test: their global minima have been
tabulated since the seventies, so a search either finds the published number or
it does not.

Nothing here writes a file.
"""

from __future__ import annotations

import numpy as np

import fastatoms as fa

#: Published global minima, in units of epsilon.
REFERENCE = {7: -16.505384, 13: -44.326801, 19: -72.659782, 38: -173.928427}


def random_cluster(n: int, seed: int) -> fa.Atoms:
    """`n` atoms thrown into a box that roughly fits them."""
    rng = np.random.default_rng(seed)
    half = 0.6 * n ** (1 / 3)
    atoms = fa.Atoms(f"Ar{n}", rng.uniform(-half, half, size=(n, 3)))
    # Unshifted, so the energies compare with the published numbers directly.
    atoms.calc = fa.LennardJones(epsilon=1.0, sigma=1.0, cutoff=100.0, shift=False)
    return atoms


def a_local_optimizer_is_not_enough() -> None:
    print("What a local optimizer does\n" + "-" * 66)
    print(f"{'start':>7} {'relaxed':>12} {'global minimum':>16}   found it?")
    for seed in range(6):
        cluster = random_cluster(13, seed)
        start = cluster.get_potential_energy()
        fa.LBFGS(fmax=1e-4, max_steps=500).run(cluster)
        relaxed = cluster.get_potential_energy()
        found = abs(relaxed - REFERENCE[13]) < 1e-4
        print(f"{seed:>7} {relaxed:>12.6f} {REFERENCE[13]:>16.6f}   "
              f"{'yes' if found else 'no, stuck in a local minimum'}")
    print("""
Six random starts, six relaxations, and the answer depends entirely on where
the structure happened to land. That is the problem basin hopping exists for.""")


def the_search() -> None:
    print("\n\nWhat basin hopping does\n" + "-" * 66)
    print(f"{'cluster':>8} {'start':>11} {'found':>12} {'reference':>12} "
          f"{'step':>6} {'accept':>7}")
    for n in (7, 13, 19):
        cluster = random_cluster(n, 7)
        search = fa.BasinHopping(steps=400, temperature=8000.0, step_size=0.45,
                                 seed=12345, fmax=1e-4, max_steps=500,
                                 minimum_distance=0.6, stall_limit=150)
        result = search.run(cluster)
        mark = "OK" if abs(result.best_energy - REFERENCE[n]) < 1e-4 else "MISSED"
        print(f"LJ{n:<6} {result.initial_energy:>11.4f} {result.best_energy:>12.6f} "
              f"{REFERENCE[n]:>12.6f} {result.best_step:>6} "
              f"{result.acceptance_ratio:>7.2f}  {mark}")

    print("""
The published minima, to six decimals, from random starts. `cluster` is left
holding the best structure found - not where the walk stopped, which is a
different thing and usually worse.""")


def watching_it_work() -> None:
    print("\n\nHow the walk gets there\n" + "-" * 66)
    cluster = random_cluster(13, 7)
    search = fa.BasinHopping(steps=400, temperature=8000.0, step_size=0.45,
                             seed=12345, fmax=1e-4, max_steps=500,
                             minimum_distance=0.6)

    improvements = []
    search.attach(lambda step: improvements.append(step) if step.improved else True)
    result = search.run(cluster)

    print(f"{'step':>6} {'energy':>12} {'step size':>11}")
    for step in improvements:
        print(f"{step.step:>6} {step.energy:>12.6f} {step.step_size:>11.4f}")
    print(f"""
{result.improved} improvements over {result.steps} steps, {result.evaluations} force evaluations in all.
The walk accepted {result.acceptance_ratio:.0%} of what it tried - the step size is scaled
towards a half automatically, since a step too small never leaves the basin and
one too large is always rejected, and both waste the relaxation.""")


def the_moves() -> None:
    print("\n\nThe perturbations\n" + "-" * 66)
    base = random_cluster(13, 3)
    for move in ("cartesian", "single", "twist"):
        cluster = base.copy()
        cluster.calc = fa.LennardJones(epsilon=1.0, sigma=1.0, cutoff=100.0, shift=False)
        fa.BasinHopping(move=move, step_size=0.5).perturb(cluster, 4242)
        shifted = np.linalg.norm(cluster.positions - base.positions, axis=1)
        print(f"  {move:<11} moved {int((shifted > 1e-9).sum()):>2} of 13 atoms, "
              f"largest displacement {shifted.max():.3f} A")
    print("""
`single` displaces one atom, `cartesian` all of them, and `twist` turns half
the cluster about a random axis - which rearranges the packing outright, and is
what shakes a cluster out of a funnel that independent displacements cannot.
`exchange` swaps two species, for alloys where the shape may already be right
and the ordering wrong.""")


def main() -> None:
    a_local_optimizer_is_not_enough()
    the_search()
    watching_it_work()
    the_moves()


if __name__ == "__main__":
    main()
