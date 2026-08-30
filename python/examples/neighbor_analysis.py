#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Structural analysis driven entirely from Python.

Builds a supercell, then uses the C++ neighbour list to compute coordination
numbers and a radial distribution function - the pair loop runs in C++ while
the histogramming stays in numpy.

    python python/examples/neighbor_analysis.py
"""

from __future__ import annotations

import time

import numpy as np

import fastatoms as fa


def diamond_silicon(a: float = 5.43) -> fa.Atoms:
    """The 8-atom conventional cell of diamond-structure silicon."""
    basis = np.array([
        [0.00, 0.00, 0.00], [0.00, 0.50, 0.50],
        [0.50, 0.00, 0.50], [0.50, 0.50, 0.00],
        [0.25, 0.25, 0.25], [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
    ])
    return fa.Atoms("Si8", scaled_positions=basis, cell=a, pbc=True)


def radial_distribution(atoms: fa.Atoms, cutoff: float, bins: int = 200):
    """g(r) from a half neighbour list."""
    pairs = fa.neighbor_list(atoms, cutoff, full=False)
    counts, edges = np.histogram(pairs.distances, bins=bins, range=(0.0, cutoff))

    centres = 0.5 * (edges[:-1] + edges[1:])
    shell_volume = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    density = len(atoms) / atoms.get_volume()
    # Two atoms per stored pair, hence the factor of 2.
    normalisation = 0.5 * len(atoms) * density * shell_volume
    return centres, counts / normalisation


def main() -> None:
    crystal = diamond_silicon().repeat(6, 6, 6)
    print(f"{crystal!r}")
    print(f"{len(crystal)} atoms, {crystal.get_volume():.1f} A^3\n")

    start = time.perf_counter()
    neighbours = fa.neighbor_list(crystal, 3.0)
    elapsed = time.perf_counter() - start

    coordination = neighbours.coordination_numbers()
    print(f"neighbour list : {len(neighbours)} pairs in {elapsed * 1e3:.2f} ms")
    print(f"coordination   : {coordination.mean():.3f} "
          f"(all four-fold: {bool((coordination == 4).all())})")
    print(f"bond length    : {neighbours.distances.mean():.4f} A "
          f"(expected {5.43 * np.sqrt(3) / 4:.4f} A)\n")

    centres, g = radial_distribution(crystal, cutoff=8.0)
    peaks = centres[(g > 1.0) & (g == np.maximum.accumulate(g))]
    print("g(r) peaks up to 8 A:")
    for radius in peaks[:6]:
        print(f"  {radius:.3f} A")

    # Bond detection from covalent radii needs no hand-picked cutoff.
    bonds = fa.neighbor_list(
        crystal, per_atom_cutoffs=fa.natural_cutoffs(crystal, mult=1.1), full=False
    )
    print(f"\ncovalent bonds : {len(bonds)} "
          f"({len(bonds) / len(crystal):.2f} per atom)")


if __name__ == "__main__":
    main()
