#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reaction paths with the nudged elastic band.

    python python/examples/neb.py

An argon adatom hops between two hollow sites of a square surface layer. The
substrate is held, the adatom is not, and the band finds the path over the
bridge site between them.

Three runs, in increasing order of what they tell you:

  plain NEB    the path, and a barrier only as good as the spacing of the
               images: with an even band no image lands on the saddle
  CI-NEB       the highest image is driven onto the saddle itself, so the
               barrier is the barrier and not a lower bound
  AutoNEB      starts coarse and adds images where the path is interesting,
               so the resolution near the top costs no force calls elsewhere

Nothing here writes a file.
"""

from __future__ import annotations

import fastatoms as fa

SPACING = 3.8      # square lattice constant of the surface layer, angstrom
CELLS = 4          # a CELLS x CELLS patch, periodic in x and y
HEIGHT = 2.70      # relaxed adatom height above a hollow site

#: The saddle here is flat, so a force tolerance converts into a much larger
#: energy tolerance than you might expect: relaxing to 5 meV/A leaves the
#: barrier 0.13 meV short, and it takes 0.1 meV/A to pin the last 0.01 meV.
#: Barriers deserve a tighter tolerance than geometries do.
FMAX = 1e-4


def surface() -> list[list[float]]:
    return [[SPACING * i, SPACING * j, 0.0] for i in range(CELLS) for j in range(CELLS)]


def slab(x: float, y: float) -> fa.Atoms:
    """The surface layer plus one adatom at (x, y, HEIGHT)."""
    positions = surface() + [[x, y, HEIGHT]]
    cell = [[SPACING * CELLS, 0.0, 0.0], [0.0, SPACING * CELLS, 0.0], [0.0, 0.0, 24.0]]
    atoms = fa.Atoms(f"Ar{len(positions)}", positions, cell=cell, pbc=[True, True, False])
    atoms.calc = fa.LennardJones(epsilon=0.0103, sigma=3.40, cutoff=9.0)
    return atoms


def endpoints() -> tuple[fa.Atoms, fa.Atoms]:
    """Adjacent hollow sites: the adatom hops one lattice vector in x."""
    half = SPACING / 2
    return slab(half, half), slab(half + SPACING, half)


#: Every atom but the last is substrate, and stays put.
SUBSTRATE = list(range(CELLS * CELLS))


def plot(profile: fa.BandProfile, width: int = 46) -> None:
    """The energy profile as an ASCII plot - no matplotlib needed."""
    energies = list(profile.energies)
    low, high = min(energies), max(energies)
    span = high - low or 1.0
    for index, (distance, energy) in enumerate(zip(profile.distances, energies)):
        column = int(round(width * (energy - low) / span))
        marker = "*" if index == profile.highest else "o"
        print(f"  {index:2d}  s={distance:6.3f} A  "
              f"{'.' * column}{marker}  {1000 * (energy - energies[0]):7.1f} meV")


def main() -> None:
    initial, final = endpoints()
    print(f"{len(initial)} atoms, {len(SUBSTRATE)} of them held\n")

    # The barrier as a function of how many images you spend on it. A plain
    # band can only report the highest image it happens to have, so it
    # approaches the barrier from below; the climbing image is driven onto the
    # saddle itself and is close to right even on a coarse band.
    print("               plain NEB    CI-NEB     difference")
    for images in (4, 6, 8, 10):
        plain = fa.NEB(initial, final, images=images, k=0.5, fix=SUBSTRATE)
        plain.run("fire", fmax=FMAX, max_steps=20000)
        climbing = fa.NEB(initial, final, images=images, k=0.5, climb=True, fix=SUBSTRATE)
        climbing.run("fire", fmax=FMAX, max_steps=20000)

        low = 1000 * plain.profile().forward_barrier
        high = 1000 * climbing.profile().forward_barrier
        print(f"  {images:2d} images   {low:7.2f} meV  {high:7.2f} meV   {high - low:5.2f} meV")

    print("\nThe plain band converges on the barrier from below, because no image of an\n"
          "evenly spaced band lands on the saddle. The climbing image is driven onto it,\n"
          "and costs nothing extra: same images, one of them projected differently.")

    # AutoNEB spends its images where they matter instead of spreading them.
    automatic = fa.AutoNEB(initial, final, images=4, max_images=10, fmax=FMAX,
                           k=0.5, fix=SUBSTRATE)
    grown = automatic.run("fire")
    print(f"\nAutoNEB   {len(grown.profile.energies)} images (from 4) in {grown.stages} stages, "
          f"{grown.evaluations} force evaluations\n"
          f"          barrier {1000 * grown.profile.forward_barrier:.2f} meV, "
          f"converged={grown.converged}")

    print("\nconverged path:")
    plot(grown.profile)


if __name__ == "__main__":
    main()
