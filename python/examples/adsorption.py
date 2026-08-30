#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Adsorption sites on a platinum nanoparticle.

    python python/examples/adsorption.py

Builds a Pt55 Mackay icosahedron with the `build` module, finds every place an
adsorbate can bind, reduces those to the symmetry-distinct ones, and writes
out trajectories of hydrogen atoms and hydroxyl groups on each.

Writes four `.extxyz` files into the current directory, so run it from a
scratch directory rather than from the repository root.
"""

from __future__ import annotations

import numpy as np

import fastatoms as fa

#: Platinum: 3.92 A lattice constant, so 2.77 A between nearest neighbours.
NEAREST_NEIGHBOUR = 2.77


def main() -> None:
    # ------------------------------------------------------------------ build
    particle = fa.build.icosahedron("Pt", shells=2, d=NEAREST_NEIGHBOUR)
    print(f"Pt{len(particle)} Mackay icosahedron, {particle.get_volume():.0f} A^3 of box\n")

    finder = fa.adsorption.AdsorptionFinder(particle, height=1.9)
    print(finder.summary())

    # The site count is not arbitrary: the atop sites are the vertices of the
    # surface polyhedron, the bridges its edges and the hollows its faces, so
    # Euler's formula has to hold.
    counts = {}
    for site in finder.sites:
        counts[site.name] = counts.get(site.name, 0) + 1
    v = len(finder.surface_atoms)
    e = counts.get("bridge", 0)
    f = counts.get("hollow", 0) + counts.get("4-fold", 0)
    print(f"\nEuler check: V - E + F = {v} - {e} + {f} = {v - e + f}")
    print("An icosahedron has no square facet, so there are no 4-fold sites.")

    # -------------------------------------------------------------- symmetry
    print("\nSymmetry-distinct sites\n" + "-" * 62)
    print(f"{'':4} {'type':<8} {'formed by':<18} {'equivalent':>10}   distance from centre")
    centre = particle.positions.mean(axis=0)
    for unique in range(len(finder.unique_sites)):
        site = finder.sites[finder.unique_sites[unique]]
        radius = np.linalg.norm(np.asarray(site.position) - centre)
        print(f"{unique:<4} {site.name:<8} {str(list(site.atoms)):<18} "
              f"{len(finder.equivalent_sites(unique)):>10}   {radius:.3f} A")

    print(f"""
{len(finder.sites)} sites reduce to {len(finder.unique_sites)}. That is the whole point of the
reduction: a binding-energy scan over the distinct sites is {len(finder.unique_sites)} calculations
rather than {len(finder.sites)}, and the rest follow by symmetry.""")

    # ------------------------------------------------------- single H atoms
    hydrogen = finder.trajectory("H", unique_only=True)
    fa.write("pt55_H_unique.extxyz", hydrogen)
    print(f"\nwrote pt55_H_unique.extxyz   {len(hydrogen)} frames, "
          f"{len(hydrogen[0])} atoms each")

    everywhere = finder.trajectory("H", unique_only=False)
    fa.write("pt55_H_all.extxyz", everywhere)
    print(f"wrote pt55_H_all.extxyz      {len(everywhere)} frames")

    # ------------------------------------------------------- OH, oxygen down
    hydroxyl = finder.trajectory("OH", unique_only=True)
    fa.write("pt55_OH_unique.extxyz", hydroxyl)
    print(f"wrote pt55_OH_unique.extxyz  {len(hydroxyl)} frames, "
          f"{len(hydroxyl[0])} atoms each")

    # The orientation is the part worth checking: the oxygen has to be the atom
    # touching the platinum, with the hydrogen pointing away from the particle.
    print("\nOH orientation, per unique site\n" + "-" * 62)
    print(f"{'site':<8} {'O from centre':>14} {'H from centre':>14}   O-H along the normal")
    for frame, unique in zip(hydroxyl, range(len(finder.unique_sites))):
        site = finder.sites[finder.unique_sites[unique]]
        oxygen = np.asarray(frame.positions[len(particle)])
        hydrogen_at = np.asarray(frame.positions[len(particle) + 1])
        bond = hydrogen_at - oxygen
        alignment = np.dot(bond / np.linalg.norm(bond), np.asarray(site.normal))
        print(f"{site.name:<8} {np.linalg.norm(oxygen - centre):>13.3f}  "
              f"{np.linalg.norm(hydrogen_at - centre):>13.3f}   {alignment:>10.6f}")
    print("\nThe hydrogen is always further out than the oxygen, and the O-H bond")
    print("always lies along the site normal - which is what 'bind through the")
    print("oxygen' has to mean on a curved particle where every facet points")
    print("somewhere different.")

    # -------------------------------------------------------------- coverage
    hollows = finder.sites_of_type("hollow")
    series = finder.coverage_series("H", candidates=hollows, steps=10)
    fractions = finder.coverage_fractions(candidates=hollows, steps=10)
    fa.write("pt55_H_coverage.extxyz", series)

    print(f"\nCoverage series on the {len(hollows)} hollow sites\n" + "-" * 62)
    print(f"{'frame':<7} {'H atoms':>8} {'coverage':>10}   closest H-H")
    for k, (frame, theta) in enumerate(zip(series, fractions)):
        adsorbed = np.asarray(frame.positions[len(particle):])
        if len(adsorbed) > 1:
            gaps = [np.linalg.norm(adsorbed[i] - adsorbed[j])
                    for i in range(len(adsorbed)) for j in range(i + 1, len(adsorbed))]
            closest = f"{min(gaps):.3f} A"
        else:
            closest = "-"
        print(f"{k:<7} {len(adsorbed):>8} {theta:>9.3f}   {closest}")

    print(f"""
wrote pt55_H_coverage.extxyz  {len(series)} frames

Each hydrogen goes on the hollow furthest from the ones already placed, so the
series spreads over the particle instead of filling one facet first. Taking the
sites in the order they were found would give a very different - and much less
useful - set of structures at the same coverage.""")


if __name__ == "__main__":
    main()
