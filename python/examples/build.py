#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generating structures.

    python python/examples/build.py

Four families, all returning an ``Atoms`` with its cell and boundary
conditions already set:

  bulk lattices   fcc, bcc, hcp, diamond and the binary ones
  2D materials    graphene, hexagonal boron nitride, the dichalcogenides
  nanotubes       any hexagonal sheet rolled by its (n, m) indices
  nanoparticles   icosahedra, cuboctahedra, decahedra, and Wulff shapes

The last section does something with them rather than merely printing them:
it relaxes icosahedra against cuboctahedra of the same size on a
Lennard-Jones potential, and recovers the known answer.

Nothing here writes a file.
"""

from __future__ import annotations

import numpy as np

import fastatoms as fa

build = fa.build


def shortest_bond(atoms: fa.Atoms) -> float:
    positions = atoms.positions
    best = np.inf
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            best = min(best, float(np.linalg.norm(positions[i] - positions[j])))
    return best


def bulk_lattices() -> None:
    print("Bulk lattices\n" + "-" * 62)
    print(f"{'':22} {'atoms':>5} {'volume':>9} {'nearest':>9}")
    rows = [
        ("fcc Al, primitive", build.fcc("Al", 4.05)),
        ("fcc Al, conventional", build.fcc("Al", 4.05, cubic=True)),
        ("bcc Fe", build.bcc("Fe", 2.87, cubic=True)),
        ("hcp Mg", build.hcp("Mg", 3.21)),
        ("diamond Si", build.diamond("Si", 5.43, cubic=True)),
        ("rocksalt NaCl", build.rocksalt("Na", "Cl", 5.64, cubic=True)),
        ("zincblende GaAs", build.zincblende("Ga", "As", 5.65, cubic=True)),
    ]
    for label, atoms in rows:
        print(f"{label:22} {len(atoms):>5} {atoms.get_volume():>9.2f} "
              f"{shortest_bond(atoms.repeat(2, 2, 2)):>9.4f}")
    print("\n`a` is always the conventional cubic constant, never the bond length:")
    print(f"  fcc     nearest neighbour = a/sqrt(2)   = {4.05 / np.sqrt(2):.4f} A")
    print(f"  bcc                       = a*sqrt(3)/2 = {2.87 * np.sqrt(3) / 2:.4f} A")
    print(f"  diamond                   = a*sqrt(3)/4 = {5.43 * np.sqrt(3) / 4:.4f} A")


def two_dimensional() -> None:
    print("\n\n2D materials\n" + "-" * 62)
    print(f"{'':22} {'atoms':>5} {'a [A]':>8} {'bond':>8} {'c [A]':>8}")
    rows = [
        ("graphene", build.graphene()),
        ("graphene, rectangular", build.graphene_rectangular()),
        ("hexagonal BN", build.honeycomb(["B", "N"], 2.51)),
        ("silicene (buckled)", build.honeycomb("Si", 3.87, buckling=0.44)),
        ("MoS2, 1H", build.tmd("Mo", "S")),
        ("MoS2, 1T", build.tmd("Mo", "S", phase="1T")),
        ("WSe2, 1H", build.tmd("W", "Se")),
    ]
    for label, atoms in rows:
        print(f"{label:22} {len(atoms):>5} {atoms.cell.lengths[0]:>8.3f} "
              f"{shortest_bond(atoms.repeat(2, 2, 1)):>8.3f} {atoms.cell.lengths[2]:>8.2f}")

    sheet = build.tmd("Mo", "S", vacuum=15.0)
    height = sheet.positions[:, 2]
    print(f"\nThe vacuum is the gap, not the cell height. MoS2 asked for 15 A:")
    print(f"  layer thickness {height.max() - height.min():.3f} A"
          f" + vacuum 15.000 A = cell {sheet.cell.lengths[2]:.3f} A")
    print(f"  {len(build.known_tmds())} dichalcogenides have tabulated lattice constants")


def nanotubes() -> None:
    print("\n\nNanotubes\n" + "-" * 62)
    print(f"{'(n, m)':>8} {'atoms':>6} {'diameter':>9} {'angle':>7} {'period':>8}  kind")
    for n, m in ((5, 5), (9, 0), (10, 0), (6, 3), (8, 4), (10, 10)):
        geometry = build.tube_geometry(n, m)
        tube = build.nanotube(n, m)
        kind = ("armchair" if n == m else "zigzag" if m == 0 else "chiral")
        print(f"{f'({n}, {m})':>8} {len(tube):>6} {2 * geometry.radius:>9.3f} "
              f"{geometry.chiral_angle:>6.1f}d {geometry.period:>8.3f}  {kind}")

    # Rolling is not limited to graphene: any hexagonal sheet works, and one
    # with thickness becomes a tube with two walls.
    tube = build.roll(build.tmd("Mo", "S"), 10, 0)
    middle = tube.positions.mean(axis=0)
    radii = np.linalg.norm(tube.positions[:, :2] - middle[:2], axis=1)
    symbols = tube.get_chemical_symbols()
    sulphur = radii[[s == "S" for s in symbols]]
    metal = radii[[s == "Mo" for s in symbols]]
    print(f"\nA (10, 0) MoS2 tube, {len(tube)} atoms: the layer's thickness becomes two walls")
    print(f"  inner S at {sulphur.min():.2f} A, Mo at {metal.mean():.2f} A, "
          f"outer S at {sulphur.max():.2f} A")


def nanoparticles() -> None:
    print("\n\nNanoparticles\n" + "-" * 62)
    print("Closed-shell sizes - the same for both, because they have the same")
    print("shell structure and differ only in how it is folded:")
    shells = [1, 2, 3, 4, 5]
    print(f"  shells        {'  '.join(f'{s:>5}' for s in shells)}")
    print(f"  icosahedron   "
          f"{'  '.join(f'{len(build.icosahedron("Ag", s, 2.89)):>5}' for s in shells)}")
    print(f"  cuboctahedron "
          f"{'  '.join(f'{len(build.cuboctahedron("Ag", s, 4.09)):>5}' for s in shells)}")

    ico = build.icosahedron("Ag", 3, 2.89)
    print(f"\nThe icosahedron is strained: asked for 2.89 A within a shell, the")
    print(f"shells themselves come out {shortest_bond(ico):.3f} A apart - a factor of")
    print(f"{2.89 / shortest_bond(ico):.4f}. Twenty tetrahedra cannot meet at a point without it.")

    deca = build.decahedron("Ag", 3, 4, a=4.09)
    marks = build.decahedron("Ag", 3, 4, r=1, a=4.09)
    print(f"\nDecahedra: Ino {len(deca)} atoms, Marks (r=1) {len(marks)} atoms - the")
    print("re-entrant notch at each twin boundary is what distinguishes them.")

    print("\nWulff shapes of silver, 4.09 A, at various {111}/{100} energy ratios:")
    print(f"  {'ratio':>7} {'atoms':>6}   shape")
    for ratio, description in ((0.80, "octahedron-like: cheap {111} takes over"),
                               (1.00, "cube and octahedron equally weighted"),
                               (1.15, "cuboctahedral: both facets present"),
                               (1.40, "cube-like: dear {111} shrinks away")):
        particle = build.wulff("Ag", "fcc", 4.09, {"100": 1.0, "111": ratio}, atoms=500)
        print(f"  {ratio:>7.2f} {len(particle):>6}   {description}")


def which_shape_wins() -> None:
    """The builders are only worth having if what they make is usable."""
    print("\n\nWhich shape a Lennard-Jones cluster prefers\n" + "-" * 62)

    epsilon, sigma = 0.0103, 3.40
    d = 2 ** (1 / 6) * sigma          # the pair minimum
    print(f"{'atoms':>6} {'icosahedron':>13} {'cuboctahedron':>15} {'difference':>12}")
    for shells, count in ((1, 13), (2, 55), (3, 147), (4, 309)):
        energies = []
        for cluster in (build.icosahedron("Ar", shells, d),
                        build.cuboctahedron("Ar", shells, d * np.sqrt(2))):
            cluster.calc = fa.LennardJones(epsilon=epsilon, sigma=sigma, cutoff=12.0)
            fa.LBFGS(fmax=1e-3, max_steps=2000).run(cluster)
            energies.append(cluster.get_potential_energy() / len(cluster))
        print(f"{count:>6} {energies[0]:>13.5f} {energies[1]:>15.5f} "
              f"{energies[1] - energies[0]:>12.5f}")

    print("""
Energy per atom in eV, after relaxing each. The icosahedron wins at every one
of these sizes, which is the known result for a Lennard-Jones potential: its
five per cent internal strain costs less than the extra surface a cuboctahedron
of the same size exposes. Real metals, with directional bonding and a much
stiffer response to strain, cross over to fcc shapes far earlier.""")


def main() -> None:
    bulk_lattices()
    two_dimensional()
    nanotubes()
    nanoparticles()
    which_shape_wins()


if __name__ == "__main__":
    main()
