#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Restricting how atoms may move.

    python python/examples/constraints.py

An argon adatom sits above a square surface layer. The substrate is frozen
throughout - that much is routine - and the adatom is then asked three
different questions by three different constraints:

  nothing extra   where does it want to sit?          the binding site
  FixPlane        where does it want to sit *at this height*?
  FixLine         how high does it want to sit *over this spot*?

Each is a real calculation with a different answer, which is the point:
a constraint is not a convenience for holding scenery still, it is how you ask
a question about part of a system while the rest is held.

Nothing here writes a file.
"""

from __future__ import annotations

import numpy as np

import fastatoms as fa

SPACING = 3.8       # square lattice constant of the surface layer, angstrom
CELLS = 4           # a CELLS x CELLS patch, periodic in x and y
#: Deliberately not above a hollow, a bridge or a top site. On a symmetry point
#: the in-plane forces cancel and a constrained relaxation has nothing to do,
#: which would make two of the three runs below look like they had failed.
START = [SPACING * 0.95, SPACING * 0.55, 3.30]

SUBSTRATE = list(range(CELLS * CELLS))
ADATOM = CELLS * CELLS


def slab(*extra: fa.Constraint) -> fa.Atoms:
    """The surface layer plus one adatom, with the substrate always frozen."""
    positions = [[SPACING * i, SPACING * j, 0.0]
                 for i in range(CELLS) for j in range(CELLS)] + [START]
    cell = [[SPACING * CELLS, 0.0, 0.0], [0.0, SPACING * CELLS, 0.0], [0.0, 0.0, 24.0]]
    atoms = fa.Atoms(f"Ar{len(positions)}", positions, cell=cell, pbc=[True, True, False])
    atoms.calc = fa.LennardJones(epsilon=0.0103, sigma=3.40, cutoff=9.0)
    atoms.constraints = [fa.FixAtoms(SUBSTRATE), *extra]
    return atoms


def three_questions() -> None:
    print("An argon adatom over a frozen square lattice, started off-symmetry at")
    print(f"[{START[0]:.2f}, {START[1]:.2f}, {START[2]:.2f}].\n")
    print(f"{'constraints':<34} {'energy [eV]':>12} {'adatom':>24} {'dof gone':>9}")

    runs = [
        ("substrate frozen", ()),
        ("+ FixPlane([adatom], 'z')", (fa.FixPlane([ADATOM], "z"),)),
        ("+ FixLine([adatom], 'z')", (fa.FixLine([ADATOM], "z"),)),
    ]
    for label, extra in runs:
        atoms = slab(*extra)
        result = fa.LBFGS(fmax=1e-5, max_steps=3000).run(atoms)
        assert result.converged, label
        position = atoms.positions[ADATOM]
        print(f"{label:<34} {atoms.get_potential_energy():>12.5f} "
              f"[{position[0]:6.3f} {position[1]:6.3f} {position[2]:6.3f}] "
              f"{atoms.constrained_degrees_of_freedom():>9}")

    print("""
The first finds the hollow site and settles onto it. The second may slide to
the hollow site but not descend, so it stops directly above one at the height
it started with. The third may descend but not slide, so it stays over the
awkward spot it began on and only finds the best height there - the least
bound of the three, which is exactly what a constrained minimum should be.

Note the last column: three degrees of freedom go per frozen atom, one more
for the plane, two for the line.""")


def where_they_act() -> None:
    """The same restriction, seen in the force, the step and the velocity."""
    print("\n" + "=" * 78)
    print("A constraint has to say the same thing three times over\n")

    atoms = slab(fa.FixPlane([ADATOM], "z"))
    forces = atoms.get_forces()
    raw = np.asarray(atoms.calc.results.forces).reshape(-1, 3)

    print(f"force on the adatom, as the model computed it : "
          f"[{raw[ADATOM][0]:7.4f} {raw[ADATOM][1]:7.4f} {raw[ADATOM][2]:7.4f}]")
    print(f"force on the adatom, as the run sees it       : "
          f"[{forces[ADATOM][0]:7.4f} {forces[ADATOM][1]:7.4f} {forces[ADATOM][2]:7.4f}]")
    print("   the z component is gone, and the model's own results are untouched")

    # A step that tries to leave the plane is projected back onto it.
    proposed = atoms.positions.copy()
    proposed[ADATOM] += [0.20, 0.00, 0.50]
    atoms.move_to(proposed)
    moved = atoms.positions[ADATOM] - np.array(START)
    print(f"\nasked to move by [ 0.20  0.00  0.50], it moved by "
          f"[{moved[0]:5.2f} {moved[1]:5.2f} {moved[2]:5.2f}]")
    print("   the plane passes through the atom, so it keeps the height it had")

    # And a velocity it should not have is projected away.
    atoms = slab(fa.FixLine([ADATOM], "z"))
    fa.maxwell_boltzmann(atoms, 300.0, seed=7)
    velocities = atoms.get_velocities()
    print(f"\na Maxwell-Boltzmann draw at 300 K gives the frozen atom 0 "
          f"|v| = {np.linalg.norm(velocities[0]):.3e}")
    print(f"and the adatom, held to a line, only v_z    "
          f"|v_xy| = {np.linalg.norm(velocities[ADATOM][:2]):.3e}")
    print("   without this a frozen atom keeps whatever velocity it was handed")
    print("   and drifts away, however carefully its force was zeroed")


#: The outer ring of the substrate. Freezing a boundary and leaving the middle
#: mobile is how a surface calculation is actually set up: the frozen part
#: stands in for the bulk that is not being simulated.
RING = [i * CELLS + j for i in range(CELLS) for j in range(CELLS)
        if i in (0, CELLS - 1) or j in (0, CELLS - 1)]


def dynamics() -> None:
    print("\n" + "=" * 78)
    print("...and the dynamics counts the degrees of freedom it actually has\n")

    atoms = slab()
    atoms.constraints = [fa.FixAtoms(RING)]     # the outer ring only
    start = atoms.positions.copy()

    total = 3 * len(atoms)
    removed = atoms.constrained_degrees_of_freedom()
    free = total - removed
    print(f"{len(atoms)} atoms, {len(RING)} of them a frozen boundary")
    print(f"degrees of freedom : {total} - {removed} = {free}")

    fa.maxwell_boltzmann(atoms, 300.0, seed=3)
    print(f"Maxwell-Boltzmann draw at 300 K starts at : "
          f"{atoms.get_temperature(free):.1f} K")

    result = fa.run_md(atoms, 6000, timestep=2.0, temperature=300.0,
                       thermostat="langevin", friction=0.02)
    tail = [step.temperature for step in result.history[2000:]]
    spread = 100.0 * np.sqrt(2.0 / free)
    print(f"Langevin at 300 K settles at              : {np.mean(tail):.0f} K"
          f"   (a {spread:.0f}% spread is expected on {free} degrees of freedom)")
    print(f"frozen ring moved  : "
          f"{np.abs(atoms.positions[RING] - start[RING]).max():.1e} A")

    print("""
Both numbers depend on getting the count right, and the failure is quiet.
Counted over all 51 directions instead, the draw would start this system far
too hot while reporting 300 K. And because freezing atoms already pins the
centre of mass, subtracting three more for it is double-counting - and going
on to remove the net momentum each step takes the mobile atoms' own momentum
away from them, which on a slab with a single adsorbate settles a 300 K run
at 74 K.""")


def a_line_can_point_anywhere() -> None:
    print("\n" + "=" * 78)
    print("A line or a plane need not lie along an axis\n")

    for direction in ("z", [1, 1, 0], [1, 1, 1]):
        atoms = slab(fa.FixLine([ADATOM], direction))
        fa.LBFGS(fmax=1e-5, max_steps=3000).run(atoms)
        travelled = atoms.positions[ADATOM] - np.array(START)
        unit = np.asarray(atoms.constraints[1].direction)
        # Whatever it did, it did along the line and nowhere else.
        perpendicular = travelled - np.dot(travelled, unit) * unit
        print(f"along {str(direction):<11} moved [{travelled[0]:6.3f} {travelled[1]:6.3f} "
              f"{travelled[2]:6.3f}]   off the line: {np.linalg.norm(perpendicular):.1e} A")


def main() -> None:
    three_questions()
    where_they_act()
    dynamics()
    a_line_can_point_anywhere()


if __name__ == "__main__":
    main()
