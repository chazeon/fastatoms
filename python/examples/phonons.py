#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harmonic phonons by finite displacements.

    python python/examples/phonons.py examples/phonons_argon.yaml

Everything here runs on the built-in Lennard-Jones calculator, so no DFT
installation is needed. Swapping in ``fa.Vasp(...)`` changes nothing else: the
module only ever asks a calculator for the forces on a displaced supercell.
"""

from __future__ import annotations

import sys

import numpy as np

import fastatoms as fa

LJ = dict(epsilon=0.0103, sigma=3.40, cutoff=8.0)


def argon(a: float = 5.26) -> fa.Atoms:
    """The primitive fcc cell, at the Lennard-Jones equilibrium volume."""
    cell = [[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]]
    atoms = fa.Atoms("Ar", [[0.0, 0.0, 0.0]], cell=cell, pbc=True)
    atoms.calc = fa.LennardJones(**LJ)
    return atoms


def rocksalt(a: float = 5.64) -> fa.Atoms:
    """NaCl in its primitive cell: two atoms, so there are optical branches."""
    cell = [[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]]
    atoms = fa.Atoms("NaCl", scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                     cell=cell, pbc=True)
    atoms.calc = fa.LennardJones(**LJ)
    return atoms


def diamond(a: float = 5.43) -> fa.Atoms:
    """Silicon: two atoms in the primitive cell, on sites that symmetry ties
    together, which is where the reduction pays best."""
    cell = [[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]]
    atoms = fa.Atoms("Si2", scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
                     cell=cell, pbc=True)
    atoms.calc = fa.LennardJones(**LJ)
    return atoms


# --------------------------------------------------------------------------
def what_symmetry_saves() -> None:
    """The point of the symmetry reduction, in one table."""
    print("=" * 74)
    print("Displacements needed, with and without symmetry")
    print("=" * 74)
    print(f"\n{'structure':<28} {'supercell':>10} {'atoms':>7} {'with':>7} "
          f"{'without':>9} {'saving':>8}")

    cases = [("argon, fcc", argon(), [2, 2, 2]),
             ("argon, fcc", argon(), [3, 3, 3]),
             ("silicon, diamond", diamond(), [2, 2, 2]),
             ("rocksalt NaCl", rocksalt(), [2, 2, 2])]
    for label, atoms, supercell in cases:
        ph = fa.Phonons(atoms, supercell=supercell)
        few, many = ph.calculation_count, ph.calculation_count_without_symmetry
        print(f"{label:<28} {str(supercell):>10} {len(ph.supercell):>7} {few:>7} "
              f"{many:>9} {many / few:>7.0f}x")

    print("\nEvery row is the same physics; the first column is how many DFT runs\n"
          "it costs. Pass symmetry=False to displace everything and compare.")


def frequencies_along_a_path() -> None:
    """Run the displacements, then ask for frequencies anywhere in the zone."""
    print("\n" + "=" * 74)
    print("Dispersion of fcc argon, 3x3x3 supercell")
    print("=" * 74)

    ph = fa.Phonons(argon(), supercell=[3, 3, 3], displacement=0.01)
    ph.run()

    print(f"\n{ph!r}")
    print(f"residual force on the undisplaced cell: {ph.residual_force:.2e} eV/A")
    print(f"acoustic sum rule violation           : "
          f"{ph.force_constants.acoustic_sum_violation():.2e} eV/A^2")

    # Gamma -> X -> W -> L in the fcc convention.
    path = {"G": [0.0, 0.0, 0.0], "X": [0.0, 0.5, 0.5], "W": [0.25, 0.5, 0.75],
            "L": [0.5, 0.5, 0.5]}
    print(f"\n{'q':<6} {'frequencies [THz]':<34} {'cm^-1':<28} meV")
    for label, q in path.items():
        f = ph.frequencies(q)
        print(f"{label:<6} {np.array2string(f, precision=4, suppress_small=True):<34} "
              f"{np.array2string(f * fa.Phonons.THZ_TO_WAVENUMBER, precision=1):<28} "
              f"{np.array2string(f * fa.Phonons.THZ_TO_MEV, precision=2)}")

    print("\nThe three branches go to zero at G: that is the acoustic sum rule,\n"
          "and it is the sharpest check that the force constants are right.")


def a_density_of_states() -> None:
    """A mesh over the zone, binned - the simplest possible phonon DOS."""
    print("\n" + "=" * 74)
    print("Phonon density of states from an 8x8x8 mesh")
    print("=" * 74)

    ph = fa.Phonons(argon(), supercell=[3, 3, 3])
    ph.run()
    frequencies = ph.mesh_frequencies(8, 8, 8)

    counts, edges = np.histogram(frequencies, bins=24, range=(0.0, 2.2))
    scale = 46.0 / max(counts.max(), 1)
    print()
    for count, low in zip(counts, edges):
        print(f"{low:5.2f} THz | {'#' * int(round(count * scale))}")
    print(f"\n{len(frequencies)} frequencies, "
          f"highest {frequencies.max():.3f} THz "
          f"({frequencies.max() * fa.Phonons.THZ_TO_WAVENUMBER:.1f} cm^-1)")


def forces_from_somewhere_else() -> None:
    """The offline workflow: write the supercells out, feed the forces back.

    This is what a queued DFT run looks like - `displaced_supercells()` gives
    the structures, and `set_forces()` takes the results whenever they arrive.
    """
    print("\n" + "=" * 74)
    print("Running the displacements yourself")
    print("=" * 74)

    ph = fa.Phonons(argon(), supercell=[2, 2, 2])
    cells = ph.displaced_supercells()
    print(f"\n{len(cells)} supercells to run, {len(cells[0])} atoms each")
    for displacement, cell in zip(ph.displacements, cells):
        print(f"  atom {displacement.atom}, "
              f"d = {np.round(displacement.vector, 4)} A -> {cell!r}")

    # Here a real workflow would write POSCARs, submit jobs and wait. The
    # forces are all that has to come back, in the same order.
    calc = fa.LennardJones(**LJ)
    forces = [calc.get_forces(cell) for cell in cells]
    ph.set_forces(forces)

    print(f"\nfrequencies at X: {np.round(ph.frequencies([0.0, 0.5, 0.5]), 4)} THz")


def symmetry_of_the_crystal() -> None:
    """What the symmetry finder sees, without any phonons involved."""
    print("\n" + "=" * 74)
    print("Space group symmetry")
    print("=" * 74)

    for label, atoms in (("argon, fcc", argon()), ("rocksalt NaCl", rocksalt())):
        symmetry = fa.find_symmetry(atoms)
        print(f"\n{label}: {symmetry.summary()}")
        print(f"  point group order      : {len(symmetry.point_group)}")
        print(f"  independent atoms      : {list(symmetry.independent_atoms)} "
              f"of {len(atoms)}")
        print(f"  site symmetry of atom 0: "
              f"{len(fa.site_symmetry(symmetry, 0))} operations")

    conventional = fa.make_supercell(argon(), [[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    print(f"\nprimitive -> conventional cubic cell: {conventional!r}, "
          f"a = {conventional.atoms.cell.lengths[0]:.4f} A")


def drive_from_the_input_file(path: str) -> None:
    """Everything above, but described by a YAML file instead of by code."""
    print("\n" + "=" * 74)
    print(f"Driving from {path}")
    print("=" * 74)

    deck = fa.read_input(path)
    atoms = deck.atoms
    atoms.calc = deck.make_calculator()
    ph = deck.make_phonons(atoms)

    print(f"\nstructure : {atoms!r}")
    print(f"calculator: {atoms.calc!r}")
    print(f"phonons   : {ph!r}")

    ph.run()
    bands = ph.band_structure([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]])
    print(f"\nband structure along [q00]:\n{np.round(bands, 4)}")


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else "examples/phonons_argon.yaml"
    print(f"FastAtoms {fa.__version__}\n")
    what_symmetry_saves()
    frequencies_along_a_path()
    a_density_of_states()
    forces_from_somewhere_else()
    symmetry_of_the_crystal()
    drive_from_the_input_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
