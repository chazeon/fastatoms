#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Driving a DFT code from Python.

Shows the two halves of the calculator interface:

  1. Input generation, which needs no DFT installation at all - useful for
     staging jobs you will submit to a queue yourself.
  2. The evaluation loop, which is identical whichever engine is attached.

    python python/examples/dft_calculator.py examples/dft_silicon.yaml
"""

from __future__ import annotations

import sys

import numpy as np

import fastatoms as fa


def show_generated_inputs(atoms: fa.Atoms) -> None:
    """Every engine's input deck, built from the same structure."""
    print("=" * 70)
    print("Input generation (no DFT installation required)")
    print("=" * 70)

    vasp = fa.VASP(xc="pbe", encut=400, kpts=[8, 8, 8], ismear=0, sigma=0.05)
    print("\n--- INCAR " + "-" * 59)
    print(vasp.incar_string(), end="")
    print("--- KPOINTS " + "-" * 57)
    print(vasp.kpoints_string(), end="")
    print("--- POSCAR (first 8 lines) " + "-" * 42)
    print("\n".join(vasp.poscar_string(atoms).splitlines()[:8]))

    espresso = fa.QuantumEspresso(
        pseudopotentials={"Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF"},
        ecutwfc=40.0,
        conv_thr=1e-8,
        kpts=[8, 8, 8],
    )
    print("\n--- pw.in (namelists) " + "-" * 47)
    deck = espresso.input_string(atoms)
    print("\n".join(deck.splitlines()[: deck.splitlines().index("ATOMIC_SPECIES")]))

    siesta = fa.SIESTA(xc="pbe", mesh_cutoff=300.0, kpts=[8, 8, 8], dm_tolerance=1e-4)
    print("--- siesta.fdf (first 12 lines) " + "-" * 37)
    print("\n".join(siesta.fdf_string(atoms).splitlines()[:12]))


def equation_of_state(atoms: fa.Atoms, calculator) -> None:
    """Energy against volume - the same loop for any calculator.

    Swap ``calculator`` for ``fa.VASP(...)`` and nothing else changes; the DFT
    run just takes rather longer than the analytic potential used here.
    """
    print("\n" + "=" * 70)
    print(f"Equation of state with {calculator.name}")
    print("=" * 70)
    print(f"\n{'scale':>7} {'a [A]':>9} {'V [A^3]':>10} {'E [eV]':>12} "
          f"{'P [GPa]':>10} {'|F|max':>10}")

    reference_cell = np.asarray(atoms.cell)
    energies, volumes = [], []

    for scale in (0.96, 0.98, 1.00, 1.02, 1.04):
        probe = atoms.copy()
        probe.set_cell(reference_cell * scale, scale_atoms=True)
        probe.calc = calculator

        energy = probe.get_potential_energy()
        forces = probe.get_forces()
        stress = probe.get_stress()
        pressure = fa.pressure_from_stress(stress) / fa.units.GPa

        energies.append(energy)
        volumes.append(probe.get_volume())
        print(f"{scale:>7.2f} {probe.cell.lengths[0]:>9.4f} {probe.get_volume():>10.3f} "
              f"{energy:>12.6f} {pressure:>10.4f} "
              f"{np.linalg.norm(forces, axis=1).max():>10.2e}")

    # A parabola through E(V) gives the equilibrium volume and bulk modulus.
    coefficients = np.polyfit(volumes, energies, 2)
    v0 = -coefficients[1] / (2 * coefficients[0])
    bulk_modulus = 2 * coefficients[0] * v0 / fa.units.GPa
    print(f"\nequilibrium volume : {v0:.3f} A^3  ({v0 ** (1 / 3):.4f} A cubic edge)")
    print(f"bulk modulus       : {bulk_modulus:.1f} GPa")
    print(f"engine invocations : {calculator.call_count}")


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else "examples/dft_silicon.yaml"
    deck = fa.read_input(path)
    atoms = deck.atoms

    print(f"FastAtoms {fa.__version__}")
    print(f"{atoms!r}\n")

    # The calculator described by the file. Building it costs nothing and does
    # not touch the DFT installation.
    dft = deck.make_calculator()
    print(f"calculator from {path}: {dft!r}")
    print(f"  implements : {dft.implemented_properties}")
    print(f"  command    : {dft.command}")
    print(f"  directory  : {dft.directory}")

    show_generated_inputs(atoms)

    # Actually running VASP needs VASP. The analytic calculator stands in so
    # this example is runnable anywhere; the loop below is engine-agnostic.
    print("\nNo DFT engine is invoked here. To run the real thing:")
    print("    atoms.calc = deck.make_calculator()")
    print("    atoms.get_potential_energy()")
    print("Or stage the inputs for a batch queue without running:")
    print("    deck.make_calculator().write_inputs(atoms, 'job-01')")

    equation_of_state(atoms, fa.LennardJones(epsilon=0.05, sigma=2.5, cutoff=8.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
