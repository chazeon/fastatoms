#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Geometry optimization: the four algorithms, and cell relaxation.

    python python/examples/relaxation.py examples/relax_argon.yaml

Everything here runs on the built-in Lennard-Jones calculator, so no DFT
installation is needed. Swapping in ``fa.VASP(...)`` changes nothing else.
"""

from __future__ import annotations

import sys

import numpy as np

import fastatoms as fa


def strained_crystal(a: float = 5.50, jitter: float = 0.12, seed: int = 11) -> fa.Atoms:
    """Face-centred-cubic argon, stretched and disordered."""
    basis = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    atoms = fa.Atoms("Ar4", scaled_positions=basis, cell=a, pbc=True).repeat(2, 2, 2)
    rng = np.random.default_rng(seed)
    atoms.positions += rng.uniform(-jitter, jitter, atoms.positions.shape)
    atoms.calc = fa.LennardJones(epsilon=0.0103, sigma=3.40, cutoff=8.5)
    return atoms


def compare_algorithms() -> None:
    """The same structure, the same threshold, four different routes downhill."""
    print("=" * 74)
    print("Relaxing 32 atoms to fmax = 0.005 eV/A")
    print("=" * 74)
    print(f"\n{'algorithm':<20} {'steps':>7} {'force calls':>12} {'energy [eV]':>15} "
          f"{'fmax':>10}")

    for name in ("sd", "cg", "bfgs", "lbfgs"):
        atoms = strained_crystal()
        result = fa.optimize(atoms, name, fmax=0.005, max_steps=5000)
        label = {"sd": "steepest descent", "cg": "conjugate gradient",
                 "bfgs": "BFGS", "lbfgs": "L-BFGS"}[name]
        print(f"{label:<20} {result.steps:>7} {result.evaluations:>12} "
              f"{result.energy:>15.6f} {result.fmax:>10.2e}")

    print("\nAll four reach the same minimum. The force calls are the column that\n"
          "matters when each one is a DFT run.")


def watch_a_relaxation() -> None:
    """A callback sees every step, which is how you log or write a trajectory."""
    print("\n" + "=" * 74)
    print("Following an L-BFGS run step by step")
    print("=" * 74)
    print(f"\n{'step':>5} {'energy [eV]':>15} {'fmax [eV/A]':>13} {'dE [eV]':>12} "
          f"{'max move [A]':>13}")

    atoms = strained_crystal()
    optimizer = fa.LBFGS(fmax=0.005)
    optimizer.attach(
        lambda s: print(f"{s.step:>5} {s.energy:>15.6f} {s.fmax:>13.3e} "
                        f"{s.energy_change:>12.2e} {s.max_displacement:>13.4f}")
    )
    result = optimizer.run(atoms)
    print(f"\n{result.message}")


def relax_the_cell() -> None:
    """Positions and lattice vectors together, and under pressure."""
    print("\n" + "=" * 74)
    print("Relaxing the cell")
    print("=" * 74)
    print(f"\n{'external pressure':>18} {'a [A]':>9} {'volume [A^3]':>14} "
          f"{'residual P':>13} {'steps':>7}")

    for pressure_gpa in (0.0, 0.5, 1.0, 2.0):
        atoms = strained_crystal(a=5.50, jitter=0.02)
        pressure = pressure_gpa * fa.units.GPa  # GPa -> eV/A^3
        result = fa.optimize(
            atoms, "lbfgs", fmax=1e-4, max_steps=500, relax_cell=True, pressure=pressure
        )
        residual = fa.pressure_from_stress(atoms.get_stress()) / fa.units.GPa
        # The cell was doubled in each direction, so halve it to report the
        # conventional lattice constant.
        print(f"{pressure_gpa:>15.1f} GPa {atoms.cell.lengths[0] / 2:>9.4f} "
              f"{atoms.get_volume() / 8:>14.3f} {residual:>10.4f} GPa {result.steps:>7}")

    print("\nThe residual pressure tracks the applied one, which is the check that\n"
          "the stress and the cell gradient agree.")


def drive_from_the_input_file(path: str) -> None:
    """Both the calculator and the optimizer come out of the YAML file."""
    print("\n" + "=" * 74)
    print(f"Driving from {path}")
    print("=" * 74)

    deck = fa.read_input(path)
    atoms = deck.atoms
    atoms.calc = deck.make_calculator()
    optimizer = deck.make_optimizer()

    print(f"\nstructure : {atoms!r}")
    print(f"calculator: {atoms.calc!r}")
    print(f"optimizer : {optimizer!r}")

    start = atoms.get_potential_energy()
    result = optimizer.run(atoms)

    print(f"\nenergy    : {start:.6f} -> {result.energy:.6f} eV "
          f"({result.energy - start:+.6f})")
    print(f"lattice   : {atoms.cell.lengths[0]:.4f} A")
    print(f"converged : {result.converged} in {result.steps} steps, "
          f"{result.evaluations} force calls")


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else "examples/relax_argon.yaml"
    print(f"FastAtoms {fa.__version__}\n")
    compare_algorithms()
    watch_a_relaxation()
    relax_the_cell()
    drive_from_the_input_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
