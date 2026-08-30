#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read a FastAtoms YAML file and report what it contains.

The Python mirror of ``examples/run_input.cpp``.

    python python/examples/read_input.py examples/silicon.yaml 3.0
"""

from __future__ import annotations

import sys

import numpy as np

import fastatoms as fa


def describe_structure(atoms: fa.Atoms) -> None:
    print(f"structure     : {atoms!r}")
    print(f"formula       : {atoms.get_chemical_formula()}")
    print(f"atoms         : {len(atoms)}")
    if atoms.cell.is_invertible:
        a, b, c = atoms.cell.lengths
        alpha, beta, gamma = atoms.cell.angles
        print(f"cell          : a={a:.4f} b={b:.4f} c={c:.4f}  "
              f"alpha={alpha:.2f} beta={beta:.2f} gamma={gamma:.2f}")
        print(f"volume        : {atoms.get_volume():.4f} A^3 "
              f"({atoms.get_volume() / len(atoms):.4f} A^3 / atom)")
    print(f"centre of mass: {np.round(atoms.get_center_of_mass(), 4)}")


def describe_simulation(simulation: fa.SimulationParameters) -> None:
    print(f"\ntask          : {simulation.task_name}")
    if simulation.calculator.name:
        print(f"calculator    : {simulation.calculator.name} "
              f"{simulation.calculator.parameters}")
    if simulation.task == fa.Task.md:
        md = simulation.md
        print(f"md            : {md.ensemble}, {md.steps} steps of "
              f"{md.timestep_fs} fs at {md.temperature_K} K")
    if simulation.task == fa.Task.optimize:
        opt = simulation.optimizer
        print(f"optimizer     : {opt.algorithm}, fmax={opt.fmax} eV/A, "
              f"max {opt.max_steps} steps")
    if simulation.output.trajectory:
        print(f"trajectory    : {simulation.output.trajectory} "
              f"every {simulation.output.interval} steps")


def describe_neighbors(atoms: fa.Atoms, cutoff: float) -> None:
    neighbours = fa.neighbor_list(atoms, cutoff)
    coordination = neighbours.coordination_numbers()
    print(f"\nneighbour list (cutoff {cutoff:.2f} A)")
    print(f"pairs         : {len(neighbours)}")
    print(f"coordination  : mean {coordination.mean():.2f}, "
          f"range {coordination.min()}..{coordination.max()}")
    if len(neighbours):
        print(f"shortest      : {neighbours.distances.min():.4f} A")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(f"usage: {argv[0]} <input.yaml> [neighbour cutoff in A]", file=sys.stderr)
        return 2

    try:
        deck = fa.read(argv[1])
    except fa.InputError as error:
        print(f"input error: {error}", file=sys.stderr)
        return 1

    print(f"FastAtoms {fa.__version__}\n")
    describe_structure(deck.atoms)
    describe_simulation(deck.simulation)

    if len(argv) > 2:
        describe_neighbors(deck.atoms, float(argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
