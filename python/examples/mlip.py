#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Machine-learning potentials: the two backends, and what each one is for.

    python python/examples/mlip.py examples/mlip_water.yaml

Unlike the other examples, this one needs a trained model. Export a MACE
checkpoint once to get the in-process path::

    ~/miniconda3/envs/mace_env/bin/python -m fastatoms.mlip.export \\
        ~/.cache/mace/MACE-OFF23_small.model -o mace-off23-small.pt

Point ``--model`` at the archive, or at a checkpoint to use the worker. Every
section skips itself with an explanation when what it needs is missing, so the
script is worth running even on a machine with none of this installed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

import fastatoms as fa


def water_box(molecules: int = 8, spacing: float = 4.93, seed: int = 0) -> fa.Atoms:
    """A small box of water, jittered so nothing sits on a symmetry point."""
    rng = np.random.default_rng(seed)
    side = int(round(molecules ** (1 / 3)))
    positions, symbols = [], []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                origin = np.array([i, j, k]) * spacing + rng.normal(0, 0.1, 3)
                positions += [origin, origin + [0.96, 0, 0], origin + [-0.24, 0.93, 0]]
                symbols += ["O", "H", "H"]
    cell = side * spacing
    return fa.Atoms("".join(symbols), positions, cell=cell, pbc=True)


# --------------------------------------------------------------------------
def what_this_build_can_do() -> None:
    print("=" * 74)
    print("What is available here")
    print("=" * 74)
    print(f"\nLibTorch in process : {fa.torch_available()}"
          f"{' (' + fa.torch_version() + ')' if fa.torch_available() else ''}")
    if not fa.torch_available():
        print("  Rebuild to evaluate models without an interpreter:")
        print("    cmake -S . -B build -DFASTATOMS_WITH_TORCH=ON \\")
        print("          -DFASTATOMS_TORCH_PYTHON=$HOME/miniconda3/envs/mace_env/bin/python")

    print(f"worker script       : {fa.mlip_worker_script()}")
    print(f"\n{'family':<12} {'conda environment':<20} interpreter")
    for family in ("mace", "mattersim", "chgnet", "nequip", "allegro"):
        environment = fa.mlip_environment_for(family)
        interpreter = fa.conda_python(environment) or "not found"
        print(f"{family:<12} {environment:<20} {interpreter}")


def the_graph_a_model_sees(atoms: fa.Atoms) -> None:
    """Step 3 of the module: an Atoms object as tensors."""
    print("\n" + "=" * 74)
    print("What a potential is actually handed")
    print("=" * 74)

    graph = fa.mlip_graph(atoms, cutoff=5.0)
    print(f"\n{graph!r}")
    print(f"  positions      {graph.positions.shape}   {graph.positions.dtype}")
    print(f"  atomic_numbers {graph.atomic_numbers.shape}      "
          f"{graph.atomic_numbers.dtype}")
    print(f"  edge_index     {graph.edge_index.shape}  receivers on row 0, senders on row 1")
    print(f"  unit_shifts    {graph.unit_shifts.shape}  integer cell offsets")

    receivers, senders = graph.edge_index
    vectors = (graph.positions[senders] + graph.unit_shifts @ graph.cell
               - graph.positions[receivers])
    distances = np.linalg.norm(vectors, axis=1)
    print(f"\n  r_ij = positions[sender] + unit_shifts @ cell - positions[receiver]")
    print(f"  {graph.nedges} edges, {distances.min():.3f} to {distances.max():.3f} A")
    print("\nThe edge list is the same O(N) neighbour list the rest of FastAtoms\n"
          "uses, so the graph costs no more than one force evaluation would.")


def a_backend_of_your_own(atoms: fa.Atoms) -> None:
    """Any object with an `evaluate` method is a potential, no model needed."""
    print("\n" + "=" * 74)
    print("A potential written in Python")
    print("=" * 74)

    class Harmonic:
        """Every atom on a spring to where it started. Energy only."""

        def __init__(self, reference, k=1.5):
            self.reference = np.array(reference, copy=True)
            self.k = k
            self.needs_edges = False

        def evaluate(self, graph, need_forces, need_stress):
            delta = graph.positions - self.reference
            out = {"energy": 0.5 * self.k * float((delta ** 2).sum())}
            if need_forces:
                out["forces"] = -self.k * delta
            return out

    probe = atoms.copy()
    calc = fa.MLIP(cutoff=5.0)
    calc.set_backend(Harmonic(probe.positions))
    probe.calc = calc

    probe.positions[0] += [0.1, -0.2, 0.05]
    print(f"\nenergy : {probe.get_potential_energy():.6f} eV "
          f"(expected {0.5 * 1.5 * (0.1**2 + 0.2**2 + 0.05**2):.6f})")
    print(f"force  : {np.round(probe.get_forces()[0], 6)} eV/A")
    print("\nUseful for testing a workflow, and for plugging in a model that\n"
          "FastAtoms has no C++ path for yet.")


def run_a_model(model: str, family: str, python: str) -> fa.Atoms | None:
    print("\n" + "=" * 74)
    print(f"Running {family}")
    print("=" * 74)

    atoms = water_box(8)
    try:
        atoms.calc = fa.make_calculator(family, model=model,
                                        **({"python": python} if python else {}))
        start = time.time()
        energy = atoms.get_potential_energy()
        first = time.time() - start
    except fa.CalculatorError as error:
        print(f"\nskipped: {error}")
        return None

    info = atoms.calc.model
    print(f"\nmodel     : {info.path}")
    print(f"backend   : {info.backend}   device {info.device}   dtype {info.dtype}")
    print(f"cutoff    : {info.cutoff} A")
    print(f"species   : {len(info.species)} elements" if info.species else "species   : any")
    print(f"returns   : forces={info.provides_forces} stress={info.provides_stress}"
          "   (what it does not, FastAtoms differentiates)")

    print(f"\n{len(atoms)} atoms")
    print(f"energy    : {energy:.6f} eV  ({energy / len(atoms):.6f} eV/atom)")
    print(f"fmax      : {np.abs(atoms.get_forces()).max():.6f} eV/A")
    print(f"pressure  : {fa.pressure_from_stress(atoms.get_stress()) * fa.units.GPa:.4f} GPa")
    print(f"first call: {first:.2f} s (loading the model)")

    steps = 5
    start = time.time()
    for _ in range(steps):
        atoms.positions[0] += [0.001, 0.0, 0.0]
        atoms.get_potential_energy()
        atoms.get_forces()
    print(f"steady    : {(time.time() - start) / steps * 1000:.1f} ms per evaluation")
    return atoms


def relax_with_a_model(atoms: fa.Atoms | None) -> None:
    if atoms is None:
        return
    print("\n" + "=" * 74)
    print("Relaxing with the same model")
    print("=" * 74)

    start = atoms.get_potential_energy()
    result = fa.optimize(atoms, "lbfgs", fmax=0.05, max_steps=100)
    print(f"\nenergy    : {start:.6f} -> {result.energy:.6f} eV "
          f"({result.energy - start:+.6f})")
    print(f"converged : {result.converged} in {result.steps} steps, "
          f"{result.evaluations} model calls")
    print(f"fmax      : {result.fmax:.4f} eV/A")
    print("\nThe optimizer, the dynamics and the phonons do not know or care\n"
          "that the forces came from a neural network.")


def drive_from_the_input_file(path: str) -> None:
    print("\n" + "=" * 74)
    print(f"Driving from {path}")
    print("=" * 74)

    if not os.path.exists(path):
        print(f"\nskipped: no such file")
        return
    deck = fa.read(path)
    print(f"\nstructure : {deck.atoms!r}")
    print(f"calculator: {deck.simulation.calculator.name} "
          f"{deck.simulation.calculator.parameters}")
    try:
        atoms = deck.atoms
        atoms.calc = deck.make_calculator()
        atoms.calc.load()
        print(f"loaded    : {atoms.calc.model!r}")
        deck.make_optimizer().run(atoms)
        print(f"relaxed   : {atoms.get_potential_energy():.6f} eV, "
              f"fmax {np.abs(atoms.get_forces()).max():.4f} eV/A")
    except fa.CalculatorError as error:
        print(f"skipped: {error}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default="examples/mlip_water.yaml")
    parser.add_argument("--model", default=os.environ.get("FASTATOMS_TEST_MLIP_MODEL", ""),
                        help="a .pt archive (in process) or a checkpoint (worker)")
    parser.add_argument("--family", default="mace")
    parser.add_argument("--python", default="", help="interpreter for the worker backend")
    args = parser.parse_args(argv[1:])

    print(f"FastAtoms {fa.__version__}\n")
    what_this_build_can_do()

    atoms = water_box(8)
    the_graph_a_model_sees(atoms)
    a_backend_of_your_own(atoms)

    if args.model:
        relax_with_a_model(run_a_model(args.model, args.family, args.python))
    else:
        print("\n" + "=" * 74)
        print("No model given")
        print("=" * 74)
        print("\nPass --model to run one. To get the in-process path:\n"
              "  ~/miniconda3/envs/mace_env/bin/python -m fastatoms.mlip.export \\\n"
              "      ~/.cache/mace/MACE-OFF23_small.model -o mace.pt\n"
              "  python python/examples/mlip.py --model mace.pt\n\n"
              "Or use a checkpoint through the worker:\n"
              "  python python/examples/mlip.py --family mattersim \\\n"
              "      --model MatterSim-v1.0.0-1M.pth")

    drive_from_the_input_file(args.input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
