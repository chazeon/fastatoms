#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark FastAtoms against the Atomic Simulation Environment.

    PYTHONPATH=build/python python benchmarks/ase_comparison.py

Both libraries are given the same atoms in the same places with the same
velocities, and the same Lennard-Jones parameters, so the numbers below compare
implementations rather than models. That the two agree is not assumed: the
`validate` section checks it, and the script refuses to report timings it could
not first show to be computing the same thing.

Each timed case runs in a fresh interpreter, so its peak resident memory is its
own and not the high-water mark of everything that ran before it. Timing covers
the work only - building the system and importing the library are measured
separately, as the `import + setup` baseline.

Requires `ase` (`pip install ase`) alongside the built extension.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import subprocess
import sys
import time
from typing import Any

# --------------------------------------------------------------------------
# The system, built from plain Python so both libraries get identical floats
# --------------------------------------------------------------------------
EPSILON = 0.0103      # eV, argon
SIGMA = 3.40          # angstrom
CUTOFF = 8.5          # angstrom
LATTICE = 5.30        # angstrom, fcc argon
TIMESTEP = 5.0        # fs
TEMPERATURE = 60.0    # K, for the initial velocities

FCC_BASIS = ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))
ARGON_MASS = 39.948   # amu
BOLTZMANN = 8.617333262145e-5   # eV/K


def build_system(repeat: int, seed: int = 20260830) -> dict[str, Any]:
    """An fcc argon supercell, jittered off its lattice sites.

    Returned as plain lists: whichever library builds from this gets bit-for-bit
    the same coordinates, which is what makes the comparison a comparison.
    """
    rng = random.Random(seed)
    side = LATTICE * repeat
    positions: list[list[float]] = []
    for i in range(repeat):
        for j in range(repeat):
            for k in range(repeat):
                for bx, by, bz in FCC_BASIS:
                    positions.append([
                        (i + bx) * LATTICE + rng.gauss(0.0, 0.05),
                        (j + by) * LATTICE + rng.gauss(0.0, 0.05),
                        (k + bz) * LATTICE + rng.gauss(0.0, 0.05),
                    ])

    # Velocities from the same stream, then the net momentum removed so that
    # neither library's centre-of-mass handling can make the runs differ.
    scale = math.sqrt(BOLTZMANN * TEMPERATURE / ARGON_MASS)
    velocities = [[rng.gauss(0.0, scale) for _ in range(3)] for _ in positions]
    for axis in range(3):
        mean = sum(v[axis] for v in velocities) / len(velocities)
        for v in velocities:
            v[axis] -= mean

    return {
        "symbols": ["Ar"] * len(positions),
        "positions": positions,
        "velocities": velocities,
        "cell": [[side, 0.0, 0.0], [0.0, side, 0.0], [0.0, 0.0, side]],
        "natoms": len(positions),
    }


# --------------------------------------------------------------------------
# Building the same system in each library
# --------------------------------------------------------------------------
def make_fastatoms(system: dict[str, Any]):
    import fastatoms as fa

    atoms = fa.Atoms("".join(system["symbols"]), system["positions"],
                     cell=system["cell"], pbc=True)
    atoms.set_masses([ARGON_MASS] * system["natoms"])
    atoms.set_velocities(system["velocities"])
    atoms.calc = fa.LennardJones(epsilon=EPSILON, sigma=SIGMA, cutoff=CUTOFF)
    return atoms


def counting_lennard_jones():
    """ASE's Lennard-Jones, plus a tally of how often it actually ran.

    FastAtoms reports `call_count`; ASE has no equivalent, and counting
    optimizer steps instead would undercount, because a step can ask for a
    force more than once. Comparing cost per force evaluation is only honest if
    both numbers mean the same thing.
    """
    from ase.calculators.lj import LennardJones

    class CountingLennardJones(LennardJones):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.evaluations = 0

        def calculate(self, *args, **kwargs):
            self.evaluations += 1
            super().calculate(*args, **kwargs)

    return CountingLennardJones


def make_ase(system: dict[str, Any]):
    from ase import Atoms

    atoms = Atoms(symbols=system["symbols"], positions=system["positions"],
                  cell=system["cell"], pbc=True)
    atoms.set_masses([ARGON_MASS] * system["natoms"])
    atoms.set_velocities(system["velocities"])
    atoms.calc = counting_lennard_jones()(epsilon=EPSILON, sigma=SIGMA, rc=CUTOFF)
    return atoms


# --------------------------------------------------------------------------
# The cases. Each returns a dict of whatever it measured.
# --------------------------------------------------------------------------
# L-BFGS is configured identically on both sides - same memory, same trust
# radius, same initial curvature, no line search - so the two are running the
# same algorithm and not merely the same name.
LBFGS_MEMORY = 100
LBFGS_MAXSTEP = 0.2
LBFGS_ALPHA = 70.0


def case_setup(library: str, system: dict[str, Any], _steps: int) -> dict[str, Any]:
    """Import the library and build the structure: the baseline every other
    case is measured against."""
    atoms = make_fastatoms(system) if library == "fastatoms" else make_ase(system)
    return {"natoms": len(atoms)}


def case_single_point(library: str, system: dict[str, Any], steps: int) -> dict[str, Any]:
    """Repeated force evaluations, with the cache invalidated each time.

    Invalidating matters: FastAtoms caches results per structure, so a loop that
    asks for the same energy twice measures a dictionary lookup the second time.
    """
    if library == "fastatoms":
        atoms = make_fastatoms(system)
        start = time.perf_counter()
        for _ in range(steps):
            atoms.calc.reset()
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
        seconds = time.perf_counter() - start
        fmax = max(max(abs(c) for c in row) for row in forces)
    else:
        atoms = make_ase(system)
        start = time.perf_counter()
        for _ in range(steps):
            atoms.calc.results.clear()
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
        seconds = time.perf_counter() - start
        fmax = float(abs(forces).max())
    return {"seconds": seconds, "evaluations": steps, "energy": float(energy), "fmax": fmax}


def ase_lbfgs(atoms, line_search: bool):
    from ase.optimize import LBFGS

    return LBFGS(atoms, maxstep=LBFGS_MAXSTEP, memory=LBFGS_MEMORY, alpha=LBFGS_ALPHA,
                 use_line_search=line_search, logfile=None)


def case_relax(library: str, system: dict[str, Any], steps: int) -> dict[str, Any]:
    """Relax to convergence with each library's own settings.

    The fixed-step case measures how fast a step is. This measures how much
    work the problem actually takes, which is what a user pays: a scheme that
    is slower per step but needs a quarter as many of them wins. Both must land
    in the same minimum for the numbers to be comparable, which `validate`
    checks.
    """
    fmax = 1.0e-6
    if library == "fastatoms":
        import fastatoms as fa

        atoms = make_fastatoms(system)
        start = time.perf_counter()
        result = fa.make_optimizer("lbfgs", fmax=fmax, max_steps=steps).run(atoms)
        seconds = time.perf_counter() - start
        return {"seconds": seconds, "steps": result.steps,
                "evaluations": atoms.calc.call_count, "energy": float(result.energy),
                "fmax": float(result.fmax), "converged": bool(result.converged)}

    atoms = make_ase(system)
    optimizer = ase_lbfgs(atoms, line_search=(library == "ase_line_search"))
    start = time.perf_counter()
    converged = optimizer.run(fmax=fmax, steps=steps)
    seconds = time.perf_counter() - start
    forces = atoms.get_forces()
    return {"seconds": seconds, "steps": optimizer.get_number_of_steps(),
            "evaluations": atoms.calc.evaluations,
            "energy": float(atoms.get_potential_energy()),
            "fmax": float(max((forces ** 2).sum(axis=1) ** 0.5)),
            "converged": bool(converged)}


def case_optimize(library: str, system: dict[str, Any], steps: int) -> dict[str, Any]:
    """A fixed number of L-BFGS steps. `fmax` is set below anything reachable
    so that both run the whole budget and the timing is comparable."""
    if library == "fastatoms":
        import fastatoms as fa

        atoms = make_fastatoms(system)
        optimizer = fa.make_optimizer("lbfgs", fmax=1e-12, max_steps=steps,
                                      memory=LBFGS_MEMORY, max_step=LBFGS_MAXSTEP,
                                      alpha=LBFGS_ALPHA, line_search="none")
        start = time.perf_counter()
        result = optimizer.run(atoms)
        seconds = time.perf_counter() - start
        return {"seconds": seconds, "steps": result.steps,
                "evaluations": atoms.calc.call_count, "energy": float(result.energy),
                "fmax": float(result.fmax)}

    atoms = make_ase(system)
    optimizer = ase_lbfgs(atoms, line_search=False)
    start = time.perf_counter()
    optimizer.run(fmax=1e-12, steps=steps)
    seconds = time.perf_counter() - start
    forces = atoms.get_forces()
    fmax = float(max((forces ** 2).sum(axis=1) ** 0.5))
    return {"seconds": seconds, "steps": optimizer.get_number_of_steps(),
            "evaluations": atoms.calc.evaluations,
            "energy": float(atoms.get_potential_energy()), "fmax": fmax}


def case_md(library: str, system: dict[str, Any], steps: int) -> dict[str, Any]:
    """Velocity Verlet, NVE, from identical initial conditions."""
    if library == "fastatoms":
        import fastatoms as fa

        atoms = make_fastatoms(system)
        dynamics = fa.make_dynamics(timestep=TIMESTEP, integrator="velocity_verlet",
                                    fix_center_of_mass=False, log=False)
        start = time.perf_counter()
        result = dynamics.run(atoms, steps)
        seconds = time.perf_counter() - start
        return {"seconds": seconds, "steps": result.steps,
                "evaluations": atoms.calc.call_count,
                "energy": float(atoms.get_potential_energy()),
                "drift": abs(result.energy_drift_per_atom(len(atoms)))}

    from ase import units
    from ase.md.verlet import VelocityVerlet

    atoms = make_ase(system)
    dynamics = VelocityVerlet(atoms, timestep=TIMESTEP * units.fs, logfile=None)
    energy_before = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    start = time.perf_counter()
    dynamics.run(steps)
    seconds = time.perf_counter() - start
    energy_after = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    return {"seconds": seconds, "steps": steps, "evaluations": atoms.calc.evaluations,
            "energy": float(atoms.get_potential_energy()),
            "drift": abs(energy_after - energy_before) / len(atoms)}


#: case -> (function, description, default step count, libraries compared).
#:
#: `relax` compares three, because reporting only ASE's default L-BFGS would
#: flatter FastAtoms: turning ASE's line search on roughly halves its step
#: count, and a benchmark that hides that is advertising.
CASES = {
    "setup": (case_setup, "import + build the structure", 0,
              ("fastatoms", "ase")),
    "single_point": (case_single_point, "repeated force evaluations", 200,
                     ("fastatoms", "ase")),
    "optimize": (case_optimize, "L-BFGS, fixed step count, matched settings", 50,
                 ("fastatoms", "ase")),
    "md": (case_md, "velocity Verlet molecular dynamics", 1000,
           ("fastatoms", "ase")),
    "relax": (case_relax, "L-BFGS to fmax = 1e-6, each library's own settings", 2000,
              ("fastatoms", "ase", "ase_line_search")),
}

#: How each library is labelled in the report.
LIBRARY_NAMES = {"fastatoms": "fastatoms", "ase": "ase",
                 "ase_line_search": "ase +ls"}


# --------------------------------------------------------------------------
# Peak memory
# --------------------------------------------------------------------------
def peak_memory_bytes() -> int:
    """This process's high-water resident set size.

    `ru_maxrss` is bytes on macOS and kilobytes on Linux, which is a
    long-standing wart worth handling rather than reporting a number that is
    wrong by a factor of a thousand on one of them.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


# --------------------------------------------------------------------------
# Worker: one case, one library, one fresh interpreter
# --------------------------------------------------------------------------
def run_worker(library: str, case: str, repeat: int, steps: int) -> int:
    system = build_system(repeat)
    function = CASES[case][0]
    try:
        measured = function(library, system, steps)
    except Exception as error:                       # noqa: BLE001
        json.dump({"error": f"{type(error).__name__}: {error}"}, sys.stdout)
        return 1
    measured["peak_bytes"] = peak_memory_bytes()
    measured["natoms"] = system["natoms"]
    json.dump(measured, sys.stdout)
    return 0


def spawn(library: str, case: str, repeat: int, steps: int) -> dict[str, Any]:
    """Run one case in a clean interpreter and bring its numbers back."""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker", library, case,
         "--repeat", str(repeat), "--steps", str(steps)],
        capture_output=True, text=True, env=environment, check=False)
    if completed.returncode != 0 and not completed.stdout.strip():
        return {"error": completed.stderr.strip().splitlines()[-1] if completed.stderr
                else f"exit {completed.returncode}"}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"error": (completed.stdout or completed.stderr).strip()[:200]}


# --------------------------------------------------------------------------
# Validation: are the two computing the same thing at all?
# --------------------------------------------------------------------------
def largest_difference(a, b) -> float:
    """Largest absolute element-wise difference between two (n, 3) blocks,
    without assuming either side hands back the same array type."""
    worst = 0.0
    for row_a, row_b in zip(a, b):
        for x, y in zip(row_a, row_b):
            worst = max(worst, abs(float(x) - float(y)))
    return worst


def validate(repeat: int, md_steps: int, verbose: bool = True) -> dict[str, Any]:
    """Check that FastAtoms and ASE agree, before believing any timing.

    A benchmark of two libraries that compute different things is not a
    benchmark, so this runs first and the report says plainly whether it
    passed.
    """
    system = build_system(repeat)
    findings: dict[str, Any] = {}

    # --- a single point ---------------------------------------------------
    fat = make_fastatoms(system)
    ase_atoms = make_ase(system)

    energy_f = fat.get_potential_energy()
    energy_a = ase_atoms.get_potential_energy()
    forces_f = fat.get_forces()
    forces_a = ase_atoms.get_forces()
    stress_f = fat.get_stress()
    stress_a = ase_atoms.get_stress()

    findings["energy_difference"] = abs(energy_f - energy_a)
    findings["energy_relative"] = abs(energy_f - energy_a) / max(abs(energy_a), 1e-30)
    findings["force_difference"] = largest_difference(forces_f, forces_a)
    findings["stress_difference"] = max(abs(float(x) - float(y))
                                        for x, y in zip(stress_f, stress_a))
    findings["fmax"] = max(max(abs(float(c)) for c in row) for row in forces_a)

    if verbose:
        print("Validation: a single point")
        print(f"  energy      FastAtoms {energy_f:.12f} eV")
        print(f"              ASE       {energy_a:.12f} eV")
        print(f"              difference {findings['energy_difference']:.3e} eV "
              f"({findings['energy_relative']:.1e} relative)")
        print(f"  forces      largest difference {findings['force_difference']:.3e} eV/A "
              f"(of {findings['fmax']:.4f} eV/A)")
        print(f"  stress      largest difference {findings['stress_difference']:.3e} eV/A^3")

    # --- the optimizer ----------------------------------------------------
    # These are not the same algorithm, whatever the shared name and matched
    # parameters: they agree for the first step and part company as soon as
    # there is curvature history, because they control the step length
    # differently. So what is checked is the thing that has to be true - that
    # they find the same minimum - and the path difference is reported rather
    # than asserted.
    import fastatoms as fa

    fat = make_fastatoms(system)
    relaxed = fa.make_optimizer("lbfgs", fmax=1.0e-6, max_steps=2000).run(fat)

    ase_atoms = make_ase(system)
    ase_optimizer = ase_lbfgs(ase_atoms, line_search=False)
    ase_optimizer.run(fmax=1.0e-6, steps=2000)
    ase_energy = ase_atoms.get_potential_energy()

    findings["optimizer_minimum_difference"] = abs(relaxed.energy - ase_energy)
    findings["optimizer_minimum_relative"] = (abs(relaxed.energy - ase_energy)
                                              / max(abs(ase_energy), 1e-30))
    findings["optimizer_position_difference"] = largest_difference(
        fat.positions, ase_atoms.get_positions())
    findings["optimizer_steps"] = (relaxed.steps, ase_optimizer.get_number_of_steps())
    findings["optimizer_evaluations"] = (fat.calc.call_count, ase_atoms.calc.evaluations)

    if verbose:
        print("\nValidation: both relaxed to fmax = 1e-6")
        print(f"  minimum     FastAtoms {relaxed.energy:.12f} eV "
              f"({relaxed.steps} steps, {fat.calc.call_count} force calls)")
        print(f"              ASE       {ase_energy:.12f} eV "
              f"({ase_optimizer.get_number_of_steps()} steps, "
              f"{ase_atoms.calc.evaluations} force calls)")
        print(f"              difference {findings['optimizer_minimum_difference']:.3e} eV "
              f"({findings['optimizer_minimum_relative']:.1e} relative)")
        print(f"  positions   largest difference "
              f"{findings['optimizer_position_difference']:.3e} A "
              f"(both converged to fmax = 1e-6, so this is the size of that tolerance)")
        print("  The two take different paths downhill - they are different step-length")
        print("  rules - so only the minimum they reach is compared.")

    # --- the trajectory ---------------------------------------------------
    # Molecular dynamics is chaotic: two implementations that agree to machine
    # precision still drift apart exponentially. What can be checked is that
    # they track for as long as the arithmetic allows, and that the divergence
    # grows from rounding rather than starting from a real difference.
    from ase import units
    from ase.md.verlet import VelocityVerlet

    fat = make_fastatoms(system)
    dynamics = fa.make_dynamics(timestep=TIMESTEP, integrator="velocity_verlet",
                                fix_center_of_mass=False, log=False)
    dynamics.prepare(fat)

    ase_atoms = make_ase(system)
    ase_dynamics = VelocityVerlet(ase_atoms, timestep=TIMESTEP * units.fs, logfile=None)

    # Sorted and de-duplicated: an unsorted list would step past a checkpoint
    # and then silently skip it.
    checkpoints = sorted({1, 10, 100, max(1, md_steps)})
    divergence: list[tuple[int, float]] = []
    taken = 0
    for target in checkpoints:
        if target <= taken:
            continue
        for _ in range(target - taken):
            dynamics.step(fat)
            ase_dynamics.run(1)
        taken = target
        divergence.append((taken, largest_difference(fat.positions,
                                                     ase_atoms.get_positions())))

    findings["trajectory"] = divergence
    findings["trajectory_short"] = divergence[0][1]
    findings["trajectory_long"] = divergence[-1][1]

    if verbose:
        print(f"\nValidation: velocity Verlet from identical initial conditions")
        print(f"  {'steps':>8} {'largest position difference [A]':>34}")
        for count, worst in divergence:
            print(f"  {count:>8} {worst:>34.3e}")
        print("  Two implementations that agree to machine precision still part")
        print("  company exponentially - that is chaos, not disagreement. What the")
        print("  first rows show is that they start together.")

    return findings


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def megabytes(value: int | None) -> float:
    return float("nan") if value is None else value / (1024.0 * 1024.0)


def bare_interpreter_peak() -> int | None:
    """Peak resident memory of an interpreter that does nothing.

    Without it, the import baselines look like library footprints when most of
    the number is CPython itself.
    """
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c",
         "import resource,sys,json;"
         "r=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;"
         "print(json.dumps(r if sys.platform=='darwin' else r*1024))"],
        capture_output=True, text=True, env=environment, check=False)
    try:
        return int(json.loads(completed.stdout))
    except (json.JSONDecodeError, ValueError):
        return None


def report(results: dict[str, dict[str, dict[str, Any]]], natoms: int,
           notes: dict[str, Any] | None = None) -> None:
    baseline = {library: results.get("setup", {}).get(library, {}).get("peak_bytes")
                for library in ("fastatoms", "ase", "ase_line_search")}
    baseline["ase_line_search"] = baseline.get("ase")

    print(f"\n{'=' * 82}")
    print(f"Timings - {natoms} atoms, Lennard-Jones, identical parameters")
    print("=" * 82)

    for case, (_function, description, _default, libraries) in CASES.items():
        if case not in results:
            continue
        print(f"\n{case}  -  {description}")
        header = (f"{'  library':<14} {'time [s]':>10} {'steps':>7} {'evals':>7} "
                  f"{'per eval [ms]':>14} {'peak [MB]':>10} {'work [MB]':>10}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for library in libraries:
            entry = results[case].get(library, {})
            name = LIBRARY_NAMES.get(library, library)
            if "error" in entry:
                print(f"  {name:<12} {'FAILED':>10}   {entry['error'][:48]}")
                continue
            seconds = entry.get("seconds")
            evaluations = entry.get("evaluations") or 0
            steps = entry.get("steps", "")
            per_eval = (seconds / evaluations * 1e3) if seconds and evaluations else float("nan")
            peak = megabytes(entry.get("peak_bytes"))
            base = baseline.get(library)
            work = peak - megabytes(base) if base else float("nan")
            shown = f"{seconds:>10.4f}" if seconds is not None else f"{'-':>10}"
            print(f"  {name:<12} {shown} {str(steps):>7} {evaluations:>7} "
                  f"{per_eval:>14.4f} {peak:>10.1f} {work:>10.1f}")

        reference = results[case].get("fastatoms", {})
        if "seconds" in reference and reference["seconds"] > 0:
            for library in libraries:
                if library == "fastatoms":
                    continue
                other = results[case].get(library, {})
                if "seconds" not in other:
                    continue
                ratio = other["seconds"] / reference["seconds"]
                label = f"speed-up vs {LIBRARY_NAMES.get(library, library)}"
                print(f"  {label:<12} {ratio:>10.1f}x")

        if case in ("optimize", "relax"):
            energies = {lib: results[case].get(lib, {}).get("energy")
                        for lib in libraries}
            if all(value is not None for value in energies.values()):
                spread = max(energies.values()) - min(energies.values())
                print(f"  energies:  " + "   ".join(
                    f"{LIBRARY_NAMES.get(k, k)} {v:.9f}" for k, v in energies.items()))
                print(f"  spread {spread:.2e} eV")
        if case == "md":
            drifts = {lib: results[case].get(lib, {}).get("drift") for lib in libraries}
            if all(value is not None for value in drifts.values()):
                print("  energy drift:  " + "   ".join(
                    f"{LIBRARY_NAMES.get(k, k)} {v:.2e} eV/atom" for k, v in drifts.items()))

    notes = notes or {}
    print("\nMemory")
    print("-" * 82)
    bare = notes.get("bare_peak")
    if bare:
        print(f"  bare interpreter                    {megabytes(bare):>8.1f} MB")
    for library in ("fastatoms", "ase"):
        peak = baseline.get(library)
        if peak:
            over = f" ({megabytes(peak) - megabytes(bare):+.1f} over bare)" if bare else ""
            print(f"  import {LIBRARY_NAMES.get(library, library):<28} "
                  f"{megabytes(peak):>8.1f} MB{over}")
    if notes.get("torch"):
        print("\n  This build links LibTorch, for the machine-learning potentials. That")
        print("  accounts for most of the fastatoms import baseline and none of the work")
        print("  measured below it; a build configured without it imports far smaller.")
        print("  Reconfigure with -DFASTATOMS_WITH_TORCH=OFF to see the difference.")

    print("\n  `work` is peak resident memory above each library's own import baseline,")
    print("  which is the number the algorithms are responsible for. It is measured")
    print("  across separate processes, so values below about 1 MB - including small")
    print("  negative ones - are process-to-process noise rather than a result.")
    print("\n  `per eval` is wall time divided by force evaluations actually performed,")
    print("  which is the only way to compare schemes needing different numbers of them.")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
#: What must hold for the timings to be comparing like with like.
#:
#: The single point and the first molecular dynamics steps are held to machine
#: precision, because the two libraries really are evaluating the same
#: expression and integrating it the same way. The optimizers are not the same
#: algorithm, so the requirement on them is the one that matters physically:
#: the same minimum, to the convergence threshold they were both given.
TOLERANCES = {
    "energy_relative": 1e-12,
    "force_difference": 1e-12,
    "stress_difference": 1e-12,
    "trajectory_short": 1e-12,
    "optimizer_minimum_relative": 1e-9,
}


def check_tolerances(findings: dict[str, Any]) -> list[str]:
    """Which agreements failed. Empty means the two libraries match."""
    failures = []
    for key, limit in TOLERANCES.items():
        value = findings.get(key)
        if value is None or not (abs(value) <= limit):
            failures.append(f"{key} = {value:.3e}, limit {limit:.0e}")
    return failures


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeat", type=int, default=4,
                        help="fcc cells per side; 4 gives 256 atoms, 5 gives 500")
    parser.add_argument("--md-steps", type=int, default=CASES["md"][2])
    parser.add_argument("--optimize-steps", type=int, default=CASES["optimize"][2])
    parser.add_argument("--single-point-steps", type=int, default=CASES["single_point"][2])
    parser.add_argument("--relax-steps", type=int, default=CASES["relax"][2],
                        help="step ceiling for the relax-to-convergence case")
    parser.add_argument("--json", default="",
                        help="write the results to this file (nothing is written otherwise)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="time the cases without first checking the two agree")
    parser.add_argument("--only", choices=sorted(CASES), action="append",
                        help="run only these cases; repeatable")
    parser.add_argument("--worker", nargs=2, metavar=("LIBRARY", "CASE"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--steps", type=int, default=0, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv[1:])

    if arguments.worker:
        library, case = arguments.worker
        return run_worker(library, case, arguments.repeat, arguments.steps)

    try:
        import ase                                    # noqa: F401
    except ImportError:
        print("This benchmark compares against ASE, which is not installed.\n"
              "    pip install ase", file=sys.stderr)
        return 2
    try:
        import fastatoms                              # noqa: F401
    except ImportError:
        print("fastatoms is not importable. Build it first, then\n"
              "    PYTHONPATH=build/python python benchmarks/ase_comparison.py",
              file=sys.stderr)
        return 2

    import ase as ase_module
    import fastatoms as fa_module

    system = build_system(arguments.repeat)
    print(f"FastAtoms {fa_module.__version__} against ASE {ase_module.__version__}")
    print(f"{system['natoms']} atoms, fcc argon, Lennard-Jones "
          f"(epsilon={EPSILON} eV, sigma={SIGMA} A, cutoff={CUTOFF} A)")
    print(f"Python {sys.version.split()[0]} on {sys.platform}\n")

    failures: list[str] = []
    if not arguments.skip_validation:
        findings = validate(arguments.repeat, min(arguments.md_steps, 1000))
        failures = check_tolerances(findings)
        print()
        if failures:
            print("VALIDATION FAILED - the two libraries are not computing the same thing:")
            for failure in failures:
                print(f"  {failure}")
            print("\nTimings below would compare different physics, so they are not run.")
            return 1
        print("Validation passed: the two agree to the tolerances above.")
    else:
        findings = {}

    counts = {"setup": 0, "single_point": arguments.single_point_steps,
              "optimize": arguments.optimize_steps, "md": arguments.md_steps,
              "relax": arguments.relax_steps}
    wanted = arguments.only or list(CASES)
    if "setup" not in wanted:
        wanted = ["setup", *wanted]

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for case, (_function, _description, _default, libraries) in CASES.items():
        if case not in wanted:
            continue
        results[case] = {}
        for library in libraries:
            label = f"  {case} / {LIBRARY_NAMES.get(library, library)}"
            print(f"{label:<40}", end="", flush=True)
            entry = spawn(library, case, arguments.repeat, counts[case])
            results[case][library] = entry
            if "error" in entry:
                print(f"FAILED  {entry['error'][:60]}")
            else:
                print(f"{entry['seconds']:.4f} s" if "seconds" in entry else "done")

    notes: dict[str, Any] = {"bare_peak": bare_interpreter_peak()}
    try:
        notes["torch"] = bool(fa_module.torch_available())
    except AttributeError:
        notes["torch"] = False
    report(results, system["natoms"], notes)

    if arguments.json:
        payload = {"natoms": system["natoms"], "repeat": arguments.repeat,
                   "python": sys.version.split()[0], "platform": sys.platform,
                   "ase": ase_module.__version__, "fastatoms": fa_module.__version__,
                   "validation": {k: v for k, v in findings.items() if k != "trajectory"},
                   "trajectory": findings.get("trajectory", []),
                   "results": results}
        with open(arguments.json, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        print(f"\nwrote {arguments.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
