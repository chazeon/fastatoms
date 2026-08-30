#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Molecular dynamics: integrators, thermostats and barostats.

    python python/examples/molecular_dynamics.py examples/md_argon.yaml

Everything runs on the built-in Lennard-Jones calculator, so no DFT
installation is needed. Swapping in ``fa.Vasp(...)`` changes nothing else.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

import fastatoms as fa


def argon(a: float = 5.30, repeat: int = 2) -> fa.Atoms:
    """Face-centred-cubic argon, the classic Lennard-Jones solid."""
    basis = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    atoms = fa.Atoms("Ar4", scaled_positions=basis, cell=a, pbc=True).repeat(
        repeat, repeat, repeat
    )
    atoms.calc = fa.LennardJones(epsilon=0.0103, sigma=3.40, cutoff=8.5)
    return atoms


def compare_integrators() -> None:
    """The same NVE run under three schemes; only one of them is any good."""
    print("=" * 78)
    print("Integrators: 400 steps of NVE at 2 fs, starting from 40 K")
    print("=" * 78)
    print(f"\n{'scheme':<18} {'symplectic':>11} {'force calls':>12} "
          f"{'final T [K]':>12} {'drift [eV/atom]':>17}")

    for name in ("euler", "velocity_verlet", "leapfrog"):
        atoms = argon()
        fa.maxwell_boltzmann(atoms, 40.0, seed=2024)
        result = fa.run_md(atoms, 400, timestep=2.0, integrator=name, interval=20)
        drift = result.energy_drift_per_atom(len(atoms))
        print(f"{name:<18} {str(fa.make_integrator(name).is_symplectic):>11} "
              f"{result.evaluations:>12} {result.history[-1].temperature:>12.2f} "
              f"{drift:>17.2e}")

    print("\nEuler pumps energy in every step; the two symplectic schemes keep it\n"
          "bounded. Leapfrog and velocity Verlet give identical positions - they are\n"
          "the same algorithm written down differently.")


def compare_thermostats() -> None:
    """Start cold and let each thermostat find the target."""
    print("\n" + "=" * 78)
    print("Thermostats: 1500 steps at 5 fs, starting at 10 K, target 80 K")
    print("=" * 78)
    expected = 80.0 * np.sqrt(2.0 / fa.momentum_degrees_of_freedom(argon()))
    print(f"\n(the canonical width for this system is sigma(T) = {expected:.2f} K)")
    print(f"\n{'thermostat':<16} {'canonical':>10} {'<T> late [K]':>14} "
          f"{'sigma(T) [K]':>13} {'drift [eV/atom]':>17}")

    for name in ("berendsen", "langevin", "nose", "nose_hoover"):
        atoms = argon()
        fa.maxwell_boltzmann(atoms, 10.0, seed=4242)
        result = fa.run_md(
            atoms, 1500, timestep=5.0, temperature=80.0, thermostat=name,
            tau=100.0, friction=0.01, seed=7,
            initialize_velocities=False, interval=5,
        )
        late = [s.temperature for s in result.history[len(result.history) // 2 :]]
        print(f"{name:<16} {str(fa.make_thermostat(name).samples_canonical):>10} "
              f"{np.mean(late):>14.2f} {np.std(late):>13.2f} "
              f"{result.energy_drift_per_atom(len(atoms)):>17.2e}")

    print("\nAll four hold the mean temperature, but only three of them sample the\n"
          "canonical distribution: Langevin, Nose and Nose-Hoover reproduce the width\n"
          "printed above, while Berendsen deliberately suppresses it. That is what\n"
          "makes Berendsen an equilibration tool rather than a sampling one.\n"
          "\nLangevin has no conserved quantity of its own - it books the heat it\n"
          "exchanges with the bath instead, so the drift column stays meaningful.")


def compare_barostats() -> None:
    """Compress the cell, then let each barostat relieve it."""
    print("\n" + "=" * 78)
    print("Barostats: 2000 steps at 5 fs on a cell squeezed to a = 5.00 A")
    print("=" * 78)
    print(f"\n{'barostat':<20} {'force calls':>12} {'final a [A]':>12} "
          f"{'<P> [GPa]':>11} {'<T> [K]':>9}")

    for name in ("berendsen", "parrinello_rahman"):
        atoms = argon(a=5.00)
        result = fa.run_md(
            atoms, 2000, timestep=5.0, temperature=30.0, thermostat="nose_hoover",
            barostat=name, pressure=0.0, tau=100.0, seed=7, interval=20,
        )
        late = result.history[len(result.history) // 2 :]
        pressure = np.mean([s.pressure for s in late]) / fa.units.GPa
        print(f"{name:<20} {result.evaluations:>12} {atoms.cell.lengths[0] / 2:>12.4f} "
              f"{pressure:>11.4f} {np.mean([s.temperature for s in late]):>9.2f}")

    print("\nBoth relieve the compression. An NPT step costs two force calls rather\n"
          "than one, because deforming the cell invalidates the cached forces.")


def show_pressure_control() -> None:
    """The cell should settle wherever the applied pressure puts it."""
    print("\n" + "=" * 78)
    print("Holding a target pressure with Parrinello-Rahman")
    print("=" * 78)
    print(f"\n{'target [GPa]':>13} {'<P> [GPa]':>11} {'a [A]':>9} {'V/atom [A^3]':>14}")

    for target in (0.0, 0.5, 1.0):
        atoms = argon()
        result = fa.run_md(
            atoms, 3000, timestep=5.0, temperature=30.0, thermostat="nose_hoover",
            barostat="parrinello_rahman", pressure=target, tau=100.0,
            barostat_tau=800.0, seed=13, interval=20,
        )
        late = result.history[len(result.history) // 2 :]
        pressure = np.mean([s.pressure for s in late]) / fa.units.GPa
        volume = np.mean([s.volume for s in late]) / len(atoms)
        print(f"{target:>13.2f} {pressure:>11.3f} {atoms.cell.lengths[0] / 2:>9.4f} "
              f"{volume:>14.3f}")


def write_a_trajectory() -> None:
    """Positions, velocities and forces, every so many steps."""
    print("\n" + "=" * 78)
    print("Trajectory output")
    print("=" * 78)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "argon.xyz"
        atoms = argon(repeat=1)
        result = fa.run_md(
            atoms, 200, timestep=5.0, temperature=60.0, thermostat="nose_hoover",
            seed=3, trajectory=str(path), trajectory_interval=50,
        )
        lines = path.read_text().splitlines()
        print(f"\nwrote {len(lines) // (len(atoms) + 2)} frames to {path.name} "
              f"({len(lines)} lines)")
        print("\nfirst frame:")
        for line in lines[: len(atoms) + 2]:
            print("  " + (line[:96] + " ..." if len(line) > 96 else line))
        print(f"\n{result.message}")


def drive_from_the_input_file(path: str) -> None:
    """Structure, calculator, integrator, thermostat and output, all from YAML."""
    print("\n" + "=" * 78)
    print(f"Driving from {path}")
    print("=" * 78)

    deck = fa.read(path)
    atoms = deck.atoms
    atoms.calc = deck.make_calculator()
    md = deck.make_dynamics(steps=400, trajectory="", log="")

    print(f"\nstructure : {atoms!r}")
    print(f"dynamics  : {md!r}")
    print(f"\n{'step':>6} {'time [fs]':>11} {'T [K]':>9} {'E_tot [eV]':>13} "
          f"{'E_cons [eV]':>13}")
    md.attach(
        lambda atoms, s: print(f"{s.step:>6} {s.time:>11.1f} {s.temperature:>9.2f} "
                               f"{s.total_energy:>13.6f} {s.conserved_energy:>13.6f}"),
        interval=100,
    )
    result = md.run(atoms)
    print(f"\n{result.message}")
    print(f"conserved-energy drift: {result.energy_drift_per_atom(len(atoms)):.2e} eV/atom")


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else "examples/md_argon.yaml"
    print(f"FastAtoms {fa.__version__}\n")
    compare_integrators()
    compare_thermostats()
    compare_barostats()
    show_pressure_control()
    write_a_trajectory()
    drive_from_the_input_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
