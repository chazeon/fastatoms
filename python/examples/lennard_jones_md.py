#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""A complete molecular-dynamics run driven entirely from Python.

Everything that scales with system size happens in the C++ core - the
structure, the periodic images and the O(N) neighbour search - while the force
model and the integrator are a few lines of numpy on top. This is the pattern
FastAtoms is built for: keep the inner loops in C++, keep the science in
Python.

    python python/examples/lennard_jones_md.py examples/argon.yaml
"""

from __future__ import annotations

import sys

import numpy as np

import fastatoms as fa


class LennardJones:
    """A cut-and-shifted Lennard-Jones pair potential."""

    def __init__(self, epsilon: float, sigma: float, cutoff: float):
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        # Shift the potential so that it goes continuously to zero at rc.
        ratio6 = (sigma / cutoff) ** 6
        self.energy_shift = 4.0 * epsilon * (ratio6 * ratio6 - ratio6)

    def __call__(self, atoms: fa.Atoms) -> tuple[float, np.ndarray]:
        """Return the potential energy in eV and the forces in eV/A."""
        # A half list: each pair is visited once, and Newton's third law gives
        # the other half for free.
        pairs = fa.neighbor_list(atoms, self.cutoff, full=False)
        forces = np.zeros((len(atoms), 3))
        if len(pairs) == 0:
            return 0.0, forces

        dr = pairs.vectors                      # r_j - r_i, minimum image
        r2 = np.einsum("ij,ij->i", dr, dr)
        inv6 = (self.sigma**2 / r2) ** 3
        inv12 = inv6 * inv6

        energy = float(np.sum(4.0 * self.epsilon * (inv12 - inv6) - self.energy_shift))

        # F_i = -(24 eps / r^2) (2 (s/r)^12 - (s/r)^6) dr,  F_j = -F_i
        coefficient = (24.0 * self.epsilon * (2.0 * inv12 - inv6) / r2)[:, None] * dr
        np.add.at(forces, pairs.i, -coefficient)
        np.add.at(forces, pairs.j, coefficient)
        return energy, forces


def maxwell_boltzmann(atoms: fa.Atoms, temperature: float, seed: int) -> None:
    """Draw velocities from a Maxwell-Boltzmann distribution."""
    rng = np.random.default_rng(seed)
    masses = atoms.get_masses()[:, None]
    sigma = np.sqrt(fa.units.kB * temperature / masses)
    atoms.set_velocities(rng.normal(scale=sigma, size=(len(atoms), 3)))
    atoms.zero_center_of_mass_momentum()
    # Rescale so the instantaneous temperature is exactly what was asked for.
    current = atoms.get_temperature()
    if current > 0.0:
        atoms.set_velocities(atoms.get_velocities() * np.sqrt(temperature / current))


def velocity_verlet(atoms: fa.Atoms, calculator, timestep: float, steps: int,
                    log_every: int) -> None:
    masses = atoms.get_masses()[:, None]
    energy, forces = calculator(atoms)

    print(f"{'step':>6} {'T [K]':>10} {'E_pot [eV]':>14} {'E_kin [eV]':>12} "
          f"{'E_tot [eV]':>14}")
    reference = None

    for step in range(steps + 1):
        if step % log_every == 0:
            kinetic = atoms.get_kinetic_energy()
            total = energy + kinetic
            reference = total if reference is None else reference
            print(f"{step:>6} {atoms.get_temperature():>10.3f} {energy:>14.6f} "
                  f"{kinetic:>12.6f} {total:>14.6f}")
        if step == steps:
            break

        # Velocity Verlet. `atoms.positions` is a live view into the C++
        # buffer, so this updates the structure in place with no copy.
        velocities = atoms.get_velocities()
        velocities += 0.5 * timestep * forces / masses
        atoms.positions += timestep * velocities
        energy, forces = calculator(atoms)
        velocities += 0.5 * timestep * forces / masses
        atoms.set_velocities(velocities)

    drift = abs(energy + atoms.get_kinetic_energy() - reference)
    print(f"\nenergy drift over {steps} steps: {drift:.2e} eV "
          f"({drift / len(atoms) * 1e3:.3e} meV/atom)")


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else "examples/argon.yaml"
    deck = fa.read_input(path)

    # Sections FastAtoms does not interpret are still available verbatim.
    repetitions = deck.simulation.raw.get("supercell", [1, 1, 1])
    atoms = deck.atoms.repeat(*repetitions)

    parameters = deck.simulation.calculator.parameters
    calculator = LennardJones(
        epsilon=parameters["epsilon"],
        sigma=parameters["sigma"],
        cutoff=parameters["cutoff"],
    )

    md = deck.simulation.md
    print(f"FastAtoms {fa.__version__}")
    print(f"{atoms!r}")
    print(f"{len(atoms)} atoms, cutoff {calculator.cutoff} A, "
          f"{md.steps} steps of {md.timestep_fs} fs\n")

    if md.initialize_velocities:
        maxwell_boltzmann(atoms, md.temperature_K, md.seed)

    velocity_verlet(
        atoms,
        calculator,
        timestep=md.timestep,          # fs converted to the internal time unit
        steps=int(md.steps),
        log_every=int(deck.simulation.output.interval),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
