#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Rigid water: SPC and SPC/E, Ewald electrostatics, SHAKE and RATTLE.

    python python/examples/water.py examples/water_spce.yaml

Everything here runs on parameters compiled into FastAtoms, so no model file
and no external program is needed.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import fastatoms as fa


def the_models() -> None:
    print("=" * 74)
    print("The rigid three-site models")
    print("=" * 74)
    print(f"\n{'model':<8} {'q_O':>9} {'q_H':>8} {'sigma':>8} {'epsilon':>11} "
          f"{'r_OH':>7} {'angle':>8} {'dipole':>8}")
    print(f"{'':<8} {'[e]':>9} {'[e]':>8} {'[A]':>8} {'[eV]':>11} {'[A]':>7} "
          f"{'[deg]':>8} {'[D]':>8}")
    for name in fa.water_model_names():
        m = fa.water_model(name)
        print(f"{m.name:<8} {m.q_oxygen:>9.4f} {m.q_hydrogen:>8.4f} {m.sigma_oxygen:>8.4f} "
              f"{m.epsilon_oxygen:>11.6f} {m.bond_length:>7.4f} {m.angle:>8.2f} "
              f"{m.dipole_moment:>8.3f}")

    spce = fa.water_model("spce")
    print(f"\nSPC/E also carries a polarisation self-energy of "
          f"{spce.self_energy / fa.units.kcal_per_mol:.2f} kcal/mol per molecule -")
    print("the work of polarising a gas-phase molecule up to the enhanced dipole")
    print("the model uses. It shifts the energy and never a force.")
    print(f"\n  {spce.reference}")


def what_a_box_looks_like() -> fa.Atoms:
    print("\n" + "=" * 74)
    print("A box of water")
    print("=" * 74)

    atoms = fa.water_box("spce", 108)
    density = sum(atoms.get_masses()) / atoms.get_volume() * 1.66053906660
    print(f"\n{atoms!r}")
    print(f"  cell     : {atoms.cell.lengths[0]:.4f} A cubic")
    print(f"  density  : {density:.4f} g/cm^3")

    topology = fa.water_topology(atoms, "spce")
    print(f"\n{topology!r}")
    print(f"  molecules      : {len(topology.molecules)}, found by connectivity")
    print(f"  constraints    : {len(topology.constraints)} - two O-H bonds and the H-H")
    print(f"                   distance, which is what fixes the angle")
    print(f"  excluded pairs : {len(topology.excluded_pairs())} - a rigid molecule's own")
    print(f"                   non-bonded energy is a constant, so it is left out")
    print(f"  net charge     : {topology.net_charge:.1e}")
    return atoms


def the_electrostatics(atoms: fa.Atoms) -> None:
    print("\n" + "=" * 74)
    print("Ewald summation")
    print("=" * 74)

    atoms.calc = fa.Forcefield(model="spce")
    energy = atoms.get_potential_energy()
    ewald = atoms.calc.ewald
    coulomb = atoms.calc.breakdown.coulomb

    print(f"\n{ewald!r}")
    print(f"\n{'term':<26} {'energy [eV]':>16}")
    print(f"{'real space (erfc)':<26} {coulomb.real:>16.6f}")
    print(f"{'reciprocal space':<26} {coulomb.reciprocal:>16.6f}")
    print(f"{'self':<26} {coulomb.self:>16.6f}")
    print(f"{'excluded pairs':<26} {coulomb.excluded:>16.6f}")
    print(f"{'':<26} {'-' * 16:>16}")
    print(f"{'total':<26} {coulomb.total:>16.6f}")

    print("\nThe individual terms are enormous and nearly cancel - that is what an")
    print("Ewald sum looks like, and it is why the pieces are reported separately.")

    breakdown = atoms.calc.breakdown
    print(f"\n{'Lennard-Jones':<26} {breakdown.lennard_jones:>16.6f}")
    print(f"{'  tail correction':<26} {breakdown.lennard_jones_tail:>16.6f}")
    print(f"{'electrostatics':<26} {coulomb.total:>16.6f}")
    print(f"{'SPC/E self-energy':<26} {breakdown.self:>16.6f}")
    print(f"{'':<26} {'-' * 16:>16}")
    print(f"{'total':<26} {energy:>16.6f}")


def the_splitting_is_arbitrary() -> None:
    """The one check that says the implementation is right."""
    print("\n" + "=" * 74)
    print("The answer does not depend on where the sum is split")
    print("=" * 74)

    # Rock salt, whose lattice sum is known exactly.
    a = 5.64
    cations = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    anions = [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]]
    atoms = fa.Atoms("Na4Cl4", scaled_positions=cations + anions, cell=a, pbc=True)
    topology = fa.Topology(len(atoms))
    for i, z in enumerate(atoms.get_atomic_numbers()):
        topology.set_charge(i, 1.0 if z == 11 else -1.0)

    print(f"\n{'alpha [1/A]':>12} {'cutoff [A]':>12} {'k-vectors':>11} "
          f"{'energy [eV]':>15} {'Madelung':>12}")
    reach = np.sqrt(-np.log(2e-12))
    for alpha in (0.8, 1.0, 1.5, 2.0):
        cutoff = reach / alpha
        ewald = fa.Ewald(accuracy=1e-12, alpha=alpha, cutoff=cutoff)
        ewald.prepare(atoms)
        energy = ewald.compute(atoms, topology)["energy"].total
        madelung = -energy * (a / 2) / (4 * fa.units.coulomb)
        print(f"{alpha:>12.2f} {cutoff:>12.3f} {ewald.kvector_count:>11} "
              f"{energy:>15.9f} {madelung:>12.9f}")

    print("\nEvery row is the same number, and it is the Madelung constant of rock")
    print("salt, 1.747564595, which is known in closed form. The splitting is an")
    print("identity, so moving it can only change the answer if something is wrong.")


def rigid_dynamics(atoms: fa.Atoms) -> None:
    print("\n" + "=" * 74)
    print("Molecular dynamics with rigid molecules")
    print("=" * 74)

    atoms.calc = fa.Forcefield(model="spce")
    dynamics = fa.make_dynamics(timestep=2.0, temperature=300.0, thermostat="nose_hoover",
                                tau=100.0, initialize_velocities=True, seed=20260830)
    dynamics.prepare(atoms)
    constraints = dynamics.constraints

    print(f"\n{constraints!r}")
    print(f"  removed degrees of freedom : {constraints.removed_degrees_of_freedom}")
    print(f"  a rigid molecule has 6, not 9, and the temperature has to know")
    print(f"  bond error before the run  : {constraints.max_deviation(atoms):.2e}")

    start = time.time()
    result = dynamics.run(atoms, 200)
    elapsed = time.time() - start

    print(f"\n{result!r}")
    print(f"  wall time    : {elapsed:.2f} s for {len(atoms)} atoms")
    print(f"  drift        : {abs(result.energy_drift_per_atom(len(atoms))):.2e} eV/atom")
    print(f"  bond error   : {constraints.max_deviation(atoms):.2e} (relative)")
    print(f"  velocity     : {constraints.max_velocity_deviation(atoms, atoms.get_velocities()):.2e}")
    print(f"  temperature  : {np.mean([s.temperature for s in result.history[-5:]]):.1f} K "
          f"(still equilibrating from a lattice start)")

    print("\nThe bonds are still exactly their model length after 400 fs, and the")
    print("velocities still have no component along them. Freezing the fastest")
    print("motion in the system is what makes a 2 fs step reasonable; a flexible")
    print("water model, whose O-H stretch has a period near 9 fs, needs about 0.5.")


def what_the_constraints_buy(atoms: fa.Atoms) -> None:
    print("\n" + "=" * 74)
    print("What happens without them")
    print("=" * 74)

    print(f"\n{'timestep [fs]':>14} {'constraints':>13} {'drift [eV/atom]':>18} {'bond error':>13}")
    for timestep, constrained in ((2.0, True), (1.0, True), (0.5, False), (2.0, False)):
        probe = fa.water_box("spce", 32)
        probe.calc = fa.Forcefield(model="spce")
        settings = dict(timestep=timestep, temperature=300.0,
                        initialize_velocities=True, seed=5)
        if not constrained:
            settings["constraints"] = False
        dynamics = fa.make_dynamics(**settings)
        # Build the yardstick from the starting geometry: after an
        # unconstrained run the molecules may have come apart, and the
        # detector would no longer recognise them as water.
        reference = fa.Constraints(fa.water_topology(probe, "spce").constraints)
        try:
            dynamics.prepare(probe)
            result = dynamics.run(probe, 100)
            drift = abs(result.energy_drift_per_atom(len(probe)))
            error = reference.max_deviation(probe)
            print(f"{timestep:>14.1f} {str(constrained):>13} {drift:>18.2e} {error:>13.2e}")
        except Exception as failure:                    # noqa: BLE001
            print(f"{timestep:>14.1f} {str(constrained):>13} {'unstable':>18} "
                  f"{str(failure).split(';')[0][:34]:>13}")

    print("\nThe last two rows are not 'the same model with a shorter timestep'.")
    print("A rigid model has no bond stretch and no bend to fall back on: the")
    print("constraints are the only thing holding O to H, so without them the")
    print("molecules simply come apart, whatever the timestep. Flexible water is")
    print("a different model with different parameters, not this one unconstrained.")


def drive_from_the_input_file(path: str) -> None:
    print("\n" + "=" * 74)
    print(f"Driving from {path}")
    print("=" * 74)

    import os
    if not os.path.exists(path):
        print("\nskipped: no such file")
        return
    deck = fa.read_input(path)
    atoms = deck.atoms
    atoms.calc = deck.make_calculator()
    print(f"\nstructure : {atoms!r}")
    print(f"calculator: {atoms.calc!r}")
    dynamics = deck.make_dynamics(log=False)
    result = dynamics.run(atoms)
    print(f"result    : {result!r}")
    print(f"drift     : {abs(result.energy_drift_per_atom(len(atoms))):.2e} eV/atom")
    print(f"bonds     : {dynamics.constraints.max_deviation(atoms):.2e} (relative)")


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else "examples/water_spce.yaml"
    print(f"FastAtoms {fa.__version__}\n")
    the_models()
    atoms = what_a_box_looks_like()
    the_electrostatics(atoms)
    the_splitting_is_arbitrary()
    rigid_dynamics(atoms)
    what_the_constraints_buy(atoms)
    drive_from_the_input_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
