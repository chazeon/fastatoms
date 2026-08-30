# SPDX-License-Identifier: MIT
"""FastAtoms - a high-performance C++ core for atomistic simulations.

The compiled core lives in :mod:`fastatoms._core`; this package re-exports it
and adds the conveniences that are easier to express in Python.

Quick start
-----------

>>> import fastatoms as fa
>>> water = fa.Atoms("H2O", [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
>>> water.get_chemical_formula()
'H2O'
>>> silicon = fa.Atoms("Si", [[0.0, 0.0, 0.0]], cell=5.43, pbc=True).repeat(4, 4, 4)
>>> len(silicon)
64
>>> neighbours = fa.neighbor_list(silicon, 6.0)
>>> len(neighbours) > 0
True

Files
-----

Extended XYZ is the default format. A negative index counts from the end, so
the default read is the last frame of a trajectory::

    atoms  = fa.read("relaxed.extxyz")           # the last frame
    first  = fa.read("md.extxyz", index=0)
    frames = fa.read_frames("md.extxyz")

    fa.write("md.extxyz", atoms, append=True)    # one frame at a time
    fa.write("traj.extxyz", frames)              # or all of them at once

Whatever the comment line carries beyond the structure lands in ``atoms.info``,
which writes through and is written back out:

>>> atoms = fa.Atoms("Ar2", [[0, 0, 0], [3.8, 0, 0]], cell=8.0, pbc=True)
>>> atoms.info["energy"] = -1.5
>>> "energy" in atoms.info, atoms.info["energy"]
(True, -1.5)

A YAML input file describes a *run* rather than a structure, which extended XYZ
has no way to express, so that is read with a different function:

>>> deck = fa.read_input("input.yaml")                 # doctest: +SKIP
>>> deck.atoms, deck.simulation.md.timestep_fs         # doctest: +SKIP

Attaching a calculator
----------------------

>>> atoms = fa.Atoms("Ar2", [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], cell=12.0, pbc=True)
>>> atoms.calc = fa.LennardJones(epsilon=0.0103, sigma=3.40)
>>> atoms.get_potential_energy() < 0.0
True
>>> atoms.get_forces().shape
(2, 3)
>>> atoms.get_stress().shape
(6,)

For a DFT run, swap in one of the file-based calculators - the rest of the
script is unchanged::

    atoms.calc = fa.VASP(xc="pbe", encut=400, kpts=[4, 4, 4], directory="run")
    atoms.calc = fa.QuantumEspresso(pseudopotentials={"Si": "Si.upf"}, ecutwfc=40.0)
    atoms.calc = fa.SIESTA(xc="pbe", mesh_cutoff=250.0)

Relaxing a structure
--------------------

>>> crystal = fa.Atoms("Ar", [[0.1, 0.0, 0.0]], cell=5.3, pbc=True).repeat(2, 2, 2)
>>> crystal.calc = fa.LennardJones()
>>> result = fa.BFGS(fmax=0.01).run(crystal)
>>> result.converged
True

Any of :class:`SteepestDescent`, :class:`ConjugateGradient`, :class:`BFGS` and
:class:`LBFGS` will do; add ``relax_cell=True`` to relax the cell as well.

Molecular dynamics
------------------

>>> run = fa.run_md(crystal, 200, timestep=5.0, temperature=60.0,
...                 thermostat="nose_hoover", seed=1)
>>> run.completed
True
>>> run.evaluations                       # one force call per step
200

Swap the pieces freely: ``integrator`` is "velocity_verlet" (default),
"leapfrog" or "euler"; ``thermostat`` is "nose_hoover", "nose" or "berendsen";
``barostat`` is "parrinello_rahman" or "berendsen".

For a stochastic, ergodic thermostat use Langevin dynamics::

    fa.run_md(atoms, 2000, temperature=300.0, thermostat="langevin",
              friction=0.01)      # 1/fs

Phonons
-------

>>> argon = fa.Atoms("Ar", [[0.0, 0.0, 0.0]], cell=5.26, pbc=True)
>>> ph = fa.Phonons(argon, supercell=[2, 2, 2])
>>> ph.calculation_count, ph.calculation_count_without_symmetry
(2, 6)

Symmetry decides which atoms and directions have to be displaced at all. Run
the displacements, then ask for frequencies anywhere in the Brillouin zone::

    ph.run()                          # uses atoms.calc
    ph.frequencies([0.5, 0.0, 0.0])   # THz

Machine-learning potentials
---------------------------

A model exported to TorchScript runs in this process through LibTorch, with no
Python in the evaluation loop::

    atoms.calc = fa.MACE(model="mace-off23-small.pt")
    atoms.get_potential_energy()

Models that cannot be scripted - MatterSim among them - are driven by one
long-lived interpreter in the environment that has the package installed, so a
step costs a pipe round trip rather than an interpreter start::

    atoms.calc = fa.MatterSim()          # mlip_env
    fa.run_md(atoms, 1000, temperature=300.0, thermostat="langevin")

>>> fa.mlip_environment_for("mace"), fa.mlip_environment_for("chgnet")
('mace_env', 'mlip_env')

Rigid water
-----------

>>> atoms = fa.water_box("spce", 32)          # 32 molecules at 0.997 g/cm^3
>>> atoms.calc = fa.Forcefield(model="spce")
>>> atoms.calc.prepare(atoms)                 # find the molecules
>>> len(atoms.calc.constraints)               # three per molecule, kept rigid
96

Electrostatics are summed by Ewald, not truncated, and the O-H bonds and H-O-H
angle are held by SHAKE and RATTLE - which is what makes a 2 fs step legitimate::

    fa.run_md(atoms, 1000, timestep=2.0, temperature=300.0,
              thermostat="nose_hoover")
"""

from __future__ import annotations

from typing import Any

from ._core import adsorption
from ._core import analysis
from ._core import build
from ._core import electronic
from ._core import symmetry
from ._core import (
    MAX_ATOMIC_NUMBER,
    Atom,
    Atoms,
    Calculator,
    CalculatorError,
    ExtxyzError,
    MLIPError,
    PropertyNotImplementedError,
    CalculatorParameters,
    BandParameters,
    AdsorptionError,
    GlobalOptimizationError,
    AnalysisError,
    ElectronicError,
    BuildError,
    SymmetryError,
    BandProfile,
    CHGNet,
    Cell,
    BFGS,
    BasinHopping,
    BasinHoppingResult,
    BasinHoppingStep,
    CG,
    Allegro,
    AutoNEB,
    AutoNEBResult,
    Barostat,
    BerendsenBarostat,
    BerendsenThermostat,
    ConjugateGradient,
    ConstraintReport,
    Constraint,
    ConstraintError,
    Constraints,
    Displacement,
    DistanceConstraint,
    Dynamics,
    DynamicsError,
    DynamicsResult,
    DynamicsStep,
    DynamicsStopReason,
    EnergyBreakdown,
    Euler,
    Ewald,
    EwaldEnergy,
    FileIOCalculator,
    FixAtoms,
    FixLine,
    FixPlane,
    Fire,
    Forcefield,
    ForceConstants,
    Integrator,
    LBFGS,
    Langevin,
    Leapfrog,
    LennardJones,
    MACE,
    MLIP,
    MLIPGraph,
    MLIPModelInfo,
    MatterSim,
    InputDeck,
    InputError,
    Molecule,
    MolecularDynamicsParameters,
    NEB,
    NEBError,
    NeighborList,
    NequIP,
    Nose,
    NoseHoover,
    OptimizationResult,
    OptimizationStep,
    Optimizer,
    OptimizerError,
    OptimizerParameters,
    OutputParameters,
    ParrinelloRahman,
    PhononError,
    PhononModes,
    PhononParameters,
    Phonons,
    Property,
    QuantumEspresso,
    Results,
    SD,
    SIESTA,
    SimulationParameters,
    SinglePointCalculator,
    SteepestDescent,
    StopReason,
    SupercellMap,
    SymmetryDataset,
    SymmetryOperation,
    Task,
    Topology,
    Thermostat,
    TrajectoryWriter,
    VASP,
    VelocityVerlet,
    WaterModel,
    __version__,
    atomic_mass,
    atoms_from_dict,
    barostat_names,
    closest_approach,
    conda_python,
    covalent_radius,
    detect_molecules,
    find_symmetry,
    instantaneous_pressure,
    integrator_names,
    kinetic_stress,
    lattice_point_group,
    make_barostat,
    idpp_interpolate,
    make_autoneb,
    make_band,
    attach_stored_results,
    make_calculator,
    make_dynamics,
    make_integrator,
    make_optimizer,
    mlip_environment_for,
    mlip_graph,
    mlip_worker_script,
    make_phonons,
    make_supercell,
    make_thermostat,
    matrix_to_voigt,
    maxwell_boltzmann,
    momentum_degrees_of_freedom,
    natural_cutoffs,
    neighbor_list,
    number_from_symbol,
    optimize,
    parse_extxyz,
    parse_extxyz_frames,
    parse_yaml,
    parse_yaml_file,
    pressure_from_stress,
    read,
    append,
    read_atoms,
    read_extxyz,
    read_extxyz_frames,
    read_frames,
    read_input,
    read_input_string,
    register_calculator,
    register_optimizer,
    registered_calculators,
    registered_optimizers,
    rescale_to_temperature,
    run_md,
    simulation_from_dict,
    site_symmetry,
    torch_available,
    torch_version,
    water_box,
    water_model,
    water_model_names,
    water_molecule,
    water_topology,
    symbol_from_number,
    thermostat_names,
    to_extxyz,
    to_yaml,
    write_yaml,
    total_stress,
    units,
    vasp_species_layout,
    voigt_to_matrix,
    write,
)

__all__ = [
    "adsorption",
    "analysis",
    "build",
    "electronic",
    "symmetry",
    "BFGS",
    "BasinHopping",
    "BasinHoppingResult",
    "BasinHoppingStep",
    "CG",
    "LBFGS",
    "Allegro",
    "AutoNEB",
    "AutoNEBResult",
    "Barostat",
    "BerendsenBarostat",
    "BerendsenThermostat",
    "MAX_ATOMIC_NUMBER",
    "SD",
    "Atom",
    "Atoms",
    "Calculator",
    "CalculatorError",
    "ExtxyzError",
    "MLIPError",
    "PropertyNotImplementedError",
    "CalculatorParameters",
    "BandParameters",
    "AdsorptionError",
    "GlobalOptimizationError",
    "AnalysisError",
    "ElectronicError",
    "BuildError",
    "SymmetryError",
    "BandProfile",
    "CHGNet",
    "Cell",
    "FileIOCalculator",
    "FixAtoms",
    "FixLine",
    "FixPlane",
    "Fire",
    "InputDeck",
    "InputError",
    "LennardJones",
    "MACE",
    "MLIP",
    "MLIPGraph",
    "MLIPModelInfo",
    "MatterSim",
    "Molecule",
    "MolecularDynamicsParameters",
    "ConjugateGradient",
    "ConstraintReport",
    "Constraint",
    "ConstraintError",
    "Constraints",
    "Displacement",
    "DistanceConstraint",
    "Dynamics",
    "DynamicsError",
    "DynamicsResult",
    "DynamicsStep",
    "DynamicsStopReason",
    "EnergyBreakdown",
    "Euler",
    "Ewald",
    "EwaldEnergy",
    "ForceConstants",
    "Forcefield",
    "Integrator",
    "Langevin",
    "Leapfrog",
    "NEB",
    "NEBError",
    "NeighborList",
    "NequIP",
    "Nose",
    "NoseHoover",
    "OptimizationResult",
    "OptimizationStep",
    "Optimizer",
    "OptimizerError",
    "OptimizerParameters",
    "OutputParameters",
    "ParrinelloRahman",
    "PhononError",
    "PhononModes",
    "PhononParameters",
    "Phonons",
    "Property",
    "QuantumEspresso",
    "Results",
    "SIESTA",
    "SimulationParameters",
    "SinglePointCalculator",
    "SteepestDescent",
    "StopReason",
    "SupercellMap",
    "SymmetryDataset",
    "SymmetryOperation",
    "Task",
    "Topology",
    "WaterModel",
    "Thermostat",
    "TrajectoryWriter",
    "VASP",
    "VelocityVerlet",
    "__version__",
    "atomic_mass",
    "atoms_from_dict",
    "barostat_names",
    "closest_approach",
    "conda_python",
    "covalent_radius",
    "detect_molecules",
    "find_symmetry",
    "from_ase",
    "instantaneous_pressure",
    "integrator_names",
    "kinetic_stress",
    "lattice_point_group",
    "make_barostat",
    "idpp_interpolate",
    "make_autoneb",
    "make_band",
    "attach_stored_results",
    "make_calculator",
    "make_dynamics",
    "make_integrator",
    "make_optimizer",
    "mlip_environment_for",
    "mlip_graph",
    "mlip_worker_script",
    "make_phonons",
    "make_supercell",
    "make_thermostat",
    "matrix_to_voigt",
    "maxwell_boltzmann",
    "momentum_degrees_of_freedom",
    "natural_cutoffs",
    "neighbor_list",
    "number_from_symbol",
    "optimize",
    "parse_extxyz",
    "parse_extxyz_frames",
    "parse_yaml",
    "parse_yaml_file",
    "pressure_from_stress",
    "read",
    "append",
    "read_atoms",
    "read_extxyz",
    "read_extxyz_frames",
    "read_frames",
    "read_input",
    "read_input_string",
    "register_calculator",
    "register_optimizer",
    "registered_calculators",
    "registered_optimizers",
    "rescale_to_temperature",
    "run_md",
    "simulation_from_dict",
    "site_symmetry",
    "torch_available",
    "torch_version",
    "water_box",
    "water_model",
    "water_model_names",
    "water_molecule",
    "water_topology",
    "symbol_from_number",
    "thermostat_names",
    "to_ase",
    "to_extxyz",
    "to_yaml",
    "write_yaml",
    "total_stress",
    "units",
    "vasp_species_layout",
    "voigt_to_matrix",
    "write",
]


def to_ase(atoms: Atoms) -> Any:
    """Convert a FastAtoms structure into an :class:`ase.Atoms`.

    ASE is imported lazily, so it is only needed if this function is called.
    Positions, atomic numbers, cell, boundary conditions, masses and the
    optional per-atom arrays are all carried over.
    """
    from ase import Atoms as AseAtoms

    converted = AseAtoms(
        numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell().array,
        pbc=atoms.get_pbc(),
    )
    converted.set_masses(atoms.get_masses())
    if atoms.get_momenta().any():
        converted.set_momenta(atoms.get_momenta())
    if atoms.get_charges().any():
        converted.set_initial_charges(atoms.get_charges())
    if atoms.get_tags().any():
        converted.set_tags(atoms.get_tags())
    for name in atoms.array_names():
        converted.set_array(name, atoms.get_array(name))
    return converted


def from_ase(atoms: Any) -> Atoms:
    """Convert an :class:`ase.Atoms` into a FastAtoms structure."""
    cell = atoms.get_cell()
    converted = Atoms(
        numbers=[int(z) for z in atoms.get_atomic_numbers()],
        positions=atoms.get_positions(),
        cell=cell.array if hasattr(cell, "array") else cell,
        pbc=[bool(p) for p in atoms.get_pbc()],
    )
    converted.set_masses(atoms.get_masses())
    momenta = atoms.get_momenta()
    if momenta is not None and momenta.any():
        converted.set_momenta(momenta)
    charges = atoms.get_initial_charges()
    if charges is not None and charges.any():
        converted.set_charges(charges)
    tags = atoms.get_tags()
    if tags is not None and tags.any():
        converted.set_tags([int(t) for t in tags])
    return converted
