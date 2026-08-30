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
>>> neighbours = fa.neighbor_list(silicon, 4.0)
>>> len(neighbours) > 0
True

Reading an input file
---------------------

>>> deck = fa.read("input.yaml")                       # doctest: +SKIP
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

    atoms.calc = fa.Vasp(xc="pbe", encut=400, kpts=[4, 4, 4], directory="run")
    atoms.calc = fa.Espresso(pseudopotentials={"Si": "Si.upf"}, ecutwfc=40.0)
    atoms.calc = fa.Siesta(xc="pbe", mesh_cutoff=250.0)
"""

from __future__ import annotations

from typing import Any

from ._core import (
    MAX_ATOMIC_NUMBER,
    Atom,
    Atoms,
    Calculator,
    CalculatorError,
    CalculatorParameters,
    Cell,
    Espresso,
    FileIOCalculator,
    LennardJones,
    InputDeck,
    InputError,
    MolecularDynamicsParameters,
    NeighborList,
    OptimizerParameters,
    OutputParameters,
    Property,
    QuantumEspresso,
    Results,
    SimulationParameters,
    Siesta,
    Task,
    Vasp,
    __version__,
    atomic_mass,
    atoms_from_dict,
    covalent_radius,
    make_calculator,
    matrix_to_voigt,
    natural_cutoffs,
    neighbor_list,
    number_from_symbol,
    parse_yaml,
    parse_yaml_file,
    pressure_from_stress,
    read,
    read_atoms,
    read_string,
    register_calculator,
    registered_calculators,
    simulation_from_dict,
    symbol_from_number,
    to_yaml,
    units,
    vasp_species_layout,
    voigt_to_matrix,
    write,
)

__all__ = [
    "MAX_ATOMIC_NUMBER",
    "Atom",
    "Atoms",
    "Calculator",
    "CalculatorError",
    "CalculatorParameters",
    "Cell",
    "Espresso",
    "FileIOCalculator",
    "InputDeck",
    "InputError",
    "LennardJones",
    "MolecularDynamicsParameters",
    "NeighborList",
    "OptimizerParameters",
    "OutputParameters",
    "Property",
    "QuantumEspresso",
    "Results",
    "SimulationParameters",
    "Siesta",
    "Task",
    "Vasp",
    "__version__",
    "atomic_mass",
    "atoms_from_dict",
    "covalent_radius",
    "from_ase",
    "make_calculator",
    "matrix_to_voigt",
    "natural_cutoffs",
    "neighbor_list",
    "number_from_symbol",
    "parse_yaml",
    "parse_yaml_file",
    "pressure_from_stress",
    "read",
    "read_atoms",
    "read_string",
    "register_calculator",
    "registered_calculators",
    "simulation_from_dict",
    "symbol_from_number",
    "to_ase",
    "to_yaml",
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
