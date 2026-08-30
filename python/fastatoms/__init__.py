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
"""

from __future__ import annotations

from typing import Any

from ._core import (
    MAX_ATOMIC_NUMBER,
    Atom,
    Atoms,
    CalculatorParameters,
    Cell,
    InputDeck,
    InputError,
    MolecularDynamicsParameters,
    NeighborList,
    OptimizerParameters,
    OutputParameters,
    SimulationParameters,
    Task,
    __version__,
    atomic_mass,
    atoms_from_dict,
    covalent_radius,
    natural_cutoffs,
    neighbor_list,
    number_from_symbol,
    parse_yaml,
    parse_yaml_file,
    read,
    read_atoms,
    read_string,
    simulation_from_dict,
    symbol_from_number,
    to_yaml,
    units,
    write,
)

__all__ = [
    "MAX_ATOMIC_NUMBER",
    "Atom",
    "Atoms",
    "CalculatorParameters",
    "Cell",
    "InputDeck",
    "InputError",
    "MolecularDynamicsParameters",
    "NeighborList",
    "OptimizerParameters",
    "OutputParameters",
    "SimulationParameters",
    "Task",
    "__version__",
    "atomic_mass",
    "atoms_from_dict",
    "covalent_radius",
    "from_ase",
    "natural_cutoffs",
    "neighbor_list",
    "number_from_symbol",
    "parse_yaml",
    "parse_yaml_file",
    "read",
    "read_atoms",
    "read_string",
    "simulation_from_dict",
    "symbol_from_number",
    "to_ase",
    "to_yaml",
    "units",
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
