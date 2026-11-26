"""ASE conversion for ``fastatoms.Atoms``.

ASE is imported lazily, so the core works without it installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Arrays that the C++ Atoms stores as built-in members, not as extra arrays.
_BUILTIN = {"numbers", "positions", "masses", "momenta", "tags", "charges"}


def _plain(obj: Any) -> Any:
    """Turn numpy values in ASE's ``info`` into plain Python objects."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


def to_ase(self: Any) -> Any:
    """Convert to ``ase.Atoms``. The labels go to ``arrays['labels']``."""
    import ase

    atoms = ase.Atoms(
        numbers=self.numbers,
        positions=self.positions,
        cell=np.asarray(self.cell),
        pbc=self.pbc,
    )
    # Labels always travel with the structure, even when they are the symbols.
    atoms.arrays["labels"] = np.array(self.labels, dtype=str)
    for name, value in self.arrays.items():
        if name in ("numbers", "positions", "labels"):
            continue
        if name == "charges":
            name = "initial_charges"
        atoms.arrays[name] = np.asarray(value)
    atoms.info = _plain(self.info)
    return atoms


def from_ase(atoms: Any) -> Any:
    """Build a ``fastatoms.Atoms`` from an ``ase.Atoms``."""
    from fastatoms._core import Atoms

    out = Atoms(
        numbers=[int(z) for z in atoms.numbers],
        positions=np.asarray(atoms.positions, dtype=float),
        cell=np.asarray(atoms.cell.array, dtype=float),
        pbc=[bool(p) for p in atoms.pbc],
    )
    for name, value in atoms.arrays.items():
        if name in ("numbers", "positions"):
            continue
        value = np.asarray(value)
        if name == "masses":
            out.masses = value
        elif name == "momenta":
            out.momenta = value
        elif name == "tags":
            out.tags = value
        elif name in ("initial_charges", "charges"):
            out.charges = value
        else:
            out.set_array(name, value)
    out.info = _plain(dict(atoms.info))
    return out
