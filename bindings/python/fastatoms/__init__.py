"""Test-only Python bindings for the fastatoms C++ core.

Not published to PyPI. The compiled module is ``fastatoms._core``; this file
re-exports it, adds the ASE conversions (plain Python) and the ASE-style
``fastatoms.io`` submodule, with ``read`` and ``write`` re-exported from it.
"""

import sys
from types import ModuleType
from typing import cast

from fastatoms import _core, io
from fastatoms._ase import from_ase, to_ase
from fastatoms._core import (
    Atoms,
    FormatError,
    InputError,
    available_formats,
    can_write,
    guess_format,
    read_all,
)
from fastatoms.io import iread, read, read_string, write, write_string

Atoms.to_ase = to_ase  # type: ignore[method-assign]  # attached at import; the stub declares it
Atoms.from_ase = staticmethod(from_ase)  # type: ignore[method-assign]  # as above

__all__ = [
    "Atoms",
    "FormatError",
    "InputError",
    "available_formats",
    "can_write",
    "guess_format",
    "io",
    "iread",
    "read",
    "read_all",
    "read_string",
    "write",
    "write_string",
]

if hasattr(_core, "neighbor_list"):
    neighbor_list = _core.neighbor_list
    __all__.append("neighbor_list")

if hasattr(_core, "symmetry"):
    # Also importable as a module: `import fastatoms.symmetry`.
    # The pybind11 submodule is a real module object at runtime; the stub types it as a namespace.
    sys.modules[__name__ + ".symmetry"] = cast(ModuleType, _core.symmetry)
    symmetry = _core.symmetry
    __all__.append("symmetry")
