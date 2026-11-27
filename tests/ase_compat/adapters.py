"""Adapters that present fastatoms readers and writers with ASE's call signatures.

Each public function here has the name and call signature of the ASE function
it replaces, so the plugin can swap it in place. Nothing is copied from ASE:
the logic is written against fastatoms' API, and ASE is only used for the
objects it defines (Atoms, constraint classes) and for its LAMMPS unit table.

Policy for keyword arguments:

* A kwarg with a fastatoms equivalent is translated.
* A kwarg with no equivalent raises ``NotImplementedError`` with the text
  ``fastatoms adapter: kwarg <name> ...``. The test then fails visibly and is
  triaged, instead of the adapter silently ignoring the request.
* A kwarg that only changes formatting, and has no effect on the values that
  round-trip, may be ignored. Each such case is listed in ``IGNORED``.

Every adapter call appends its name to ``CALLS`` so the plugin can tell which
tests actually reached fastatoms.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import PurePath
from typing import Any

import numpy as np

import fastatoms as fa
from fastatoms import io as faio

# Names of fastatoms adapters called since the last reset (one entry per call).
CALLS: list[str] = []

# Keyword arguments that are accepted and deliberately ignored, with the reason.
IGNORED: dict[str, str] = {
    "write_xyz.fmt": (
        "ASE fixed-decimal fmt is not honoured; fastatoms writes shortest "
        "round-trip numbers. Values round-trip exactly, only the text differs."
    ),
}


def _unsupported(where: str, kwarg: str, value: Any = None) -> NotImplementedError:
    shown = "" if value is None else f"={value!r}"
    return NotImplementedError(f"fastatoms adapter: kwarg {kwarg}{shown} is not supported by {where}")


def _fspath(source: Any) -> Any:
    if isinstance(source, PurePath):
        return os.fspath(source)
    return source


def _emit(target: Any, text: str) -> None:
    """Write serialised text to a path or to an open (text or binary) file."""
    if isinstance(target, (str, os.PathLike)):
        with open(os.fspath(target), "w", encoding="utf-8") as handle:
            handle.write(text)
        return
    try:
        target.write(text)
    except TypeError:
        target.write(text.encode("utf-8"))


def _as_list(images: Any) -> list[Any]:
    if isinstance(images, (list, tuple)):
        return list(images)
    if hasattr(images, "get_chemical_symbols"):  # a single ASE Atoms
        return [images]
    if isinstance(images, Iterable):  # e.g. a generator of Atoms
        return list(images)
    raise TypeError(f"expected Atoms or a sequence of Atoms, got {type(images).__name__}")


def _to_ase(frame: Any) -> Any:
    """fastatoms frame -> ase.Atoms.

    Labels are kept only where they differ from the chemical symbols, because
    ASE has no notion of them and they carry no information otherwise.
    """
    atoms = frame.to_ase()
    labels = atoms.arrays.pop("labels", None)
    if labels is not None and not np.array_equal(np.asarray(labels), atoms.get_chemical_symbols()):
        atoms.arrays["labels"] = np.asarray(labels)
    return atoms


# ASE calls the per-atom molecule number "mol-id"; fastatoms calls it "mol".
_MOL_FA_TO_ASE = {"mol": "mol-id"}


def _mol_to_ase(atoms: Any) -> Any:
    """Rename fastatoms' ``mol`` to ASE's ``mol-id``. Like ASE's reader, drop it when
    every atom is in molecule 0."""
    if "mol" in atoms.arrays:
        mol = atoms.arrays.pop("mol")
        if np.any(np.asarray(mol) != 0):
            atoms.arrays["mol-id"] = np.asarray(mol)
    return atoms


def _mol_to_fa(atoms: Any) -> Any:
    """Rename ASE's ``mol-id`` to fastatoms' ``mol`` on a copy of ``atoms``."""
    if "mol-id" not in atoms.arrays:
        return atoms
    work = atoms.copy()
    work.arrays["mol"] = np.asarray(work.arrays.pop("mol-id"), dtype=np.int64)
    return work


def _convert_frames(result: Any) -> Any:
    """Turn fastatoms' result (one frame or a list) into ASE's Atoms or list."""
    if isinstance(result, list):
        return [_to_ase(frame) for frame in result]
    return _to_ase(result)


def _read(source: Any, fmt: str, index: Any, **options: Any) -> Any:
    return faio.read(_fspath(source), index=index, format=fmt, **options)


def _to_fa(atoms: Any, *, vasp_selective: bool = False, drop_constraints: bool = False) -> Any:
    """ase.Atoms -> fastatoms.Atoms, with ASE-only state translated or removed.

    ``vasp_selective`` writes constraints as a ``selective_dynamics`` int array
    (1 = free, 0 = fixed), the convention fastatoms' POSCAR reader and writer
    use. Calculator results are not transferred: fastatoms has no calculator.
    """
    flags = None if drop_constraints else _vasp_fixed_flags(atoms)
    work = atoms.copy()
    work.calc = None
    work.set_constraint()
    frame = fa.Atoms.from_ase(work)
    if vasp_selective and flags is not None and flags.any():
        frame.set_array("selective_dynamics", (~flags).astype(np.int64))
    return frame


# -- constraints (ASE <-> VASP selective dynamics) --------------------------------


def _vasp_fixed_flags(atoms: Any) -> np.ndarray | None:
    """(N, 3) bool, True where the lattice-direction is fixed, from ASE constraints."""
    from ase.constraints import FixAtoms, FixedLine, FixedPlane, FixScaled

    if not atoms.constraints:
        return None
    flags = np.zeros((len(atoms), 3), dtype=bool)
    cell = np.asarray(atoms.cell.array)
    for con in atoms.constraints:
        if isinstance(con, FixAtoms):
            flags[np.asarray(con.index)] = True
        elif isinstance(con, FixScaled):
            flags[np.asarray(con.index)] = np.asarray(con.mask, dtype=bool)
        elif isinstance(con, (FixedPlane, FixedLine)):
            # VASP can only express these along a lattice vector.
            parallel = np.all(np.abs(np.cross(con.dir, cell)) < 1e-5, axis=1)
            if parallel.sum() != 1:
                raise RuntimeError(
                    "fastatoms adapter: FixedPlane/FixedLine direction must be parallel to one cell vector for POSCAR"
                )
            if isinstance(con, FixedPlane):
                flags[np.asarray(con.index)] = parallel
            else:
                flags[np.asarray(con.index)] = ~parallel
        else:
            raise _unsupported("POSCAR writer", f"constraint {type(con).__name__}")
    return flags


def _apply_move_mask(atoms: Any, move_mask: np.ndarray) -> None:
    """Set ASE constraints from an extxyz move_mask column (nonzero = free to move).

    One column (N,) gives a single FixAtoms for the fixed atoms. Three columns
    (N, 3) give one FixCartesian per atom. This matches the column-width rule
    in the extxyz format, so the same file yields the same constraint layout.
    """
    from ase.constraints import FixAtoms, FixCartesian

    movable = move_mask.astype(bool)
    if movable.ndim == 1 or movable.shape[1] == 1:
        fixed = ~movable.reshape(len(atoms))
        indices = np.nonzero(fixed)[0]
        if len(indices):
            atoms.set_constraint(FixAtoms(indices=indices))
        return
    atoms.set_constraint([FixCartesian(a, mask=~movable[a]) for a in range(len(atoms))])


def _movable_from_constraints(atoms: Any) -> np.ndarray | None:
    """move_mask for extxyz writing, or None when there is nothing to write.

    Follows the extxyz rule: any FixCartesian makes it (N, 3), otherwise (N,).
    Only FixAtoms and FixCartesian are represented, as in ASE's extxyz writer.
    """
    from ase.constraints import FixAtoms, FixCartesian

    relevant = [c for c in atoms.constraints if isinstance(c, (FixAtoms, FixCartesian))]
    if not relevant:
        return None
    per_axis = any(isinstance(c, FixCartesian) for c in relevant)
    movable = np.ones((len(atoms), 3) if per_axis else (len(atoms),), dtype=bool)
    for con in relevant:
        index = np.asarray(con.index)
        if isinstance(con, FixAtoms):
            movable[index] = False
        else:
            movable[index] = ~np.asarray(con.mask, dtype=bool) & movable[index]
    return movable.astype(np.int64)


def _apply_selective(atoms: Any, frame: Any) -> None:
    """Set ASE constraints from fastatoms' selective_dynamics array (0 = fixed)."""
    from ase.constraints import FixAtoms, FixScaled

    if "selective_dynamics" not in frame.arrays:
        return
    flags = np.asarray(frame.arrays["selective_dynamics"]).reshape(-1, 3)
    fixed = flags == 0
    constraints: list[Any] = []
    all_fixed: list[int] = []
    for index, row in enumerate(fixed):
        if row.all():
            all_fixed.append(index)
        elif row.any():
            constraints.append(FixScaled(index, row.astype(int), atoms.get_cell()))
    if all_fixed:
        constraints.append(FixAtoms(all_fixed))
    if constraints:
        atoms.set_constraint(constraints)


# -- VASP ------------------------------------------------------------------------


def _vasp_velocity_unit() -> float:
    """Multiply POSCAR velocities (Angstrom/fs) by this to get ASE velocities."""
    from ase.units import Ang, fs

    return Ang / fs


def read_vasp(fd: Any) -> Any:
    """ase.io.vasp.read_vasp(fd): one POSCAR/CONTCAR structure.

    POSCAR velocities are Angstrom/fs. fastatoms keeps them as written, so the
    adapter scales them to ASE's internal units.
    """
    CALLS.append("read_vasp")
    frame = _read(fd, "vasp", -1)
    atoms = _to_ase(frame)
    _apply_selective(atoms, frame)
    if "momenta" in atoms.arrays:
        atoms.set_velocities(atoms.get_velocities() * _vasp_velocity_unit())
    return atoms


def read_vasp_xdatcar(filename: Any = "XDATCAR", index: Any = -1) -> Any:
    """ase.io.vasp.read_vasp_xdatcar: frames from an XDATCAR (int -> Atoms, slice/':' -> list)."""
    CALLS.append("read_vasp_xdatcar")
    frames = _read(filename, "xdatcar", index)
    return _convert_frames(frames)


def write_vasp(
    fd: Any,
    atoms: Any,
    direct: bool = False,
    sort: bool = False,
    symbol_count: Any = None,
    vasp5: bool = True,
    vasp6: bool = False,
    ignore_constraints: bool = False,
    potential_mapping: Any = None,
) -> None:
    """ase.io.vasp.write_vasp: one POSCAR structure."""
    CALLS.append("write_vasp")
    if symbol_count is not None:
        raise _unsupported("POSCAR writer", "symbol_count")
    if vasp6 or potential_mapping:
        raise _unsupported("POSCAR writer", "vasp6/potential_mapping")
    if not hasattr(atoms, "get_chemical_symbols"):
        raise TypeError("write_vasp expects one Atoms object")
    work = atoms.copy()
    if "momenta" in work.arrays:
        # ASE velocities back to POSCAR units (Angstrom/fs).
        work.set_velocities(work.get_velocities() / _vasp_velocity_unit())
    frame = _to_fa(work, vasp_selective=True, drop_constraints=ignore_constraints)
    text = faio.write_string(frame, "vasp", direct=bool(direct), sort=bool(sort), vasp5=bool(vasp5))
    _emit(fd, text)


# -- XYZ / extended XYZ ------------------------------------------------------------


def _extxyz_frame_to_ase(frame: Any) -> Any:
    """One extxyz frame: move_mask becomes constraints, and an empty comment is dropped."""
    atoms = _to_ase(frame)
    if "move_mask" in atoms.arrays:
        _apply_move_mask(atoms, np.asarray(atoms.arrays.pop("move_mask")))
    if atoms.info.get("comment") == "":
        # ASE keeps no 'comment' key for a blank comment line.
        del atoms.info["comment"]
    return atoms


def read_xyz(fileobj: Any, index: Any = -1, properties_parser: Any = None) -> Any:
    """ase.io.extxyz.read_xyz (also ase.io.extxyz.read_extxyz)."""
    CALLS.append("read_extxyz")
    if properties_parser is not None:
        raise _unsupported("extxyz reader", "properties_parser", properties_parser)
    result = _read(fileobj, "extxyz", index)
    if isinstance(result, list):
        return [_extxyz_frame_to_ase(frame) for frame in result]
    return _extxyz_frame_to_ase(result)


def read_plain_xyz(fileobj: Any, index: Any = -1) -> Any:
    """ase.io.xyz.read_xyz: plain XYZ, read by the same fastatoms reader."""
    CALLS.append("read_xyz")
    return _convert_frames(_read(fileobj, "xyz", index))


def _xyz_frames(images: Any, comment: str | None, write_info: bool, columns: Any = None) -> list[Any]:
    frames = []
    keep = None if columns is None else set(columns) | {"numbers", "positions", "symbols"}
    for atoms in _as_list(images):
        work = atoms.copy()
        work.calc = None
        if not write_info:
            work.info = {}
        if comment is not None:
            work.info["comment"] = comment
        movable = _movable_from_constraints(work)
        work.set_constraint()
        if keep is not None:
            for name in list(work.arrays):
                if name not in keep:
                    del work.arrays[name]
        if movable is not None and (keep is None or "move_mask" in keep):
            work.new_array("move_mask", movable)
        frames.append(_to_fa(work))
    return frames


def write_xyz(
    fileobj: Any,
    images: Any,
    comment: str = "",
    columns: Any = None,
    write_info: bool = True,
    write_results: bool = True,
    plain: bool = False,
    vec_cell: bool = False,
) -> None:
    """ase.io.extxyz.write_xyz (also write_extxyz)."""
    CALLS.append("write_extxyz")
    if vec_cell:
        raise _unsupported("extxyz writer", "vec_cell", vec_cell)
    frames = _xyz_frames(images, comment or None, write_info, columns)
    fmt = "xyz" if plain else "extxyz"
    _emit(fileobj, faio.write_string(frames, fmt))


def write_plain_xyz(fileobj: Any, images: Any, comment: str = "", fmt: str = "%22.15f") -> None:
    """ase.io.xyz.write_xyz: plain XYZ. ``fmt`` only changes the number text."""
    CALLS.append("write_xyz")
    frames = _xyz_frames(images, comment or None, True)
    _emit(fileobj, faio.write_string(frames, "xyz"))


# -- CIF -----------------------------------------------------------------------------


def read_cif(
    fileobj: Any,
    index: Any = -1,
    *,
    store_tags: bool = False,
    primitive_cell: bool = False,
    subtrans_included: bool = True,
    fractional_occupancies: bool = True,
    reader: str = "ase",
) -> Any:
    """ase.io.cif.read_cif.

    ASE's default keeps the majority species on a disordered site, which is
    fastatoms' ``disorder="max_occupancy"``. fastatoms' own default
    (``keep_all``) would include the minority species, so the adapter sets it.
    ``fractional_occupancies=False`` only drops ``info["occupancy"]``, as in ASE.
    """
    CALLS.append("read_cif")
    if store_tags:
        raise _unsupported("CIF reader", "store_tags", store_tags)
    if primitive_cell:
        raise _unsupported("CIF reader", "primitive_cell", primitive_cell)
    if not subtrans_included:
        raise _unsupported("CIF reader", "subtrans_included", subtrans_included)
    if reader != "ase":
        raise _unsupported("CIF reader", "reader", reader)
    frames = _read(fileobj, "cif", index, disorder="max_occupancy")
    converted = _convert_frames(frames)
    if not fractional_occupancies:
        for atoms in converted if isinstance(converted, list) else [converted]:
            atoms.info.pop("occupancy", None)
    return converted


def write_cif(
    fd: Any, images: Any, cif_format: Any = None, wrap: bool = True, labels: Any = None, loop_keys: Any = None
) -> None:
    """ase.io.cif.write_cif."""
    CALLS.append("write_cif")
    if cif_format is not None:
        raise _unsupported("CIF writer", "cif_format", cif_format)
    if labels is not None:
        raise _unsupported("CIF writer", "labels", labels)
    if loop_keys is not None:
        raise _unsupported("CIF writer", "loop_keys", loop_keys)
    if not wrap:
        raise _unsupported("CIF writer", "wrap", wrap)
    frames = [_to_fa(atoms) for atoms in _as_list(images)]
    # One data block per structure; a single structure is written as before.
    faio.write(fd, frames if len(frames) > 1 else frames[0], format="cif")


# -- LAMMPS data -----------------------------------------------------------------------


def _lammps_convert(value: Any, quantity: str, units: str) -> Any:
    """LAMMPS units -> ASE units, with ASE's own table (ase.calculators.lammps.convert)."""
    from ase.calculators.lammps import convert

    return convert(value, quantity, units, "ASE")


def _lammps_convert_from_ase(value: Any, quantity: str, units: str) -> Any:
    """ASE units -> LAMMPS units, with ASE's own table."""
    from ase.calculators.lammps import convert

    return convert(value, quantity, "ASE", units)


def read_lammps_data(
    fileobj: Any,
    Z_of_type: dict | None = None,
    sort_by_id: bool = True,
    read_image_flags: bool = True,
    units: str = "metal",
    atom_style: str | None = None,
    style: str | None = None,
) -> Any:
    """ase.io.lammpsdata.read_lammps_data.

    * ``Z_of_type``: each type is labelled with the element symbol, so the
      reader resolves it to the same atomic number.
    * ``sort_by_id``: the frame is reordered by the ``id`` column, as ASE does.
    * ``read_image_flags=True`` (ASE default): fastatoms' ``unwrap``.
    * ``units``: positions, cell, masses and velocities are converted from the
      LAMMPS unit system to ASE units with ASE's conversion table. fastatoms
      keeps values in the file's units. The same name is also passed to
      fastatoms, which uses it to infer elements from the stated masses.
    * ``mol`` <-> ``mol-id``: the molecule column is exposed under ASE's name.
    """
    from ase.data import chemical_symbols

    CALLS.append("read_lammps_data")
    if style is not None:
        atom_style = style
    options: dict[str, Any] = {"unwrap": bool(read_image_flags), "units": units}
    if atom_style:
        options["atom_style"] = atom_style
    if Z_of_type:
        options["types"] = {str(int(t)): chemical_symbols[int(z)] for t, z in Z_of_type.items()}
    frame = _read(fileobj, "lammps-data", None, **options)
    atoms = _mol_to_ase(_to_ase(frame))

    if sort_by_id and "id" in atoms.arrays:
        atoms = atoms[np.argsort(np.asarray(atoms.arrays["id"]), kind="stable")]

    if units != "ASE":
        # ASE's convert() is the unit table in ase.calculators.lammps.
        has_velocities = "momenta" in atoms.arrays
        velocities = atoms.get_velocities() if has_velocities else None
        atoms.set_cell(_lammps_convert(np.asarray(atoms.cell.array), "distance", units))
        atoms.positions = _lammps_convert(atoms.positions, "distance", units)
        if "masses" in atoms.arrays:
            atoms.set_masses(_lammps_convert(atoms.get_masses(), "mass", units))
        if velocities is not None:
            # Masses are converted first, so set_velocities uses the new masses.
            atoms.set_velocities(_lammps_convert(velocities, "velocity", units))
    return atoms


def write_lammps_data(
    fd: Any,
    atoms: Any,
    *,
    specorder: list | None = None,
    reduce_cell: bool = False,
    force_skew: bool = False,
    prismobj: Any = None,
    write_image_flags: bool = False,
    masses: bool = False,
    velocities: bool = False,
    atom_type_labels: bool = False,
    units: str = "metal",
    bonds: bool = True,
    atom_style: str = "atomic",
) -> None:
    """ase.io.lammpsdata.write_lammps_data."""
    CALLS.append("write_lammps_data")
    if reduce_cell:
        raise _unsupported("lammps-data writer", "reduce_cell", reduce_cell)
    if prismobj is not None:
        raise _unsupported("lammps-data writer", "prismobj")
    if write_image_flags:
        raise _unsupported("lammps-data writer", "write_image_flags", write_image_flags)
    if atom_type_labels:
        raise _unsupported("lammps-data writer", "atom_type_labels", atom_type_labels)
    if units != "metal":
        raise _unsupported("lammps-data writer", "units", units)
    if bonds and "bonds" in atoms.arrays:
        raise _unsupported("lammps-data writer", "bonds")
    work = atoms.copy()
    if not velocities:
        if "momenta" in work.arrays:
            del work.arrays["momenta"]
    elif "momenta" in work.arrays:
        # Same order as the reader: masses first, then velocities, so that
        # momenta = mass * velocity stays consistent in LAMMPS units.
        velocity = work.get_velocities()
        work.set_masses(_lammps_convert_from_ase(work.get_masses(), "mass", units))
        work.set_velocities(_lammps_convert_from_ase(velocity, "velocity", units))
    else:
        work.set_masses(_lammps_convert_from_ase(work.get_masses(), "mass", units))
    if specorder is None and "type" not in atoms.arrays:
        # ASE numbers the types in alphabetical order of the element symbols.
        specorder = sorted(set(atoms.get_chemical_symbols()))
    frame = _to_fa(_mol_to_fa(work))
    options: dict[str, Any] = {
        "atom_style": atom_style,
        "masses": bool(masses),
        "force_skew": bool(force_skew),
    }
    if specorder:
        options["specorder"] = list(specorder)
    _emit(fd, faio.write_string(frame, "lammps-data", **options))


# -- LAMMPS dump --------------------------------------------------------------------------


def _dump_units_to_ase(atoms: Any, units: str) -> Any:
    """Convert a dump frame from LAMMPS ``units`` to ASE units, as ASE's reader does:
    cell and positions (distance), and velocities.

    ``to_ase`` sets ASE velocities from fastatoms' stored velocities, so dummy
    (mass-0) atoms keep theirs.
    """
    if units == "ASE":
        return atoms
    atoms.set_cell(_lammps_convert(np.asarray(atoms.cell.array), "distance", units))
    atoms.positions = _lammps_convert(atoms.positions, "distance", units)
    if "momenta" in atoms.arrays:
        velocities = atoms.get_velocities()
        atoms.set_velocities(_lammps_convert(velocities, "velocity", units))
    return atoms


def read_lammps_dump_text(fd: Any, index: Any = -1, units: str = "metal", **kwargs: Any) -> Any:
    """ase.io.lammpsrun.read_lammps_dump_text.

    ``units`` (default "metal", as in ASE) converts the cell, positions and
    velocities to ASE units, with ASE's table. Image flags are not applied (ASE's
    default, and fastatoms' ``unwrap=False``).
    """
    CALLS.append("read_lammps_dump_text")
    if kwargs:
        raise _unsupported("lammps-dump reader", ", ".join(sorted(kwargs)))
    # ASE takes positions as written and ignores the ix/iy/iz image flags, and
    # fastatoms' default (unwrap=False) does the same.
    frames = _convert_frames(_read(fd, "lammps-dump", index))
    if isinstance(frames, list):
        return [_dump_units_to_ase(frame, units) for frame in frames]
    return _dump_units_to_ase(frames, units)


# -- Registry table ---------------------------------------------------------------------

# ASE module, attribute -> adapter. The attribute name is the one ASE's
# registry looks up (read_<format>), plus the public aliases the tests import.
MODULE_PATCHES: dict[tuple[str, str], Any] = {
    ("ase.io.vasp", "read_vasp"): read_vasp,
    ("ase.io.vasp", "read_vasp_xdatcar"): read_vasp_xdatcar,
    ("ase.io.vasp", "write_vasp"): write_vasp,
    ("ase.io.extxyz", "read_xyz"): read_xyz,
    ("ase.io.extxyz", "read_extxyz"): read_xyz,
    ("ase.io.extxyz", "write_xyz"): write_xyz,
    ("ase.io.extxyz", "write_extxyz"): write_xyz,
    ("ase.io.xyz", "read_xyz"): read_plain_xyz,
    ("ase.io.xyz", "write_xyz"): write_plain_xyz,
    ("ase.io.cif", "read_cif"): read_cif,
    ("ase.io.cif", "write_cif"): write_cif,
    ("ase.io.lammpsdata", "read_lammps_data"): read_lammps_data,
    ("ase.io.lammpsdata", "write_lammps_data"): write_lammps_data,
    ("ase.io.lammpsrun", "read_lammps_dump_text"): read_lammps_dump_text,
}

# ASE io-registry name -> (reader, writer). None means ASE keeps its own.
REGISTRY: dict[str, tuple[Any, Any]] = {
    "vasp": (read_vasp, write_vasp),
    "vasp-xdatcar": (read_vasp_xdatcar, None),
    "xyz": (read_plain_xyz, write_plain_xyz),
    "extxyz": (read_xyz, write_xyz),
    "cif": (read_cif, write_cif),
    "lammps-data": (read_lammps_data, write_lammps_data),
    "lammps-dump-text": (read_lammps_dump_text, None),
}

# Modes select which half of each pair is swapped.
READ_MODES = ("read", "both")
WRITE_MODES = ("write", "both")
