"""Expectations for ASE's I/O tests run against fastatoms.

Each entry matches tests by ``"<dotted module>::<test name>"`` with fnmatch
patterns. Actions:

* ``deselect``: the test exercises ASE internals, not file semantics.
* ``xfail``: a known difference. The test is strict, so a fix turns it into an
  XPASS failure that forces the entry to be removed.

Categories (see the report): ``a`` adapter bug (fixed in adapters.py, never
listed here), ``b`` genuine fastatoms bug (never listed here), ``c`` deliberate
or scope difference, ``d`` ASE internal.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Expectation:
    pattern: str
    action: str  # "deselect" or "xfail"
    category: str  # "c" or "d"
    reason: str
    modes: tuple[str, ...] = field(default=("read", "write", "both"))


_CIF = "ase.test.fio.test_cif::"
_EXTXYZ = "ase.test.fio.test_extxyz::"
_DUMP = "ase.test.fio.test_lammpsdump::"
_POSCAR = "ase.test.fio.vasp.test_poscar::"
_LDATA_READ = "ase.test.lammpsdata.test_lammpsdata_read::"
_LDATA_WRITE = "ase.test.lammpsdata.test_lammpsdata_write::"
_LDATA_RW = "ase.test.lammpsdata.test_write_and_read::"
_LDATA_CONN = "ase.test.lammpsdata.test_read_connectivity::"

_OCC = (
    "ASE CIF bookkeeping: info['occupancy'] (and spacegroup_kinds) keeps minority "
    "species on disordered sites. fastatoms stores occupancy as an array and drops "
    "the minority species, so this test checks ASE internals, not file semantics."
)
_CIFBLOCK = (
    "ASE CIFBlock tags (store_tags, labels=, loop_keys=) are ASE's CIF object model. "
    "fastatoms has no equivalent, and the adapter raises NotImplementedError."
)
_CALC = (
    "Calculator results (energy, forces, stress, charges from a calculator or "
    "SinglePointCalculator) are ASE internals. fastatoms has no calculator."
)
_LDATA_MASS = (
    "fastatoms infers an element only from a mass within 0.1 amu of one; any other "
    "type becomes the dummy X with label 'k'. ASE takes the nearest element. Deliberate."
)
_LDATA_SCOPE = (
    "fastatoms skips the Bonds/Angles/Dihedrals sections of LAMMPS data files and "
    "does not write bonds. Deliberate scope (no connectivity arrays)."
)

READ = ("read", "both")
WRITE = ("write", "both")

EXPECTATIONS: list[Expectation] = [
    # -- CIF: ASE internals -------------------------------------------------------
    Expectation(_CIF + "test_cif", "deselect", "d", _OCC),
    Expectation(_CIF + "test_cif_icsd", "deselect", "d", _OCC),
    Expectation(_CIF + "test_bad_occupancies", "deselect", "d", _OCC),
    Expectation(_CIF + "test_cif_labels", "deselect", "d", _CIFBLOCK),
    Expectation(_CIF + "test_cif_loop_keys", "deselect", "d", _CIFBLOCK),
    Expectation(_CIF + "test_cif_writer_label_numbers", "deselect", "d", _CIFBLOCK),
    # -- CIF: deliberate scope ----------------------------------------------------
    Expectation(
        _CIF + "test_cif_roundtrip_nonperiodic",
        "xfail",
        "c",
        "fastatoms reads and writes only periodic CIFs with fractional sites. ASE writes "
        "a molecule as Cartesian sites (_atom_site_Cartn_*), which fastatoms rejects "
        "with an explicit error.",
    ),
    Expectation(
        _CIF + "test_empty_or_atomless[[]*",
        "xfail",
        "c",
        "fastatoms raises FormatError on a CIF with no atoms. ASE returns an empty list for index=':'.",
        modes=READ,
    ),
    Expectation(
        _CIF + "test_cif_missingvector",
        "xfail",
        "c",
        "Both reject a cell with a missing vector. fastatoms's ValueError text differs (wording only).",
        modes=WRITE,
    ),
    # -- extended XYZ -----------------------------------------------------------------
    Expectation(
        _EXTXYZ + "test_comment",
        "xfail",
        "c",
        "fastatoms's extxyz writer does not write vec_cell=True (VEC1..3 cell lines), and "
        "the adapter refuses it. The reader does read VEC lines.",
        modes=WRITE,
    ),
    Expectation(
        _EXTXYZ + "test_vec_cell",
        "xfail",
        "c",
        "fastatoms's extxyz writer does not write vec_cell=True (VEC1..3 cell lines), and "
        "the adapter refuses it. The reader does read VEC lines.",
        modes=WRITE,
    ),
    Expectation(
        _EXTXYZ + "test_array_shape",
        "xfail",
        "c",
        "fastatoms writes a nested list of mixed arrays as _JSON (as ASE does for nested "
        "lists) but emits no UserWarning, where ASE warns that it skipped an unwritable key.",
        modes=WRITE,
    ),
    Expectation(_EXTXYZ + "test_basic_functionality", "deselect", "d", _CALC),
    Expectation(_EXTXYZ + "test_stress", "deselect", "d", _CALC),
    Expectation(_EXTXYZ + "test_conflicting_fields", "deselect", "d", _CALC),
    Expectation(_EXTXYZ + "test_write_read_charges*", "deselect", "d", _CALC),
    # -- LAMMPS dump -----------------------------------------------------------------
    Expectation(
        _DUMP + "test_lammpsdump_no_element",
        "xfail",
        "c",
        "X for untyped dump types: with no element column fastatoms does not guess an "
        "element from the mass column (ASE gives C for mass 12). The atom is kept as X.",
        modes=READ,
    ),
    Expectation(
        _DUMP + "test_lammpsdump_no_element_no_type_no_mass",
        "xfail",
        "c",
        "Both reject a dump with neither a type nor an element column. fastatoms's "
        "FormatError text differs (wording only).",
        modes=READ,
    ),
    Expectation(
        _DUMP + "test_lammpsdump_no_positions",
        "xfail",
        "c",
        "Both reject a dump with no coordinate columns. fastatoms's FormatError text differs (wording only).",
        modes=READ,
    ),
    # -- POSCAR ------------------------------------------------------------------------
    Expectation(
        _POSCAR + "test_read_poscar_no_species",
        "xfail",
        "c",
        "fastatoms needs the species in the POSCAR or a species= option, and does not "
        "read the OUTCAR/POTCAR next to the file as ASE does.",
        modes=READ,
    ),
    # -- LAMMPS data -------------------------------------------------------------------
    Expectation(_LDATA_READ + "test_lammpsdata_read", "xfail", "c", _LDATA_MASS, modes=READ),
    Expectation(_LDATA_RW + "test_types", "xfail", "c", "bonds are not written", modes=("write",)),
    Expectation(_LDATA_RW + "test_bonds", "xfail", "c", _LDATA_SCOPE),
    Expectation(_LDATA_CONN + "test_connectivity*", "xfail", "c", _LDATA_SCOPE, modes=READ),
    Expectation(
        _LDATA_CONN + "test_positions?False?",
        "xfail",
        "c",
        "fastatoms always sorts atoms by id, so sort_by_id=False (file order) is not "
        "available. Deliberate: the reader's atom order is id order.",
        modes=READ,
    ),
    Expectation(
        _LDATA_RW + "test_atom_type_labels",
        "xfail",
        "c",
        "Atom Type Labels section is not written (scope). The adapter raises "
        "NotImplementedError for atom_type_labels=True.",
        modes=WRITE,
    ),
    Expectation(
        _LDATA_RW + "test_image_flags?True*",
        "xfail",
        "c",
        "write_image_flags=True is not implemented: fastatoms's writer omits image flags "
        "(scope). The adapter raises NotImplementedError.",
        modes=WRITE,
    ),
]
