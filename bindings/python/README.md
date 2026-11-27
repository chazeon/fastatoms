# fastatoms Python bindings

A pybind11 module, `fastatoms._core`, with an ASE-style `fastatoms.io` layer on
top. It serves two purposes: the parity suite in `tests/py/` compares the C++
core with ASE, vesin and spglib through it, and downstream code (gemmy's server)
uses it as a structure reader. It is not published to PyPI; install it from git.

## Install

```sh
pip install "fastatoms @ git+https://github.com/chazeon/fastatoms@<sha>"
# or: uv pip install ...   /   uv add "fastatoms @ git+..."
```

Pin a commit sha. The build needs Python >= 3.10, a C++20 compiler, git and
network access: CMake fetches gemmi, vesin, spglib and pybind11 at pinned tags,
and scikit-build-core provides cmake and ninja. A clean build takes about 15 s.
The only runtime dependency is numpy.

Every feature can be switched off at build time; YAML, MLIP and the other legacy
modules are never built:

```sh
pip install "fastatoms @ git+..." -C cmake.define.FASTATOMS_WITH_SYMMETRY=OFF
# switches: FASTATOMS_WITH_{VASP,XYZ,CIF,LAMMPS,NEIGHBORS,SYMMETRY}, all ON by default
```

`fastatoms.available_formats()` reports what a given build contains.

## Stable API

The names below are what downstream code may rely on: their signatures and
semantics change only together with a note here. Everything else (`_core`
internals, `tests/`, private helpers) may change at any time.

- `fastatoms.io.read(filename, index=None, format=None, **options)`: `filename` is
  a path, a text or binary file object, or a `.gz` path; ASE's `file@index`
  syntax is honoured. `index` follows `ase.io.read`: `None` is the last frame
  (one `Atoms`), an int is one `Atoms`, a slice or a string such as `":"` or
  `"::2"` is a list. `fastatoms.read` is the same function.
- `fastatoms.io.read_string(text, format, index=None, **options)`,
  `fastatoms.io.iread(...)` (every frame by default),
  `fastatoms.io.write(filename, images, format=None, **options)` and
  `write_string(...)`. A list of images is accepted for `xyz`, `extxyz` and `cif`.
- Format names: `vasp`, `xdatcar`, `xyz` (plain), `extxyz`, `cif`, `lammps-data`,
  `lammps-dump`, with ASE's names accepted as aliases. Guessing follows ASE's
  precedence: special file names, then strong content signatures, then the
  extension, then a sniff for unnamed XYZ.
- Options are passed through per format; an unknown option raises `FormatError`.
  The important ones: `types={int: label}` (LAMMPS; named types resolve from the
  map, every other unresolved type becomes `X` with label `"k"`, a label naming no
  element raises), `unwrap` (LAMMPS image flags, default `False`), `atom_style`,
  `units` (LAMMPS data mass inference), `disorder` (`keep_all`, `max_occupancy`,
  `error`) and `merge_tolerance` (CIF, in Å), `species` (VASP 4).
- `fastatoms.Atoms`: `labels` (the species label as written: `Fe1`, `Fe_pv`,
  `H1`), `symbols`, `numbers`, `positions`, `cell` (rows are lattice vectors),
  `pbc`, `scaled_positions(wrap=False)`, `velocities` (as written, file units),
  `arrays` (`type`, `id`, `image`, `occupancy`, `selective_dynamics`, ... when the
  file provides them; CIF adds `spacegroup_kinds`), `info` (always a dict; CIF
  adds ASE's `occupancy` map when the file has an occupancy column),
  `repeat(reps)` (ASE order), `copy()`, `to_ase()`
  and `Atoms.from_ase(atoms)`.
- `fastatoms.neighbor_list(quantities, atoms, cutoff)`: the pair set, `S`, `D` and
  `d` of `ase.neighborlist.neighbor_list`, sorted by `(i, j, S)`.
- `fastatoms.FormatError` (a `ValueError`) for every malformed or unsupported
  input.
- Deliberate differences from ASE are listed in `tests/py/test_formats.py`
  (`DELIBERATE_DIFFERENCES`) and asserted by the test suite.

## What is bound

- `Atoms`: numbers, positions `(N, 3)`, cell `(3, 3)`, pbc, symbols, labels,
  `scaled_positions(wrap)`, `arrays` (a dict of numpy arrays, string arrays as
  lists), `info` (a dict), `repeat`, `to_ase()` and `Atoms.from_ase(atoms)`.
- `_core.read`, `read_all`, `write`, `guess_format`, `available_formats`, `can_write`
  (text in, text out; `fastatoms.read_string` and `write_string` wrap them).
- `fastatoms.io`, ASE-style and written in Python: `read(filename, index, format,
  **options)`, `iread`, `write(filename, images, format, **options)`, plus
  `read_string` and `write_string`. `fastatoms.read` and `fastatoms.write` are
  aliases of the same functions. Paths, text and binary file objects, and `.gz`
  paths are accepted.
- `symmetry` (`fastatoms.symmetry`, present when built with
  `FASTATOMS_WITH_SYMMETRY`): `get_symmetry_dataset` (spglib's attribute names),
  `standardize_cell`, `find_primitive`, `niggli_reduce`, `analyse`,
  `space_group_number` and `space_group_symbol`.
- `neighbor_list(quantities, atoms, cutoff)`, present when the core is built with
  `FASTATOMS_WITH_NEIGHBORS`. It returns a tuple in the order requested, like
  `ase.neighborlist.neighbor_list`.

Getters that return numpy arrays copy the data once and transfer ownership of
that buffer to numpy through a capsule. Nothing returned can dangle when the
`Atoms` later reallocates. Results that are computed on the fly (neighbour lists,
scaled positions) are moved without a copy.

## Build and test

The build uses [uv](https://docs.astral.sh/uv/) and scikit-build-core, with the
cmake define `FASTATOMS_BUILD_PYTHON=ON` set in `pyproject.toml`.

Point the build at local gemmi and vesin checkouts so that nothing is downloaded:

```sh
export SKBUILD_CMAKE_DEFINE="FASTATOMS_GEMMI_SOURCE_DIR=/path/to/gemmi;FASTATOMS_VESIN_SOURCE_DIR=/path/to/vesin"
uv sync                 # builds and installs fastatoms plus the dev group
uv run pytest           # runs tests/py (add -s to see the 2000-atom timing)
```

Leave out the `SKBUILD_CMAKE_DEFINE` line and CMake fetches gemmi and vesin
instead. To turn a feature off, add it to the same variable, for example
`FASTATOMS_WITH_CIF=OFF`. The suite then skips the tests for that format.

After editing C++ sources, rebuild with `uv sync --reinstall-package fastatoms`.
uv keys its build cache on `pyproject.toml`, so a plain `uv sync` does not pick
up C++ edits.

numpy is a runtime dependency (the package imports it). The dev group holds pytest, ase, vesin and spglib. vesin is used only as a
neighbour-list reference and spglib only as the symmetry reference.

Check everything in one go with `scripts/check-python.sh`. It runs
`ruff format --check`, `ruff check`, `mypy --strict` and pytest, and exits non-zero
on the first failure. Ruff and mypy are configured in `pyproject.toml`.

Types: `fastatoms/_core.pyi` is the stub for the compiled module. It starts from
`pybind11-stubgen` output and is hand-tidied (numpy types, the `symmetry` namespace).
`tests/py/test_stubs.py` regenerates it and checks that the public names and
parameter names still match the built module, so a stale stub fails the suite.
`fastatoms/py.typed` marks the package as typed.

## Layout

- `src/module.cpp`: module entry point and exception types (`InputError` and
  `FormatError`, both `ValueError` subclasses).
- `src/common.{hpp,cpp}`: numpy ownership (`move_to_array`), Python to `Value`
  conversion and the cell and pbc helpers.
- `src/bind_atoms.cpp`, `src/bind_formats.cpp`, `src/bind_neighbors.cpp`,
  `src/bind_symmetry.cpp`.
- `fastatoms/__init__.py`: re-exports `_core`, attaches the ASE conversion from
  `fastatoms/_ase.py` to `Atoms`, and exposes `fastatoms/io.py` as `fastatoms.io`.
