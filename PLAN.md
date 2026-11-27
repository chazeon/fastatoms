# PLAN — fastatoms → slim C++20 core, WASM + TS, pybind11 for tests

Source of truth for the goal: `GOAL.md`. Downstream consumer: `chazeon/gemmy`
(Python `compile` stage: structure load → supercell → periodic images →
neighbour graph → bonds/polyhedra ligands). This library provides that first
stage in the browser; the convex hull and the scene logic stay out of scope.

Work happens on the branch `wasm-core`; `main` is not touched.

## 1. What gemmy actually needs (from the code dive)

- `read(text, format)` for POSCAR/CONTCAR (v4/v5), extxyz, CIF and LAMMPS,
  **keeping the raw species label per atom** (`Fe1`, `H1`, `Fe_pv`) next to
  the chemical symbol. gemmy uses the label as `style_key`.
- `Atoms`: cell (rows are lattice vectors), pbc, positions,
  `getScaledPositions(wrap=false)`, symbols, labels, and `repeat([nx,ny,nz])`
  in ASE order (atom index fastest, so labels survive via `i % n0`).
- `neighborList("ijSdD", atoms, cutoff)`, ASE semantics:
  - no self pair at S=0, but self-images are included;
  - triclinic cells, mixed pbc, and cutoff > cell all work;
  - **deterministic order** (sorted by i, then j, then S).
- Small lattice helpers (wrap shift, snap) are trivial and stay in TS/Python
  for now.

## 2. What the existing code gives us (from the code dive)

**Keep, trimmed:**
- `vec3.hpp`, `units.hpp`, `cell.*`, `elements.*`.
- `atoms.*`: remove the calculator, constraints, momenta and thermodynamics
  members, and **add per-atom string arrays** (`labels`).
- The `Value` part of `io/parameters` (`info` dict).
- `io/extxyz.*`: strip the calculator `Results` coupling.
- The util/text helpers.
- CMake patterns: find system package, else FetchContent with
  `GIT_SHALLOW` and tags as cache variables; explicit source lists;
  `FASTATOMS_IS_TOP_LEVEL`.
- The pybind11 zero-copy helpers: `move_to_array`, `make_view`,
  `as_flat_doubles`.

**Replace:**
- **POSCAR:**
  - labels such as `Fe1`, `Fe_pv`, `Fe/abc`, `Fe_sv_GW` are misread as the
    counts line;
  - three scale factors are silently misread;
  - a mode line with leading whitespace is read as Cartesian;
  - per-axis selective dynamics flags are collapsed;
  - velocities are dropped;
  - the comment is lost.
  We rewrite the reader.
- **CIF:**
  - multi-line loop rows are dropped and text fields are skipped;
  - there is no space-group table;
  - occupancy is ignored.
  We switch to **gemmi** (header subset plus `symmetry.cpp` and
  `sprintf.cpp`; no zlib, no mmCIF).
- **Neighbour list:** switch to **vesin** (an external library, which
  GOAL.md prefers), with `n_threads=1` under WASM. The old binned list is
  deleted. It is correct, but we don't keep two.

**New:**
- LAMMPS `data` and `dump` IO. The old code has none.

**Drop:**
- calculators, MLIP, constraints, MD, optimize, NEB, phonon, symmetry,
  electronic, build, adsorption, analysis, forcefield, globalopt;
- yaml-cpp, process, OUTCAR, conda, and the old examples and benchmarks.
- XDATCAR stays only as a cheap frames reader.

## 3. Target layout

Per the user: **out-of-scope modules stay in the tree but are not built**, and
since the code is MIT **reused files are edited in place** (no parallel copies).
P1 agents write to `formats/` and `neighbor/` so they don't collide; P1.5 then
moves those over the legacy `io/` files and `core/neighbor_list` in place. The old top-level build lives in
`cmake/legacy/CMakeLists.legacy.txt` (and the old pyproject next to it). New
code goes in new directories, so no legacy file gets clobbered:

```
CMakeLists.txt                      new: options per feature (§5), builds only the slim core
cmake/Dependencies.cmake            gemmi + vesin (compiled subset), pybind11
include/fastatoms/core/             reused: vec3, cell, elements (+ try_number_from_label),
                                    atoms (trimmed; + int/string arrays), value (carved from io/parameters)
include/fastatoms/io/               IO, replaced in place: format (registry), vasp, xyz
                                    (was extxyz), cif (gemmi), lammps (new)
include/fastatoms/core/neighbor_list.hpp   replaced in place: vesin neighbour pairs
include/fastatoms/symmetry/         replaced in place: spglib
src/io/, src/core/, src/symmetry/   implementations
tests/core/                         new C++ tests (testing.hpp harness reused)
tests/data/<format>/                fixtures shared by C++, Python and JS tests
tests/py/                           pytest parity vs ASE / vesin (uv)
bindings/wasm/                      embind module
bindings/python/                    pybind11 test module
js/                                 npm package "fastatoms" (TS wrapper, node:test)
legacy, unbuilt:                    io/{parameters,yaml_input}, md/, optimize/, phonon/, ...
                                    python/, tests/cpp, tests/python
```

## 4. Format edge cases (the checklist auditors verify)

### VASP POSCAR/CONTCAR (read + write)

**Comment line.** Kept in `info["comment"]`.

**Scale line.**
- One value is a scale.
- A negative value is a target volume.
- Three values are per-axis scales. They multiply the lattice columns, as
  VASP does. Three negative values are an error.

**Species line.**
- The line is detected as a species line if its **first token is not an
  integer**. This replaces the old "every token is an element" check.
- Each token is kept verbatim as the label.
- The element comes from `label_to_element`:
  - strip a `/hash` suffix (VASP 6 POTCAR hash);
  - strip `_` followed by anything (`_pv`, `_sv_GW`, `_h`, `_s`, `_GW`);
  - strip trailing digits, `+` or `-` (`Fe1` → Fe, `O2-` → O);
  - then match 2 letters before 1, case-insensitive.
  - Phonopy/ASE-style labels all go through this path: `Fe1`, `Fe_pv1`,
    `Fe_pv`, `Fe1_pv`.
- Labels that don't resolve to an element are an error that names the label.
- Duplicate species blocks (`Fe O Fe`) are allowed.

**VASP 4 (no species line).** Species are taken from:
1. an explicit `species=[...]` option, else
2. comment-line tokens that all resolve as labels, else
3. an error.

**Counts line.** Integer count per species; the number of species must match.

**Selective dynamics.** Optional line whose first non-space char is `s`/`S`.
Kept per axis as a `bool[3]` array `selective_dynamics` (true = free, ASE's
`FixCartesian` semantics inverted, documented).

**Coordinate mode.** First non-space char: `c`/`C`/`k`/`K` → Cartesian,
scaled by the scale factor(s); anything else → Direct.
- Leading whitespace is tolerated.
- An empty mode line is an error.

**Positions.**
- Exactly N rows, each with 3 numbers.
- Flags T/F follow if selective dynamics is on.
- Trailing tokens are ignored (a site label such as `! Fe1`).
- Fortran `1.0D-03` is accepted.

**Velocity block** (optional, after a blank line, with an optional mode
line). Read into `velocities`.
- Predictor-corrector blocks are skipped without error.
- CONTCAR from a relaxation, MD and phonopy `SPOSCAR`/`POSCAR-001` are all
  covered.

**Line endings.** CRLF, tabs, and blank lines at the end are all fine.

**Writer.**
- Groups species consecutively, writes labels when present (opt out with
  `labels=false`), Direct by default, and selective dynamics if present.
- It uses shortest round-trip output. Cartesian read → write → read is
  bit-identical; Direct is within ~1 ulp, because positions are stored
  Cartesian and fractions are re-derived.

### LAMMPS data (read + write)

**Header.**
- Counts: `atoms`, `atom types`, and so on; other sections (bonds etc.) are
  skipped by count.
- Box: `xlo xhi`, `ylo yhi`, `zlo zhi`, and `xy xz yz` (restricted
  triclinic).
- The 2024+ **general triclinic** `avec` / `bvec` / `cvec` / `abc origin`
  is also accepted.
- The cell is built as `a=(lx,0,0)`, `b=(xy,ly,0)`, `c=(xz,yz,lz)`, and the
  positions are shifted by the box origin.

**Atom styles.**
- The style is taken from an option, else the `Atoms # style` comment, else
  guessed from the field count of the first row (as ASE).
- Supported: atomic, charge, molecular, full, bond, angle, sphere.
- Image flags in the 3 trailing ints are applied optionally (`unwrap`);
  otherwise they are kept as an `image` array.

**Type → element map, in priority order:**
1. an explicit `types={1:"Fe",2:"O"}` (or a list); a partial map is allowed,
   and an entry that names no element is an error;
2. the `Atom Type Labels` section (LAMMPS 2023+; labels may replace type
   numbers in Masses, Atoms and dump type columns);
3. `Masses` comments (`1 55.845 # Fe`) whose element agrees with the mass; a
   second comment word that names the element (`# C14 C`, written for
   isotopes) is taken as written when it agrees with the label;
4. mass matching against the element table (nearest within 0.1 u);
5. otherwise the type is the dummy element X (Z = 0) labelled by its number
   (`"2"`), the same as a dump type with no map. A Masses comment that names
   an element whose mass disagrees with the stated mass, and no element
   matches that mass, is an error (inconsistent input, not merely unknown).
- The type number is kept as the `type` array; the label is the map value,
  so `Fe1` and `Fe2` can both be iron.

**Other.**
- Atoms are sorted by id, and the ids are kept as an `id` array.
- `Velocities` is read.

**Writer.**
- atomic/charge/full style, with `Masses` + `# El` comments.
- The type order is stable (first appearance, or a given `specorder`).
- A non-LAMMPS-oriented cell is rotated to the restricted-triclinic form
  (QR); the rotation is reported in `info`. Tilt factors are written only if
  non-zero, or if forced.

### LAMMPS dump (text, read; multi-frame)

- `ITEM: TIMESTEP`, `NUMBER OF ATOMS`, and
  `BOX BOUNDS [xy xz yz] pp pp ff` (pbc from the flags).
- The **triclinic bounds conversion** follows the LAMMPS docs: `xlo_bound`
  includes the min/max of the tilts.
- Columns:
  - `id`, `type`, `element`;
  - coordinates: `x y z` | `xs ys zs` | `xu yu zu` | `xsu ysu zsu`, chosen
    in that priority order (unwrapped preferred, if asked);
  - `ix iy iz`, `vx vy vz`, `fx fy fz`, `q`;
  - any other column becomes a float array.
- An explicit `types` entry wins over the element column (the element column
  is then not consulted); the element column wins over the file's own types.
- Rows are sorted by id.
- `index` selects frames (`0`, `-1`, slice) without parsing the others in full.

### XYZ / extxyz

- Reuse the existing reader.
- A plain XYZ has no cell and pbc F.
- An extxyz `Lattice` comment sets pbc T by default, and `pbc` overrides it.
- `Properties` columns map to arrays; extra string columns (e.g. `label:S:1`)
  are now **kept** as string arrays.
- Multi-frame.
- Writer: extxyz with shortest round-trip floats.

### CIF (gemmi)

- `cif::read_memory`, then `make_small_structure_from_block`, then
  `get_all_unit_cell_sites()`, which applies the symmetry ops. Hall and HM
  symbols come from gemmi's table when there are no ops.
- The element comes from the `_atom_site_type_symbol`, else from the label.
- Labels are kept (`Fe1`).
- Occupancy is kept as `occupancy`.
- ASE's `info["occupancy"]` (only when the file has an occupancy column: row
  index as text → {symbol: occupancy} of every row at that position, own first)
  and the `spacegroup_kinds` int array linking each atom to its row.
- **Disorder policy option:** `keep_all` (default), `max_occupancy` (one atom
  per site), `error`.
- The cell uses the same convention as `Cell::from_parameters` (matches ASE).
- Multi-block files: `index` selects the block.
- Writer: a simple P1 CIF with labels (no gemmi needed).

## 5. Customisable WASM ("cut to what we need")

CMake feature options, each adding its sources, defines and dependency:

| Option | Default | What it adds |
| --- | --- | --- |
| `FASTATOMS_WITH_VASP` | ON | POSCAR, XDATCAR |
| `FASTATOMS_WITH_XYZ` | ON | xyz, extxyz |
| `FASTATOMS_WITH_CIF` | ON | pulls in gemmi |
| `FASTATOMS_WITH_LAMMPS` | ON | LAMMPS data and dump |
| `FASTATOMS_WITH_NEIGHBORS` | ON | pulls in vesin |
| `FASTATOMS_BUILD_PYTHON` | | Python test binding |
| `FASTATOMS_BUILD_TESTS` | | C++ tests |

- **No YAML.** The embind layer registers only the enabled formats, and
  `formats()` reports them at runtime.
- **Build script:** `js/scripts/build-wasm.sh [--minimal|--full|-D...]`
  (emsdk from `~/emsdk`).
- **Output:** an ES module with `-sMODULARIZE -sEXPORT_ES6`,
  `-sALLOW_MEMORY_GROWTH`, `-O3`/`-Oz` and `--closure 1`. Optionally
  `-sSINGLE_FILE` for bundler-free use.
- **Size budget** (recorded in the README): minimal (vasp+xyz+nl) and full.

## 6. JS/TS API (ASE-flavoured, camelCase)

```ts
import { init, read, readAll, write, Atoms, neighborList, formats } from "fastatoms";
await init();                                  // loads wasm once
const atoms = read(text, { format: "vasp" });  // format auto-detected from filename option
atoms.length; atoms.cell /* Float64Array(9) */; atoms.pbc;
atoms.getPositions(); atoms.getScaledPositions({ wrap: false });
atoms.getChemicalSymbols(); atoms.getAtomicNumbers();
atoms.getArray("labels"); atoms.arrays; atoms.info;
atoms.repeat([2, 2, 2]); atoms.wrap(); atoms.copy(); atoms.getVolume();
const nl = neighborList("ijSdD", atoms, 3.0);  // {i,j,S,d,D} typed arrays, sorted
write(atoms, { format: "extxyz" });            // → string
```

- Data crosses as typed-array **copies**, so there are no dangling views
  after memory growth.
- The C++ `Atoms` is held behind a handle with `dispose()`, plus a
  `FinalizationRegistry` as a safety net.

## 7. Python (tests only)

- pybind11 module `fastatoms._core`, built with uv + scikit-build-core.
- pytest parity against `ase.io.read`:
  - vasp, cif, lammps-data, lammps-dump and extxyz on the same fixtures;
  - `ase.neighborlist.neighbor_list` and `vesin` (Python) for the neighbour
    sets, over random triclinic cells, mixed pbc and cutoff > cell.
- Not packaged for PyPI.

## 8. Execution plan and agents

| Phase | Who | Work |
| --- | --- | --- |
| P0 | me | Branch; Atoms/Value/labels core; contract headers in `formats/` and `neighbor/`; registry; new CMake; core tests. ✅ |
| P1 | haiku (parallel, each owns its files and its own feature-only build dir) | **A** VASP · **B** LAMMPS data+dump · **C** CIF via gemmi · **D** vesin neighbour pairs · **E** xyz/extxyz port |
| P1.5 | me | Fold `formats/` → `io/` (replacing vasp/cif/extxyz→xyz) and `neighbor/` → `core/neighbor_list`, in place; namespace `fastatoms::io`; full build. ✅ |
| P2 | haiku (parallel) | **E** embind + TS package + node tests + build script · **F** pybind11 + uv + pytest ASE parity |
| P3 | sonnet | **Audit 1:** format edge cases against §4 (writes failing tests/fixtures) · **Audit 2:** NL correctness, API and WASM build/size, C++ quality |
| P4 | haiku | Fix the audit findings; me: final verify (C++ tests, pytest, node tests, wasm size) and README |

**Done when** all of the following hold:
- `ctest`, `uv run pytest` and `npm test` pass;
- the full and minimal WASM builds succeed;
- parity with ASE holds on every fixture;
- the gemmy examples (`examples/*/POSCAR`, extxyz) load in node with
  labels intact.
