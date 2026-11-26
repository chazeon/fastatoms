# fastatoms

Read and write atomic structures (VASP POSCAR/CONTCAR/XDATCAR, extended XYZ,
CIF, LAMMPS data and dump) and compute neighbour lists, in Node and the
browser. The C++ core is compiled to WebAssembly; this package wraps it in an
ASE-style API.

```js
import { init, read, readAll, write, neighborList, formats, Atoms } from "fastatoms";

await init();                                   // loads the wasm once

const atoms = read(poscarText, { filename: "POSCAR" });  // last frame, like ASE
atoms.length;                 // number of atoms
atoms.cell;                   // Float64Array(9), rows are lattice vectors
atoms.pbc;                    // [boolean, boolean, boolean]
atoms.getPositions();         // Float64Array(3N) copy
atoms.getScaledPositions({ wrap: true });
atoms.getChemicalSymbols();   // ["Fe", "O", ...]
atoms.getLabels();            // ["Fe1", "O", ...]  as written in the file
atoms.getArray("labels");     // any per-atom array; see below
atoms.info;                   // frame information (comment, energy, ...)
atoms.repeat([2, 2, 2]);      // supercell, labels kept in ASE order
atoms.copy(); atoms.wrap(); atoms.getVolume();

const { i, j, d } = neighborList("ijd", atoms, 3.0);   // ASE semantics
const text = write(atoms, { format: "extxyz" });
const frames = readAll(trajectoryText, { format: "extxyz" });

formats();                    // e.g. ["vasp", "xdatcar", "xyz", "extxyz"]

const nacl = Atoms.fromArrays({
  symbols: ["Na", "Cl"],
  positions: [0, 0, 0, 2.82, 2.82, 2.82],
  cell: [5.64, 0, 0, 0, 5.64, 0, 0, 0, 5.64],
  pbc: true,
});
nacl.dispose();               // optional: frees wasm memory now
```

## ASE-style `io`

`io.read` and `io.write` follow ASE's signatures. The top-level `read`,
`readAll` and `write` are kept and wrap them.

```js
import { init, io } from "fastatoms";
await init();

io.read(text, { format: "vasp" });                  // Atoms: last frame
io.read(text, { filename: "traj.xyz", index: -2 }); // Atoms: frame -2
io.read(text, { format: "extxyz", index: ":" });    // Atoms[]: all frames
io.read(text, { format: "extxyz", index: "::2" });  // Atoms[]: every 2nd
const atoms = await io.read(file);                  // File or Blob: Promise
io.write(atoms, { filename: "CONTCAR" });           // string
io.write(frames, { format: "extxyz" });             // list: xyz/extxyz only
```

- Input: a string, an `ArrayBuffer` or `Uint8Array` (UTF-8, BOM stripped), or a
  `Blob`/`File`. A Blob or File makes the call async and returns a Promise.
  A File supplies its name for the format guess.
- `index`: `undefined` is the last frame; an integer (or integer string) is one
  frame, negatives count from the end; a slice string (`":"`, `"::2"`,
  `"-3:"`, `"1:5"`) gives `Atoms[]`. The return type follows `index`.
- Format: from `format`, else from the filename, else from the content. A
  `.gz` suffix is ignored for the guess. An unknown format throws an Error that
  names the file.
- Gzip: bytes that start with `1f 8b` are inflated with `DecompressionStream`
  on the Blob/File path. Gzipped strings or buffers are an error.
- `io.write` returns text. A list is accepted only for xyz and extxyz, as in
  ASE. A `.gz` filename is refused, since the result is text.

## Symmetry (spglib)

```js
import { symmetry } from "fastatoms";

if (symmetry.hasSymmetry()) {
  const ds = symmetry.getSymmetryDataset(atoms, { symprec: 1e-3 });
  ds.number;          // 225 for fcc Cu
  ds.international;   // "Fm-3m"
  ds.wyckoffs;        // one letter per atom
  ds.rotations;       // Int32Array, 9 per operation
  const conv = symmetry.standardizeCell(atoms, { toPrimitive: false });
  const prim = symmetry.findPrimitive(atoms, { useLabels: true });
  const reduced = symmetry.niggliReduce(atoms);
}
```

Options: `symprec` (angstrom, default `1e-5`), `angleTolerance` (degrees,
default `-1`, spglib's rule), `useLabels` (type atoms by label, so `Fe1` and
`Fe2` are different sites). Without the feature (`--minimal`, `--gemmy`) every
call throws an Error saying to rebuild with `--full`.

Dataset fields: `number`, `international`, `hall`, `hallNumber`, `choice`,
`pointgroup` (symbol), `rotations`, `translations`, `wyckoffs`,
`siteSymmetrySymbols`, `equivalentAtoms`, `crystallographicOrbits`,
`stdLattice`, `stdTypes`, `stdPositions`, `transformationMatrix`,
`originShift`, `primitiveLattice`, `mappingToPrimitive`,
`stdMappingToPrimitive`. Lattices are row-major 3x3, rows as vectors.

## API

| Function | Notes |
| --- | --- |
| `init()` | Loads the module once. Every other call needs it resolved first. |
| `read(text, options?)` | One frame. `options.index` (default `-1`, the last frame). |
| `readAll(text, options?)` | Every frame, in file order. |
| `write(atoms, { format, ...options })` | Returns the file contents as a string. |
| `formats()`, `canWrite(format)` | What this build has. |
| `neighborList(quantities, atoms, cutoff, { full? })` | Letters from `ijSdD`; only those keys come back. `full` defaults to `true`. |
| `hasNeighbors()` | Whether `neighborList` is compiled in. |
| `Atoms.fromArrays({ symbols \| numbers, positions, cell?, pbc? })` | `cell` is 9 row-major numbers or three rows. `pbc` defaults to `false`, as in ASE. |
| `io.read`, `io.write` | ASE-style, see above. |
| `symmetry.*` | spglib, see above. `symmetry.hasSymmetry()` tells whether it is built in. |

`read` picks the format from `format`, else from `filename`, else from the
text. Format-specific options go in the same object, e.g.
`read(text, { format: "vasp", species: ["Fe", "O"] })` for VASP 4 files.
Unknown options are errors.

### Atoms

- Getters return **copies**. A typed array stays valid after further calls and
  after wasm memory grows.
- `getArray(name)` returns a `Float64Array` for float arrays. Integer arrays
  (LAMMPS `type`, `id`, image flags, `tags`) are converted to `Float64Array`
  too, since embind has no `BigInt64Array` mapping. String arrays (such as
  `labels`) return `string[]`. `atoms.arrayInfo()` gives the name, kind and
  `ncols` of each array.
- `info` is a fresh parsed copy on every access. Mutating it does not change the
  atoms.
- `dispose()` frees the wasm-side memory. A `FinalizationRegistry` does the
  same if you forget, but it runs whenever the garbage collector decides, so
  call `dispose()` for large structures.
- After `dispose()`, every method throws.

### Errors

Anything the C++ core rejects becomes a JS `Error` whose `message` names the
problem (unknown format, malformed line, wrong array length, bad option, ...).

### Labels

Species labels are kept per atom: a POSCAR species line `O H1 H2` gives labels
`O`, `H1`, `H2`, and symbols `O`, `H`, `H`. `repeat` keeps labels in ASE order,
so the label of repeated atom `k` is the label of atom `k % N`.

## Build

The package ships a prebuilt module in `dist/wasm/`. To rebuild it, you need
emsdk (default `~/emsdk`, or set `EMSDK`) and the repository's sources.

```sh
npm run build:wasm            # --full (default)
npm run build:wasm:minimal    # --minimal
bash scripts/build-wasm.sh --gemmy
bash scripts/build-wasm.sh --minimal -DFASTATOMS_WASM_SINGLE_FILE=ON
```

The script configures `<repo>/build/wasm-<preset>` with
`emcmake cmake -G Ninja`, builds `fastatoms_wasm`, and copies
`fastatoms.mjs` and `fastatoms.wasm` into `dist/wasm/`. Any `-D...` argument is
passed to CMake and wins over the preset.

| Preset | Formats | Neighbours | Symmetry | `fastatoms.wasm` | gzip -9 |
| --- | --- | --- | --- | --- | --- |
| `--full` (default) | VASP, XDATCAR, XYZ, extXYZ, CIF, LAMMPS data and dump | yes | yes (spglib) | 839,722 bytes (820 KiB) | 328,099 bytes |
| `--minimal` | VASP, XDATCAR, XYZ, extXYZ | yes | no | 419,114 bytes (409 KiB) | 188,399 bytes |
| `--gemmy` | VASP, XDATCAR, XYZ, extXYZ, CIF | yes | no | 533,734 bytes (521 KiB) | 244,260 bytes |

Sizes are for `-Oz` (`MinSizeRel`), with LTO, emmalloc and no file system, on
Emscripten 7.0.1. Before these changes the same presets measured 592,873 bytes
(full, without spglib since nothing called it), 436,686 bytes (minimal) and
266,720 bytes gzipped (full). The spglib build adds about 268 KB raw to the full
preset.

What the flags buy on the minimal preset (raw bytes, measured one at a time):
`-flto` on every object, -16 KiB; `-sMALLOC=emmalloc`, -6 KiB; `-sFILESYSTEM=0`,
no change in the wasm but -13.6 KiB of JS glue (`fastatoms.mjs` 34.5 to 20.9
KB); dropping `<iostream>` from vesin, -7 KiB (not applied, the vesin source is
fetched by CMake).

Environment variables:

- `FASTATOMS_GEMMI_SOURCE_DIR`, `FASTATOMS_VESIN_SOURCE_DIR`,
  `FASTATOMS_SPGLIB_SOURCE_DIR`: existing checkouts of gemmi, vesin and spglib. Without them CMake downloads the pinned tags.
- `EMSDK`: the emsdk directory (default `~/emsdk`).

Single-file output (`-DFASTATOMS_WASM_SINGLE_FILE=ON`) embeds the wasm in
`fastatoms.mjs`. It needs no separate `.wasm` file and suits bundler-free
use.

## Development

```sh
npm install
npm test          # builds the TypeScript, then runs node --test
npm run check     # tsc --noEmit, then biome check (lint and format)
npm run lint      # biome check only
npm run format    # biome format --write
```

`npm run check` is the style gate: the Biome config is `biome.json`, the same
as the gemmy repository. Run `npm run format` before committing.

Tests run against whatever is in `dist/wasm/`. Each feature-dependent test is
skipped when its format or neighbour list is not compiled in. Fixtures are read
from `../tests/data` and from the gemmy examples; set
`FASTATOMS_GEMMY_EXAMPLES` to point at them elsewhere. Tests that need a
missing example are skipped.
