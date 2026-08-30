# FastAtoms

A high-performance C++ core for atomistic simulations, driven from Python.

FastAtoms gives you an ASE-shaped API — `Atoms`, `Cell`, chemical symbols,
periodic boundary conditions — on top of contiguous C++ storage, an O(N)
neighbour list, and a YAML front end for structures and run settings. The
Python layer is a thin binding: `atoms.positions` is a zero-copy numpy view
straight into the C++ buffer, so a Python script can drive a simulation without
paying for the convenience.

```python
import fastatoms as fa

deck = fa.read("examples/argon.yaml")       # structure + run settings
atoms = deck.atoms.repeat(3, 3, 3)          # 108-atom supercell

pairs = fa.neighbor_list(atoms, cutoff=8.5, full=False)
print(len(pairs), "pairs")
print(pairs.vectors.shape)                  # (npairs, 3) minimum-image vectors

atoms.positions += 0.01                     # writes into the C++ buffer
```

## Layout

```
include/fastatoms/     public headers (the C++ API)
  core/                vec3, cell, atoms, neighbor_list, elements, units
  io/                  parameters, yaml_input
src/                   implementation
python/
  bindings/            pybind11 extension, one file per bound area
  fastatoms/           the importable package (__init__.py)
  examples/            Python driver scripts
tests/cpp, tests/python
examples/              example YAML inputs and a C++ driver
cmake/                 dependency resolution and the installed package config
```

## Building

Requirements: a C++17 compiler, CMake ≥ 3.20, and Python ≥ 3.9 with numpy for
the bindings. **yaml-cpp** and **pybind11** are found on the system if present
and downloaded automatically otherwise, so a plain checkout builds as-is.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The build tree is directly importable:

```sh
PYTHONPATH=build/python python -c "import fastatoms; print(fastatoms.__version__)"
```

To install the Python package instead:

```sh
pip install .
```

### Build options

| Option | Default | Effect |
| --- | --- | --- |
| `FASTATOMS_BUILD_PYTHON` | `ON` | Build the pybind11 extension |
| `FASTATOMS_BUILD_TESTS` | `ON` | Build the C++ test suite |
| `FASTATOMS_BUILD_EXAMPLES` | `ON` | Build the `fastatoms-run` driver |
| `FASTATOMS_ENABLE_OPENMP` | `ON` | Parallelise the neighbour search |
| `FASTATOMS_ENABLE_NATIVE` | `OFF` | `-march=native`; faster, not portable |
| `FASTATOMS_ENABLE_LTO` | `OFF` | Link-time optimisation in release builds |
| `FASTATOMS_SHARED_CORE` | `OFF` | Build the core as a shared library |
| `FASTATOMS_INSTALL` | `ON` | Generate install rules for the C++ library |

Set `FETCHCONTENT_FULLY_DISCONNECTED=ON` to forbid downloads and require system
dependencies.

## The input format

One YAML file carries both the structure and the run settings. Every section
except `structure` is optional, and most fields accept several spellings.

```yaml
structure:
  cell: {a: 5.43, b: 5.43, c: 5.43, alpha: 90, beta: 90, gamma: 90}
  pbc: [true, true, true]           # or `true`, or "xy"
  coordinates: fractional           # or cartesian (the default)
  symbols: [Si, Si]
  positions: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]

simulation:
  task: md                          # single_point | md | optimize
  calculator:
    name: lennard_jones
    parameters: {epsilon: 0.0103, sigma: 3.4, cutoff: 8.5}
  md: {ensemble: nvt, timestep: 1.0, steps: 5000, temperature: 300.0}
  output: {trajectory: traj.yaml, interval: 50}
```

`cell` accepts a single number (cubic), three lengths (orthorhombic), six
lattice parameters, a 3×3 matrix, or the mapping form above. Atoms can be given
row-wise instead of column-wise, with per-atom extras:

```yaml
structure:
  atoms:
    - {symbol: O, position: [0.0, 0.0, 0.0], charge: -0.834}
    - {symbol: H, position: [0.96, 0.0, 0.0], velocity: [0.0, 0.0, 0.0]}
    - [H, -0.24, 0.93, 0.0]         # the compact form
```

`masses`, `charges`, `tags`, `momenta`, `velocities` and free-form `arrays` are
all read if present. Keys FastAtoms does not interpret are still reachable
through `deck.document` and `deck.simulation.raw`, so a downstream driver can
carry its own settings in the same file.

Malformed input raises `fastatoms.InputError` (a `ValueError`) with the section
and, for syntax errors, the line and column.

## What's in the core

**`Atoms`** — positions, atomic numbers, cell and boundary conditions, plus
optional masses, momenta, charges, tags and arbitrary named arrays. Optional
arrays stay empty until first written and follow the atoms through `extend`,
slicing and `repeat`. ASE-style methods (`get_positions`, `get_scaled_positions`,
`wrap`, `center`, `rotate`, `get_distance(..., mic=True)`, `repeat`, `__getitem__`
with slices and boolean masks) sit alongside Python properties.

**`Cell`** — lattice vectors as rows, cached inverse, volume, lengths, angles,
perpendicular widths, reciprocal vectors, and an exact minimum-image search
that stays correct for strongly sheared triclinic cells (fractional rounding
alone does not).

**`neighbor_list`** — cell-linked-list pair search, linear in system size. It
handles any mix of periodic and free directions, triclinic cells, and cutoffs
larger than the cell itself (each periodic image of a pair is returned with its
own integer shift). The result is compressed-sparse-row: pairs are sorted by
`i`, so `first[a]:first[a+1]` is one atom's block. Pass `per_atom_cutoffs` (see
`natural_cutoffs`) for covalent-radius bond detection.

Shifts refer to the positions you passed in, wrapped or not:

```
r_ij = positions[j] + shifts @ cell - positions[i]
```

**Units** follow ASE exactly — ångström, eV, amu, and a time unit of
Å·√(amu/eV) ≈ 10.18 fs — so values pass between the two libraries unconverted.
`fastatoms.units` holds the constants.

## Performance

Design choices that matter:

- Every per-atom quantity is one contiguous `std::vector`, vectors stored
  atom-major. That is both numpy-compatible without copying and friendly to
  auto-vectorisation; a whole array can be handed to a force kernel as a raw
  pointer.
- The cell caches its inverse matrix, since converting to fractional
  coordinates is on the hot path of wrapping, neighbour search and distances.
- The neighbour search bins atoms with a counting sort, splits atoms into
  contiguous per-thread chunks (so the CSR layout falls out sorted, with no
  locks and no final sort), and releases the GIL.

Measured on an Apple M-series laptop, diamond silicon, 5 Å cutoff (~28
neighbours per atom), 8 threads:

| atoms | pairs | build |
| ---: | ---: | ---: |
| 512 | 14,336 | 0.26 ms |
| 4,096 | 114,688 | 2.03 ms |
| 32,768 | 917,504 | 16.0 ms |
| 64,000 | 1,792,000 | 34.9 ms |

Flat at ~0.5 µs/atom across two orders of magnitude — the search is linear, not
quadratic. The same 32,768-atom case takes 55 ms single-threaded.

## Examples

```sh
# C++ driver: summarise an input file and its neighbour statistics
./build/examples/fastatoms-run examples/silicon.yaml 3.0

# The same, from Python
PYTHONPATH=build/python python python/examples/read_input.py examples/silicon.yaml 3.0

# Coordination numbers and g(r) for a 1728-atom silicon cell
PYTHONPATH=build/python python python/examples/neighbor_analysis.py

# A complete Lennard-Jones MD run: C++ neighbour list, numpy forces
PYTHONPATH=build/python python python/examples/lennard_jones_md.py examples/argon.yaml
```

The MD example conserves energy to under 1 µeV/atom over 200 steps, which is
the end-to-end check that positions, shifts, masses and units all agree.

## Using the C++ library directly

```cmake
find_package(FastAtoms REQUIRED)
target_link_libraries(my_code PRIVATE FastAtoms::core)
```

```cpp
#include "fastatoms/fastatoms.hpp"

const auto deck = fastatoms::io::read_yaml_file("input.yaml");
const auto pairs = fastatoms::build_neighbor_list(deck.atoms, 5.0);
```

## Testing

`ctest` runs four C++ suites and the Python suite. The neighbour list is
checked against a brute-force O(N² × images) reference for cubic, triclinic,
slab, isolated and smaller-than-cutoff cells; the minimum-image convention is
checked the same way.

```sh
ctest --test-dir build --output-on-failure
PYTHONPATH=build/python pytest tests/python
```

## Interoperating with ASE

```python
import fastatoms as fa
ase_atoms = fa.to_ase(atoms)      # needs `pip install ase`
atoms = fa.from_ase(ase_atoms)
```

## Licence

MIT. See `LICENSE`.
