# FastAtoms

[![Build](https://github.com/seixas-research/fastatoms/actions/workflows/ci.yml/badge.svg)](https://github.com/seixas-research/fastatoms/actions/workflows/ci.yml)
[![Conda](https://img.shields.io/conda/vn/conda-forge/fastatoms.svg)](https://anaconda.org/conda-forge/fastatoms)
[![PyPI](https://img.shields.io/pypi/v/fastatoms.svg)](https://pypi.org/project/fastatoms/)
[![Python](https://img.shields.io/pypi/pyversions/fastatoms.svg)](https://pypi.org/project/fastatoms/)
[![Coverage](https://codecov.io/gh/seixas-research/fastatoms/branch/main/graph/badge.svg)](https://codecov.io/gh/seixas-research/fastatoms)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A high-performance C++ core for atomistic simulations, driven from Python.**

---

## Introduction

FastAtoms is an atomistic simulation toolkit whose numerical work happens in
C++ and whose interface looks like ASE. It is built on three commitments.

**A C++ backend that does the work.** Every per-atom quantity is one contiguous
`std::vector`, stored atom-major. The neighbour search bins atoms with a
counting sort and is linear in the number of atoms — flat at about 0.5 µs per
atom from 512 to 64,000 — releasing the GIL while it runs. Force kernels take
raw pointers.

**No external mathematical dependencies.** There is no BLAS, no LAPACK, no FFT
library, no spglib. The optimizers, the eigensolvers, the Ewald summation, the
space-group determination, the linear tetrahedron integration and the symmetry
detection are all written here, from scratch. The only third-party libraries
are yaml-cpp for the input format and pybind11 for the bindings — both resolved
from the system when present and vendored only as a fallback, so a plain
checkout builds as-is.

**A Python interface that costs nothing.** `atoms.positions` is a zero-copy
numpy view straight into the C++ buffer; analysis results come back as
read-only views onto the C++ storage rather than copies. A Python script drives
the simulation without paying for the convenience.

```python
import fastatoms as fa

atoms = fa.build.diamond("Si", 5.43, cubic=True)
atoms.positions += 0.01          # writes into the C++ buffer, no copy
print(atoms.get_space_group().hermann_mauguin)      # Fd-3m
```

## Key features

| Module | What it covers |
| --- | --- |
| **Calculators** | Lennard-Jones and a classical force field in C++; VASP, Quantum ESPRESSO and SIESTA through generated inputs and parsed output |
| **Machine-learning potentials** | MACE, MatterSim, CHGNet, NequIP and Allegro, evaluated in-process through LibTorch or through a persistent worker |
| **Geometry optimizers** | Steepest descent, conjugate gradient, BFGS, L-BFGS and FIRE, with line searches and combined position–cell relaxation |
| **Molecular dynamics** | NVE, NVT and NPT; velocity Verlet, leapfrog and Euler; Berendsen, Langevin, Nosé and Nosé–Hoover thermostats; Berendsen and Parrinello–Rahman barostats |
| **Constraints** | `FixAtoms`, `FixLine` and `FixPlane`, respected by every optimizer, integrator and band alike, with SHAKE/RATTLE for rigid geometry |
| **Reaction paths** | NEB, climbing-image NEB and AutoNEB, with IDPP interpolation for starting bands |
| **Phonons** | Harmonic phonons by finite displacements, with a symmetry-reduced displacement set and no Phonopy dependency |
| **Classical force fields** | SPC and SPC/E rigid water, Ewald electrostatics, SHAKE and RATTLE |
| **Structure builders** | Bulk lattices, graphene and transition-metal dichalcogenides, nanotubes by (n, m), icosahedra, decahedra, cuboctahedra and Wulff constructions |
| **Crystallographic symmetry** | Space groups 1–230 with derived Hermann–Mauguin symbols, Wyckoff positions, and primitive ↔ conventional cell standardization |
| **Trajectory analysis** | Radial distribution functions, static structure factors by three routes, and interfacial tension from the pressure tensor |
| **Electronic structure** | High-symmetry band paths, densities of state by linear tetrahedron and by smearing, direct and indirect band gaps |
| **I/O** | Extended XYZ as the default format, plus a YAML input deck that describes a whole run |

## Installation

### With conda

```sh
conda install -c conda-forge fastatoms
```

A recipe lives in [`conda/`](conda/) and can be built locally in the meantime:

```sh
conda build conda/ -c conda-forge --override-channels
conda install --use-local fastatoms
```

### With pip

```sh
pip install fastatoms
```

Or from a checkout, which builds the extension as part of the install:

```sh
pip install .
```

### From source with CMake

Requirements: a C++17 compiler, CMake ≥ 3.20, and Python ≥ 3.9 with numpy for
the bindings. **yaml-cpp** and **pybind11** are found on the system if present
and downloaded automatically otherwise.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The build tree is directly importable:

```sh
PYTHONPATH=build/python python -c "import fastatoms; print(fastatoms.__version__)"
```

#### Build options

| Option | Default | Effect |
| --- | --- | --- |
| `FASTATOMS_BUILD_PYTHON` | `ON` | Build the pybind11 extension |
| `FASTATOMS_BUILD_TESTS` | `ON` | Build the C++ test suite |
| `FASTATOMS_BUILD_EXAMPLES` | `ON` | Build `fastatoms-run` and `fastatoms-interpolate` |
| `FASTATOMS_ENABLE_OPENMP` | `ON` | Parallelise the neighbour search |
| `FASTATOMS_ENABLE_NATIVE` | `OFF` | `-march=native`; faster, not portable |
| `FASTATOMS_ENABLE_LTO` | `OFF` | Link-time optimisation in release builds |
| `FASTATOMS_SHARED_CORE` | `OFF` | Build the core as a shared library |
| `FASTATOMS_WITH_TORCH` | `OFF` | Evaluate machine-learning potentials in process |
| `FASTATOMS_INSTALL` | `ON` | Generate install rules for the C++ library |

Set `FETCHCONTENT_FULLY_DISCONNECTED=ON` to forbid downloads and require system
dependencies.

## Quickstart

Build a structure, attach a calculator, relax it:

```python
import fastatoms as fa

# A 32-atom argon crystal, deliberately compressed so there is work to do.
atoms = fa.build.fcc("Ar", 5.0, cubic=True).repeat(2, 2, 2)

atoms.calc = fa.LennardJones(epsilon=0.0103, sigma=3.40, cutoff=8.0)
print(f"start   {atoms.get_potential_energy():.4f} eV")

result = fa.LBFGS(fmax=0.01, relax_cell=True).run(atoms)

print(f"relaxed {atoms.get_potential_energy():.4f} eV in {result.steps} steps")
print(f"lattice constant {atoms.cell.lengths[0] / 2:.4f} A")
print(f"space group {atoms.get_space_group().hermann_mauguin}")
```

```
start   -2.0051 eV
relaxed -2.3887 eV in 9 steps
lattice constant 5.2741 A
space group Fm-3m
```

The cell relaxation finds the equilibrium lattice constant, and the symmetry
survives it. The same three lines work with a machine-learning potential
(`atoms.calc = fa.MACE()`) or a DFT code (`atoms.calc = fa.Vasp(xc="pbe")`)
without anything else changing.

More, in [`python/examples/`](python/examples/):

```sh
python python/examples/build.py         # lattices, sheets, tubes, nanoparticles
python python/examples/constraints.py   # three questions about one adatom
python python/examples/neb.py           # reaction paths and barriers
python python/examples/phonons.py       # dispersions and densities of state
```

## Layout

```
include/fastatoms/     public headers (the C++ API)
  build/               bulk, twod (graphene, TMDs), nanotube, cluster (Wulff)
  analysis/            rdf, structure_factor, surface_tension
  electronic/          bandpath, bands, dos (tetrahedron and smearing)
  symmetry/            pointgroup, spacegroup, standardize
  core/                vec3, cell, atoms, constraints, neighbor_list, elements,
                       units
  io/                  extxyz (the default format), formats, parameters, yaml_input
  calculators/         calculator, file_io_calculator, vasp, espresso, siesta,
                       mlip (base), mlip_torch, mlip_worker, mlip_models
  optimize/            optimizer, sd, cg, bfgs, lbfgs, fire, line_search, variables
  forcefield/          topology, water (SPC, SPC/E, TIP3P), ewald, forcefield
  md/                  dynamics, integrator, thermostat, barostat, trajectory,
                       constraints (SHAKE/RATTLE)
  phonon/              phonons, supercell, symmetry, force_constants, eigen
  neb/                 neb, autoneb, idpp, registry
  util/                process, text
src/                   implementation
python/
  bindings/            pybind11 extension, one file per bound area
  fastatoms/           the importable package (__init__.py)
    mlip/              the model exporter and the potential worker
  examples/            Python driver scripts
tests/cpp, tests/python
benchmarks/            head-to-head timing against ASE
conda/                 conda recipe: meta.yaml, build.sh, bld.bat
examples/              example YAML inputs, the `fastatoms-run` driver and
                       `fastatoms-interpolate`
cmake/                 dependency resolution and the installed package config
```

## Files

Extended XYZ is the default. It carries what a simulation produces — many
frames, per-atom forces and velocities, a cell, per-frame energies — in a form
every other atomistic code already reads.

```python
atoms  = fa.read("relaxed.extxyz")          # the last frame
first  = fa.read("md.extxyz", index=0)
frames = fa.read_frames("md.extxyz")        # all of them

fa.write("out.extxyz", atoms)
fa.write("out.extxyz", atoms, append=True)  # one more frame
fa.write("traj.extxyz", frames)             # a whole trajectory at once
```

The extension picks the format, and anything unrecognised — `dump`,
`trajectory.out` — is read as extended XYZ, which is what "the default" has to
mean to be worth saying.

### What the comment line carries

```
3
Lattice="10 0 0 0 11 0 0 0 12" Properties=species:S:1:pos:R:3:forces:R:3 pbc="T T F" energy=-1.5
O   0.000  0.000  0.000    0.10 -0.20  0.30
H   0.958  0.000  0.000   -0.05  0.10  0.00
H  -0.240  0.927  0.000   -0.05  0.10  0.00
```

`Lattice` becomes the cell (nine numbers, or three for a diagonal one), `pbc`
the boundary conditions, and `Properties` says what the columns are:
`species`, `pos`, `forces`, `velo`, `momenta`, `masses`, `charges`, `tags`, and
anything else becomes a per-atom array of its own name. Everything else on the
line — `energy`, a step number, a label — lands in `atoms.info`, a
write-through mapping shaped like ASE's:

```python
atoms.info["energy"]          # -1.5
atoms.info["step"] = 7        # sticks, and is written back out
```

A plain `.xyz` file with a free-form comment and no `Properties` reads too: the
columns are then the historical `species` and `pos`.

### Numbers come back the way they went in

Real values are written in the shortest text that reads back as *exactly* the
same double. That is not a compromise between size and accuracy, it is both:
`0.1` is shorter than twelve fixed significant digits and, unlike them, exact.
`precision=` trades it for a smaller file when a trajectory is long and the
last digits do not matter.

### Trajectories

Appending is what the format is for, and what molecular dynamics and geometry
optimization need. The C++ writer holds the file open across frames:

```cpp
fastatoms::io::ExtxyzWriter writer("md.extxyz");
for (...) {
    writer.write(atoms, info);   // buffered, one open for the whole run
}
```

Reading back does not require holding the file in memory. The reader maps it
and hands out one frame at a time, and an index over the frame offsets makes
`at(-1)` cheap on a file with a million of them:

```cpp
fastatoms::io::ExtxyzReader reader("md.extxyz");
Atoms frame;
while (reader.next(frame)) { ... }
reader.size();       // frames, from one scan of the newlines
reader.at(-1);       // the last
```

### Speed

The reader maps the file rather than copying it, scans with pointers rather
than streams, and converts with `std::from_chars`, which neither allocates nor
consults the locale. Measured on one frame of a million argon atoms with
forces, 51 MB, on an Apple M-series laptop:

| | throughput |
| --- | ---: |
| write | 173 MB/s |
| read | 326 MB/s |
| read, one `istringstream` per line | 63 MB/s |

The naive implementation is the one everybody writes first; it is 5.2× slower,
and the difference is what makes a million-atom trajectory worth reading at
all.

### YAML has not gone away

It cannot: an extended XYZ file has nowhere to say which calculator to attach
or how long to integrate for, so an input deck is still YAML and is still how a
*run* is described.

```python
deck = fa.read_input("run.yaml")     # an InputDeck: structure + simulation
atoms = fa.read("structure.extxyz")  # just a structure
```

What changed is the default for structures, which is most reading and nearly
all writing. `fa.read` now returns `Atoms` for any format; the deck reader is
`fa.read_input`.

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
  task: md                          # single_point | md | optimize | phonon | neb
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

A `constraints:` list restricting how the atoms may move is read here too; see
[Constraints](#constraints).

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

**`Constraint`** — `FixAtoms`, `FixLine` and `FixPlane`, attached to an `Atoms`
and respected by the optimizers, the dynamics and the band alike. See
[Constraints](#constraints).

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

## Calculators

A calculator evaluates energies, forces and stresses for an `Atoms` object.
Attach one and the structure can answer for itself:

```python
atoms.calc = fa.Vasp(xc="pbe", encut=400, kpts=[4, 4, 4], directory="run")

energy = atoms.get_potential_energy()   # eV
forces = atoms.get_forces()             # eV/A, shape (N, 3)
stress = atoms.get_stress()             # eV/A^3, Voigt (xx, yy, zz, yz, xz, xy)
```

That is one VASP run, not three: results are cached and only recomputed when
the structure actually changes. Forces are also written onto the structure as
the per-atom array `forces`, so they follow it through slicing and writing.

### The three DFT interfaces

| | writes | runs | reads |
| --- | --- | --- | --- |
| `fa.Vasp` | POSCAR, INCAR, KPOINTS, POTCAR | `$VASP_COMMAND`, else `vasp_std` | OUTCAR |
| `fa.Espresso` | pw.in | `$ASE_ESPRESSO_COMMAND`, else `pw.x -in pw.in > pw.out` | pw.out |
| `fa.Siesta` | *label*.fdf, pseudopotentials | `$SIESTA_COMMAND`, else `siesta < siesta.fdf > siesta.out` | *label*.out, *label*.FORCE_STRESS |

Each takes the settings its own code expects, and anything FastAtoms does not
interpret is passed straight through to the right place:

```python
# encut, ismear, sigma... become INCAR tags
fa.Vasp(xc="pbe", encut=400, ismear=0, sigma=0.05, kpts=[4, 4, 4])

# each keyword is routed to the namelist that owns it
fa.Espresso(pseudopotentials={"Si": "Si.upf"}, ecutwfc=40.0, conv_thr=1e-8)

# underscores become dots: dm_tolerance -> dm.tolerance
fa.Siesta(xc="pbe", mesh_cutoff=250.0, dm_tolerance=1e-4, kpts=[4, 4, 4])
```

`fa.LennardJones` is a built-in analytic calculator that needs no external
program — useful for checking a workflow before pointing it at a DFT code, and
what the test suite uses to verify the interface against finite differences.

### Choosing the engine from the input file

```yaml
simulation:
  calculator:
    name: vasp          # or espresso / qe / pw / siesta / lennard_jones
    parameters:
      directory: vasp-run
      xc: pbe
      kpts: [8, 8, 8]
      encut: 400
```

```python
deck = fa.read("dft.yaml")
atoms = deck.atoms
atoms.calc = deck.make_calculator()          # keywords here override the file
energy = atoms.get_potential_energy()
```

Swapping engines is one edit to that block; nothing in the driving script
changes. `fa.register_calculator(name, factory)` adds your own.

### Running elsewhere

The two halves work separately, which is what an HPC workflow usually needs:

```python
calc = fa.Vasp(xc="pbe", encut=400)
calc.write_inputs(atoms, "job-01")     # stage a directory, run nothing
...                                    # submit job-01 to a queue
calc.read_results("job-01")            # parse it afterwards
calc.results.energy, calc.results.stress
```

`command` is a shell expression, so `mpirun -np 64 vasp_std` and job-launcher
wrappers work as written. A non-zero exit, a signal, or a `timeout` produces a
`CalculatorError` quoting the command, the directory, and the tail of the log.

### Conventions

Energies are in eV, forces in eV/Å, and the stress is in eV/Å³ in Voigt order
`(xx, yy, zz, yz, xz, xy)` with the ASE sign convention `σ = (1/V) ∂E/∂ε`. Each
interface converts from its code's own units and sign — kB and `XX YY ZZ XY YZ
ZX` for VASP, Ry/bohr³ with the opposite sign for Quantum ESPRESSO, Ry/bohr³
for SIESTA's `.FORCE_STRESS`. `results.extra` carries what is left over: Fermi
level, iteration counts, convergence flags.

Because VASP requires atoms grouped by species, `fa.Vasp` reorders them in the
POSCAR and permutes the forces back, so what you read matches the order you
gave.

## Electronic structure

Band paths, band structures and densities of state.

```python
path = fa.electronic.bandpath(atoms, path="GXWKGL", points=200)
bands = fa.electronic.BandStructure(kpoints, energies, bands=8, fermi_level=5.2)
gap = bands.band_gap()
dos = fa.electronic.density_of_states_tetrahedron(bands, (12, 12, 12))
```

Arrays are zero-copy read-only views, as in the trajectory analysis module.

### Band paths

The Bravais lattice is identified from the cell itself — by handing a
one-atom structure to the symmetry module, whose point group is then the
lattice's own holohedry — so a cell in any setting gets the right answer.

```python
fa.electronic.identify_bravais(atoms)     # BravaisLattice.FCC
fa.electronic.default_path(atoms)         # "GXWKGLUWLK,UX"
fa.electronic.special_points(atoms)       # {"G": ..., "X": ..., ...}
```

A comma makes the path *jump* rather than interpolate, and contributes no
length to the plotted axis. Points are shared between segments in proportion
to their length in reciprocal space, so the sampling is even along that axis
rather than even per segment. `special_distances` and `special_labels` are the
tick marks.

**Coverage is stated, not implied.** CUB, FCC, BCC, TET, ORC, HEX and RHL (both
variants) have their special points tabulated. The other seven — BCT, ORCF,
ORCI, ORCC, MCL, MCLC, TRI — are *identified* correctly but have no table here,
and asking for their points raises an error naming the lattice rather than
returning something plausible. Use `bandpath_from_points` with corners of your
own for those.

### Density of states

Two methods, which differ in how they interpolate between the k-points that
were actually computed:

| | needs | gives |
| --- | --- | --- |
| `density_of_states_tetrahedron` | a regular grid | sharp van Hove singularities, error falling as 1/n² |
| `density_of_states` | any k-points | a curve smoothed by the smearing width |

The tetrahedron method cuts each microcell of the grid into six tetrahedra
along its **shortest body diagonal**, takes the bands to be linear inside each,
and does the resulting isosurface integral in closed form. Checked against the
free electron gas, where the density of states is `4π√E` and its integral
`(8/3)πE^{3/2}`: the error is 1.2% on a 24³ grid and falls as the grid
squared — 12% at 12³, 3.3% at 24³, 1.8% at 32³ on the integral. That is the
discretisation, not a defect: a parabola is not linear inside a tetrahedron.

Smearing offers Gaussian, Methfessel–Paxton of any order, and Fermi–Dirac.
Methfessel–Paxton converges faster in the width, at the cost of **going
slightly negative** — which is correct, and is asserted by the tests rather
than worked around. Both methods obey the sum rule: the area under the density
of states is the number of states there are.

The two are compared on the same eigenvalues at 24³ and 32³ points a side. The
tetrahedron error keeps falling; the smeared one is unchanged, because past
that point the width is what is wrong and no number of k-points fixes it.

### Band structures and gaps

```python
gap = bands.band_gap()
gap.gap           # the fundamental gap, zero for a metal
gap.direct_gap    # the smallest separation at a single k-point
gap.direct        # whether the two coincide
```

The fundamental gap is between the highest occupied state *anywhere* and the
lowest empty state *anywhere* — the two need not share a k-point, and that is
the whole distinction between an indirect gap and a direct one. Both are
reported along with which k-point and band each was found at.

`BandStructure.from_calculator` reads eigenvalues out of a calculator's extra
results. None of the calculators shipped here produce them yet, so it is the
hook for one that does rather than something with a caller today; build from
arrays in the meantime.

## Trajectory analysis

Pair correlations, structure factors and interfacial tension, over a sequence
of `Atoms`.

```python
frames = fa.read_frames("run.extxyz")

rdf = fa.analysis.radial_distribution(frames, rmax=10.0, bins=200)
oh  = fa.analysis.radial_distribution(frames, first="O", second="H")
sq  = fa.analysis.structure_factor(frames)
gamma = fa.analysis.surface_tension(frames)
```

Every array comes back as a **zero-copy read-only numpy view** onto the C++
storage, with the owning object as the array's base — `rdf.g.base is rdf`.
Nothing is copied on the way out, and nothing can outlive its storage.

### g(r)

Pairs come from the neighbour list, so the cost is linear in the number of
atoms rather than quadratic, and the minimum-image convention and periodic
boundaries are handled there. `first` and `second` select which species a pair
may be made of; leaving them out gives the total g(r).

Alongside `g` you get `coordination`, the running integral, whose plateaux are
the coordination shells — for fcc aluminium it steps 12, 18, 42 at exactly the
right radii, and for rock salt the Na–Cl curve reaches 6 while Na–Na reaches
12.

The normalisation uses `N_A(N_B − δ_AB)/V` rather than `N_A N_B/V`, which is
what makes g(r) come out at exactly one for an ideal gas of finitely many
atoms instead of one minus 1/N.

`RadialDistributionAccumulator` takes frames one at a time, so a trajectory of
ten thousand need not be held in memory:

```python
accumulator = fa.analysis.RadialDistributionAccumulator(rmax=10.0, bins=200)
for frame in fa.read_frames("run.extxyz"):
    accumulator.add(frame)
rdf = accumulator.result()
```

`rdf.reliable_rmax` is half the smallest cell width. Beyond it a shell no
longer fits inside the box and g(r) is biased by the periodic images — the
number is reported rather than the request being refused, because sometimes
you do want to look out there.

### S(q)

Three routes, because they are good at different things and disagreeing is
diagnostic:

| method | exact for | needs |
| --- | --- | --- |
| `"reciprocal"` | a periodic cell | nothing — no cutoff, no window |
| `"debye"` | a finite cluster | a cutoff, and a window |
| `structure_factor_from_rdf` | — | a g(r) you already have |

`"reciprocal"` sums `|Σ exp(iq·r)|²/N` over the wavevectors the cell actually
supports. It is the one to trust for bulk MD, and on a conventional fcc cell it
reproduces the extinction rule exactly: 4 at (111), (200) and (220), and 0 at
(100) and (110).

One thing worth knowing about the Debye route: for a **periodic** system the
truncated sum has to have its uniform background subtracted, or the result
carries the ideal-gas contribution instead of the deviation from it — which
shows up as S(q) sitting well above one at small q for a structureless liquid.
A finite cluster in vacuum has no such background and gets the Debye equation
as it is usually written. This module does the subtraction when the frame is
periodic and not otherwise.

### Surface tension

```
gamma = (L_n / 2) [ P_nn - (P_t1t1 + P_t2t2) / 2 ]
```

averaged over frames, with `normal=` choosing the axis (z by default). The half
is there because a periodic slab has **two** interfaces — it is not a
normalisation of the average.

The pressure tensor is the negative of the stress with the kinetic term
included, so a compressed solid reads positive. Results carry a
`standard_error`: surface tension converges slowly, and a value without one
next to it says very little. `per_frame` gives the individual values so a run
can be blocked or plotted, and `surface_tension_from_pressures` takes tensors
computed elsewhere.

## Crystallographic symmetry

Space groups, point groups, Wyckoff positions and cell standardization,
implemented from scratch — no spglib or any other symmetry library.

```python
group = atoms.get_space_group()
print(group.number, group.hermann_mauguin)      # 227 Fd-3m

primitive = atoms.get_primitive_cell()          # 8 atoms -> 2
conventional = primitive.get_conventional_cell()
```

### How the symbol is arrived at

The Hermann–Mauguin symbol is **derived**, not matched against a stored list of
operation sets. The crystal system fixes a set of symmetry directions; each
direction is asked which axis and which plane it carries; and the symbol is
assembled from the answers. That is how the symbol is *defined*, so a group
this code has never seen still comes out right.

What makes it work is the **intrinsic translation** — the part of an
operation's translation that no choice of origin can remove, obtained by
averaging it over the powers of its own rotation. That component is what makes
a screw a screw and a glide a glide, and it is origin-independent, so the
answer does not depend on where the structure happens to sit in its cell.

Only the last step is a lookup: turning the derived symbol into the number
between 1 and 230 that International Tables assigns it. That numbering is a
convention, and no amount of group theory recovers it.

| | point group | space group |
| --- | --- | --- |
| copper | m-3m | Fm-3m (225) |
| diamond | m-3m | **Fd-3m** (227) |
| graphene | 6/mmm | P6/mmm (191) |
| magnesium | 6/mmm | **P6₃/mmc** (194) |

Each pair shares a point group and differs only in translations — a `d` glide
in the first, a 6₃ screw in the second. Anything that only classified
rotations would call them the same.

### What comes back

```python
group.number                 # 227
group.hermann_mauguin        # "Fd-3m"
group.hermann_mauguin_full   # "F 4_1/d -3 2/m"
group.schoenflies            # "Oh-7"
group.point_group            # m-3m (Oh, cubic, order 48)
group.centring               # Centring.F
group.operations             # every (rotation, translation)
group.wyckoff                # the orbits
group.primitive              # the cell the analysis ran on
```

Wyckoff orbits carry a multiplicity, the site's own point group, and which
atoms belong to them — rock salt comes back as `4a (m-3m)` and `4b (m-3m)`,
diamond as a single `8a (-43m)`. The letters are assigned by increasing
multiplicity as the Tables do; they are not looked up from them, so for a group
with several sites of equal multiplicity the lettering may differ from the
published one while the multiplicities and site symmetries do not.

### Standardization

`primitive_cell` finds the smallest cell that tiles the crystal, from the pure
translations the symmetry search turns up, reduced to a basis by Hermite normal
form. `conventional_cell` goes the other way: it asks the point group where the
symmetry directions are and the lattice for its shortest vector along each,
which is what turns a one-atom fcc cell into the four-atom cube.
`niggli_reduce` gives the unique shortest, most orthogonal basis for a lattice,
and is what the triclinic case falls back on.

**Everything is analysed on the primitive cell**, whatever you hand in. A
supercell admits far fewer operations than the crystal has — its lattice is a
sublattice, and most of the rotations do not map it onto itself — so a 2×1×1
box of diamond would otherwise come back monoclinic, which is true of the box
and false of the crystal.

### Tolerance

`tolerance` is a distance in ångström. Too tight and real symmetry is missed;
too loose and operations appear that are not there, which surfaces as a
`SymmetryError` saying the rotations found are not one of the thirty-two
crystallographic point groups rather than as a quietly wrong answer.

## Building structures

Generators for the structures you would otherwise write a script to produce.
Everything comes back with its cell and boundary conditions already set.

```python
from fastatoms import build

si = build.diamond("Si", 5.43, cubic=True)          # 8 atoms, periodic
sheet = build.tmd("Mo", "S")                        # MoS2 monolayer, 15 A vacuum
tube = build.nanotube(10, 0, length=4)              # a zigzag carbon nanotube
particle = build.wulff("Ag", "fcc", 4.09,           # the equilibrium shape
                       {"111": 1.0, "100": 1.15}, atoms=500)
```

### Bulk lattices

| | primitive | conventional | nearest neighbour |
| --- | ---: | ---: | --- |
| `build.fcc(symbol, a)` | 1 | 4 | `a/√2` |
| `build.bcc(symbol, a)` | 1 | 2 | `a√3/2` |
| `build.hcp(symbol, a, c=…)` | 2 | — | `a` |
| `build.diamond(symbol, a)` | 2 | 8 | `a√3/4` |
| `build.rocksalt(first, second, a)` | 2 | 8 | `a/2` |
| `build.zincblende(first, second, a)` | 2 | 8 | `a√3/4` |
| `build.cscl(first, second, a)` | 2 | — | `a√3/2` |

`a` is always the conventional cubic lattice constant, never the bond length.
Pass `cubic=True` for the conventional cell — the primitive one is what you
want for a band structure, the conventional one for anything you intend to look
at. `hcp` defaults to the ideal `c = a√(8/3)`. `build.bulk(symbols, name, a)`
takes any of them by name, which is what the Python and input-file layers use.

### 2D materials

```python
build.graphene(a=2.46, vacuum=15.0, nx=4, ny=4)   # 32 atoms
build.graphene_rectangular()                       # the 4-atom orthorhombic cell
build.honeycomb(["B", "N"], 2.51)                  # hexagonal boron nitride
build.honeycomb("Si", 3.87, buckling=0.44)         # silicene, which is not flat
build.tmd("Mo", "S")                               # 1H MoS2 from measured values
build.tmd("W", "Se", phase="1T", vacuum=20.0)      # the octahedral polymorph
```

**Vacuum is the gap, not the cell height.** Ask a dichalcogenide for 15 Å and
you get 15 Å of empty space between one layer and the next image of it —
`3.13 + 15.00 = 18.13 Å` of cell for MoS₂ — with the layer centred so it never
straddles the boundary. `build.set_vacuum(atoms, gap)` re-pads an existing
sheet without touching its in-plane lattice.

Sixteen dichalcogenides have tabulated lattice constants and thicknesses
(`build.known_tmds()`); give `a=` and `thickness=` for anything else. The 1H
and 1T phases differ only in where the lower chalcogen sits, and in almost
every property that follows from it.

### Nanotubes

```python
build.nanotube(10, 0, length=4)              # carbon
build.roll(build.tmd("Mo", "S"), 10, 0)      # any hexagonal sheet
build.tube_geometry(6, 3)                    # the numbers, without the atoms
```

| (n, m) | atoms | diameter | chiral angle |
| --- | ---: | ---: | ---: |
| (5, 5) | 20 | 6.78 Å | 30° — armchair |
| (9, 0) | 36 | 7.05 Å | 0° — zigzag |
| (6, 3) | 84 | 6.22 Å | 19.1° — chiral |

A sheet with thickness rolls into a tube with two walls, because each atom
keeps its height above the mid-plane as a radial offset: a (10, 0) MoS₂ tube
has its sulphur at 3.46 Å and 6.59 Å with the molybdenum between them at
5.03 Å.

One subtlety worth knowing about: the chiral indices are defined against a
basis whose vectors meet at 60°, and the sheets here are built on the
120° description of the same lattice. `roll` converts before applying the
standard formulas, so `(5, 5)` means the 6.78 Å armchair tube everyone else
means by it. Getting this wrong does not fail loudly — it builds a rectangle
that is not a lattice rectangle, and the tube comes out with several times too
many atoms.

### Nanoparticles

```python
build.icosahedron("Ag", shells=3, d=2.89)        # 147 atoms
build.cuboctahedron("Ag", shells=3, a=4.09)      # 147 atoms
build.decahedron("Ag", p=3, q=4, r=1, a=4.09)    # Marks; r=0 gives Ino
```

Icosahedra and cuboctahedra close their shells at the same magic numbers — 13,
55, 147, 309, 561 — because they have the same shell structure and differ only
in how it is folded. The icosahedron is **strained**: ask for 2.89 Å within a
shell and the shells themselves come out 2.749 Å apart, a factor of 1.0515.
Twenty tetrahedra cannot meet at a point without it, and it is why icosahedra
win at small sizes and lose at large ones.

Decahedra are five fcc tetrahedra twinned about a common axis. Five times the
70.53° dihedral is 352.6°, not a full turn, so each wedge is stretched
azimuthally to exactly 72° — which is what real five-fold twinned particles do
too. `r` cuts the Marks re-entrance into each twin boundary.

### The Wulff construction

```python
build.wulff("Ag", "fcc", 4.09, {"111": 1.0, "100": 1.15}, atoms=500)
build.wulff("Ag", "fcc", 4.09, [[1, 1, 1, 1.0], [1, 0, 0, 1.15]], atoms=500)
```

Each facet sits at a distance proportional to its surface energy — that is
Wulff's theorem — so cheap facets grow and dear ones shrink away. Only the
ratios matter; the overall scale comes from the atom count you ask for, and
since the count jumps as whole facets of atoms appear, the closest achievable
size is used rather than the exact one. Symmetry-equivalent directions are
generated for you: `build.equivalent_directions(1, 1, 0)` gives all twelve.

### Does it produce anything usable?

`python/examples/build.py` ends by relaxing icosahedra against cuboctahedra of
the same size on a Lennard-Jones potential:

| atoms | icosahedron | cuboctahedron |
| ---: | ---: | ---: |
| 13 | −0.03499 | −0.03226 |
| 55 | −0.05173 | −0.04969 |
| 147 | −0.06027 | −0.05875 |
| 309 | −0.06528 | −0.06414 |

Energy per atom in eV. The icosahedron wins at every one of these sizes, which
is the known result for a Lennard-Jones potential: its five per cent internal
strain costs less than the extra surface a cuboctahedron of the same size
exposes. Real metals cross over to fcc shapes far earlier.

## Constraints

Constraints restrict how atoms may move, and everything that moves them obeys.

```python
atoms.constraints = [
    fa.FixAtoms(range(16)),         # the substrate: frozen entirely
    fa.FixLine([16], "z"),          # an adatom, free only along z
    fa.FixPlane([17], [1, 1, 0]),   # held to a plane, free within it
]
```

| | motion left | degrees of freedom removed |
| --- | --- | ---: |
| `fa.FixAtoms(indices)` | none | 3 per atom |
| `fa.FixLine(indices, direction)` | along `direction` | 2 per atom |
| `fa.FixPlane(indices, normal)` | perpendicular to `normal` | 1 per atom |

Indices can be a sequence, a `range`, or a boolean `mask=`; a direction can be
`"x"`, `"y"`, `"z"` or a vector, which is normalised for you. Attach with
`atoms.constraints = [...]` (a single constraint need not be wrapped in a
list), add to the list with `atoms.add_constraint(...)`, and remove everything
with `atoms.clear_constraints()`.

The line and the plane pass through wherever the atom already is, because it is
the *displacement* that is constrained, not the position: attaching
`FixPlane([3], "z")` holds atom 3 at its current height rather than teleporting
it to `z = 0`.

### Where they act

A constraint answers three questions - what may a force do, what may a
displacement do, what may a velocity do - and it has to answer all three
consistently. Removing a component from the force alone is the classic mistake:
the atom then stays put in a relaxation and drifts in dynamics, because it kept
whatever velocity it started with.

- **Forces** are projected on the way out of `get_forces()`, so every consumer
  sees them constrained. The calculator's own `results` stay raw — they are
  what the model said, and reading them back should show that.
- **Velocities** are projected on the way *in*, in `set_momenta` and
  `set_velocities`. Between the two, every thermostat, barostat, integrator and
  velocity initialiser is covered without having to know constraints exist — a
  frozen atom handed a Maxwell–Boltzmann draw does not quietly wander off.
- **Positions** are projected by `atoms.move_to(...)`, which is what the
  optimizers and integrators use. Plain `atoms.positions = ...` stays
  unconstrained, because building a structure is not a move.
- **The band** projects again after its own tangent projection, since the
  spring term and the climbing image's reversed component are under no
  obligation to respect anything.
- **Degrees of freedom** are subtracted before the temperature is computed, so
  a thermostat holds the system at the temperature you asked for — and the
  centre-of-mass term is counted only for the translations the constraints
  leave as symmetries. Freezing atoms already pins the centre of mass; charging
  three more for it makes a slab with one adsorbate look like it has no degrees
  of freedom at all. The Maxwell–Boltzmann draw is rescaled over the same
  reduced count, since `set_velocities` has already projected most of what it
  drew away.

### Counting degrees of freedom

`atoms.constrained_degrees_of_freedom()` measures what the projection actually
removes rather than summing what each constraint claims, so an atom named twice
is not charged twice:

```python
atoms.constraints = [fa.FixAtoms([0]), fa.FixPlane([0], "z")]
atoms.constrained_degrees_of_freedom()      # 3 - the plane adds nothing
atoms.constraints = [fa.FixPlane([0], "z"), fa.FixPlane([0], "x")]
atoms.constrained_degrees_of_freedom()      # 2 - two planes leave a line
```

The count is obtained by pushing the three unit directions of every atom
through the same projection the rest of the code enforces and counting how many
survive, so it cannot disagree with what is actually applied.

### From the input file

Constraints belong to the `structure:`, not to the run, which is what makes
them apply to every task: a deck that freezes a substrate freezes it whether it
goes on to relax, run dynamics or build a band.

```yaml
structure:
  cell: [[15.2, 0, 0], [0, 15.2, 0], [0, 0, 24.0]]
  pbc: xy
  atoms: [[Ar, 0.0, 0.0, 0.0], ...]
  constraints:
    - {type: fix_atoms, indices: ["0-15"]}
    - {type: fix_line,  indices: [16], direction: z}
    - {type: fix_plane, indices: [17], normal: [1, 1, 0]}
```

`indices` takes integers and `"first-last"` ranges; `mask: [true, false, ...]`
is the alternative spelling. A direction is `x`, `y`, `z` or three numbers.
`type` tolerates `fix_atoms`, `fix-atoms`, `FixAtoms` and `fix`; `atoms`,
`index` and `along` work where `indices` and `direction` do. A single
constraint need not be written as a list.

`examples/neb_argon.yaml` holds its substrate this way, and
`./build/examples/fastatoms-run` reports what it found before running:

```
constraints : 1, removing 48 degrees of freedom
              FixAtoms(16 atoms)
```

Structures written back out with `fa.to_yaml` carry their constraints, so a
round trip gives the same structure rather than a freer one.

### Not the same as SHAKE and RATTLE

The rigid-geometry constraints of [the water models](#rigid-geometry-shake-and-rattle)
are a different thing: `DistanceConstraint` is holonomic, couples atoms to each
other, and needs an iterative solver. These are linear projections, exact in one
step. The two compose — rigid water molecules on a frozen slab is both at once.

`fa.NEB(..., fix=[...])` and `neb: {fix: [...]}` also still exist. They are
shorthands that hold atoms for one band only; a `FixAtoms` on the structure is
the general mechanism and gives the same answer to the digit.

## Geometry optimization

Five algorithms, all written from scratch — no external linear algebra or
optimization library is used anywhere in the module.

```python
atoms.calc = fa.Vasp(xc="pbe", encut=400, kpts=[4, 4, 4])

result = fa.LBFGS(fmax=0.02).run(atoms)      # relaxes atoms in place
print(result.converged, result.steps, result.evaluations, result.energy)
```

| | direction | memory | best for |
| --- | --- | --- | --- |
| `fa.SteepestDescent` | `-g/α` | none | the robust baseline |
| `fa.ConjugateGradient` | `-g/α + β d_prev` | one vector | large systems, cheap |
| `fa.BFGS` | `-H g`, dense inverse Hessian | O(n²) | up to a few thousand DOF |
| `fa.LBFGS` | two-loop recursion over `m` pairs | O(m·n) | everything larger |
| `fa.Fire` | damped dynamics, adaptive timestep | one vector | forces with no energy |

`ConjugateGradient` offers Polak–Ribière⁺ (default), Fletcher–Reeves and
Hestenes–Stiefel via `variant=`. `BFGS` stores the *inverse* Hessian, so the
search direction is a matrix-vector product and no linear solve is ever needed.

`Fire` is the odd one out, and the reason it is here. It integrates a damped
equation of motion — mixing the velocity toward the force direction, growing
the timestep while going downhill and freezing on the first uphill step —
rather than searching along a line. It therefore never needs an energy that
the force is the gradient of, which is exactly the situation the nudged
elastic band is in below. It is slower than L-BFGS on an ordinary relaxation
and much harder to break.

### Convergence

A run stops when the largest force on any single atom — not the largest
Cartesian component — falls below `fmax`, or when `max_steps` is used up. The
result says which:

```python
result.converged        # bool
result.reason           # converged / max_steps / line_search_failed / no_progress / interrupted
result.steps            # accepted steps
result.evaluations      # force calls — the number that matters for DFT
result.history          # every step: energy, fmax, dE, displacement
```

`max_step` (default 0.2 Å) is a trust radius bounding how far any atom may move
in one step. `attach(callback)` gets every step, for logging or writing a
trajectory; returning `False` stops the run.

### Relaxing the cell

```python
fa.optimize(atoms, "lbfgs", fmax=1e-3, relax_cell=True)
fa.optimize(atoms, "lbfgs", relax_cell=True, pressure=2.0 * fa.units.GPa)
fa.optimize(atoms, "lbfgs", relax_cell=True, mask=[1, 1, 0, 0, 0, 0])  # a and b only
```

The cell is parameterised by a deformation `S` relative to the starting cell,
with atoms carried along: `cell = cell₀ S` and `rᵢ = yᵢ S`. The gradients follow
from the definition of the stress, `∂E/∂ε = V σ`:

```
∂E/∂S = V S⁻ᵀ σ            ∂E/∂yᵢ = -Fᵢ Sᵀ
```

Both are checked against finite differences in the test suite. With a `pressure`
the enthalpy `E + pV` is minimised instead; relaxing under 2 GPa leaves a
residual pressure of 2 GPa to four decimals, which is the cross-check that the
stress and the cell gradient agree.

### Line searches

Step lengths come from a backtracking (Armijo) search or a strong Wolfe search
— bracketing plus `zoom`, following Nocedal & Wright — with safeguarded
quadratic interpolation. Wolfe is the default for all four algorithms.

Skipping it (`line_search="none"`) saves force calls only while the curvature
model is good. On a stretched or concave starting geometry every quasi-Newton
curvature update is correctly discarded, the method degenerates to a fixed
`-g/α` step, and it stalls. Measured on a strained argon crystal:

| starting point | `line_search="none"` | `"wolfe"` |
| --- | ---: | ---: |
| near equilibrium | 9 force calls | 13 |
| strongly disordered | 25 | 26 |
| stretched, cell relaxed | 201, **not converged** | 26, converged |

### From the input file

```yaml
simulation:
  task: optimize
  calculator:
    name: lennard_jones
    parameters: {epsilon: 0.0103, sigma: 3.40, cutoff: 8.5}
  optimizer:
    algorithm: lbfgs        # sd | cg | bfgs | lbfgs
    fmax: 1.0e-4
    max_steps: 300
    relax_cell: true
```

```python
deck = fa.read("relax.yaml")
atoms = deck.atoms
atoms.calc = deck.make_calculator()
deck.make_optimizer().run(atoms)          # keywords here override the file
```

`./build/examples/fastatoms-run examples/relax_argon.yaml` does the same from
C++. `fa.register_optimizer(name, factory)` adds your own.

### Using the algorithms on their own

The optimizers minimise any differentiable function, which is how they are
tested independently of any physics:

```python
def rosenbrock(x):
    ...
    return energy, gradient

result, x = fa.BFGS(fmax=1e-6, alpha=1.0).minimize(rosenbrock, x0)
```

## Molecular dynamics

```python
atoms.calc = fa.LennardJones()

result = fa.run_md(atoms, 2000, timestep=5.0, temperature=300.0,
                   thermostat="nose_hoover", trajectory="run.xyz")
print(result.completed, result.evaluations,
      result.energy_drift_per_atom(len(atoms)))
```

One step is composed from three pluggable pieces — an integrator, an optional
thermostat, an optional barostat — applied in a symmetric Trotter splitting:

```
thermostat(dt/2)  barostat kick(dt/2), deform(dt)  integrate(dt)  barostat kick(dt/2)  thermostat(dt/2)
```

The thermostat and the barostat kicks only read the stress and rescale
velocities, so they land where the forces are already cached: **an NVE or NVT
step costs exactly one force evaluation, NPT costs two.**

### Integrators

| | order | symplectic | notes |
| --- | --- | --- | --- |
| `fa.VelocityVerlet` | 2 | yes | the default; positions and velocities at the same instant |
| `fa.Leapfrog` | 2 | yes | velocities half a step off; identical positions to Verlet |
| `fa.Euler` | 1 | **no** | drifts without bound — for teaching, not for running |

Leapfrog and velocity Verlet are algebraically the same scheme, and the test
suite checks that they produce matching positions. Measured on argon, 400 NVE
steps at 2 fs:

| scheme | drift, eV/atom |
| --- | ---: |
| velocity Verlet | 2 × 10⁻⁷ |
| leapfrog | 4 × 10⁻⁶ |
| Euler | 7 × 10⁻⁴ |

### Thermostats (NVT)

| | samples canonical | notes |
| --- | --- | --- |
| `fa.NoseHoover` | yes | the workhorse; `chain_length` > 1 gives Nosé–Hoover chains |
| `fa.Langevin` | yes | stochastic and ergodic by construction; set `friction` in 1/fs |
| `fa.Nose` | yes | Nosé's original, with `s` and `g = N_f + 1` |
| `fa.BerendsenThermostat` | **no** | fast equilibration; suppresses fluctuations |

`Langevin` adds friction and a matching random force,
`dv/dt = F/m − γv + √(2γk_BT/m)·η`. Because every degree of freedom is
thermostatted independently it is ergodic where a single Nosé–Hoover variable is
not — a lone oscillator samples correctly. The Ornstein–Uhlenbeck half of the
step is solved *exactly* rather than discretised, so all the timestep error
comes from the Verlet part. It conserves nothing of its own, so instead it
books the heat it exchanges with the bath: `potential + kinetic + heat` stays
flat and can be checked for drift like the deterministic thermostats.

Nosé's method and Nosé–Hoover describe the same extended system — Hoover's
change of variables turns Nosé's virtual-time equations into the real-time form
used here. They differ in the degree-of-freedom count: Nosé's `g = N_f + 1`
carries the extra one from the time scaling in his derivation, which is right
for sampling in virtual time and slightly off in real time. That is exactly why
Hoover's `g = N_f` version superseded it. `fa.Nose` keeps `s` and reports Nosé's
Hamiltonian; prefer `fa.NoseHoover` for production, with a chain of three or
four for small or stiff systems where a single variable is not ergodic.

Starting cold at 10 K with a target of 80 K, 1500 steps at 5 fs:

| thermostat | ⟨T⟩ | σ(T) | drift, eV/atom |
| --- | ---: | ---: | ---: |
| Berendsen | 79.98 | 6.2 | 1.8 × 10⁻² |
| Langevin | 81.50 | 11.5 | −7 × 10⁻⁶ |
| Nosé | 80.49 | 11.9 | −3 × 10⁻⁶ |
| Nosé–Hoover | 79.69 | 11.0 | −4 × 10⁻⁶ |

All four hold the mean. The canonical width for this system is
σ(T) = T√(2/N_f) = 11.7 K: Langevin and the Nosé family reproduce it, and
Berendsen deliberately damps it, which is what makes it an equilibration tool
rather than a sampling one. Only the ones that sample correctly keep a
conserved quantity flat.

### Barostats (NPT)

| | samples isobaric | notes |
| --- | --- | --- |
| `fa.ParrinelloRahman` | yes | the cell carries its own velocity; `isotropic=False` frees the shape |
| `fa.BerendsenBarostat` | **no** | weak coupling; volume fluctuations come out too small |

Parrinello–Rahman is implemented in the strain-rate form: the cell velocity `ε̇`
is driven by the imbalance between the internal and applied stress, and atoms
and their velocities transform affinely with the box, so the volume is a genuine
dynamical variable. Always pair a barostat with a thermostat — the cell degree
of freedom heats up otherwise.

```python
fa.run_md(atoms, 2000, temperature=300.0, thermostat="nose_hoover",
          barostat="parrinello_rahman", pressure=1.0)   # GPa
```

### What comes back

`conserved_energy` is the quantity to watch: potential + kinetic plus whatever
the thermostat and barostat are holding. For NVE it is just the total energy;
under Nosé–Hoover it is the extended Hamiltonian. Its drift is the honest
measure of whether the timestep was small enough.

```python
result.history[-1].temperature, .pressure, .volume, .conserved_energy
result.energy_drift_per_atom(len(atoms))
md.attach(lambda atoms, step: print(step.step, step.temperature), interval=10)
```

A run stops early if a coordinate or momentum stops being finite. That catches
arithmetic that has overflowed — it is not a guarantee the physics is sound,
because with a truncated pair potential a blown-up system simply flies apart at
large but finite speed. Watch the drift for that.

### Trajectories

`trajectory="run.xyz"` writes extended XYZ — positions, velocities and forces
per atom, plus the cell and the per-frame scalars in the comment line, which
ASE, OVITO and VMD all read:

```
4
Lattice="5.3 0 0 0 5.3 0 0 0 5.3" Properties=species:S:1:pos:R:3:velo:R:3:forces:R:3
  pbc="T T T" step=20 time=100 energy=-0.291 temperature=24.7 pressure=3.3e-04 ...
Ar  0.0132 -0.0071 0.0043   0.00042 -0.00021 0.00013   -0.0091 0.0048 -0.0029
```

`log="run.log"` writes a table of the per-step scalars; `log=True` sends it to
standard output.

### From the input file

```yaml
simulation:
  task: md
  calculator: {name: lennard_jones, parameters: {epsilon: 0.0103, sigma: 3.40}}
  md:
    ensemble: nvt            # nve | nvt | npt
    integrator: velocity_verlet
    thermostat: nose_hoover  # nose_hoover | nose | langevin | berendsen
    timestep: 5.0            # fs
    steps: 2000
    temperature: 60.0        # K
    tau: 100.0               # fs  (Langevin: friction, in 1/fs)
    chain_length: 3
    initialize_velocities: true
    seed: 20260830
  output:
    trajectory: argon-md.xyz
    log: argon-md.log
    interval: 20
```

```python
deck = fa.read("md.yaml")
atoms = deck.atoms
atoms.calc = deck.make_calculator()
deck.make_dynamics().run(atoms)          # keywords here override the file
```

`./build/examples/fastatoms-run examples/md_argon.yaml` does the same from C++.
Pressures are in GPa everywhere in the `md:` section; the optimizer's `pressure`
is in eV/Å³, or use `pressure_GPa` there too.

## Phonons

Harmonic phonons by finite displacements, self-contained: the supercells, the
space group, the force constants, the dynamical matrix and its eigensolver are
all in this repository. There is no dependency on Phonopy or on any external
linear-algebra library.

```python
atoms = fa.Atoms("Ar", [[0, 0, 0]], cell=[[0, 2.63, 2.63], [2.63, 0, 2.63],
                                          [2.63, 2.63, 0]], pbc=True)
atoms.calc = fa.LennardJones(epsilon=0.0103, sigma=3.40, cutoff=8.0)

ph = fa.Phonons(atoms, supercell=[3, 3, 3], displacement=0.01)
ph.run()                                  # 2 force calls, not 6

ph.frequencies([0.5, 0.0, 0.0])           # THz, ascending, negative if unstable
ph.frequencies([0, 0, 0])                 # [0, 0, 0]: the three acoustic modes
```

The structure has to be at a minimum — the expansion is taken about the
positions you give — and `ph.residual_force` is the check that it is.

### Supercells

`make_supercell` takes an integer transformation matrix, or a diagonal
shorthand, and keeps the map back to the unit cell:

```python
fa.make_supercell(atoms, 2)                                   # 2x2x2
fa.make_supercell(atoms, [2, 2, 3])
fa.make_supercell(atoms, [[-1, 1, 1], [1, -1, 1], [1, 1, -1]])  # fcc -> cubic
```

The result carries `unit_cell_index` (which unit-cell atom each copy is) and
`lattice_points` (the integer translation of each), which is what the Fourier
sum in the dynamical matrix needs. The first `unit_cell_size` atoms are the
original cell at the origin.

### Symmetry

`find_symmetry` returns the space group of a structure: the rotations in the
lattice basis (integer matrices), their Cartesian forms, the fractional
translations, and how each operation permutes the atoms.

```python
symmetry = fa.find_symmetry(atoms)        # tolerance=1e-5 by default
len(symmetry)                             # 48 operations
symmetry.point_group                      # the rotations, without translations
symmetry.independent_atoms                # one representative per orbit
symmetry.equivalent_atoms                 # orbit label for every atom
fa.site_symmetry(symmetry, 0)             # operations that fix atom 0
```

Two atoms in the same orbit need only one of them displaced; a direction
related to another by a site-symmetry operation need not be displaced at all.
Both reductions are applied:

| structure | supercell | displacements | without symmetry |
| --- | ---: | ---: | ---: |
| argon, fcc | 2×2×2 | 2 | 6 |
| argon, fcc | 3×3×3 | 2 | 6 |
| silicon, diamond | 2×2×2 | 2 | 12 |
| rocksalt NaCl | 2×2×2 | 4 | 12 |

`ph.calculation_count` and `ph.calculation_count_without_symmetry` report the
two numbers; `symmetry=False` turns the reduction off, which is what the test
suite compares against. The missing rows of the force-constant matrix are
rebuilt from the representatives by `Φ(a) = RᵀΦ(rep)R`, so the answer is the
same either way — to 10⁻⁸ eV/Å², asserted.

### Force constants

Central differences by default (`central=False` halves the cost at the price
of an O(d) error instead of O(d²)), solved as a least-squares problem over the
symmetry-expanded displacement set, then projected onto the acoustic sum rule:

```python
ph.force_constants.array                  # (3 * n_unit, 3 * n_super) eV/A^2
ph.force_constants.acoustic_sum_violation()
ph.residual_force                         # forces on the undisplaced supercell
```

The sum rule is what makes the acoustic branches reach exactly zero at Γ;
`acoustic_sum_rule=False` leaves the raw numbers alone.

### Anywhere in the Brillouin zone

The dynamical matrix is a Fourier sum over the periodic images of each
supercell atom, weighted by multiplicity when an atom is equidistant from
several — which is what keeps the dispersion smooth on a small supercell.

```python
ph.dynamical_matrix(q)                    # (3n, 3n) complex, Hermitian
ph.modes(q).frequencies                   # THz
ph.modes(q).eigenvectors                  # (3n, 3n), polarisations in columns
ph.band_structure([[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]])
ph.mesh_frequencies(8, 8, 8)              # flattened, for a density of states
```

q is given in fractional reciprocal coordinates of the unit cell, so
`[0.5, 0, 0]` is the zone boundary along the first reciprocal vector.
Frequencies are in THz; `fa.Phonons.THZ_TO_WAVENUMBER` and
`fa.Phonons.THZ_TO_MEV` convert. Unstable modes come back as negative
frequencies rather than as imaginary numbers, which is the usual convention.

### Running the displacements yourself

`run()` needs a calculator attached. When the forces come from a queue
instead, ask for the structures and hand the results back in the same order:

```python
ph = fa.Phonons(atoms, supercell=[2, 2, 2])
cells = ph.displaced_supercells()         # one Atoms per displacement
forces = [run_dft_somehow(cell) for cell in cells]
ph.set_forces(forces)                     # then frequencies() as usual
```

### From the input file

```yaml
simulation:
  task: phonon
  calculator: {name: lennard_jones, parameters: {epsilon: 0.0103, sigma: 3.40}}
  phonons:
    supercell: [3, 3, 3]     # or all nine entries of a transformation matrix
    displacement: 0.01       # A
    symmetry: true
    central: true
    acoustic_sum_rule: true
    symmetry_tolerance: 1.0e-5
```

```python
deck = fa.read("phonons.yaml")
atoms = deck.atoms
atoms.calc = deck.make_calculator()
ph = deck.make_phonons(atoms)            # keywords here override the file
ph.run()
```

`./build/examples/fastatoms-run examples/phonons_argon.yaml` does the same from
C++.

## Classical force fields and rigid water

SPC and SPC/E are calculators like any other, and the constraints that keep
their molecules rigid are picked up by the dynamics without being restated:

```python
atoms = fa.water_box("spce", 108)          # 324 atoms at 0.997 g/cm^3
atoms.calc = fa.Forcefield(model="spce")

fa.run_md(atoms, 1000, timestep=2.0, temperature=300.0, thermostat="nose_hoover")
```

The 2 fs step is the point. It is only legitimate because the O-H stretch — the
fastest motion in the system, with a period near 9 fs — has been frozen out
rather than integrated.

### The models

| | q(O) | q(H) | σ [Å] | ε [kcal/mol] | r(OH) [Å] | ∠HOH | µ [D] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SPC | −0.82 | +0.41 | 3.166 | 0.1554 | 1.0 | 109.47° | 2.27 |
| SPC/E | −0.8476 | +0.4238 | 3.166 | 0.1553 | 1.0 | 109.47° | 2.35 |
| TIP3P | −0.834 | +0.417 | 3.15061 | 0.1521 | 0.9572 | 104.52° | 2.35 |

The dipole moments are not inputs — they follow from the charges and the
geometry, and the test suite checks them against the published values, which is
a check on the parameters that reading them back would not be.

SPC/E carries one thing SPC does not: a polarisation self-energy of
5.22 kcal/mol per molecule, the work of polarising a gas-phase molecule up to
the enhanced dipole the model uses. It shifts the energy and never a force;
`self_energy=False` removes it.

```python
fa.water_model("spce").dipole_moment       # 2.351 debye
fa.water_model("spce").self_energy         # 0.2264 eV per molecule
```

### Electrostatics: Ewald, not a cutoff

A lattice sum of `1/r` is only conditionally convergent — its value depends on
the order of summation — so truncating it at a cutoff does not approximate it,
it computes something else. Ewald splits the sum with an identity,

```
1/r = erfc(αr)/r  +  erf(αr)/r
```

leaving a short-ranged part for the neighbour list and a smooth part that
converges quickly in reciprocal space, plus a self term and a correction that
removes the intramolecular pairs the reciprocal sum inevitably included.
Conducting ("tinfoil") boundary conditions are used, so there is no surface
term.

```python
atoms.calc.breakdown.coulomb
# EwaldEnergy(real=4.14, reciprocal=15.62, self=-462.92, excluded=444.38,
#             total=1.22 eV)
```

Those four enormous, nearly cancelling numbers are what an Ewald sum looks
like, which is why they are reported separately: their relative sizes are how
you tell whether α is sensible.

Everything not given is derived from one number. `accuracy` (1e-6 by default)
fixes α from the real-space cutoff, and the reciprocal cutoff from α; the
reciprocal vectors are cached and only rebuilt when the cell moves, and the
structure factors are accumulated by recursion so the trigonometry is O(N)
rather than O(N·K).

The splitting is fixed once and then kept, deliberately: it is a property of
the model, not of the cell it is used in. Letting the cutoff follow the cell
would make the energy depend on the volume through the truncation as well as
through the physics, and the stress — a derivative with respect to exactly that
volume — would stop being the derivative of the energy.

### Rigid geometry: SHAKE and RATTLE

Three distance constraints per molecule — the two O-H bonds and the H-H
distance that fixes the angle — make a three-site model rigid with no angular
machinery at all.

SHAKE solves for the constraint forces one at a time, sweeping until they hold
together, with the forces acting along the *old* bond directions, which is what
keeps the scheme time-reversible. RATTLE then projects the velocities so that
each constrained distance also has zero rate of change; without that second
stage the distances stay right while their rates drift, and the reported
temperature drifts with them.

```python
dynamics = fa.make_dynamics(timestep=2.0, temperature=300.0)
dynamics.prepare(atoms)
dynamics.constraints.removed_degrees_of_freedom     # 324, three per molecule
dynamics.constraints.max_deviation(atoms)           # ~1e-11, relative
```

Each constraint removes one degree of freedom, and `Dynamics` subtracts them
before computing a temperature — a rigid water box has six per molecule, not
nine, and using nine reports a temperature a third too low. All three
integrators enforce them, and `constraints=False` turns them off.

That last option is worth a warning: a rigid model has no bond stretch to fall
back on, so running one unconstrained does not give flexible water, it gives no
molecules at all.

### What is checked

The Ewald sum is pinned three ways that between them leave nowhere for an error
to hide.

Against a number that is known exactly — the Madelung constant of rock salt:

```
 alpha [1/A]   cutoff [A]   k-vectors     energy [eV]     Madelung
        0.80        6.488         895   -35.694057588  1.747564595
        1.00        5.190        1787   -35.694057588  1.747564595
        1.50        3.460        5918   -35.694057588  1.747564595
        2.00        2.595       14128   -35.694057588  1.747564595
```

which is also the second check: the split is an identity, so moving α — here
over a factor of sixteen in the number of reciprocal vectors — cannot change
the total. And the third is finite differences: the forces and all six stress
components are checked against numerical derivatives of the energy, which is
the only practical way to be sure of the reciprocal-space stress term.

Beyond that, a single molecule alone in a box must have essentially no
electrostatic energy — all its pairs are excluded — and what is left falls off
as 1/L³ like the dipole-image term it is; and a dimer in a large box converges
to the direct pairwise sum computed without any periodicity at all.

For the constraints, the claim tested is not that the drift is small — that
depends on the box and the cutoff — but that it is *second order in the
timestep*, since a constrained integrator with its impulse bookkeeping wrong
loses energy at first order:

| timestep | drift [eV/atom] | bond error |
| ---: | ---: | ---: |
| 2.0 fs | 3.0 × 10⁻⁴ | 5 × 10⁻¹¹ |
| 1.0 fs | 8.2 × 10⁻⁵ | 5 × 10⁻¹¹ |
| 0.5 fs | 2.0 × 10⁻⁵ | 5 × 10⁻¹¹ |

### From the input file

```yaml
simulation:
  task: md
  calculator:
    name: forcefield       # or spc / spce / tip3p, which preset the model
    parameters:
      model: spce          # spc | spce | tip3p
      accuracy: 1.0e-6     # everything else is derived from this
      # cutoff: 9.0        # real-space, A; default is half the shortest width
      # alpha: 0.35        # splitting parameter, 1/A
      shift: true          # shift the Lennard-Jones to zero at the cutoff
      tail: true           # isotropic correction beyond it
      self_energy: true    # SPC/E's polarisation constant
  md:
    ensemble: nvt
    thermostat: nose_hoover
    timestep: 2.0
    steps: 500
    temperature: 300.0
    # constraints: true    # taken from the calculator unless turned off
```

`./build/examples/fastatoms-run examples/water_spce.yaml` runs that: 500 steps
of rigid SPC/E at 2 fs, with the conserved energy flat to 3 × 10⁻⁹ eV/atom.

A structure with no cell is not a special case: Ewald exists because a *lattice*
sum is conditionally convergent, and a cluster has no such problem, so it gets
the plain sum instead. The two agree to 4 × 10⁻⁴ eV for a water dimer in a 60 Å
box, the residue being the dimer's interaction with its own images.

## Machine-learning potentials

MACE, MatterSim, CHGNet, NequIP and Allegro are calculators like any other:
assign one and the optimizers, the dynamics and the phonon module carry on
without knowing the forces came from a neural network.

```python
atoms.calc = fa.MACE(model="mace-off23-small.pt")   # in process, via LibTorch
atoms.calc = fa.MatterSim()                          # worker, in mlip_env

fa.optimize(atoms, "lbfgs", fmax=0.02)
fa.run_md(atoms, 1000, temperature=300.0, thermostat="langevin")
```

### Two backends, and why both exist

A trained model is a pickled Python object. Loading one means importing its
package, which means an interpreter. FastAtoms takes that interpreter out of
the loop where it can, and keeps it alive where it cannot.

| | in process | worker |
| --- | --- | --- |
| what it runs | a TorchScript archive, through LibTorch | the package's own ASE calculator |
| Python in the loop | none at all | one process, started once |
| works for | MACE, `nequip-compile`d models | everything, MatterSim included |
| needs | `-DFASTATOMS_WITH_TORCH=ON` | the model's conda environment |

The split is not a preference, it is what the models allow. MACE scripts
cleanly — `torch.jit.script` compiles it, and the result is a 7 MB archive C++
loads directly. MatterSim does not: its `M3Gnet` builds the convolution stack
with `functools.partial`, and scripting it fails with *"Can't get source for
`functools.partial`"*. So MatterSim, CHGNet and friends are driven by a
long-lived worker instead.

That worker is emphatically not the `FileIOCalculator` pattern the DFT
interfaces use. There is no input file, no output file and no process start per
step: one interpreter is forked, it imports and loads the model once, and every
evaluation afterwards is a binary request and reply over a pipe.

Measured here on 72 atoms of water with MACE-OFF23-small, on CPU:

| | first call | per evaluation |
| --- | ---: | ---: |
| in process (LibTorch) | 0.87 s | 163 ms |
| worker (persistent) | 5.73 s | 214 ms |

The persistent worker is what removes most of the overhead — a fresh
interpreter per step would cost seconds, not milliseconds. Running in process
removes what is left: the model loads 6.6x faster and each step is 1.3x
quicker, because the dominant cost, the network itself, is the same either way.

### Exporting a model

Run this once, inside the environment that has the package:

```sh
~/miniconda3/envs/mace_env/bin/python -m fastatoms.mlip.export     ~/.cache/mace/MACE-OFF23_small.model -o mace-off23-small.pt --check
```

The archive follows one small ABI, so the C++ side is model-agnostic:

```
forward(Dict[str, Tensor]) -> Dict[str, Tensor]

  in   positions (n,3), atomic_numbers (n,), cell (3,3),
       edge_index (2,m), unit_shifts (m,3)
  out  energy (), and optionally forces (n,3) and stress (3,3)
```

plus a JSON description carried as a TorchScript extra file: the cutoff, the
species the model was trained on, and whether it returns forces and stress. A
new family means one adapter in `python/fastatoms/mlip/export.py`; no C++
changes.

### Structures in, tensors out

The graph is built by the same O(N) neighbour list everything else uses, so it
costs no more than one force evaluation would:

```python
graph = fa.mlip_graph(atoms, cutoff=5.0)
graph.edge_index      # (2, nedges): receivers on row 0, senders on row 1
graph.unit_shifts     # (nedges, 3), integer cell offsets
```

with the convention MACE, MatterSim, CHGNet and NequIP share:

```
r_ij = positions[sender] + unit_shifts @ cell - positions[receiver]
```

Each model's own layout — MACE's one-hot `node_attrs` and batch pointers,
MatterSim's `pbc_offsets` and three-body indices — is derived from those five
tensors by the adapter, so nothing model-specific reaches C++.

### Forces and stress by automatic differentiation

A model that returns its own forces is believed. Anything it does not return is
differentiated out of its energy, in C++, through LibTorch's autograd:

```
strain      = zeros(3, 3, requires_grad=True)
deformation = I + (strain + strainᵀ)/2
positions  ← positions · deformation      cell ← cell · deformation
E           = model(...)
forces      = -dE/d(positions)
stress      =  dE/d(strain) / V
```

Symmetrising the strain makes the stress symmetric and a rigid rotation free.
`model.provides_forces` and `model.provides_stress` say which path was taken.

That path is checked against a model that knows the answer. Exporting MACE
twice — once returning its own forces and stress, once with `--energy-only` so
FastAtoms has to differentiate — gives **identical forces and stresses agreeing
to 9 × 10⁻¹⁹ eV/Å³**, which pins down the sign convention, the symmetrisation
and the 1/V together.

One caveat the error message will tell you about: a model that differentiates
its own energy inside `forward` frees the graph on the way out, so it cannot
then be differentiated again. Such a model must declare `provides_forces` and
`provides_stress`, which is what the MACE exporter does by default.

### Bringing your own

Any object with an `evaluate` method is a potential — useful for a model
FastAtoms has no C++ path for, and for testing a workflow with no model at all:

```python
class Harmonic:
    needs_edges = False
    def evaluate(self, graph, need_forces, need_stress):
        delta = graph.positions - self.reference
        return {"energy": 0.5 * self.k * (delta ** 2).sum(),
                "forces": -self.k * delta if need_forces else None}

calc = fa.MLIP(cutoff=5.0)
calc.set_backend(Harmonic(...))
```

### Environments

MACE defaults to `mace_env` and every other family to `mlip_env`, following
where the packages are installed. Override per calculator with
`conda_env=` or `python=`, or globally with `$FASTATOMS_MACE_ENV` and
`$FASTATOMS_MLIP_ENV`.

### From the input file

```yaml
simulation:
  task: optimize
  calculator:
    name: mace             # mace | mattersim | chgnet | nequip | allegro | mlip
    parameters:
      model: mace-off23-small.pt
      backend: auto        # auto | torchscript | worker
      device: cpu          # cpu | cuda | mps
      dtype: float64
      # conda_env: mace_env
      # cutoff: 5.0        # overrides what the model reports
      # skin: 0.0          # added to it
  optimizer: {algorithm: bfgs, fmax: 0.02}
```

`./build/examples/fastatoms-run examples/mlip_water.yaml` relaxes that structure
with MACE from C++, with no Python process anywhere.

### Building with LibTorch

LibTorch ships inside the `torch` wheel, so point CMake at an interpreter that
has it:

```sh
cmake -S . -B build -DFASTATOMS_WITH_TORCH=ON \
      -DFASTATOMS_TORCH_PYTHON=$HOME/miniconda3/envs/mace_env/bin/python
cmake --build build -j
```

Two things to know. FastAtoms stops linking its own OpenMP in this
configuration, because LibTorch already loads one and a second copy aborts the
process at startup (`OMP: Error #15`); only the neighbour-list threading is
lost, which is a rounding error beside a network evaluation. And the extension
then carries an rpath into that environment, so deleting it breaks
`import fastatoms`. Without `FASTATOMS_WITH_TORCH` everything still works
through the worker.

## Reaction paths: NEB, CI-NEB and AutoNEB

A band of images is strung between two relaxed endpoints and relaxed as a
single object. What comes back is the minimum energy path and the barrier
along it.

```python
initial, final = hollow_site(), neighbouring_hollow_site()

band = fa.NEB(initial, final, images=9, k=0.5, climb=True, fix=range(16))
band.run("fire", fmax=1e-4)

profile = band.profile()
print(profile.forward_barrier, profile.reverse_barrier, profile.highest)
```

`fa.NEB` takes two endpoints and interpolates, or a list of images you have
prepared yourself. `images` counts the endpoints, which never move.

### What the projection does

Relaxing a band naively does two wrong things: the images slide downhill into
the endpoints, and the springs holding them apart pull the path away from the
valley floor by cutting corners. So each force is split along the local
tangent τ̂ and only the useful half of each is kept:

```
F = F_true − (F_true·τ̂)τ̂  +  k(|R₊ − R| − |R − R₋|) τ̂
    ⌊________________⌋       ⌊_________________________⌋
    true force, perpendicular  spring force, parallel only
```

The tangent is the improved (upwind) estimate of Henkelman and Jónsson: it
points at whichever neighbour is higher in energy, and interpolates between
them only at an extremum. `tangent="bisect"` and `tangent="upwind"` select the
older estimates, which kink on steep barriers — the default is the one you
want.

### The climbing image

A plain band reports the highest image it happens to have, and an evenly
spaced band puts no image exactly on the saddle, so the barrier comes out
low. `climb=True` picks the highest image, drops its springs, and *inverts*
its force along the tangent, so it walks uphill along the path while relaxing
in every other direction. Its converged position is the saddle point itself.
From `python/examples/neb.py`, an argon adatom hopping between hollow sites:

| band | plain NEB | CI-NEB | short by |
| ---: | ---: | ---: | ---: |
| 4 images | 11.97 meV | 13.77 meV | 1.80 meV |
| 6 images | 13.14 meV | 13.77 meV | 0.63 meV |
| 8 images | 13.45 meV | 13.77 meV | 0.32 meV |
| 10 images | 13.58 meV | 13.77 meV | 0.19 meV |

The plain band creeps up on the answer; the climbing image has it at four
images, and costs nothing extra — the same images, one of them projected
differently. `climb_threshold=` delays the climb until the band as a whole
has settled below a force, which is worth doing when the interpolation starts
somewhere silly. The gate is tested against the same number `fmax` reports, so
a run converging tighter than the threshold always starts climbing first.

### AutoNEB

`fa.AutoNEB` starts from a coarse band, relaxes it loosely, and then spends
images where the path is actually interesting — inserting into the widest
energy gap, dropping images that have become redundant, and redistributing
the rest — before a final tight relaxation with the climbing image on.

```python
result = fa.AutoNEB(initial, final, images=4, max_images=10,
                    fmax=1e-4, k=0.5, fix=range(16)).run("fire")
print(result.profile.forward_barrier, result.stages, result.evaluations)
```

On the hop above this reaches the same 13.7743 meV as a fixed 10-image
climbing band, to four decimal places, having started from four images.

### Two things that will bite you

**Hold the substrate.** The projected band force is not required to sum to
zero — the component removed from one atom is not the component removed from
another. A spectator atom therefore keeps a force the tangent knows nothing
about, and the whole band drifts bodily along the reaction coordinate with
nothing to restore it. `fix=` takes the indices held still in every image, and
on a surface calculation it is not optional.

**Barriers want a tighter `fmax` than geometries do.** A saddle is usually
flat along its soft modes, so a force tolerance converts into a much larger
energy tolerance than intuition suggests. On the hop above, 5 meV/Å leaves the
barrier 0.13 meV short and 1 meV/Å leaves 0.01 meV; it takes 0.1 meV/Å to pin
the last digit.

### Building the starting band

The endpoints are yours to relax; the images in between come from
`fastatoms-interpolate`, which writes the multi-frame extended XYZ file the
input deck reads:

```sh
./build/examples/fastatoms-interpolate initial.extxyz final.extxyz \
    -n 9 --idpp -o band.extxyz
```

It reports the closest approach in every image, which is the one number that
says whether the band is usable at all:

```
image   closest pair [A]
    0             3.8000   (endpoint)
    1             2.5333
    2             1.2667
    3             0.0000
```

A straight line between endpoints is fine when little moves and the atoms stay
out of each other's way. When they do not — an exchange, a diffusion hop, a
molecule turning over — it walks them straight through one another, and there
is no recovering from that: the first force call lands at 5748 eV and the band
never converges. `--idpp` interpolates the *pair distances* instead of the
coordinates and then finds the coordinates that reproduce them (Smidstrup et
al., *J. Chem. Phys.* **140**, 214106), which routes the atoms around each
other. On the exchange above:

| | closest approach | highest image | NEB result |
| --- | ---: | ---: | --- |
| straight line | 0.000 Å | 5748 eV | never converges |
| `--idpp` | 3.725 Å | −0.028 eV | converges in 117 steps |

Available from Python as `fa.idpp_interpolate(initial, final, images)` and
`fa.closest_approach(atoms)`.

### From the input file

```yaml
structure:                 # the initial state; `neb.initial` defaults to it
  cell: [[15.2, 0, 0], [0, 15.2, 0], [0, 0, 24.0]]
  pbc: xy
  atoms: [[Ar, 1.90, 1.90, 2.70], ...]

simulation:
  task: neb
  calculator:
    name: lennard_jones
    parameters: {epsilon: 0.0103, sigma: 3.40, cutoff: 9.0}

  neb:
    final: product.extxyz    # a file, or a structure written inline
    # band: band.extxyz      # or a prepared band, which overrides both ends
    images: 9                # counting the endpoints
    spring_constant: 0.5
    climb: true
    fix: ["0-15"]            # ranges, not sixteen separate entries
    # tangent: improved      # improved | bisect | upwind
    # variable_springs: false
    # climb_threshold: 0.0
    # auto: true             # AutoNEB instead of a fixed band
    # max_images: 11
    # coarse_fmax: 0.2
    # insert: energy         # energy | distance
    # prune: false
    # redistribute: false

  optimizer: {algorithm: fire, fmax: 0.0001, max_steps: 2000}
  output: {trajectory: band.extxyz}    # the converged band
```

`./build/examples/fastatoms-run examples/neb_argon.yaml` runs the adatom hop
above and prints the profile:

```
image        s [A]      E-E0 [eV]  fmax [eV/A]
   0       0.0000       0.000000        fixed
   ...
   4       1.9853       0.013774    3.207e-05   <- highest

forward barrier : 0.013774 eV
```

`fmax` and `max_steps` come from the `optimizer:` section, including for
AutoNEB, since that is what they control. The algorithm defaults to FIRE here
rather than the deck-wide BFGS, for the reason below. `output.trajectory`
receives the converged band, which is also how a run is restarted: point
`neb.band` at what the last one wrote.

### Driving it

`band.run("fire")` is shorthand; any optimizer object works, and `fa.NEB` is
itself usable as a plain objective through `positions()`, `set_positions()`
and `forces()` if you would rather drive the loop yourself.

Line searches are rejected, deliberately and loudly. The projected band force
is not the gradient of any scalar function — the spring term is added by hand
and the climbing image's parallel component is reversed — so there is nothing
for a line search to minimise along its direction. Pass one and you get an
error rather than a wrong path:

```python
band.run(fa.LBFGS(line_search="strong_wolfe"))   # NEBError
band.run(fa.LBFGS(line_search="none"))           # fine
band.run("fire")                                 # fine, and the default choice
```

FIRE needs no line search by construction, which is why it is the one to reach
for here.

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

# Calculators: generated VASP/QE/SIESTA inputs, then an equation of state
PYTHONPATH=build/python python python/examples/dft_calculator.py examples/dft_silicon.yaml

# Geometry optimization: the algorithms compared, then a cell relaxation
PYTHONPATH=build/python python python/examples/relaxation.py examples/relax_argon.yaml

# Molecular dynamics: integrators, thermostats and barostats compared
PYTHONPATH=build/python python python/examples/molecular_dynamics.py examples/md_argon.yaml

# Phonons: what symmetry saves, a dispersion, a density of states
PYTHONPATH=build/python python python/examples/phonons.py examples/phonons_argon.yaml

# Machine-learning potentials: the graph, both backends, a relaxation
PYTHONPATH=build/python python python/examples/mlip.py --model mace-off23-small.pt

# Structure builders: lattices, sheets, tubes, particles, and which shape wins
PYTHONPATH=build/python python python/examples/build.py

# Constraints: three questions about one adatom, and the bookkeeping they need
PYTHONPATH=build/python python python/examples/constraints.py

# Reaction paths: plain NEB against CI-NEB, then AutoNEB, on an adatom hop
PYTHONPATH=build/python python python/examples/neb.py

# The same hop from an input file, with the converged band written out
./build/examples/fastatoms-run examples/neb_argon.yaml

# Build a starting band for one, avoiding a straight line through the atoms
./build/examples/fastatoms-interpolate initial.extxyz final.extxyz -n 9 --idpp

# Rigid water: SPC/SPC/E, the Ewald split, and what the constraints buy
PYTHONPATH=build/python python python/examples/water.py examples/water_spce.yaml
```

The MD example conserves energy to under 1 µeV/atom over 200 steps, which is
the end-to-end check that positions, shifts, masses and units all agree.

### Against ASE

`benchmarks/ase_comparison.py` runs the same system through both libraries and
reports what it costs. Both get the same atoms in the same places with the same
velocities and the same Lennard-Jones parameters, and the script will not print
a timing until it has shown the two agree:

```sh
pip install ase
PYTHONPATH=build/python python benchmarks/ase_comparison.py
```

256 atoms of fcc argon, Apple M-series, single-threaded, ASE 3.29:

| case | FastAtoms | ASE | speed-up | work [MB] |
| --- | ---: | ---: | ---: | --- |
| force evaluation | 0.69 ms | 4.00 ms | 5.8× | 0.5 vs 3.8 |
| L-BFGS, 50 steps | 0.036 s | 0.227 s | 6.2× | 1.1 vs 18.9 |
| velocity Verlet, 1000 steps | 0.737 s | 4.710 s | 6.4× | 0.1 vs 19.9 |
| relax to fmax = 10⁻⁶ | 0.023 s | 0.714 s | 31.7× | 0.2 vs 19.5 |

The first three are like-for-like: same algorithm, same number of force
evaluations, so the ratio is implementation speed. The last is not, and it is
the more useful number — FastAtoms reaches the same minimum in 23 steps against
ASE's 162, so the cost of *solving the problem* falls by more than the cost of
a step does. ASE's L-BFGS does better with its line search enabled (59 steps,
0.478 s), and the benchmark reports that row too rather than comparing only
against the slower default.

`work` is peak resident memory above each library's own import baseline. ASE's
per-step allocation of numpy arrays is the whole of the difference; the
absolute footprints are the other way round, because this build links LibTorch
for the machine-learning potentials, which costs about 80 MB at import and
nothing thereafter. The script prints all of that rather than the flattering
half.

### What the benchmark checks before it believes itself

Two libraries computing different things cannot be compared, so the script
validates first and refuses to time anything that fails:

| | agreement |
| --- | ---: |
| energy, single point | 3 × 10⁻¹⁵ relative |
| forces | 1 × 10⁻¹⁵ eV/Å |
| stress | 5 × 10⁻¹⁹ eV/Å³ |
| velocity Verlet, 1 step | 7 × 10⁻¹⁸ Å |
| velocity Verlet, 100 steps | 1 × 10⁻¹⁴ Å |
| relaxed minimum | 8 × 10⁻¹² relative |

The trajectory row is worth reading carefully. Molecular dynamics is chaotic:
two implementations that agree to machine precision still separate
exponentially, so the benchmark asserts agreement on the first steps, where
rounding is all that separates them, and reports the growth rather than
pretending a 1000-step trajectory should match. The optimizers are held to a
different standard again, because they are not the same algorithm — they part
company as soon as there is curvature history — so what is required of them is
that they find the same minimum.

## Releasing

The version lives in two places that nothing reads together, so they drift:
`CMakeLists.txt` is what the binary reports (`PROJECT_VERSION` becomes
`FASTATOMS_VERSION`, which `fastatoms.__version__` returns), and
`pyproject.toml` is what a wheel is stamped with. The `versioning_fastatoms`
command writes both at once. It is a standalone script kept on `PATH` rather
than in the tree; run it from the top of the source tree, or pass `--root`.

```sh
versioning_fastatoms                 # interactive spinboxes
versioning_fastatoms --show          # what each file says, and why
versioning_fastatoms --auto -n       # the bump the commits ask for
versioning_fastatoms --minor         # or say it outright
versioning_fastatoms --check         # exit 1 on drift; for CI
```

`--auto` reads the commit log since the last tag and follows Conventional
Commits: `feat:` asks for a minor bump, `fix:` and `perf:` for a patch, and a
`!` after the type or a `BREAKING CHANGE:` trailer for a major. While the major
version is 0 a breaking change bumps the minor instead, which is what semver
says 0.y.z means — the tool prints that reason rather than looking like it
refused.

A pre-release is written differently in each file, which is the part a simpler
bumper gets wrong:

```
versioning_fastatoms --set 1.0.0-rc.1 -n
  CMakeLists.txt   0.1.2  -> 1.0.0        project(VERSION) takes only numbers
  pyproject.toml   0.1.1  -> 1.0.0rc1     PEP 440 spells it this way
```

Every file is written as a unit: if one write fails, the ones already touched
are restored, because half a bump is the drift the tool exists to prevent. It
writes files and nothing else — no commit, no tag, no push.

## Packaging for conda

`conda/` holds a recipe that builds the C++ core, the Python extension, the
headers and the CMake package files in one pass, and installs the lot into the
environment. Nobody has to run CMake:

```sh
conda install -c conda-forge --override-channels conda-build
conda build conda/ -c conda-forge --override-channels
conda install -c local fastatoms
```

`--override-channels` keeps the resolve on conda-forge, hosted on
anaconda.org, instead of also consulting the commercial `repo.anaconda.com`
channels that a default `.condarc` usually lists. To publish:

```sh
anaconda upload $(conda build conda/ --output -c conda-forge --override-channels)
```

The package carries both halves, so after installing it

```python
import fastatoms as fa          # the extension, from site-packages
```

and

```cmake
find_package(FastAtoms REQUIRED) # headers and lib, from $CONDA_PREFIX
target_link_libraries(my_code PRIVATE FastAtoms::core)
```

both work against the environment. `fastatoms-run` lands on `PATH`.

### What the recipe has to get right

Three things, each of which was found by building the recipe rather than by
reading it:

**No network.** `find_package(yaml-cpp)` falls back to downloading a copy when
it finds nothing, which on a sandboxed builder fails and on an unsandboxed one
silently vendors a dependency the package then carries invisibly.
`FETCHCONTENT_FULLY_DISCONNECTED=ON` turns that into an error at configure
time, so the yaml-cpp in `host:` is the one that gets used or the build stops.

**The OpenMP runtime has to come from the prefix.** On macOS,
`find_package(OpenMP)` reaches `/opt/homebrew` before the conda prefix, and the
result builds clean, passes its tests, and is still broken: the package links a
library conda does not ship, so it fails on machines without Homebrew and
aborts with `OMP: Error #15` on machines with it. The recipe names the library
outright on macOS. Linux needs none of this — conda's gcc brings its own
libgomp and there is no second one to find.

**Installed binaries need an rpath.** `fastatoms-run` links
`@rpath/libyaml-cpp.dylib` and was installed with no `LC_RPATH` at all, so it
could not start. `CMAKE_INSTALL_RPATH_USE_LINK_PATH=ON` fixes it. conda-build's
relocation pass would probably have papered over this, but a prefix that is
correct before post-processing is one less thing depending on post-processing.

### Two deliberate choices

The recipe produces **one package** carrying both the library and the bindings.
conda-forge convention for a C++ library with Python bindings is to split them
— a `libfastatoms` output with the headers, library and CMake files, and a
`fastatoms` output with the extension — so that a C++ consumer need not pull in
Python. Splitting is a `outputs:` block over the same `build.sh`; it is worth
doing before submitting a feedstock, and is unnecessary noise for an internal
channel.

**LibTorch is off.** Adding pytorch to `host:` would multiply the build for a
feature that also works out of process: machine-learning potentials still run
through the persistent Python worker. A build that wants in-process inference
sets `FASTATOMS_WITH_TORCH=ON` and adds `pytorch` to `host:`.

## Using the C++ library directly

```cmake
find_package(FastAtoms REQUIRED)
target_link_libraries(my_code PRIVATE FastAtoms::core)
```

```cpp
#include "fastatoms/fastatoms.hpp"

auto deck = fastatoms::io::read_yaml_file("input.yaml");
const auto pairs = fastatoms::build_neighbor_list(deck.atoms, 5.0);

deck.atoms.set_calculator(fastatoms::make_calculator(deck.simulation.calculator));
const double energy = deck.atoms.get_potential_energy();

auto optimizer = fastatoms::optimize::make_optimizer(deck.simulation.optimizer);
const auto relaxed = optimizer->run(deck.atoms);

auto dynamics = fastatoms::md::make_dynamics(deck.simulation);
const auto trajectory = dynamics->run(deck.atoms);

// A rigid SPC/E water box: Ewald electrostatics and three constraints per
// molecule, which the dynamics picks up from the calculator.
fastatoms::Atoms water = fastatoms::ff::water_box(fastatoms::ff::water_model("spce"), 108);
water.set_calculator(fastatoms::make_calculator("forcefield", spce_parameters));

auto phonons = fastatoms::phonon::make_phonons(deck.atoms, deck.simulation);
phonons->run();
const auto frequencies = phonons->frequencies({0.5, 0.0, 0.0});   // THz

// A neural potential, evaluated in this process through LibTorch.
deck.atoms.set_calculator(fastatoms::make_calculator("mace", parameters));
const double mlip_energy = deck.atoms.get_potential_energy();
```

## Testing

`ctest` runs twenty-one C++ suites and the Python suite. The neighbour list is
checked against a brute-force O(N² × images) reference for cubic, triclinic,
slab, isolated and smaller-than-cutoff cells; the minimum-image convention is
checked the same way.

The calculator layer is checked in two parts. Forces and the stress sign and
Voigt ordering are verified against finite differences of the analytic
calculator. The DFT interfaces are checked against representative excerpts of
each code's real output, and the full write/execute/parse pipeline runs against
a stand-in engine — a shell script that drops a canned output file in place —
so everything but the physics of the DFT codes themselves is exercised. **The
parsers have not been run against live VASP, Quantum ESPRESSO or SIESTA
installations**; treat the first real run as the check that your version's
output format matches.

The optimizers are tested as pure numerics first — analytic objectives, the
exact secant identity `H y = s` for both BFGS and the L-BFGS two-loop
recursion, the Armijo and strong Wolfe conditions, and finite termination on a
quadratic (conjugate gradient reaches the minimum of a six-variable quadratic
in six steps) — and then on real structures, where the Lennard-Jones dimer has
an exact answer to check against.

The MD module is checked the same way: a free particle travels in a straight
line under every scheme, velocity Verlet tracks a harmonic oscillator against
its analytic solution, leapfrog and velocity Verlet agree to 10⁻¹⁰ on the
positions, Euler's drift is more than fifty times Verlet's, momentum is
conserved to 10⁻⁹, and the extended-system energies of Nosé and Nosé–Hoover —
and Langevin's heat accounting — stay flat while the physical energy moves. The
canonical thermostats are checked against the *width* of the temperature
distribution, σ(T) = T√(2/N_f), not just its mean, which is what separates
genuine sampling from holding an average. The force-evaluation counts are
asserted exactly, so a regression that quietly doubles the cost of a step fails
the suite.

The phonon module is checked against systems whose answers are known in
closed form: an Einstein solid, where every branch is √(k/m) at every q, and a
chain of central springs, whose longitudinal branch along [100] is
ω(q)² = (2k/m)(1 − cos 2πq). The transverse residual that finite displacements
leave behind on the spring lattice is not merely bounded but predicted —
ω_L(q)·d/(a√2), matched to 10⁻⁴ — so the test fails if the error changes
character rather than only if it grows. The symmetry reduction is checked the
way it matters: the same structure run with `symmetry=True` and
`symmetry=False` must give the same force constants, to 10⁻⁸. The eigensolver
is verified on 2×2 and 9×9 Hermitian matrices by reconstructing UᴴAU and
checking it is diagonal, unitary and ordered; the space-group finder against
the known orders of the cubic, tetragonal, orthorhombic, hexagonal and
triclinic point groups, and by applying every operation it returns and
confirming the structure maps onto itself.

The nudged elastic band is checked against a surface whose saddle point is
known in closed form: the barrier of a tilted double well, solved by Newton
iteration inside the test itself. The climbing image finds it to 2 × 10⁻¹⁶ eV,
where the same band without climbing is short by 2.5 × 10⁻² — the test asserts
both numbers, so it fails if climbing stops working *or* if the plain band
silently starts sitting on the saddle by accident.

The projection is then checked property by property, which is what catches a
tangent that is subtly wrong rather than absent. On an evenly spaced band the
surviving true force must be exactly perpendicular to the local tangent at
every image; with climbing on, exactly one image — the highest — may keep a
parallel component, and the test counts them. The springs must push along the
path and nowhere else on a deliberately bunched band, the tangent must be a
unit vector pointing towards the higher neighbour, and a symmetric barrier must
relax to a symmetric profile.

The electronic structure module is checked against the free electron gas,
whose density of states is `4π√E` in closed form and whose integral is
`(8/3)πE^{3/2}`. Both are matched, and the tetrahedron error is required to
fall when the grid is refined while the smeared one is required not to — which
is the statement that the two methods are limited by different things. Every
smearing function has to integrate to one, since a delta replacement without
unit area miscounts the states everywhere and silently. A flat band, where
every tetrahedron has four equal corner energies and every denominator in the
closed forms vanishes, has to give a finite non-negative answer. Band paths
have to pass exactly through the points they name, with Γ→X of an fcc lattice
landing at 2π/a.

The analysis module is checked against answers that are known before the code
runs. The coordination shells of fcc aluminium have to come out at `a/√2`, `a`,
`a√1.5` and `a√2` holding 12, 18 and 42 atoms cumulatively; rock salt has to
give six chlorines around a sodium and twelve sodiums, with the cross
correlation starting at `a/2` and the like-species one at `a/√2`. An ideal gas
has to give g(r) = 1 and S(q) = 1, which is what says the normalisations are
right. The reciprocal-lattice structure factor has to reproduce the fcc
extinction rule — 4 at (111), (200) and (220), 0 at (100) and (110) — which no
part of the code is told about. The tension formula is checked against tensors
whose answer can be worked out by hand, an isotropic pressure has to give
exactly zero whatever its magnitude, and an unstrained cubic crystal has to
give zero all the way through a real calculator.

That ideal-gas test earned its place: the Debye sum was returning S(q)
noticeably above one at small q, because a truncated sum over a periodic system
carries the uniform background that the `1` is supposed to be measured against.

The symmetry module is checked against structures whose space groups are in
every textbook: simple cubic, fcc, bcc, diamond, rock salt, zinc blende,
caesium chloride, hcp, graphene and both dichalcogenide phases — twelve
answers, across six crystal systems and four centrings, none of which depends
on this code. The pairs that share a point group and differ only in
translations are checked as pairs, because those are what a symbol derived from
rotations alone would get wrong. The 230-entry table is checked to be 230
distinct symbols that round-trip to their numbers, with the crystal-system
boundaries falling where the Tables put them, and the thirty-two point groups
to have thirty-two distinct operation-count signatures — which is the property
the identification rests on.

Three bugs surfaced there and are worth recording. Fractional coordinates are
row vectors in this codebase, so an operation's invariant axis is a *left*
eigenvector; taking columns of the projector instead gave the right answer for
every cubic case, where the matrices are symmetric, and put the six-fold axis
in the wrong slot for every hexagonal one. The axis finder in the standardizer
looped `k < order`, which runs zero times for a rotoinversion's negative order
and returned whichever coordinate direction came first. And the symbol slot
names the *generator* of the axial subgroup, so a body diagonal carrying both a
three-fold and a three-fold rotoinversion is written `-3`, not `3` — the
difference between `Pm3m` and `Pm-3m`.

The builders are checked against things that do not depend on the code that
produced them. Coordination numbers are counted, not asserted from the
construction: six for simple cubic, twelve for fcc and hcp, eight for bcc, four
for diamond, three for every carbon in a graphene sheet and in a rolled tube.
Nearest-neighbour distances are checked against `a/√2` and the rest, and the
primitive and conventional cells are required to describe the same crystal by
volume. Nanotube atom counts and diameters are checked against the published
values for four different (n, m). Clusters have to hit the magic numbers 13,
55, 147, 309 and 561 — nothing in the construction guarantees those, they come
out or they do not — and a decahedron has to map onto itself under a fifth of a
turn, atom by atom. The Wulff construction is checked by the property that
defines it: raising the cost of {111} relative to {100} must move atoms out of
the corners of the cube and into its faces.

Two of those tests earned their place immediately. The nanotube count guard
caught a (10, 0) tube coming out with 140 atoms instead of 40, from applying
the 60° chiral-index formulas to a 120° lattice basis; and the five-fold
symmetry test caught a decahedron whose twin boundaries were being
deduplicated asymmetrically, which is why the wedge is now replicated after
deduplication rather than before.

The constraints are checked twice over: the projections themselves, and - the
part that actually matters - that every consumer honours them. The constrained
force is required to equal the true gradient of the energy over the moves the
constraints allow, checked component by component against finite differences of
the energy, stepping and letting the constraint have its say before measuring
the distance actually travelled. Without that property an optimizer is
minimising one function while measuring another. Every optimizer must then
converge a doubly constrained relaxation and leave the frozen atoms bit-for-bit
where they started; the dynamics must do the same over three hundred steps and
report a temperature computed with the reduced count; the band must reach the
same barrier through a `FixAtoms` as through its own `fix=`.

That optimizer test earned its place. The optimizers read the calculator's
`Results` rather than calling `get_forces`, so constraining only the latter left
them minimising the unconstrained gradient: the geometry came out exactly right
and the reported force never fell, so a converged run reported failure while
sitting on the answer. The degrees-of-freedom count is checked on overlapping
constraints, where summing what each one claims gives four removed out of three.

The interpolator is checked on the case it exists for. That a straight line
puts two exchanging atoms exactly on top of each other is asserted, not assumed
— if it stopped being true the IDPP tests would stop testing anything — and
IDPP then has to separate them to better than 3 Å while leaving both endpoints
bit-for-bit unchanged. Its objective is checked against central differences of
its own energy, and two runs of it must agree to 10⁻¹⁵, since the nudge that
breaks the symmetric saddle is deterministic rather than random. The `neb:`
section of a deck is parsed, built into a band, and relaxed to the same barrier
the hand-built band finds; the decks that must be rejected all carry a valid
structure, so they fail for the reason under test rather than for a missing
section.

The Python suite runs a periodic channel where the straight-line interpolation
happens to be the exact path, so every band is started from a displaced one
instead: the images are pushed off the axis and off their spacing, and have to
relax back to the same barrier. Without that the tests would pass on a broken
projection. AutoNEB must agree with a fixed band of the size it grows to, and
the three optimizers that can drive a band — FIRE, L-BFGS with the line search
off, and steepest descent — must all reach the same saddle. A line-searching
optimizer must be refused rather than quietly produce a wrong path.

The machine-learning layer is tested without needing a model at all: a stub
backend computes Lennard-Jones from exactly the graph a real one would receive,
which makes the conventions checkable — the shift convention is verified edge by
edge against `r_ij = r[sender] + shift·cell - r[receiver]`, and the forces the
calculator reports are checked against finite differences of the energy it
reports. That one model call must serve energy, forces and stress together is
asserted by counting evaluations, because the alternative silently doubles the
cost of every MD step.

The tests that need a real model skip themselves unless pointed at one with
`FASTATOMS_TEST_MLIP_MODEL`, `FASTATOMS_TEST_MLIP_PYTHON` and
`FASTATOMS_TEST_MLIP_ENERGY_ONLY_MODEL`. With those set,
the sharpest check in the suite runs: the same MACE model, exported to
TorchScript and evaluated in C++ on a FastAtoms graph, against the `mace`
package's own ASE calculator building its own graph in Python. They agree to
**0 eV on the energy and 7 × 10⁻¹⁸ eV/Å on the forces**, which is what makes the
tensor conversion, the edge convention and the units trustworthy rather than
merely plausible.

```sh
ctest --test-dir build --output-on-failure
PYTHONPATH=build/python pytest tests/python

# including the model-backed tests
export FASTATOMS_TEST_MLIP_MODEL=mace-off23-small.pt
export FASTATOMS_TEST_MLIP_ENERGY_ONLY_MODEL=mace-energy-only.pt
export FASTATOMS_TEST_MLIP_PYTHON=$HOME/miniconda3/envs/mace_env/bin/python
export FASTATOMS_TEST_MLIP_WORKER_MODEL=$HOME/.cache/mace/MACE-OFF23_small.model
ctest --test-dir build --output-on-failure
```

## Interoperating with ASE

```python
import fastatoms as fa
ase_atoms = fa.to_ase(atoms)      # needs `pip install ase`
atoms = fa.from_ase(ase_atoms)
```

## Contributing

Issues and pull requests are welcome. Two things make a change easy to accept:

- **Tests that would fail without it.** Every module here is checked against
  something that does not depend on the code producing it — a textbook space
  group, a coordination number counted rather than assumed, an analytic saddle
  point solved inside the test. See [Testing](#testing) for what the suite
  already covers.
- **The house style.** C++17, four-space indent, 100-column lines, clean under
  `-Wall -Wextra -Wpedantic -Wshadow`. Units follow ASE: ångström, eV, amu,
  with stress in eV/Å³ in Voigt order.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Licence

MIT. See [`LICENSE`](LICENSE).
