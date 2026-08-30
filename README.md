# FastAtoms

A high-performance C++ core for atomistic simulations, driven from Python.

FastAtoms gives you an ASE-shaped API — `Atoms`, `Cell`, chemical symbols,
periodic boundary conditions — on top of contiguous C++ storage, an O(N)
neighbour list, a YAML front end for structures and run settings, calculators
that drive VASP, Quantum ESPRESSO and SIESTA, machine-learning potentials that
run through LibTorch with no interpreter in the loop, a classical force field
with Ewald electrostatics and rigid-water constraints, geometry optimizers that
relax positions and cells, molecular dynamics in NVE, NVT and NPT, and harmonic
phonons by finite displacements with a symmetry-reduced displacement set. The
Python layer is a thin binding:
`atoms.positions` is a zero-copy numpy view straight into the C++ buffer, so a
Python script can drive a simulation without paying for the convenience.

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
  calculators/         calculator, file_io_calculator, vasp, espresso, siesta,
                       mlip (base), mlip_torch, mlip_worker, mlip_models
  optimize/            optimizer, sd, cg, bfgs, lbfgs, line_search, variables
  forcefield/          topology, water (SPC, SPC/E, TIP3P), ewald, forcefield
  md/                  dynamics, integrator, thermostat, barostat, trajectory,
                       constraints (SHAKE/RATTLE)
  phonon/              phonons, supercell, symmetry, force_constants, eigen
  util/                process, text
src/                   implementation
python/
  bindings/            pybind11 extension, one file per bound area
  fastatoms/           the importable package (__init__.py)
    mlip/              the model exporter and the potential worker
  examples/            Python driver scripts
tests/cpp, tests/python
benchmarks/            head-to-head timing against ASE
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

## Geometry optimization

Four algorithms, all written from scratch — no external linear algebra or
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

`ConjugateGradient` offers Polak–Ribière⁺ (default), Fletcher–Reeves and
Hestenes–Stiefel via `variant=`. `BFGS` stores the *inverse* Hessian, so the
search direction is a matrix-vector product and no linear solve is ever needed.

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

# Geometry optimization: the four algorithms compared, then a cell relaxation
PYTHONPATH=build/python python python/examples/relaxation.py examples/relax_argon.yaml

# Molecular dynamics: integrators, thermostats and barostats compared
PYTHONPATH=build/python python python/examples/molecular_dynamics.py examples/md_argon.yaml

# Phonons: what symmetry saves, a dispersion, a density of states
PYTHONPATH=build/python python python/examples/phonons.py examples/phonons_argon.yaml

# Machine-learning potentials: the graph, both backends, a relaxation
PYTHONPATH=build/python python python/examples/mlip.py --model mace-off23-small.pt

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

`ctest` runs fourteen C++ suites and the Python suite. The neighbour list is
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

## Licence

MIT. See `LICENSE`.
