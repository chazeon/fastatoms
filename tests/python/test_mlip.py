# SPDX-License-Identifier: MIT
"""Tests for the machine-learning potential calculators.

Most of these need no model and no torch: a backend written in Python stands in
for the network, which makes the plumbing - the graph, the units, the caching,
the error paths - testable anywhere.

The tests that do need a model skip themselves unless pointed at one:

    FASTATOMS_TEST_MLIP_MODEL    an exported TorchScript archive
    FASTATOMS_TEST_MLIP_PYTHON   an interpreter with a potential installed
    FASTATOMS_TEST_MLIP_FAMILY   which family that interpreter should load
    FASTATOMS_TEST_MLIP_WORKER_MODEL   the checkpoint it should load
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import fastatoms as fa

ARCHIVE = os.environ.get("FASTATOMS_TEST_MLIP_MODEL", "")
INTERPRETER = os.environ.get("FASTATOMS_TEST_MLIP_PYTHON", "")
FAMILY = os.environ.get("FASTATOMS_TEST_MLIP_FAMILY", "mace")
CHECKPOINT = os.environ.get("FASTATOMS_TEST_MLIP_WORKER_MODEL", "")
ENERGY_ONLY = os.environ.get("FASTATOMS_TEST_MLIP_ENERGY_ONLY_MODEL", "")

needs_archive = pytest.mark.skipif(
    not ARCHIVE or not fa.torch_available(),
    reason="set FASTATOMS_TEST_MLIP_MODEL and build with -DFASTATOMS_WITH_TORCH=ON")
needs_worker = pytest.mark.skipif(
    not INTERPRETER, reason="set FASTATOMS_TEST_MLIP_PYTHON to an interpreter with a model")


def argon(a: float = 5.30, repeat: int = 1) -> fa.Atoms:
    basis = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    atoms = fa.Atoms("Ar4", scaled_positions=basis, cell=a, pbc=True)
    return atoms.repeat(repeat, repeat, repeat) if repeat > 1 else atoms


def water() -> fa.Atoms:
    return fa.Atoms("OH2", [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
                    cell=12.0, pbc=True)


class HarmonicBackend:
    """A stand-in for a network: every atom on a spring to where it started.

    Returns only an energy, so FastAtoms has to differentiate it - which is
    exactly the path a model that reports no forces of its own would take, and
    the answer is known analytically.
    """

    def __init__(self, reference: np.ndarray, k: float = 1.5, cutoff: float = 4.0):
        self.reference = np.array(reference, dtype=float)
        self.k = k
        self.needs_edges = True
        self.info = None
        self.calls = 0
        self._cutoff = cutoff

    def evaluate(self, graph, need_forces, need_stress):
        self.calls += 1
        delta = graph.positions - self.reference
        result = {"energy": 0.5 * self.k * float((delta * delta).sum())}
        if need_forces:
            result["forces"] = -self.k * delta
        return result


# --------------------------------------------------------------------------
# The graph
# --------------------------------------------------------------------------
def test_graph_shapes_and_convention():
    atoms = argon(5.30, 1)
    graph = fa.mlip_graph(atoms, cutoff=6.0)

    assert len(graph) == len(atoms) == 4
    assert graph.natoms == 4
    assert graph.positions.shape == (4, 3)
    assert graph.atomic_numbers.tolist() == [18, 18, 18, 18]
    assert graph.edge_index.shape == (2, graph.nedges)
    assert graph.unit_shifts.shape == (graph.nedges, 3)
    assert graph.cutoff == 6.0
    assert graph.pbc.tolist() == [True, True, True]
    assert "MLIPGraph" in repr(graph)

    # r_ij = positions[sender] + unit_shifts @ cell - positions[receiver],
    # and every edge is inside the cutoff.
    receivers, senders = graph.edge_index
    vectors = (graph.positions[senders] + graph.unit_shifts @ graph.cell
               - graph.positions[receivers])
    distances = np.linalg.norm(vectors, axis=1)
    assert distances.min() > 1e-8
    assert distances.max() <= 6.0 + 1e-9

    # Integer shifts, and a full list (both directions present).
    assert np.allclose(graph.unit_shifts, np.round(graph.unit_shifts))
    assert np.count_nonzero(receivers == 0) == np.count_nonzero(senders == 0)


def test_graph_matches_the_neighbour_list():
    atoms = argon(5.30, 2)
    graph = fa.mlip_graph(atoms, cutoff=5.0)
    pairs = fa.neighbor_list(atoms, cutoff=5.0, full=True)
    assert graph.nedges == len(pairs)


def test_graph_without_edges():
    atoms = argon(5.30, 2)
    graph = fa.mlip_graph(atoms, cutoff=0.0, edges=False)
    assert graph.natoms == len(atoms)
    assert graph.nedges == 0


def test_graph_errors():
    with pytest.raises(Exception):
        fa.mlip_graph(fa.Atoms(), 5.0)
    with pytest.raises(Exception):
        fa.mlip_graph(argon(), 0.0)


# --------------------------------------------------------------------------
# The calculator, against a Python backend
# --------------------------------------------------------------------------
def test_python_backend_drives_the_calculator():
    atoms = argon(5.30, 1)
    reference = np.array(atoms.positions, copy=True)
    backend = HarmonicBackend(reference)

    calc = fa.MLIP(cutoff=4.0)
    calc.set_backend(backend)
    atoms.calc = calc

    # At the reference geometry the springs are relaxed.
    assert atoms.get_potential_energy() == pytest.approx(0.0)
    assert np.abs(atoms.get_forces()).max() == pytest.approx(0.0)

    atoms.positions[0] += [0.1, -0.2, 0.05]
    displacement = np.array([0.1, -0.2, 0.05])
    assert atoms.get_potential_energy() == pytest.approx(0.5 * 1.5 * (displacement ** 2).sum())
    forces = atoms.get_forces()
    assert forces[0] == pytest.approx(-1.5 * displacement)
    assert np.abs(forces[1:]).max() == pytest.approx(0.0)


def test_one_model_call_serves_energy_and_forces():
    atoms = argon(5.30, 1)
    backend = HarmonicBackend(np.array(atoms.positions, copy=True))
    calc = fa.MLIP(cutoff=4.0)
    calc.set_backend(backend)
    atoms.calc = calc

    atoms.positions[0] += [0.05, 0.0, 0.0]
    atoms.get_potential_energy()
    atoms.get_forces()
    assert backend.calls == 1        # an MD step must not evaluate twice

    atoms.positions[0] += [0.05, 0.0, 0.0]
    atoms.get_potential_energy()
    assert backend.calls == 2


def test_backend_can_be_removed():
    calc = fa.MLIP(cutoff=4.0)
    calc.set_backend(HarmonicBackend(np.zeros((4, 3))))
    assert calc.loaded
    calc.set_backend(None)
    assert not calc.loaded


def test_a_model_is_required():
    atoms = argon()
    atoms.calc = fa.MLIP()
    with pytest.raises(fa.CalculatorError):
        atoms.get_potential_energy()


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
def test_families_are_registered_and_named():
    assert fa.MACE().family == "mace"
    assert fa.MatterSim().family == "mattersim"
    assert fa.CHGNet().family == "chgnet"
    assert fa.NequIP().family == "nequip"
    assert fa.Allegro().family == "allegro"

    for name in ("mace", "mattersim", "chgnet", "nequip", "allegro", "mlip"):
        assert name in fa.registered_calculators()
    assert fa.make_calculator("MatterSim").name == "mattersim"
    assert fa.make_calculator("mace_mp").name == "mace"


def test_default_environments():
    # MACE has its own environment; everything else shares one.
    assert fa.mlip_environment_for("mace") == "mace_env"
    for family in ("mattersim", "chgnet", "nequip", "allegro"):
        assert fa.mlip_environment_for(family) == "mlip_env"


def test_worker_script_is_installed_next_to_the_package():
    path = fa.mlip_worker_script()
    assert path and os.path.isfile(path)
    assert path.endswith("worker.py")


def test_torch_availability_is_self_consistent():
    if fa.torch_available():
        assert fa.torch_version()
    else:
        assert fa.torch_version() == ""
        atoms = argon()
        atoms.calc = fa.MACE(model="model.pt", backend="torchscript")
        with pytest.raises(fa.CalculatorError):
            atoms.get_potential_energy()


def test_a_missing_environment_is_reported_clearly():
    atoms = argon()
    atoms.calc = fa.MACE(model="whatever", backend="worker",
                         conda_env="no-such-environment-42")
    with pytest.raises(fa.CalculatorError, match="conda environment"):
        atoms.get_potential_energy()


def test_unknown_backend_is_rejected():
    atoms = argon()
    atoms.calc = fa.MACE(model="model.pt", backend="magic")
    with pytest.raises(fa.CalculatorError):
        atoms.get_potential_energy()


def test_yaml_selects_a_model():
    deck = fa.read_input_string("""
structure:
  cell: {a: 5.3, b: 5.3, c: 5.3}
  pbc: true
  coordinates: fractional
  atoms:
    - {symbol: Ar, position: [0.0, 0.0, 0.0]}

simulation:
  task: single_point
  calculator:
    name: mattersim
    parameters:
      model: MatterSim-v1.0.0-1M.pth
      device: cpu
""")
    assert deck.simulation.calculator.name == "mattersim"
    calc = deck.make_calculator()
    assert calc.name == "mattersim"
    assert calc.family == "mattersim"


# --------------------------------------------------------------------------
# Real models
# --------------------------------------------------------------------------
@needs_archive
def test_archive_runs_in_process():
    atoms = water()
    atoms.calc = fa.MACE(model=ARCHIVE, backend="torchscript")

    info = atoms.calc.model
    assert info.backend == "torchscript"
    assert info.cutoff > 0.0
    assert 1 in info.species and 8 in info.species
    assert info.supports(1) and not info.supports(92)

    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
    forces = atoms.get_forces()
    assert forces.shape == (3, 3)

    # The forces must be the gradient of the energy the same model reports.
    step = 1e-4
    for atom in range(len(atoms)):
        for axis in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.calc = minus.calc = atoms.calc
            plus.positions[atom, axis] += step
            minus.positions[atom, axis] -= step
            numerical = -(plus.get_potential_energy()
                          - minus.get_potential_energy()) / (2 * step)
            assert forces[atom, axis] == pytest.approx(numerical, abs=2e-4)


@needs_archive
def test_unknown_species_are_refused():
    atoms = fa.Atoms("Ar2", [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], cell=10.0, pbc=True)
    atoms.calc = fa.MACE(model=ARCHIVE, backend="torchscript")
    with pytest.raises(fa.CalculatorError, match="does not cover"):
        atoms.get_potential_energy()


@needs_worker
def test_worker_evaluates_and_stays_alive():
    atoms = water()
    parameters = {"python": INTERPRETER, "backend": "worker"}
    if CHECKPOINT:
        parameters["model"] = CHECKPOINT
    atoms.calc = fa.make_calculator(FAMILY, **parameters)

    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
    assert atoms.calc.model.backend == "worker"

    for _ in range(5):
        atoms.positions[1, 0] += 0.005
        assert np.isfinite(atoms.get_potential_energy())
    assert atoms.calc.call_count == 6      # one process, six evaluations


@pytest.mark.skipif(not (ARCHIVE and INTERPRETER and CHECKPOINT and fa.torch_available()),
                    reason="needs both an archive and an interpreter")
def test_both_backends_agree():
    """The claim the whole design rests on."""
    atoms = water()
    atoms.positions[1] += [0.07, -0.03, 0.02]
    atoms.positions[2] += [-0.02, 0.05, 0.01]

    compiled = atoms.copy()
    compiled.calc = fa.MACE(model=ARCHIVE, backend="torchscript")
    worker = atoms.copy()
    worker.calc = fa.MACE(model=CHECKPOINT, backend="worker", python=INTERPRETER)

    assert compiled.get_potential_energy() == pytest.approx(
        worker.get_potential_energy(), abs=1e-7)
    assert compiled.get_forces() == pytest.approx(worker.get_forces(), abs=1e-7)


@pytest.mark.skipif(not (ARCHIVE and ENERGY_ONLY and fa.torch_available()),
                    reason="needs an archive and its energy-only twin")
def test_autodiff_reproduces_the_models_own_derivatives():
    """Forces and stress differentiated in C++ must match what MACE computes.

    The same weights are exported twice: once returning the model's own forces
    and stress, once returning nothing but the energy. The second archive
    forces FastAtoms down its LibTorch autograd path - forces from the gradient
    with respect to the positions, stress from the gradient with respect to an
    injected strain, divided by the volume. If the two disagree, the sign
    convention, the symmetrisation or the 1/V is wrong.

    Build the second archive with::

        python -m fastatoms.mlip.export <checkpoint> -o energy-only.pt --energy-only
    """
    native = water()
    native.positions[1] += [0.03, 0.02, 0.0]
    autodiff = native.copy()

    native.calc = fa.MACE(model=ARCHIVE, backend="torchscript")
    autodiff.calc = fa.MACE(model=ENERGY_ONLY, backend="torchscript")

    assert native.calc.model.provides_forces and native.calc.model.provides_stress
    assert not autodiff.calc.model.provides_forces
    assert not autodiff.calc.model.provides_stress

    assert native.get_potential_energy() == pytest.approx(
        autodiff.get_potential_energy(), abs=1e-10)
    assert native.get_forces() == pytest.approx(autodiff.get_forces(), abs=1e-10)
    assert native.get_stress() == pytest.approx(autodiff.get_stress(), abs=1e-14)


@needs_archive
def test_a_model_drives_the_optimizer():
    atoms = water()
    atoms.positions[1, 0] += 0.15          # stretch one O-H bond
    atoms.calc = fa.MACE(model=ARCHIVE, backend="torchscript")

    start = atoms.get_potential_energy()
    result = fa.optimize(atoms, "bfgs", fmax=0.05, max_steps=50)
    assert result.converged
    assert result.energy < start
    assert np.abs(atoms.get_forces()).max() < 0.05
