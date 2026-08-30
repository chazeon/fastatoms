# SPDX-License-Identifier: MIT
"""Tests for the geometry optimization module."""

from __future__ import annotations

import numpy as np
import pytest

import fastatoms as fa

ALGORITHMS = ["sd", "cg", "bfgs", "lbfgs"]
CLASSES = [fa.SteepestDescent, fa.ConjugateGradient, fa.BFGS, fa.LBFGS]


def lennard_jones(**kwargs):
    settings = dict(epsilon=0.0103, sigma=3.40, cutoff=8.5)
    settings.update(kwargs)
    return fa.LennardJones(**settings)


def fcc_argon(a=5.26, repeat=1, jitter=0.0, seed=7) -> fa.Atoms:
    basis = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    atoms = fa.Atoms("Ar4", scaled_positions=basis, cell=a, pbc=True)
    if repeat > 1:
        atoms = atoms.repeat(repeat, repeat, repeat)
    if jitter:
        rng = np.random.default_rng(seed)
        atoms.positions += rng.uniform(-jitter, jitter, atoms.positions.shape)
    atoms.calc = lennard_jones()
    return atoms


def dimer(distance=4.2) -> fa.Atoms:
    atoms = fa.Atoms("Ar2", [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    atoms.calc = lennard_jones(cutoff=20.0, shift=False)
    return atoms


# --------------------------------------------------------------------------
# The pure-math entry point
# --------------------------------------------------------------------------
def quadratic(curvature, centre):
    curvature = np.asarray(curvature, float)
    centre = np.asarray(centre, float)

    def objective(x):
        d = x - centre
        return 0.5 * float(np.sum(curvature * d * d)), curvature * d

    return objective


def rosenbrock(x):
    energy = 0.0
    gradient = np.zeros_like(x)
    for k in range(0, len(x), 2):
        a, b = x[k], x[k + 1]
        t = b - a * a
        energy += 100.0 * t * t + (1.0 - a) ** 2
        gradient[k] += -400.0 * a * t - 2.0 * (1.0 - a)
        gradient[k + 1] += 200.0 * t
    return energy, gradient


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_minimize_a_python_function(algorithm):
    centre = np.array([0.5, -1.0, 2.0, 0.0, -0.5, 1.5])
    optimizer = fa.make_optimizer(
        algorithm, fmax=1e-6, max_steps=500, alpha=20.0, max_step=10.0
    )
    result, x = optimizer.minimize(
        quadratic([1.0, 3.0, 7.0, 12.0, 20.0, 35.0], centre), np.full(6, 5.0)
    )

    assert result.converged
    assert result.reason == fa.StopReason.converged
    assert x == pytest.approx(centre, abs=1e-4)
    assert len(result.history) == result.steps + 1


@pytest.mark.parametrize("algorithm", ["cg", "bfgs", "lbfgs"])
def test_rosenbrock(algorithm):
    optimizer = fa.make_optimizer(
        algorithm, fmax=1e-5, max_steps=3000, alpha=1.0, max_step=1.0
    )
    result, x = optimizer.minimize(
        rosenbrock, np.array([-1.2, 1.0, -1.2, 1.0, -1.2, 1.0])
    )
    assert result.converged
    assert x == pytest.approx(np.ones(6), abs=1e-3)


def test_minimize_rejects_a_bad_objective():
    with pytest.raises(TypeError, match="energy, gradient"):
        fa.BFGS().minimize(lambda x: 1.0, np.zeros(3))
    with pytest.raises(ValueError, match="exactly"):
        fa.BFGS().minimize(lambda x: (1.0, x, x), np.zeros(3))
    # The convergence test groups the gradient in threes, so the length has to
    # be a multiple of three.
    with pytest.raises(fa.OptimizerError):
        fa.BFGS().minimize(quadratic([1, 1], [0, 0]), np.zeros(2))


# --------------------------------------------------------------------------
# Relaxing structures
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls", CLASSES)
def test_dimer_relaxes_to_the_analytic_minimum(cls):
    atoms = dimer()
    result = cls(fmax=1e-6).run(atoms)

    assert result.converged
    assert bool(result) is True
    assert atoms.get_distance(0, 1) == pytest.approx(2.0 ** (1 / 6) * 3.40, abs=1e-5)
    assert result.energy == pytest.approx(-0.0103, abs=1e-8)
    # The structure is left where the result says, with the cache still valid.
    assert atoms.get_potential_energy() == pytest.approx(result.energy)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_crystal_relaxation_meets_the_force_threshold(algorithm):
    atoms = fcc_argon(repeat=2, jitter=0.25, seed=3)
    result = fa.optimize(atoms, algorithm, fmax=0.005, max_steps=2000)

    assert result.converged
    assert np.linalg.norm(atoms.get_forces(), axis=1).max() <= 0.005
    assert result.fmax <= 0.005


def test_energy_decreases_monotonically():
    atoms = fcc_argon(repeat=2, jitter=0.3, seed=17)
    result = fa.LBFGS(fmax=0.01).run(atoms)

    energies = [step.energy for step in result.history]
    assert result.converged
    assert len(energies) > 3
    assert all(b <= a + 1e-12 for a, b in zip(energies, energies[1:]))


def test_atoms_optimize_shortcut():
    atoms = fcc_argon(repeat=1, jitter=0.2)
    result = atoms.optimize("lbfgs", fmax=0.01)
    assert result.converged
    assert atoms.calc.call_count == result.evaluations


def test_quasi_newton_is_cheaper_than_steepest_descent():
    def evaluations(algorithm):
        atoms = fcc_argon(repeat=2, jitter=0.25, seed=5)
        result = fa.optimize(atoms, algorithm, fmax=0.01, max_steps=5000)
        assert result.converged
        return result.evaluations

    assert evaluations("bfgs") < evaluations("sd")
    assert evaluations("lbfgs") < evaluations("sd")


# --------------------------------------------------------------------------
# Convergence handling
# --------------------------------------------------------------------------
def test_step_budget_and_reasons():
    atoms = fcc_argon(repeat=1, jitter=0.2)
    result = fa.BFGS(fmax=1e-12, max_steps=3).run(atoms)
    assert not result.converged
    assert result.reason == fa.StopReason.max_steps
    assert result.steps == 3
    assert "without reaching fmax" in result.message


def test_an_already_converged_start_costs_one_evaluation():
    atoms = fcc_argon()  # a perfect lattice: forces vanish by symmetry
    result = fa.BFGS(fmax=1e-6).run(atoms)
    assert result.converged
    assert result.steps == 0
    assert result.evaluations == 1


def test_callback_receives_steps_and_can_stop():
    atoms = fcc_argon(repeat=1, jitter=0.25)
    optimizer = fa.LBFGS(fmax=1e-8, max_steps=100)

    seen = []
    optimizer.attach(lambda step: seen.append(step) or step.step < 2)
    result = optimizer.run(atoms)

    assert result.reason == fa.StopReason.interrupted
    assert len(seen) == 3
    assert seen[0].step == 0
    assert all(hasattr(s, "fmax") and hasattr(s, "max_displacement") for s in seen)
    assert "OptimizationStep(" in repr(seen[0])


def test_a_callback_that_returns_nothing_keeps_going():
    atoms = fcc_argon(repeat=1, jitter=0.2)
    optimizer = fa.LBFGS(fmax=0.01)
    collected = []
    optimizer.attach(lambda step: collected.append(step.energy))
    assert optimizer.run(atoms).converged
    assert len(collected) > 1


def test_missing_calculator_is_reported():
    atoms = fa.Atoms("Ar2", [[0, 0, 0], [4.0, 0, 0]])
    with pytest.raises(fa.OptimizerError):
        fa.BFGS().run(atoms)
    assert issubclass(fa.OptimizerError, RuntimeError)


def test_bad_settings_are_rejected():
    atoms = fcc_argon(jitter=0.1)
    for bad in (dict(fmax=-1.0), dict(max_step=0.0), dict(alpha=0.0)):
        with pytest.raises(fa.OptimizerError):
            fa.BFGS(**bad).run(atoms)
    with pytest.raises(fa.OptimizerError):
        fa.LBFGS(memory=0).run(atoms)
    with pytest.raises(fa.OptimizerError):
        fa.BFGS(line_search="golden").run(atoms)


# --------------------------------------------------------------------------
# Cell relaxation
# --------------------------------------------------------------------------
def test_cell_relaxation_drives_the_stress_to_zero():
    atoms = fcc_argon(a=5.6, jitter=0.02, seed=23)
    initial_volume = atoms.get_volume()

    result = fa.LBFGS(fmax=1e-4, relax_cell=True).run(atoms)

    assert result.converged
    assert atoms.get_volume() < initial_volume
    assert np.abs(atoms.get_stress()).max() < 1e-5
    lengths = atoms.cell.lengths
    assert lengths[0] == pytest.approx(lengths[1], rel=1e-3)


def test_external_pressure_compresses_the_cell():
    def relaxed_volume(pressure):
        atoms = fcc_argon(a=5.3)
        result = fa.optimize(
            atoms, "lbfgs", fmax=1e-4, relax_cell=True, pressure=pressure
        )
        assert result.converged
        return atoms.get_volume()

    assert relaxed_volume(0.005) < relaxed_volume(0.0)


def test_a_mask_freezes_cell_directions():
    atoms = fcc_argon(a=5.6)
    original_c = atoms.cell.lengths[2]
    fa.optimize(
        atoms, "lbfgs", fmax=1e-4, relax_cell=True, mask=[1, 1, 0, 0, 0, 0]
    )
    assert atoms.cell.lengths[2] == pytest.approx(original_c, abs=1e-9)
    assert atoms.cell.lengths[0] < original_c


def test_cell_relaxation_needs_periodicity():
    atoms = fa.Atoms("Ar2", [[0, 0, 0], [4.0, 0, 0]])
    atoms.calc = lennard_jones()
    with pytest.raises(fa.OptimizerError):
        fa.BFGS(relax_cell=True).run(atoms)


def test_a_concave_start_still_converges():
    # Simple cubic argon far past the potential minimum: a region where every
    # curvature update is legitimately discarded, so only a line search that
    # can grow the step gets there.
    atoms = fa.Atoms("Ar", [[0.1, 0, 0]], cell=5.6, pbc=True).repeat(2, 2, 2)
    atoms.calc = lennard_jones()
    result = fa.optimize(atoms, "lbfgs", fmax=1e-4, max_steps=300, relax_cell=True)

    assert result.converged
    assert result.steps < 100
    assert atoms.cell.lengths[0] / 2 < 4.5


# --------------------------------------------------------------------------
# Algorithm details
# --------------------------------------------------------------------------
def test_bfgs_exposes_its_inverse_hessian():
    atoms = fcc_argon(repeat=1, jitter=0.2)
    optimizer = fa.BFGS(fmax=0.01)
    optimizer.run(atoms)

    hessian = optimizer.inverse_hessian
    assert hessian.shape == (3 * len(atoms), 3 * len(atoms))
    assert hessian == pytest.approx(hessian.T)  # symmetric by construction
    assert optimizer.skipped_updates >= 0


def test_lbfgs_keeps_a_bounded_history():
    atoms = fcc_argon(repeat=1, jitter=0.3)
    optimizer = fa.LBFGS(fmax=1e-4, memory=3)
    optimizer.run(atoms)
    assert optimizer.memory == 3
    assert optimizer.history_size <= 3


@pytest.mark.parametrize(
    "variant", ["polak_ribiere", "fletcher_reeves", "hestenes_stiefel"]
)
def test_conjugate_gradient_variants(variant):
    optimizer = fa.ConjugateGradient(
        variant=variant, fmax=1e-6, max_steps=2000, alpha=100.0, max_step=10.0
    )
    assert optimizer.variant == variant
    result, _ = optimizer.minimize(
        quadratic([1.0, 5.0, 25.0, 50.0, 100.0, 200.0], np.zeros(6)), np.full(6, 2.0)
    )
    assert result.converged
    # Conjugate gradient finishes a quadratic in about as many steps as there
    # are variables.
    assert result.steps <= 12


def test_line_search_choices():
    assert fa.BFGS().line_search == "wolfe"
    assert fa.BFGS(line_search="none").line_search == "none"
    assert fa.BFGS(line_search="armijo").line_search == "backtracking"

    atoms = fcc_argon(repeat=1, jitter=0.2)
    for choice in ("none", "backtracking", "wolfe"):
        probe = fcc_argon(repeat=1, jitter=0.2)
        assert fa.optimize(probe, "lbfgs", fmax=0.01, line_search=choice).converged
    assert fa.optimize(atoms, "lbfgs", fmax=0.01).converged


def test_repr_and_parameters():
    optimizer = fa.LBFGS(fmax=0.02, max_steps=50, memory=7)
    assert optimizer.fmax == 0.02
    assert optimizer.max_steps == 50
    assert optimizer.parameters["memory"] == 7
    assert "lbfgs(" in repr(optimizer)

    optimizer.set("fmax", 0.005)
    assert optimizer.fmax == 0.005


# --------------------------------------------------------------------------
# Registry and the YAML path
# --------------------------------------------------------------------------
def test_registry():
    assert set(fa.registered_optimizers()) == {"sd", "cg", "bfgs", "lbfgs"}
    assert fa.make_optimizer("BFGS").name == "bfgs"
    assert fa.make_optimizer("Steepest-Descent").name == "sd"
    assert fa.make_optimizer("L-BFGS").name == "lbfgs"
    assert fa.make_optimizer("conjugate_gradient").name == "cg"
    with pytest.raises(fa.OptimizerError):
        fa.make_optimizer("newton")


def test_custom_optimizer_from_python():
    fa.register_optimizer("python_bfgs", lambda params: fa.BFGS(**params))
    optimizer = fa.make_optimizer("python_bfgs", fmax=0.02)
    assert optimizer.fmax == 0.02

    atoms = fcc_argon(repeat=1, jitter=0.2)
    assert optimizer.run(atoms).converged


RELAX_INPUT = """
structure:
  cell: {a: 5.3, b: 5.3, c: 5.3}
  coordinates: fractional
  symbols: [Ar, Ar, Ar, Ar]
  positions:
    - [0.02, 0.00, 0.00]
    - [0.00, 0.50, 0.50]
    - [0.50, 0.00, 0.50]
    - [0.50, 0.50, 0.00]

simulation:
  task: optimize
  calculator:
    name: lennard_jones
    parameters: {epsilon: 0.0103, sigma: 3.40, cutoff: 8.5}
  optimizer:
    algorithm: lbfgs
    fmax: 0.001
    max_steps: 300
    relax_cell: true
    memory: 20
"""


def test_optimizer_comes_from_the_input_file():
    deck = fa.read_string(RELAX_INPUT)
    assert deck.simulation.optimizer.algorithm == "lbfgs"
    assert deck.simulation.optimizer.fmax == 0.001
    assert deck.simulation.optimizer.relax_cell is True

    optimizer = deck.make_optimizer()
    assert optimizer.name == "lbfgs"
    assert optimizer.fmax == 0.001
    assert optimizer.max_steps == 300
    # Settings the typed struct does not know about still reach the optimizer.
    assert optimizer.parameters["memory"] == 20

    # Keyword arguments override the file.
    assert deck.make_optimizer(fmax=0.05).fmax == 0.05
    assert fa.make_optimizer(deck.simulation.optimizer).name == "lbfgs"


def test_full_yaml_relaxation():
    deck = fa.read_string(RELAX_INPUT)
    atoms = deck.atoms
    atoms.calc = deck.make_calculator()
    result = deck.make_optimizer().run(atoms)

    assert result.converged
    assert np.abs(atoms.get_stress()).max() < 1e-3
    assert np.linalg.norm(atoms.get_forces(), axis=1).max() < 0.001
