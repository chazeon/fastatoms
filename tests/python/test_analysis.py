# SPDX-License-Identifier: MIT
"""Trajectory analysis from Python.

The C++ suite checks the physics. What is checked here is the Python surface,
and the promise the bindings make: the arrays come back as views onto the C++
storage, not copies of it.
"""

from __future__ import annotations

import numpy as np
import pytest

import fastatoms as fa

analysis = fa.analysis
build = fa.build


def ideal_gas(count: int, side: float, seed: int) -> fa.Atoms:
    """Atoms at random positions: an ideal gas, whose g(r) is one."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, side, size=(count, 3))
    return fa.Atoms(f"Ar{count}", positions, cell=side, pbc=True)


# --------------------------------------------------------------------------
# The zero-copy promise
# --------------------------------------------------------------------------
def test_the_arrays_are_views_not_copies():
    rdf = analysis.radial_distribution(build.fcc("Al", 4.05, cubic=True), rmax=6.0, bins=300)

    for name in ("r", "g", "counts", "coordination"):
        array = getattr(rdf, name)
        assert isinstance(array, np.ndarray)
        assert array.base is rdf                 # it points at the C++ storage
        assert not array.flags.writeable         # which must not be written through
        assert array.flags.c_contiguous

    # Two accesses give the same memory, so nothing was copied on the way out.
    assert rdf.g.__array_interface__["data"][0] == rdf.g.__array_interface__["data"][0]
    with pytest.raises(ValueError):
        rdf.g[0] = 5.0


def test_structure_factor_and_tension_arrays_are_views_too():
    sq = analysis.structure_factor([build.fcc("Al", 4.05, cubic=True)], qmax=5.0, points=100)
    assert sq.q.base is sq and sq.s.base is sq
    assert not sq.s.flags.writeable

    gamma = analysis.surface_tension_from_pressures(
        [np.diag([1.0, 1.0, 2.0])], [10.0])
    assert gamma.per_frame.base is gamma


# --------------------------------------------------------------------------
# g(r)
# --------------------------------------------------------------------------
def test_the_fcc_shells():
    a = 4.05
    rdf = analysis.radial_distribution(build.fcc("Al", a, cubic=True).repeat(4, 4, 4),
                                       rmax=7.0, bins=700)
    peaks = [rdf.r[k] for k in range(1, len(rdf) - 1)
             if rdf.g[k] > 1.5 and rdf.g[k] >= rdf.g[k - 1] and rdf.g[k] > rdf.g[k + 1]]
    assert peaks[:4] == pytest.approx(
        [a / np.sqrt(2), a, a * np.sqrt(1.5), a * np.sqrt(2)], abs=0.02)

    # Twelve nearest neighbours, then six more, then twenty-four more.
    def coordination_at(radius):
        return float(rdf.coordination[np.searchsorted(rdf.r, radius) - 1])

    assert coordination_at(3.2) == pytest.approx(12.0)
    assert coordination_at(4.4) == pytest.approx(18.0)
    assert coordination_at(5.3) == pytest.approx(42.0)


def test_an_ideal_gas_is_flat():
    frames = [ideal_gas(400, 25.0, seed) for seed in range(40)]
    rdf = analysis.radial_distribution(frames, rmax=8.0, bins=40)
    assert rdf.frames == 40
    far = rdf.r > 2.0
    assert np.abs(rdf.g[far] - 1.0).max() < 0.06


def test_species_pairs():
    a = 5.64
    salt = build.rocksalt("Na", "Cl", a, cubic=True).repeat(3, 3, 3)
    cross = analysis.radial_distribution(salt, rmax=6.0, bins=600, first="Na", second="Cl")
    like = analysis.radial_distribution(salt, rmax=6.0, bins=600, first="Na", second="Na")

    assert cross.label == "Na-Cl"
    assert like.label == "Na-Na"
    assert cross.r[int(np.argmax(cross.g))] == pytest.approx(a / 2, abs=0.02)
    assert like.r[int(np.argmax(like.g))] == pytest.approx(a / np.sqrt(2), abs=0.02)

    # Six chlorines around a sodium; twelve sodiums.
    assert float(cross.coordination[np.searchsorted(cross.r, 3.5) - 1]) == pytest.approx(6.0)
    assert float(like.coordination[np.searchsorted(like.r, 4.5) - 1]) == pytest.approx(12.0)

    # A selection may be a list as well as a single symbol.
    both = analysis.radial_distribution(salt, rmax=6.0, first=["Na", "Cl"])
    assert both.label == "Na,Cl-all"


def test_the_accumulator_streams():
    frames = [ideal_gas(200, 20.0, seed) for seed in range(5)]
    accumulator = analysis.RadialDistributionAccumulator(rmax=6.0, bins=60)
    for frame in frames:
        accumulator.add(frame)
    assert accumulator.frames == 5

    streamed = accumulator.result()
    at_once = analysis.radial_distribution(frames, rmax=6.0, bins=60)
    assert streamed.g == pytest.approx(at_once.g)


# --------------------------------------------------------------------------
# S(q)
# --------------------------------------------------------------------------
def test_the_reciprocal_sum_finds_the_bragg_peaks():
    a = 4.05
    sq = analysis.structure_factor([build.fcc("Al", a, cubic=True)],
                                   method="reciprocal", qmax=5.0, points=500)
    assert sq.method == "reciprocal"
    strong = sorted(sq.q[k] for k in range(len(sq)) if sq.s[k] > 1.0)
    # (111), (200) and (220); the odd-plus-even reflections are extinct.
    assert strong == pytest.approx(
        [2 * np.pi * np.sqrt(3) / a, 2 * np.pi * 2 / a, 2 * np.pi * 2 * np.sqrt(2) / a],
        abs=0.02)


def test_the_three_routes_agree_on_an_ideal_gas():
    frames = [ideal_gas(400, 26.0, seed) for seed in range(15)]

    debye = analysis.structure_factor(frames, method="debye", qmin=1.0, qmax=8.0, points=25)
    rdf = analysis.radial_distribution(frames, rmax=12.0, bins=240)
    transformed = analysis.structure_factor_from_rdf(rdf, qmin=1.0, qmax=8.0, points=25)

    assert np.abs(np.asarray(debye.s) - 1.0).max() < 0.15
    assert np.abs(np.asarray(transformed.s) - 1.0).max() < 0.15
    assert debye.method == "debye" and transformed.method == "rdf"


def test_an_unknown_method_is_refused():
    with pytest.raises(fa.AnalysisError, match="not a method"):
        analysis.structure_factor([build.fcc("Al", 4.05)], method="magic")


# --------------------------------------------------------------------------
# Surface tension
# --------------------------------------------------------------------------
def test_the_tension_formula():
    # gamma = (L/2) [P_zz - (P_xx + P_yy)/2] = (20/2)(10 - 2) = 80
    gamma = analysis.surface_tension_from_pressures([np.diag([1.0, 3.0, 10.0])], [20.0])
    assert gamma.gamma == pytest.approx(80.0)
    assert gamma.gamma_mN_per_m == pytest.approx(80.0 * analysis.mN_per_m)
    assert gamma.normal == 2

    # Along x instead: (20/2)(1 - 6.5) = -55
    along_x = analysis.surface_tension_from_pressures(
        [np.diag([1.0, 3.0, 10.0])], [20.0], normal=0)
    assert along_x.gamma == pytest.approx(-55.0)


def test_an_isotropic_pressure_has_no_tension():
    for pressure in (-0.05, 0.0, 2.0):
        gamma = analysis.surface_tension_from_pressures(
            [np.diag([pressure] * 3)], [30.0])
        assert gamma.gamma == pytest.approx(0.0, abs=1e-14)


def test_the_average_and_its_error():
    pressures = [np.diag([0.0, 0.0, zz]) for zz in (10.0, 20.0, 30.0)]
    gamma = analysis.surface_tension_from_pressures(pressures, [2.0] * 3)
    assert list(gamma.per_frame) == pytest.approx([10.0, 20.0, 30.0])
    assert gamma.gamma == pytest.approx(20.0)
    assert gamma.standard_error == pytest.approx(10.0 / np.sqrt(3))
    assert gamma.frames == 3


def test_a_bulk_crystal_has_no_interface():
    crystal = build.fcc("Ar", 5.26, cubic=True).repeat(3, 3, 3)
    crystal.calc = fa.LennardJones(epsilon=0.0103, sigma=3.40, cutoff=8.0)
    gamma = analysis.surface_tension([crystal])
    assert abs(gamma.gamma_mN_per_m) < 1e-6

    pressure = np.asarray(analysis.pressure_tensor(crystal))
    assert pressure[0][0] == pytest.approx(pressure[2][2])


def test_bad_input_is_refused():
    with pytest.raises(fa.AnalysisError):
        analysis.radial_distribution([])
    with pytest.raises(fa.AnalysisError):
        analysis.radial_distribution(build.fcc("Al", 4.05), rmax=-1.0)
    with pytest.raises(fa.AnalysisError):
        analysis.surface_tension_from_pressures([np.eye(3)], [1.0], normal=3)
    with pytest.raises(fa.AnalysisError):
        analysis.surface_tension_from_pressures([np.eye(3), np.eye(3)], [1.0])
