# SPDX-License-Identifier: MIT
"""The ASE benchmark is a script, not a library, so it rots quietly.

This runs it at a size that finishes in a second, which is enough to catch the
way it actually breaks: an API it drives changing under it. The physics
assertions are the benchmark's own - it refuses to report timings for two
libraries it cannot first show to be computing the same thing - so checking
that its validation passes checks both the script and the agreement.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

ase = pytest.importorskip("ase", reason="the benchmark compares against ASE")

BENCHMARK = pathlib.Path(__file__).resolve().parents[2] / "benchmarks" / "ase_comparison.py"


@pytest.fixture(scope="module")
def benchmark():
    if not BENCHMARK.is_file():
        pytest.skip(f"{BENCHMARK} is missing")
    spec = importlib.util.spec_from_file_location("ase_comparison", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_both_libraries_get_identical_input(benchmark):
    """The comparison is only a comparison if the inputs are the same."""
    system = benchmark.build_system(repeat=2)
    fat = benchmark.make_fastatoms(system)
    ase_atoms = benchmark.make_ase(system)

    assert len(fat) == len(ase_atoms) == system["natoms"]

    # Positions and the cell are stored as given, so they must match bit for bit.
    assert benchmark.largest_difference(fat.positions, ase_atoms.get_positions()) == 0.0
    assert benchmark.largest_difference(fat.cell.array, ase_atoms.cell.array) == 0.0

    # Velocities are not stored: both libraries keep momenta and divide by the
    # mass on the way out, and that round trip is not exact in binary floating
    # point. Machine precision is the right expectation, not equality.
    assert benchmark.largest_difference(fat.get_velocities(),
                                        ase_atoms.get_velocities()) < 1e-15


def test_validation_passes(benchmark):
    """The agreement the benchmark asserts before it will report anything."""
    findings = benchmark.validate(repeat=2, md_steps=10, verbose=False)
    failures = benchmark.check_tolerances(findings)
    assert not failures, "; ".join(failures)

    # The headline claims, restated here so a regression names itself.
    assert findings["energy_relative"] < 1e-12
    assert findings["force_difference"] < 1e-12
    assert findings["trajectory_short"] < 1e-12
    assert findings["optimizer_minimum_relative"] < 1e-9


@pytest.mark.parametrize("case", ["single_point", "optimize", "md", "relax"])
def test_every_case_runs_for_both_libraries(benchmark, case):
    system = benchmark.build_system(repeat=2)
    function, _description, _default, libraries = benchmark.CASES[case]
    steps = 3 if case != "relax" else 200

    energies = {}
    for library in libraries:
        measured = function(library, system, steps)
        assert measured["seconds"] >= 0.0
        assert measured["evaluations"] > 0
        energies[library] = measured["energy"]

    # `relax` runs to convergence, so every library must find the same minimum.
    if case == "relax":
        spread = max(energies.values()) - min(energies.values())
        assert spread < 1e-8, f"different minima: {energies}"


def test_peak_memory_is_reported_in_bytes(benchmark):
    peak = benchmark.peak_memory_bytes()
    # A Python process with numpy loaded is megabytes, not kilobytes: this
    # catches the ru_maxrss unit trap, where Linux reports kB and macOS bytes.
    assert peak > 8 * 1024 * 1024
    assert peak < 8 * 1024 * 1024 * 1024
