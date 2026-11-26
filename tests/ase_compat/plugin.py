"""pytest plugin: run ASE's own I/O tests against the fastatoms readers and writers.

Load with ``-p tests.ase_compat.plugin``. ASE's test modules are collected from
the installed package with ``--pyargs``, so no ASE code is copied into the repo.
This plugin only swaps functions at runtime:

* ``pytest_configure`` (before ASE's test modules are imported) replaces the
  public ASE reader/writer functions, both on their modules (for direct
  imports) and on the ASE io registry (``IOFormat._readfunc`` /
  ``_writefunc``), so ``ase.io.read``/``write`` and direct calls both go
  through the adapters in ``adapters.py``.
* ``pytest_collection_modifyitems`` applies the expectations table in
  ``expectations.py``: ``deselect`` removes a test, ``xfail`` marks it strict.
* ``pytest_runtest_call`` records which adapters a test reached, so a pass that
  never touched fastatoms is reported rather than counted as compatibility.
* ``pytest_terminal_summary`` prints the per-module, per-mode table and the
  list of unexpected failures to triage.

Modes (``--fa-mode`` or ``FA_ASE_MODE``): ``read`` swaps only readers, ``write``
only writers, ``both`` both, ``none`` nothing (ASE's own code, the baseline).
Default ``read``.
"""

from __future__ import annotations

import fnmatch
import importlib
import os
from collections import Counter, defaultdict
from typing import Any

import pytest
from tests.ase_compat import adapters
from tests.ase_compat.expectations import EXPECTATIONS, Expectation

MODES = ("none", "read", "write", "both")  # "none" is the ASE-only baseline

# (module, attribute, original) for every module attribute we replaced.
_module_originals: list[tuple[Any, str, Any]] = []
# (IOFormat, attribute name, original) for every registry slot we replaced.
_registry_originals: list[tuple[Any, str, Any]] = []

# nodeid -> dotted module name, recorded at collection.
_module_of: dict[str, str] = {}
# nodeid -> adapter names reached during the call phase.
_adapter_hits: dict[str, list[str]] = {}
# nodeid -> final outcome (passed, failed, xfailed, xpassed, skipped) and the first error line.
_outcomes: dict[str, str] = {}
_failure_text: dict[str, str] = {}
_deselected: list[tuple[str, Expectation]] = []
_xfailed_by: dict[str, Expectation] = {}
_mode: str = "read"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("fastatoms-ase-compat", "fastatoms vs ASE I/O tests")
    group.addoption(
        "--fa-mode",
        action="store",
        default=os.environ.get("FA_ASE_MODE", "read"),
        choices=MODES,
        help="Which side of ASE's I/O to replace with fastatoms: read, write or both. "
        "'none' runs ASE's own implementation with no patches and no expectations (baseline).",
    )


def _swap(obj: Any, name: str, new: Any, store: list) -> None:
    store.append((obj, name, getattr(obj, name)))
    setattr(obj, name, new)


def install(mode: str) -> None:
    """Replace ASE readers/writers for ``mode``. Idempotent per process."""
    from ase.io import formats

    want_read = mode in ("read", "both")
    want_write = mode in ("write", "both")

    for (module_name, attr), adapter in adapters.MODULE_PATCHES.items():
        is_read = attr.startswith("read")
        if (is_read and not want_read) or (not is_read and not want_write):
            continue
        module = importlib.import_module(module_name)
        _swap(module, attr, adapter, _module_originals)

    for ase_name, (reader, writer) in adapters.REGISTRY.items():
        fmt = formats.ioformats[ase_name]
        if reader is not None and want_read:
            _swap(fmt, "_readfunc", (lambda fn=reader: fn), _registry_originals)
        if writer is not None and want_write:
            _swap(fmt, "_writefunc", (lambda fn=writer: fn), _registry_originals)


def uninstall() -> None:
    for obj, name, original in reversed(_module_originals + _registry_originals):
        setattr(obj, name, original)
    _module_originals.clear()
    _registry_originals.clear()


# Warnings that come from the environment, not from either I/O implementation.
# ASE's test conftest turns every warning into an error. ASE 3.29 sets array
# shapes in place, which NumPy 2.5 deprecates, so every Atoms() with zero
# atoms errors. Ignored here so the I/O tests run; reported in the summary.
ENVIRONMENT_WARNING_FILTERS = [
    "ignore:Setting the shape on a NumPy array:DeprecationWarning",
]


def pytest_configure(config: pytest.Config) -> None:
    global _mode
    _mode = config.getoption("fa_mode")
    config.addinivalue_line("markers", "fa_ase: test from ASE's I/O suite run against fastatoms")
    install(_mode)


def pytest_unconfigure(config: pytest.Config) -> None:
    uninstall()


def _match(item: pytest.Item) -> Expectation | None:
    key = f"{_module_of.get(item.nodeid, '')}::{item.name}"
    for expectation in EXPECTATIONS:
        if _mode not in expectation.modes:
            continue
        if fnmatch.fnmatchcase(key, expectation.pattern):
            return expectation
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    kept: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        module = getattr(item, "module", None)
        _module_of[item.nodeid] = module.__name__ if module is not None else ""
        # Item marks are applied after the ini filters, so they take precedence.
        for spec in ENVIRONMENT_WARNING_FILTERS:
            item.add_marker(pytest.mark.filterwarnings(spec))
        expectation = _match(item)
        if expectation is None:
            kept.append(item)
            continue
        if expectation.action == "deselect":
            deselected.append(item)
            _deselected.append((item.nodeid, expectation))
            continue
        item.add_marker(pytest.mark.xfail(strict=True, reason=f"[{expectation.category}] {expectation.reason}"))
        _xfailed_by[item.nodeid] = expectation
        kept.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


def _record_phase(item: pytest.Item):
    """Collect the adapter calls made in one phase (setup, call or teardown).

    Hits from every phase are kept, so a test whose fixtures write or read a file
    through fastatoms is not counted as bypassing fastatoms.
    """
    adapters.CALLS.clear()
    try:
        return (yield)
    finally:
        _adapter_hits.setdefault(item.nodeid, []).extend(adapters.CALLS)
        adapters.CALLS.clear()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item):
    return (yield from _record_phase(item))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item):
    return (yield from _record_phase(item))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item):
    return (yield from _record_phase(item))


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "setup" and report.skipped:
        _outcomes[report.nodeid] = "skipped"
        return
    if report.when == "setup" and report.failed:
        _outcomes[report.nodeid] = "failed"
        _failure_text[report.nodeid] = _first_line(report)
        return
    if report.when != "call":
        return
    if hasattr(report, "wasxfail"):
        _outcomes[report.nodeid] = "xpassed" if report.passed else "xfailed"
    elif report.failed and "XPASS(strict)" in str(report.longrepr):
        # A strict xfail that now passes. Listed separately so the entry gets removed.
        _outcomes[report.nodeid] = "xpassed"
    elif report.passed:
        _outcomes[report.nodeid] = "passed"
    elif report.skipped:
        _outcomes[report.nodeid] = "skipped"
    else:
        _outcomes[report.nodeid] = "failed"
        _failure_text[report.nodeid] = _first_line(report)


def _first_line(report: pytest.TestReport) -> str:
    text = str(report.longrepr) if report.longrepr else ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("E "):
            return stripped[2:][:200]
    return (text.strip().splitlines() or [""])[-1][:200]


def _module_label(module: str) -> str:
    return module.replace("ase.test.", "")


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    write = terminalreporter.write_line
    rows: dict[str, Counter] = defaultdict(Counter)
    bypass: Counter = Counter()
    for nodeid, module in _module_of.items():
        outcome = _outcomes.get(nodeid)
        if outcome is None:
            continue
        rows[module][outcome] += 1
        if outcome == "passed" and not _adapter_hits.get(nodeid):
            bypass[module] += 1
    for nodeid, _expectation in _deselected:
        rows[_module_of.get(nodeid, "")]["deselected"] += 1

    write("")
    write(f"fastatoms vs ASE I/O compatibility, mode={_mode}")
    header = f"{'module':<40} {'passed':>7} {'failed':>7} {'xfailed':>8} {'xpassed':>8} {'deselected':>11} {'no-fa':>6}"
    write(header)
    write("-" * len(header))
    totals: Counter = Counter()
    for module in sorted(rows):
        c = rows[module]
        totals.update(c)
        write(
            f"{_module_label(module):<40} {c['passed']:>7} {c['failed']:>7} {c['xfailed']:>8} "
            f"{c['xpassed']:>8} {c['deselected']:>11} {bypass[module]:>6}"
        )
    write("-" * len(header))
    write(
        f"{'total':<40} {totals['passed']:>7} {totals['failed']:>7} {totals['xfailed']:>8} "
        f"{totals['xpassed']:>8} {totals['deselected']:>11} {sum(bypass.values()):>6}"
    )
    write("no-fa: passed without reaching any fastatoms adapter (not evidence of compatibility)")

    unexpected = [
        (nodeid, _failure_text.get(nodeid, "")) for nodeid, outcome in _outcomes.items() if outcome == "failed"
    ]
    if unexpected:
        write("")
        write(f"unexpected failures to triage ({len(unexpected)}):")
        for nodeid, text in sorted(unexpected):
            write(f"  {nodeid}: {text}")
    if any(outcome == "xpassed" for outcome in _outcomes.values()):
        write("")
        write("xpassed (strict xfail now passes; remove the expectation):")
        for nodeid, outcome in sorted(_outcomes.items()):
            if outcome == "xpassed":
                write(f"  {nodeid}")

    counts: Counter = Counter()
    for expectation in _xfailed_by.values():
        counts[(expectation.action, expectation.category)] += 1
    for _nodeid, expectation in _deselected:
        counts[(expectation.action, expectation.category)] += 1
    write("")
    write("expectation entries applied (action, category: count):")
    for (action, category), count in sorted(counts.items()):
        write(f"  {action:<9} {category}: {count}")
