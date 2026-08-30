# SPDX-License-Identifier: MIT
"""Tests for the versioning_fastatoms command.

A version bumper is exactly the kind of tool nobody notices is broken until a
release goes out wrong, so the parts that can be checked cheaply are: the
dialect conversion (a pre-release is spelled differently in CMake and in
pyproject), the bump arithmetic, what the commit log is read to mean, and the
promise that a failed write leaves nothing half-applied.

The command lives outside the repository, alongside the other `versioning_*`
scripts, so it is looked up rather than assumed: `$VERSIONING_FASTATOMS`, then
`PATH`, then `~/bin`, then an in-tree copy. These tests skip when it is not
installed, which is what a clone without it should do.

Everything here works on a temporary copy of a project tree. Nothing touches
the real CMakeLists.txt.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import pathlib
import shutil
import stat
import subprocess
import sys

import pytest

def find_tool():
    """Where the command is, or None. Explicit first, then the usual places."""
    override = os.environ.get("VERSIONING_FASTATOMS")
    if override:
        return pathlib.Path(override)
    found = shutil.which("versioning_fastatoms")
    if found:
        return pathlib.Path(found)
    for candidate in (pathlib.Path.home() / "bin" / "versioning_fastatoms",
                      pathlib.Path(__file__).resolve().parents[2] / "tools"
                      / "versioning_fastatoms"):
        if candidate.is_file():
            return candidate
    return None


TOOL = find_tool()

#: Skip the whole module when the command is not installed, rather than
#: guarding each test: a path that was given explicitly but does not exist is
#: as absent as one that was never given.
pytestmark = pytest.mark.skipif(
    TOOL is None or not TOOL.is_file(),
    reason="versioning_fastatoms is not installed (set $VERSIONING_FASTATOMS, "
           "or put it on PATH)")


@pytest.fixture(scope="module")
def tool():
    # The tool has no .py extension - it is a command - so the loader has to be
    # named rather than inferred from the suffix.
    loader = importlib.machinery.SourceFileLoader("versioning_fastatoms", str(TOOL))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path):
    """A minimal tree carrying the two files the tool writes, already drifted."""
    (tmp_path / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.20)\n\n'
        'project(FastAtoms\n'
        '    VERSION 0.1.2\n'
        '    DESCRIPTION "test"\n'
        '    LANGUAGES CXX)\n', encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\n'
        'name = "fastatoms"\n'
        'version = "0.1.1"\n'
        'requires-python = ">=3.9"\n\n'
        '[tool.scikit-build]\n'
        'minimum-version = "0.10"\n', encoding="utf-8")
    return tmp_path


def run(project_dir, *arguments):
    return subprocess.run([sys.executable, str(TOOL), "--root", str(project_dir),
                           *arguments],
                          capture_output=True, text=True, check=False)


# --------------------------------------------------------------------------
# The version itself
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text, core, pep440", [
    ("1.2.3", "1.2.3", "1.2.3"),
    ("0.1.2", "0.1.2", "0.1.2"),
    ("1.0.0-rc.1", "1.0.0", "1.0.0rc1"),
    ("1.0.0-alpha.2", "1.0.0", "1.0.0a2"),
    ("1.0.0-beta.3", "1.0.0", "1.0.0b3"),
    ("2.0.0-rc.1+build.5", "2.0.0", "2.0.0rc1+build.5"),
])
def test_each_file_gets_its_own_spelling(tool, text, core, pep440):
    """CMake takes only numbers; Python packaging is PEP 440. Writing the semver
    form into pyproject.toml gets it normalised behind your back."""
    version = tool.Version.parse(text)
    assert str(version) == text
    assert version.core == core          # what CMake can accept
    assert version.to_pep440() == pep440  # what pip expects


@pytest.mark.parametrize("bad", ["1.2", "v1.2.3", "1.2.3.4", "", "one.two.three"])
def test_nonsense_versions_are_refused(tool, bad):
    assert tool.Version.try_parse(bad) is None


@pytest.mark.parametrize("start, part, expected", [
    ("1.2.3", "major", "2.0.0"),
    ("1.2.3", "minor", "1.3.0"),
    ("1.2.3", "patch", "1.2.4"),
    ("0.1.2", "patch", "0.1.3"),
    # Releasing a pre-release drops the tag rather than bumping past it.
    ("1.2.0-rc.1", "patch", "1.2.0"),
    ("1.2.0-rc.1", "minor", "1.3.0"),
])
def test_bump_arithmetic(tool, start, part, expected):
    assert str(tool.Version.parse(start).bumped(part)) == expected


# --------------------------------------------------------------------------
# What the commit log is read to mean
# --------------------------------------------------------------------------
def test_conventional_commits_choose_the_bump(tool):
    one = tool.Version.parse("1.4.2")
    assert tool.suggest_bump(one, [("feat: add phonons", "")])[0] == "minor"
    assert tool.suggest_bump(one, [("fix: correct the stress", "")])[0] == "patch"
    assert tool.suggest_bump(one, [("perf: faster neighbour list", "")])[0] == "patch"
    assert tool.suggest_bump(one, [("docs: tidy the README", "")])[0] == "patch"
    assert tool.suggest_bump(one, [("feat!: drop the old API", "")])[0] == "major"
    assert tool.suggest_bump(one, [("feat: x", "BREAKING CHANGE: y")])[0] == "major"
    # The strongest signal in the batch wins.
    assert tool.suggest_bump(one, [("fix: a", ""), ("feat: b", "")])[0] == "minor"
    assert tool.suggest_bump(one, [("fix: a", ""), ("feat!: b", "")])[0] == "major"


def test_a_breaking_change_before_1_0_bumps_the_minor(tool):
    """Semver puts 0.y.z outside the stability promise, so a breaking change
    there moves the minor. Silently declining to jump to 1.0.0 would look like
    a bug, so the reason says why."""
    zero = tool.Version.parse("0.1.2")
    part, reason = tool.suggest_bump(zero, [("feat!: rework the calculator", "")])
    assert part == "minor"
    assert "major version is 0" in reason
    assert str(zero.bumped(part)) == "0.2.0"


def test_no_history_is_a_patch_and_says_so(tool):
    one = tool.Version.parse("1.0.0")
    assert tool.suggest_bump(one, [])[0] == "patch"
    assert "no commits" in tool.suggest_bump(one, [])[1]
    # Not a git repository at all.
    part, reason = tool.suggest_bump(one, None)
    assert part == "patch"
    assert "git" in reason


# --------------------------------------------------------------------------
# Reading and writing a tree
# --------------------------------------------------------------------------
def test_check_reports_drift(project):
    finished = run(project, "--check")
    assert finished.returncode == 1
    assert "drift" in finished.stderr.lower()
    assert "0.1.2" in finished.stderr and "0.1.1" in finished.stderr


def test_sync_repairs_drift_from_the_source_of_truth(project):
    assert run(project, "--sync").returncode == 0
    assert 'version = "0.1.2"' in (project / "pyproject.toml").read_text()
    assert "VERSION 0.1.2" in (project / "CMakeLists.txt").read_text()
    assert run(project, "--check").returncode == 0


def test_a_bump_writes_every_file(project):
    assert run(project, "--minor").returncode == 0
    assert "VERSION 0.2.0" in (project / "CMakeLists.txt").read_text()
    assert 'version = "0.2.0"' in (project / "pyproject.toml").read_text()
    assert run(project, "--check").returncode == 0


def test_a_prerelease_is_written_in_each_dialect(project):
    assert run(project, "--set", "1.0.0-rc.1").returncode == 0
    # CMake cannot express a pre-release and would reject the semver form.
    assert "VERSION 1.0.0\n" in (project / "CMakeLists.txt").read_text()
    assert 'version = "1.0.0rc1"' in (project / "pyproject.toml").read_text()


def test_dry_run_writes_nothing(project):
    before = {path.name: path.read_text() for path in project.iterdir()}
    finished = run(project, "--major", "--dry-run")
    assert finished.returncode == 0
    assert "Nothing was written" in finished.stdout
    assert {path.name: path.read_text() for path in project.iterdir()} == before


def test_surrounding_formatting_is_left_alone(project):
    original = (project / "CMakeLists.txt").read_text()
    run(project, "--patch")
    updated = (project / "CMakeLists.txt").read_text()
    assert updated == original.replace("VERSION 0.1.2", "VERSION 0.1.3")
    # The unrelated pyproject keys are untouched.
    text = (project / "pyproject.toml").read_text()
    assert 'minimum-version = "0.10"' in text
    assert 'requires-python = ">=3.9"' in text


def test_a_failed_write_leaves_nothing_half_applied(project):
    """The promise that makes this safe to run: CMakeLists moved and pyproject
    not is exactly the drift the tool exists to prevent, so a failure part-way
    through has to undo itself."""
    cmake = project / "CMakeLists.txt"
    pyproject = project / "pyproject.toml"
    before = cmake.read_text()

    # Make the second target unwritable, so the pass fails after the first.
    pyproject.chmod(stat.S_IRUSR)
    try:
        finished = run(project, "--major")
        assert finished.returncode == 1
        assert "No files were changed" in finished.stderr
        assert cmake.read_text() == before, "the first file was not rolled back"
    finally:
        pyproject.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_a_missing_optional_file_is_skipped(project, tool):
    (project / "pyproject.toml").unlink()
    targets = tool.collect_targets(str(project))
    assert [target.relpath for target in targets] == ["CMakeLists.txt"]
    assert run(project, "--patch").returncode == 0


def test_a_missing_cmakelists_is_an_error(tmp_path):
    finished = run(tmp_path, "--show")
    assert finished.returncode == 1
    assert "CMakeLists.txt not found" in finished.stderr


def test_an_unparseable_version_is_reported_not_guessed(project):
    (project / "CMakeLists.txt").write_text(
        "project(FastAtoms VERSION not-a-version LANGUAGES CXX)\n", encoding="utf-8")
    finished = run(project, "--show")
    assert finished.returncode == 1
    assert "source of truth" in finished.stderr


def test_the_tool_never_writes_git_history(tool):
    """It writes files and says so. History is the user's to make."""
    source = TOOL.read_text(encoding="utf-8")
    # The only git it runs is `describe` and `log`, both read-only.
    for forbidden in ("commit", "tag'", "push"):
        assert f"'{forbidden}" not in source.replace("last_tag", "").replace(
            "--tags", ""), f"the tool appears to run git {forbidden}"
    assert "'describe'" in source and "'log'" in source
