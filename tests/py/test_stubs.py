"""The committed ``_core.pyi`` matches what pybind11-stubgen generates from the built module.

The committed stub is hand-tidied (concrete numpy types, the nested ``symmetry``
namespace), so exact text equality would be brittle. Instead the test compares
the public names and the parameter names of every function, which is what a
stale stub gets wrong when the C++ bindings change.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
COMMITTED = REPO / "bindings" / "python" / "fastatoms" / "_core.pyi"


def _params(node: ast.FunctionDef) -> tuple[str, ...]:
    args = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
    if node.args.vararg is not None:
        args.append("*" + node.args.vararg.arg)
    if node.args.kwarg is not None:
        args.append("**" + node.args.kwarg.arg)
    return tuple(a for a in args if a != "self")


def _walk(body: list[ast.stmt], prefix: str, out: dict[str, tuple[str, ...]]) -> None:
    for node in body:
        if isinstance(node, ast.ClassDef):
            out[prefix + node.name] = ()
            _walk(node.body, prefix + node.name + ".", out)
        elif isinstance(node, ast.FunctionDef):
            # Setter argument names are pybind11's `arg1`; only the property matters.
            is_setter = any(isinstance(d, ast.Attribute) and d.attr == "setter" for d in node.decorator_list)
            out[prefix + node.name] = ("<setter>",) if is_setter else _params(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith("__"):
                out[prefix + node.target.id] = ()


def _public_api(stub_text_by_prefix: dict[str, str]) -> dict[str, tuple[str, ...]]:
    api: dict[str, tuple[str, ...]] = {}
    for prefix, text in stub_text_by_prefix.items():
        _walk(ast.parse(text).body, prefix, api)
    # The submodule itself is a namespace, not a member.
    api.pop("symmetry", None)
    return api


def _generate(tmp: Path) -> dict[str, str]:
    subprocess.run(
        [sys.executable, "-m", "pybind11_stubgen", "--ignore-all-errors", "-o", str(tmp), "fastatoms._core"],
        check=True,
        capture_output=True,
        cwd=REPO,
    )
    pkg = tmp / "fastatoms" / "_core"
    return {
        "": (pkg / "__init__.pyi").read_text(),
        "symmetry.": (pkg / "symmetry.pyi").read_text(),
    }


def test_committed_stub_matches_the_built_module(tmp_path: Path) -> None:
    pytest.importorskip("pybind11_stubgen")
    import fastatoms

    if not hasattr(fastatoms, "symmetry"):
        pytest.skip("symmetry not compiled; the stub covers the full build")

    generated = _public_api(_generate(tmp_path))
    committed = _public_api({"": COMMITTED.read_text()})
    # The committed nested namespace is `symmetry.X`; the generated file is a module
    # with the same members, so both sides already use the same prefix.
    committed_sym = {k: v for k, v in committed.items() if k.startswith("symmetry.")}
    generated_sym = {k: v for k, v in generated.items() if k.startswith("symmetry.")}
    assert set(committed) == set(generated), (
        f"names only in committed: {sorted(set(committed) - set(generated))}; "
        f"names only in generated: {sorted(set(generated) - set(committed))}"
    )
    assert committed_sym == generated_sym
    for name, params in generated.items():
        assert committed[name] == params, f"parameters differ for {name}"
