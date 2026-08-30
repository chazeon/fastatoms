#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export a trained potential to a TorchScript archive FastAtoms can run in C++.

    $CONDA_PREFIX/bin/python -m fastatoms.mlip.export <checkpoint> -o model.pt

A checkpoint like ``MACE-OFF23_small.model`` is a pickled Python object: it
needs the ``mace`` package, and therefore a Python interpreter, to be loaded at
all. This script is run once, inside whichever environment has that package,
and writes a self-contained TorchScript archive. Afterwards C++ loads the
archive through LibTorch and no interpreter is involved in the simulation.

The archive follows one small ABI, so the C++ side is model-agnostic::

    forward(Dict[str, Tensor]) -> Dict[str, Tensor]

      in   positions      (n, 3)  angstrom
           atomic_numbers (n,)    int64
           cell           (3, 3)  rows are the lattice vectors
           edge_index     (2, m)  int64, receivers then senders
           unit_shifts    (m, 3)  integer cell offsets of the senders
      out  energy         ()      eV
           forces         (n, 3)  eV/angstrom   (optional)
           stress         (3, 3)  eV/angstrom^3 (optional)

plus attributes describing the model: ``cutoff``, ``atomic_numbers``,
``provides_forces``, ``provides_stress``, ``family`` and ``fastatoms_abi``.
Anything the model does not return, FastAtoms differentiates out of the energy
in C++, so an energy-only model is a perfectly good citizen here.

Adding a family means writing one adapter module and registering it below.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch  # noqa: E402

ABI_VERSION = 1
METADATA_FILE = "fastatoms_mlip.json"


# ---------------------------------------------------------------------------
# MACE
# ---------------------------------------------------------------------------
class MACEAdapter(torch.nn.Module):
    """The FastAtoms ABI on top of a MACE model.

    MACE wants one-hot species, cartesian shift vectors and the batch pointers
    of a single-graph batch. All of that is derived here from the five tensors
    the ABI defines, so the C++ side never learns anything MACE-specific.

    MACE differentiates its own energy for forces and stress, so those come
    back directly and FastAtoms does not differentiate a second time.
    """

    def __init__(self, model: torch.nn.Module, head: int = -1, energy_only: bool = False):
        super().__init__()
        self.model = model
        # With `energy_only` the model is asked for nothing but the energy and
        # FastAtoms differentiates it in C++ instead. That is the path every
        # model that reports no forces of its own takes, and exporting MACE
        # this way is how the test suite checks it: the same model, once
        # reporting its own forces and once differentiated by LibTorch, must
        # give the same answer.
        self.energy_only: bool = energy_only
        self.register_buffer("cutoff", model.r_max.double().reshape(()))
        self.register_buffer("atomic_numbers", model.atomic_numbers.long())

        # A dense z -> column table, so the one-hot encoding is a gather rather
        # than a search and TorchScript can see straight through it.
        table = torch.zeros(120, dtype=torch.long)
        for column, z in enumerate(model.atomic_numbers.tolist()):
            table[int(z)] = column
        self.register_buffer("z_index", table)
        self.n_species: int = int(model.atomic_numbers.numel())

        heads = list(getattr(model, "heads", ["Default"]))
        chosen = head if head >= 0 else len(heads) - 1
        self.register_buffer("head_index", torch.tensor([chosen], dtype=torch.long))

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        positions = data["positions"]
        numbers = data["atomic_numbers"]
        cell = data["cell"]
        edge_index = data["edge_index"]
        unit_shifts = data["unit_shifts"]

        n = positions.shape[0]
        columns = self.z_index[numbers]
        node_attrs = torch.zeros(n, self.n_species, dtype=positions.dtype,
                                 device=positions.device)
        node_attrs.scatter_(1, columns.unsqueeze(-1), 1.0)

        inputs: Dict[str, torch.Tensor] = {
            "positions": positions,
            "node_attrs": node_attrs,
            "edge_index": edge_index,
            "shifts": torch.matmul(unit_shifts, cell),
            "unit_shifts": unit_shifts,
            "cell": cell,
            "batch": torch.zeros(n, dtype=torch.long, device=positions.device),
            "ptr": torch.tensor([0, n], dtype=torch.long, device=positions.device),
            "head": self.head_index,
        }
        want = not self.energy_only
        out = self.model(inputs, training=False, compute_force=want,
                         compute_virials=False, compute_stress=want,
                         compute_displacement=want)

        result: Dict[str, torch.Tensor] = {}
        energy = out["energy"]
        result["energy"] = energy.reshape(()) if energy is not None else torch.zeros(
            (), dtype=positions.dtype, device=positions.device)
        if not self.energy_only:
            forces = out["forces"]
            result["forces"] = forces if forces is not None else torch.zeros_like(positions)
            stress = out["stress"]
            result["stress"] = stress[0] if stress is not None else torch.zeros(
                3, 3, dtype=positions.dtype, device=positions.device)
        return result


def export_mace(path: str, dtype: str, head: int, energy_only: bool = False):
    from e3nn.util import jit

    model = torch.load(path, map_location="cpu")
    model = model.double() if dtype == "float64" else model.float()
    model = model.to("cpu")
    for parameter in model.parameters():
        parameter.requires_grad = False

    adapter = MACEAdapter(model, head=head, energy_only=energy_only)
    # e3nn's helper scripts the equivariant submodules that need their own
    # treatment; the wrapper on top is ordinary TorchScript.
    prepared = jit.compile(adapter)
    scripted = (prepared if isinstance(prepared, torch.jit.ScriptModule)
                else torch.jit.script(prepared))
    return scripted, {"provides_forces": not energy_only,
                      "provides_stress": not energy_only}


# ---------------------------------------------------------------------------
# Families that cannot be exported (yet), with a useful explanation
# ---------------------------------------------------------------------------
def export_unsupported(family: str, reason: str):
    def failing(path: str, dtype: str, head: int, energy_only: bool = False):
        raise SystemExit(
            f"{family}: {reason}\n"
            f"Use the worker backend instead, which drives the package directly:\n"
            f"    atoms.calc = fa.{family.capitalize()}(model={path!r})   "
            f"# backend='worker' is the default for this family"
        )

    return failing


EXPORTERS = {
    "mace": export_mace,
    "mattersim": export_unsupported(
        "mattersim",
        "M3GNet builds its convolution stack with functools.partial, which "
        "torch.jit.script cannot compile ('Can't get source for functools.partial')",
    ),
    "chgnet": export_unsupported(
        "chgnet", "the released model is not TorchScript-scriptable"),
}


def detect_family(path: str) -> str:
    name = os.path.basename(path).lower()
    if "mace" in name:
        return "mace"
    if "mattersim" in name:
        return "mattersim"
    if "chgnet" in name:
        return "chgnet"
    return "mace"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="the trained model to convert")
    parser.add_argument("-o", "--output", default="", help="archive to write")
    parser.add_argument("--family", default="", choices=["", *sorted(EXPORTERS)],
                        help="model family (guessed from the file name by default)")
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    parser.add_argument("--head", type=int, default=-1,
                        help="which head of a multi-head model to export (default: the last)")
    parser.add_argument("--energy-only", action="store_true",
                        help="export only the energy, leaving forces and stress to "
                             "FastAtoms' own automatic differentiation")
    parser.add_argument("--check", action="store_true",
                        help="reload the archive and evaluate it on a small test structure")
    args = parser.parse_args(argv[1:])

    family = args.family or detect_family(args.checkpoint)
    exporter = EXPORTERS.get(family)
    if exporter is None:
        raise SystemExit(f"unknown family '{family}'; known: {', '.join(sorted(EXPORTERS))}")

    if not os.path.exists(args.checkpoint) and family == "mace":
        raise SystemExit(f"no such checkpoint: {args.checkpoint}")

    output = args.output or (os.path.splitext(args.checkpoint)[0] + ".fastatoms.pt")
    print(f"exporting {family} model {args.checkpoint} -> {output}", file=sys.stderr)

    scripted, capabilities = exporter(args.checkpoint, args.dtype, args.head,
                                     args.energy_only)

    # The metadata the C++ backend reads back off the archive. Plain attributes
    # set on a ScriptModule after the fact do not survive `save`, so this rides
    # along as an extra file, which is TorchScript's own mechanism for exactly
    # this and needs no cooperation from the scripted code.
    metadata = {
        "fastatoms_abi": ABI_VERSION,
        "family": family,
        "model": os.path.basename(args.checkpoint),
        "model_version": _package_version(family),
        "dtype": args.dtype,
        "provides_forces": capabilities["provides_forces"],
        "provides_stress": capabilities["provides_stress"],
    }
    if hasattr(scripted, "cutoff"):
        metadata["cutoff"] = float(scripted.cutoff)
    if hasattr(scripted, "atomic_numbers"):
        metadata["species"] = [int(z) for z in scripted.atomic_numbers.tolist()]

    scripted.save(output, _extra_files={METADATA_FILE: json.dumps(metadata).encode()})
    size = os.path.getsize(output) / 1e6
    print(f"wrote {output} ({size:.1f} MB)", file=sys.stderr)

    if args.check:
        _check(output, family)
    return 0


def _package_version(family: str) -> str:
    try:
        import importlib

        return str(getattr(importlib.import_module(family), "__version__", ""))
    except Exception:  # noqa: BLE001
        return ""


def _check(path: str, family: str) -> None:
    """Reload the archive the way C++ will and evaluate a two-atom molecule."""
    files = {METADATA_FILE: ""}
    module = torch.jit.load(path, _extra_files=files)
    metadata = json.loads(files[METADATA_FILE])
    print(f"check: metadata {metadata}", file=sys.stderr)
    cutoff = float(module.cutoff) if hasattr(module, "cutoff") else 5.0
    species = module.atomic_numbers.tolist() if hasattr(module, "atomic_numbers") else [1]
    z = int(species[0])

    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]], dtype=torch.float64)
    data = {
        "positions": positions,
        "atomic_numbers": torch.tensor([z, z], dtype=torch.long),
        "cell": torch.eye(3, dtype=torch.float64) * 20.0,
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "unit_shifts": torch.zeros(2, 3, dtype=torch.float64),
    }
    out = module(data)
    forces = ("fmax %.6f eV/A" % float(out["forces"].detach().abs().max())
              if "forces" in out else "energy only (FastAtoms differentiates)")
    print(f"check: cutoff {cutoff} A, {len(species)} species, "
          f"energy {float(out['energy'].detach()):.6f} eV, {forces}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
