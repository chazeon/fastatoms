#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""A long-lived worker that evaluates a machine-learning potential.

FastAtoms starts this script once, in whichever Python environment has the
model's package installed, and then sends it one binary request per structure
over a pipe. Nothing is written to disk and no interpreter is started again, so
the cost of a step is a pipe round trip plus the model itself.

It is deliberately standalone: it imports no ``fastatoms`` module, because the
environment it runs in - ``mace_env``, ``mlip_env`` - has the potential
installed but not necessarily anything else.

Protocol (little-endian, native sizes), all of it on stdin/stdout:

  greeting  "FAMH" int32 version int32 length  <length bytes of JSON>
  request   "FAMR" int32 version int32 natoms
            uint8 need_forces uint8 need_stress uint8 pbc[3]
            float64 positions[3 * natoms]
            int64   numbers[natoms]
            float64 cell[9]
  reply     "FAMS" int32 status
            status 0: uint8 has_forces uint8 has_stress
                      float64 energy
                      float64 forces[3 * natoms]   (if has_forces)
                      float64 stress[6]            (if has_stress, Voigt)
            status 1: int32 length <length bytes of message>

Run it by hand to check an environment::

    $CONDA_PREFIX/bin/python worker.py --family mace --model small --probe
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import traceback

MAGIC_HELLO = b"FAMH"
MAGIC_REQUEST = b"FAMR"
MAGIC_REPLY = b"FAMS"
PROTOCOL_VERSION = 1


# ---------------------------------------------------------------------------
# The binary channel
# ---------------------------------------------------------------------------
def take_stdout() -> "object":
    """Claim file descriptor 1 for the protocol and send prints to stderr.

    Model packages are chatty - MACE prints its default dtype, MatterSim logs
    the checkpoint it loaded - and a single stray byte on stdout would desync
    the stream. So the real stdout is duplicated away and fd 1 is pointed at
    stderr, where anything printed becomes diagnostics instead of corruption.
    """
    channel_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(channel_fd, "wb", buffering=0)


def read_exact(stream, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("the driver closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Loading a model
# ---------------------------------------------------------------------------
def load_mace(model: str, device: str, dtype: str, options: dict):
    """MACE, from a .model checkpoint or one of the published names."""
    from mace.calculators import MACECalculator

    dtype = dtype if dtype in ("float32", "float64") else "float64"
    if os.path.exists(model):
        calc = MACECalculator(model_paths=model, device=device,
                              default_dtype=dtype, **options)
        return calc, model

    name = model.lower().replace("_", "-")
    if name.startswith("off"):
        from mace.calculators import mace_off

        calc = mace_off(model=name.replace("off-", "") or "small", device=device,
                        default_dtype=dtype, **options)
    elif name.startswith("anicc"):
        from mace.calculators import mace_anicc

        calc = mace_anicc(device=device, **options)
    else:
        from mace.calculators import mace_mp

        calc = mace_mp(model=model, device=device, default_dtype=dtype, **options)
    return calc, model


def load_mattersim(model: str, device: str, dtype: str, options: dict):
    """MatterSim's M3GNet. Cannot be TorchScripted, which is why this exists."""
    from mattersim.forcefield import MatterSimCalculator

    load_path = model or "MatterSim-v1.0.0-1M.pth"
    calc = MatterSimCalculator(load_path=load_path, device=device, **options)
    return calc, load_path


def load_chgnet(model: str, device: str, dtype: str, options: dict):
    from chgnet.model.dynamics import CHGNetCalculator

    if model and os.path.exists(model):
        from chgnet.model import CHGNet

        calc = CHGNetCalculator(model=CHGNet.from_file(model), use_device=device, **options)
    else:
        calc = CHGNetCalculator(use_device=device, **options)
    return calc, model or "chgnet-pretrained"


def load_nequip(model: str, device: str, dtype: str, options: dict):
    """NequIP and Allegro both deploy through the nequip package."""
    from nequip.ase import NequIPCalculator

    for factory in ("from_compiled_model", "from_deployed_model"):
        builder = getattr(NequIPCalculator, factory, None)
        if builder is None:
            continue
        try:
            return builder(model, device=device, **options), model
        except Exception:  # noqa: BLE001 - try the next entry point
            continue
    raise RuntimeError(
        f"could not load '{model}' with nequip {getattr(__import__('nequip'), '__version__', '?')}; "
        "compile it first with `nequip-compile`"
    )


def load_ase(model: str, device: str, dtype: str, options: dict):
    """Any ASE calculator, named as `package.module:ClassName`."""
    module_name, _, class_name = model.partition(":")
    if not class_name:
        raise ValueError("an 'ase' model looks like 'package.module:ClassName'")
    import importlib

    factory = getattr(importlib.import_module(module_name), class_name)
    return factory(**options), model


LOADERS = {
    "mace": load_mace,
    "mattersim": load_mattersim,
    "chgnet": load_chgnet,
    "nequip": load_nequip,
    "allegro": load_nequip,
    "ase": load_ase,
}


def describe(family: str, calculator, path: str) -> dict:
    """What the C++ side wants to know about the loaded model."""
    info = {"family": family, "path": path, "provides_forces": True,
            "provides_stress": True}

    for attribute in ("r_max", "cutoff", "rcut"):
        value = getattr(calculator, attribute, None)
        if value is not None:
            try:
                info["cutoff"] = float(value)
                break
            except (TypeError, ValueError):
                pass

    try:
        import importlib

        package = importlib.import_module(family if family != "allegro" else "allegro")
        info["version"] = str(getattr(package, "__version__", ""))
    except Exception:  # noqa: BLE001 - a version is a nicety, not a requirement
        pass

    species = getattr(calculator, "z_table", None)
    numbers = getattr(species, "zs", None) if species is not None else None
    if numbers is None:
        numbers = getattr(calculator, "atomic_numbers", None)
    if numbers is not None:
        try:
            info["species"] = [int(z) for z in numbers]
        except (TypeError, ValueError):
            pass

    return info


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def serve(channel, stream, calculator) -> None:
    import numpy as np
    from ase import Atoms
    from ase.stress import full_3x3_to_voigt_6_stress

    while True:
        try:
            magic = stream.read(4)
        except (EOFError, OSError):
            return
        if not magic:
            return
        if magic != MAGIC_REQUEST:
            raise RuntimeError(f"bad request magic {magic!r}")

        version, natoms = struct.unpack("<ii", read_exact(stream, 8))
        if version != PROTOCOL_VERSION:
            raise RuntimeError(f"protocol version {version} is not supported")
        need_forces, need_stress, px, py, pz = struct.unpack("<5B", read_exact(stream, 5))

        positions = np.frombuffer(read_exact(stream, 24 * natoms), dtype="<f8").reshape(natoms, 3)
        numbers = np.frombuffer(read_exact(stream, 8 * natoms), dtype="<i8")
        cell = np.frombuffer(read_exact(stream, 72), dtype="<f8").reshape(3, 3)

        try:
            atoms = Atoms(numbers=numbers, positions=positions, cell=cell,
                          pbc=(bool(px), bool(py), bool(pz)))
            atoms.calc = calculator

            energy = float(atoms.get_potential_energy())
            forces = None
            stress = None
            if need_forces:
                forces = np.ascontiguousarray(atoms.get_forces(), dtype="<f8")
            if need_stress and atoms.cell.rank == 3:
                raw = atoms.get_stress()
                raw = np.asarray(raw, dtype="<f8")
                if raw.shape == (3, 3):
                    raw = full_3x3_to_voigt_6_stress(raw)
                stress = np.ascontiguousarray(raw, dtype="<f8")

            reply = bytearray(MAGIC_REPLY)
            reply += struct.pack("<i", 0)
            reply += struct.pack("<2B", forces is not None, stress is not None)
            reply += struct.pack("<d", energy)
            if forces is not None:
                reply += forces.tobytes()
            if stress is not None:
                reply += stress.tobytes()
            channel.write(bytes(reply))
        except Exception:  # noqa: BLE001 - any failure is reported, not fatal
            message = traceback.format_exc().encode("utf-8", "replace")[-8192:]
            reply = bytearray(MAGIC_REPLY)
            reply += struct.pack("<i", 1)
            reply += struct.pack("<i", len(message))
            reply += message
            channel.write(bytes(reply))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="mace")
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--options", default="")
    parser.add_argument("--probe", action="store_true",
                        help="load the model, print what was found, and exit")
    args = parser.parse_args(argv[1:])

    options = {}
    if args.options:
        try:
            options = json.loads(args.options)
        except json.JSONDecodeError:
            options = {}

    loader = LOADERS.get(args.family.lower())
    if loader is None:
        known = ", ".join(sorted(LOADERS))
        if args.probe:
            print(f"unknown family '{args.family}'; known: {known}")
            return 2
        channel = take_stdout()
        payload = json.dumps({"error": f"unknown family '{args.family}'; known: {known}"})
        channel.write(MAGIC_HELLO + struct.pack("<ii", PROTOCOL_VERSION, len(payload)))
        channel.write(payload.encode())
        return 2

    if args.probe:
        calculator, path = loader(args.model, args.device, args.dtype, options)
        print(json.dumps(describe(args.family, calculator, path), indent=2))
        return 0

    channel = take_stdout()
    try:
        calculator, path = loader(args.model, args.device, args.dtype, options)
        payload = json.dumps(describe(args.family, calculator, path))
    except Exception:  # noqa: BLE001 - report the failure through the protocol
        detail = traceback.format_exc()
        sys.stderr.write(detail)
        payload = json.dumps({"error": detail.strip().splitlines()[-1][:400]})
        channel.write(MAGIC_HELLO + struct.pack("<ii", PROTOCOL_VERSION, len(payload)))
        channel.write(payload.encode())
        return 1

    channel.write(MAGIC_HELLO + struct.pack("<ii", PROTOCOL_VERSION, len(payload)))
    channel.write(payload.encode())

    serve(channel, sys.stdin.buffer, calculator)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
