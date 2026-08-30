# SPDX-License-Identifier: MIT
"""Machine-learning potential support: the model exporter and the worker.

Neither module imports :mod:`fastatoms`. They are meant to be run inside the
environment that has the potential installed - ``mace_env`` for MACE,
``mlip_env`` for MatterSim, CHGNet, NequIP and Allegro - which is generally not
the environment FastAtoms itself lives in.

    # once, to get an archive C++ can run without Python
    mace_env/bin/python -m fastatoms.mlip.export MACE-OFF23_small.model -o mace.pt

    # or check that an environment can serve a model at all
    mlip_env/bin/python -m fastatoms.mlip.worker --family mattersim \
        --model MatterSim-v1.0.0-1M.pth --probe
"""

__all__ = ["export", "worker"]
