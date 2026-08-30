// SPDX-License-Identifier: MIT
#pragma once

/// \file mlip_worker.hpp
/// \brief The out-of-process backend: one long-lived Python interpreter, fed
///        over pipes.
///
/// Not every potential can be exported to TorchScript. MatterSim's M3GNet, for
/// one, builds its convolution stack with `functools.partial`, which
/// `torch.jit.script` cannot see through. For those models the choice is
/// between reimplementing their data pipeline and calling the package that
/// already has one - and calling it is both cheaper and more likely to be
/// right.
///
/// What this must not become is a `FileIOCalculator`: writing an input file,
/// starting an interpreter and parsing an output file costs seconds, and an MD
/// run needs milliseconds. So the interpreter is started once, imports its
/// model once, and then answers a binary request per evaluation over a pipe.
/// The per-step overhead is two `write`s and two `read`s of packed doubles.

#include <memory>
#include <string>

#include "fastatoms/calculators/mlip.hpp"

namespace fastatoms::mlip {

/// Start a worker for `model` and talk to it over pipes.
///
/// The interpreter is chosen in this order: the `python` parameter, the
/// `conda_env` parameter resolved against the usual conda locations, then
/// whatever `python3` is on PATH. The worker script itself is found from the
/// `worker` parameter, then `$FASTATOMS_MLIP_WORKER`, then the installed or
/// in-tree copy of `fastatoms/mlip/worker.py`.
///
/// \throws MLIPError if the interpreter, the script, or the model is missing,
///         or if the worker fails to load the model.
[[nodiscard]] std::shared_ptr<Backend> make_worker_backend(const std::string& family,
                                                           const std::string& model,
                                                           const Value& parameters);

/// Absolute path of the interpreter for a conda environment name, or an empty
/// string when no such environment can be found.
///
/// Looks under `$CONDA_PREFIX/../<name>`, `$CONDA_ROOT/envs/<name>`,
/// `~/miniconda3/envs/<name>`, `~/anaconda3/envs/<name>`, `~/mambaforge`,
/// `~/miniforge3` and `~/micromamba`. A name that is already a path is used
/// as it stands.
[[nodiscard]] std::string conda_python(const std::string& environment);

/// Path of the worker script that `make_worker_backend` would use.
[[nodiscard]] std::string worker_script_path();

}  // namespace fastatoms::mlip
