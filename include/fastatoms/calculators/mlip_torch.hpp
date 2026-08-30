// SPDX-License-Identifier: MIT
#pragma once

/// \file mlip_torch.hpp
/// \brief The in-process backend: a TorchScript model evaluated through the
///        PyTorch C++ API, with no interpreter anywhere near the loop.
///
/// This header deliberately mentions no LibTorch type. Whether FastAtoms was
/// built with torch is a build-time detail that must not leak into the public
/// API, so everything torch-shaped lives behind `make_torch_backend`.

#include <memory>
#include <string>

#include "fastatoms/calculators/mlip.hpp"

namespace fastatoms::mlip {

/// Load a TorchScript model and evaluate it in this process.
///
/// The archive must follow the FastAtoms MLIP ABI: a module whose `forward`
/// takes `Dict[str, Tensor]` with keys `positions`, `atomic_numbers`, `cell`,
/// `edge_index` and `unit_shifts`, and returns a `Dict[str, Tensor]` with an
/// `energy` and optionally `forces` and `stress`. `python/fastatoms/mlip/`
/// contains the exporter that wraps MACE and friends into that shape.
///
/// Whatever the model does not return is obtained here by differentiating the
/// energy: forces from the gradient with respect to the positions, stress from
/// the gradient with respect to a strain applied to both the positions and the
/// cell.
///
/// \throws MLIPError if FastAtoms was built without LibTorch, if the file is
///         not a TorchScript archive, or if the model does not follow the ABI.
[[nodiscard]] std::shared_ptr<Backend> make_torch_backend(const std::string& family,
                                                          const std::string& model,
                                                          const Value& parameters);

}  // namespace fastatoms::mlip
