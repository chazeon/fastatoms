// SPDX-License-Identifier: MIT
#pragma once

/// \file fastatoms.hpp
/// \brief Umbrella header: includes the whole public API.

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/core/cell.hpp"
#include "fastatoms/core/elements.hpp"
#include "fastatoms/core/neighbor_list.hpp"
#include "fastatoms/core/units.hpp"
#include "fastatoms/core/vec3.hpp"
#include "fastatoms/io/parameters.hpp"
#include "fastatoms/io/yaml_input.hpp"

/// FastAtoms - a high-performance C++ core for atomistic simulations.
namespace fastatoms {

/// Library version string, e.g. "0.1.0".
[[nodiscard]] const char* version() noexcept;

}  // namespace fastatoms
