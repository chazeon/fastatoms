// SPDX-License-Identifier: MIT
#pragma once

/// \file yaml_input.hpp
/// \brief Reading structures and run settings from YAML.
///
/// Expected document layout (every section is optional except `structure`):
///
/// \code{.yaml}
/// structure:
///   cell: [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
///   # or:  cell: {a: 5.43, b: 5.43, c: 5.43, alpha: 90, beta: 90, gamma: 90}
///   # or:  cell: 10.0            (cubic)
///   # or:  cell: [10.0, 10.0, 10.0]  (orthorhombic)
///   pbc: [true, true, true]      # or a single bool, or "xyz" / "xy" / ""
///   coordinates: cartesian       # or "scaled" / "fractional" / "direct"
///   atoms:
///     - {symbol: O, position: [0.0, 0.0, 0.0]}
///     - {symbol: H, position: [0.96, 0.0, 0.0], velocity: [0, 0, 0]}
///   # or the columnar form:
///   # symbols:   [O, H, H]
///   # positions: [[...], [...], [...]]
///   # masses / charges / tags / momenta / velocities are optional arrays
///
/// simulation:
///   task: md
///   calculator: {name: lennard_jones, parameters: {epsilon: 0.0103, sigma: 3.4}}
///   md: {ensemble: nvt, timestep: 1.0, steps: 1000, temperature: 300.0}
///   optimizer: {algorithm: bfgs, fmax: 0.02, max_steps: 500}
///   output: {trajectory: traj.xyz, interval: 10}
/// \endcode

#include <string>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/io/parameters.hpp"

namespace fastatoms::io {

/// A parsed input document.
struct InputDeck {
    Atoms atoms{};
    SimulationParameters simulation{};
    /// The complete document, including any sections FastAtoms does not
    /// interpret itself.
    Value document{};
};

/// Parse a YAML file. \throws InputError on a missing file or invalid content.
[[nodiscard]] InputDeck read_yaml_file(const std::string& path);

/// Parse a YAML document held in memory.
[[nodiscard]] InputDeck read_yaml_string(const std::string& text);

/// Parse just a `structure:` subtree into an `Atoms` object.
[[nodiscard]] Atoms atoms_from_value(const Value& structure);

/// Parse just a `simulation:` subtree.
[[nodiscard]] SimulationParameters simulation_from_value(const Value& simulation);

/// Convert an arbitrary YAML document into a `Value` tree.
[[nodiscard]] Value parse_yaml_string(const std::string& text);
[[nodiscard]] Value parse_yaml_file(const std::string& path);

/// Serialise a structure back to the `structure:` layout above.
///
/// `scaled` writes fractional instead of Cartesian coordinates. Optional
/// arrays are only emitted when present.
[[nodiscard]] std::string write_yaml_string(const Atoms& atoms, bool scaled = false);

/// As `write_yaml_string`, but writes to `path`.
/// \throws InputError if the file cannot be opened.
void write_yaml_file(const std::string& path, const Atoms& atoms, bool scaled = false);

}  // namespace fastatoms::io
