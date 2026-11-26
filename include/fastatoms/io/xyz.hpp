// SPDX-License-Identifier: MIT
#pragma once

/// \file xyz.hpp
/// \brief Plain XYZ and extended XYZ (read + write, multi-frame).
///
/// Plain XYZ: no cell, pbc all false, comment kept in info["comment"].
/// Extended XYZ: Lattice (pbc defaults to T T T, `pbc=` overrides),
/// Properties columns -> positions / arrays (string columns kept as string
/// arrays; a `species` column also fills "labels"), other key=value pairs ->
/// info. Number formatting on write is shortest round-trip.

#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/core/atoms.hpp"

namespace fastatoms::io {

[[nodiscard]] std::vector<Atoms> read_xyz(std::string_view text);

struct XyzWriteOptions {
    bool extended{true};   ///< false writes plain XYZ (symbols + positions, comment line)
    int precision{0};      ///< "precision": 0 = shortest round-trip
};

[[nodiscard]] std::string write_xyz(const Atoms& atoms, const XyzWriteOptions& options = {});
[[nodiscard]] std::string write_xyz(const std::vector<Atoms>& frames,
                                    const XyzWriteOptions& options = {});

}  // namespace fastatoms::io
