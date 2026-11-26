// SPDX-License-Identifier: MIT
#pragma once

/// \file xyz.hpp
/// \brief Plain XYZ and extended XYZ (read + write, multi-frame).
///
/// Plain XYZ: no cell, pbc all false, the comment line kept verbatim in
/// info["comment"] and never parsed as key=value (as ASE's ase.io.xyz reader).
/// Extended XYZ: Lattice (pbc defaults to T T T, `pbc=` overrides),
/// Properties columns -> positions / arrays (string columns kept as string
/// arrays, one space-joined value per atom when a column has several tokens;
/// a `species` column also fills "labels"), other key=value pairs -> info.
/// Number formatting on write is shortest round-trip.

#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/core/atoms.hpp"

namespace fastatoms::io {

struct XyzReadOptions {
    bool extended{true};  ///< false reads plain XYZ: the comment line is never parsed
};

[[nodiscard]] std::vector<Atoms> read_xyz(std::string_view text,
                                          const XyzReadOptions& options = {});

struct XyzWriteOptions {
    bool extended{true};   ///< false writes plain XYZ (symbols + positions, comment line)
    int precision{0};      ///< 0 = shortest round-trip, 1 to 17 significant digits; else FormatError
};

[[nodiscard]] std::string write_xyz(const Atoms& atoms, const XyzWriteOptions& options = {});
[[nodiscard]] std::string write_xyz(const std::vector<Atoms>& frames,
                                    const XyzWriteOptions& options = {});

}  // namespace fastatoms::io
