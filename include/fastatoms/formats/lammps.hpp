// SPDX-License-Identifier: MIT
#pragma once

/// \file lammps.hpp
/// \brief LAMMPS data files (read + write) and text dumps (read).
///
/// See PLAN.md §4 "LAMMPS" for the full checklist: restricted triclinic
/// (xy xz yz) and general triclinic (avec/bvec/cvec + abc origin) boxes, atom
/// styles, the type -> element map priority (explicit `types` > `Atom Type
/// Labels` section > Masses comment whose element agrees with the mass >
/// mass match > error), image flags, dump BOX BOUNDS with tilt-inclusive
/// bounds, x/xs/xu/xsu coordinate columns, element column, id sorting.
/// Type columns may hold type labels. A truncated last frame is dropped and
/// the last good frame gets info["truncated"] = true.

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/core/atoms.hpp"

namespace fastatoms::formats {

/// LAMMPS type -> label. The label sets "labels" and, through
/// `number_from_label`, the element ("Fe1" and "Fe2" may both be iron).
/// Option key "types": a map {"1": "Fe", ...} or a list ["Fe", "O"]
/// (list entry k is type k+1).
using LammpsTypeMap = std::map<int, std::string>;

struct LammpsDataReadOptions {
    std::string atom_style{};   ///< "atom_style": empty = the `Atoms # style` hint, else
                                ///< guessed from the field count (as ASE)
    LammpsTypeMap types{};      ///< "types"
    bool unwrap{false};         ///< "unwrap": apply image flags to positions
};

struct LammpsDataWriteOptions {
    std::string atom_style{"atomic"};      ///< "atom_style": atomic | charge | full
    std::vector<std::string> specorder{};  ///< "specorder": labels (or symbols) in type order;
                                           ///< empty = the "type" array, else first appearance
    bool masses{true};                     ///< "masses": write a Masses section
    bool force_skew{false};                ///< "force_skew": write xy xz yz even if zero
};

[[nodiscard]] Atoms read_lammps_data(std::string_view text,
                                     const LammpsDataReadOptions& options = {});
[[nodiscard]] std::string write_lammps_data(const Atoms& atoms,
                                            const LammpsDataWriteOptions& options = {});

struct LammpsDumpReadOptions {
    LammpsTypeMap types{};  ///< "types"; without it and without an element column,
                            ///< type k becomes the dummy element X with label "k"
    bool unwrap{true};      ///< "unwrap": prefer xu/xsu, or apply ix iy iz
};

[[nodiscard]] std::vector<Atoms> read_lammps_dump(std::string_view text,
                                                  const LammpsDumpReadOptions& options = {});

}  // namespace fastatoms::formats
