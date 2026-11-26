// SPDX-License-Identifier: MIT
#pragma once

/// \file lammps.hpp
/// \brief LAMMPS data files (read + write) and text dumps (read).
///
/// See PLAN.md §4 "LAMMPS" for the full checklist: restricted triclinic
/// (xy xz yz) and general triclinic (avec/bvec/cvec + abc origin) boxes, atom
/// styles, the type -> element map priority (explicit `types` > `Atom Type
/// Labels` section > Masses comment whose element agrees with the mass >
/// mass match > dummy X), image flags, dump BOX BOUNDS with tilt-inclusive
/// bounds, x/xs/xu/xsu coordinate columns, element column, id sorting.
/// Type columns may hold type labels. A truncated last frame is dropped and
/// the last good frame gets info["truncated"] = true.
///
/// A type that nothing resolves (not in `types`, and no label, Masses comment
/// or mass that names an element) becomes the dummy element X (Z = 0) labelled
/// by its type number, e.g. "2". Errors are reserved for bad input: a `types`
/// entry that names no element, or a Masses comment whose element disagrees
/// with the stated mass.

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/core/atoms.hpp"

namespace fastatoms::io {

/// LAMMPS type -> label. The label sets "labels" and, through
/// `number_from_label`, the element ("Fe1" and "Fe2" may both be iron).
/// Option key "types": a map {"1": "Fe", ...} or a list ["Fe", "O"]
/// (list entry k is type k+1). A partial map is fine: it resolves the types
/// it names, and the file (data) or the dummy X (dump) resolves the rest.
using LammpsTypeMap = std::map<int, std::string>;

struct LammpsDataReadOptions {
    std::string atom_style{};   ///< "atom_style": empty = the `Atoms # style` hint, else
                                ///< guessed from the field count (as ASE)
    LammpsTypeMap types{};      ///< "types"
    bool unwrap{false};         ///< "unwrap": apply image flags to positions
    std::string units{"metal"}; ///< "units": the LAMMPS unit style of the file's masses,
                                ///< used only to infer elements from masses (raw masses
                                ///< are kept as written). lj has no mass scale, so it
                                ///< disables mass inference. Others: real, metal, si,
                                ///< cgs, electron, micro, nano.
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
    LammpsTypeMap types{};  ///< "types"; an entry decides before any element column;
                            ///< without one and without an element column,
                            ///< type k becomes the dummy element X with label "k"
    bool unwrap{false};     ///< "unwrap": apply the ix iy iz image flags to x, y, z
                            ///< columns. Off (default), x, y, z are taken as written, as
                            ///< ASE does; xu and xsu are unwrapped whatever this is
};

[[nodiscard]] std::vector<Atoms> read_lammps_dump(std::string_view text,
                                                  const LammpsDumpReadOptions& options = {});

}  // namespace fastatoms::io
