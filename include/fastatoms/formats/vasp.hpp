// SPDX-License-Identifier: MIT
#pragma once

/// \file vasp.hpp
/// \brief POSCAR / CONTCAR (read + write) and XDATCAR (read).
///
/// See PLAN.md §4 "VASP" for every edge case the reader must accept. In short:
/// VASP 4 and 5+, labels such as Fe1 / Fe_pv / Fe/hash on the species line
/// (kept in "labels", element via `number_from_label`), negative (volume) and
/// three-component scale factors, selective dynamics per axis, Direct /
/// Cartesian keyed on the first non-blank letter, an optional velocity block,
/// CRLF, Fortran `D` exponents.

#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/core/atoms.hpp"

namespace fastatoms::formats {

struct PoscarReadOptions {
    /// Species for a VASP 4 file (no species line), one entry per count.
    /// Ignored when the file has a species line. Option key: "species".
    std::vector<std::string> species{};
};

struct PoscarWriteOptions {
    bool direct{true};              ///< "direct"
    bool write_labels{true};        ///< "labels": species line from "labels", else symbols
    bool vasp5{true};               ///< "vasp5": false omits the species line
    bool sort{false};               ///< "sort": group by label in first-appearance order;
                                    ///< false groups only consecutive runs (keeps order)
    std::string comment{};          ///< "comment": empty -> info["comment"] or the formula
};

[[nodiscard]] Atoms read_poscar(std::string_view text, const PoscarReadOptions& options = {});
[[nodiscard]] std::string write_poscar(const Atoms& atoms, const PoscarWriteOptions& options = {});

/// Every frame of an XDATCAR, fixed-cell and variable-cell layouts.
[[nodiscard]] std::vector<Atoms> read_xdatcar(std::string_view text,
                                              const PoscarReadOptions& options = {});

}  // namespace fastatoms::formats
