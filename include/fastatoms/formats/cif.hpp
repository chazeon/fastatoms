// SPDX-License-Identifier: MIT
#pragma once

/// \file cif.hpp
/// \brief CIF (read via gemmi, write P1).
///
/// Reading parses with gemmi, expands the asymmetric unit with the symmetry
/// operations (the symop loop, else gemmi's table from the H-M / Hall symbol
/// or IT number), merges images closer than `merge_tolerance`, keeps labels
/// ("labels") and occupancies ("occupancy"), and builds the cell with the same
/// convention as `Cell::from_parameters` (a along x, b in the xy plane).

#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/core/atoms.hpp"

namespace fastatoms::formats {

enum class CifDisorder {
    KeepAll,       ///< "keep_all": every site, partial occupancy kept as data
    MaxOccupancy,  ///< "max_occupancy": one atom per position, the most occupied
    Error,         ///< "error": throw on any occupancy < 1
};

struct CifReadOptions {
    CifDisorder disorder{CifDisorder::KeepAll};  ///< "disorder"
    double merge_tolerance{0.01};               ///< "merge_tolerance", Cartesian distance in Å: images within this are one atom
};

/// One structure per data block that has atom sites.
[[nodiscard]] std::vector<Atoms> read_cif(std::string_view text,
                                          const CifReadOptions& options = {});

/// P1 CIF with the labels as _atom_site_label.
[[nodiscard]] std::string write_cif(const Atoms& atoms);

}  // namespace fastatoms::formats
