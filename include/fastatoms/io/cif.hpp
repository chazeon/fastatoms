#pragma once

/// Crystallographic Information File.
///
/// The format crystallography actually publishes in: a structure is given as
/// an asymmetric unit plus the symmetry that generates the rest of it, so
/// reading one means expanding the operations rather than just listing atoms.
/// That expansion is the whole job - a CIF read as if it were a coordinate
/// list gives a structure missing most of its atoms.

#include <iosfwd>
#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/io/extxyz.hpp"

namespace fastatoms::io {

/// How to write a CIF.
struct CifOptions {
    /// The `data_` block name. Empty means the chemical formula.
    std::string name{};

    /// Write positions as fractions of the cell, which is what the format is
    /// for. Cartesian CIFs exist but nothing else reads them reliably.
    bool fractional{true};

    int precision{8};
};

/// Read a CIF, expanding its symmetry.
///
/// The symmetry comes from the file's own `_space_group_symop_operation_xyz`
/// loop (or its older `_symmetry_equiv_pos_as_xyz` spelling), which is what
/// every published CIF carries. A file that names only a space-group number
/// and leaves the operations to be looked up is read when that number is 1,
/// and otherwise refused rather than silently returned as an asymmetric unit
/// pretending to be a structure.
///
/// Atoms generated onto the same site by different operations are merged, with
/// `tolerance` in fractional units deciding what "the same site" means.
[[nodiscard]] Atoms read_cif(const std::string& path, double tolerance = 1e-4);
[[nodiscard]] Atoms parse_cif(std::istream& stream, const std::string& origin = "<string>",
                              double tolerance = 1e-4);

/// Every `data_` block in a multi-structure CIF.
[[nodiscard]] std::vector<Atoms> read_cif_frames(const std::string& path,
                                                 double tolerance = 1e-4);

/// Write a CIF.
///
/// Always in P1 with every atom listed: the structure here is a set of
/// positions, and claiming a symmetry it has not been checked for would be
/// asserting something unverified. Run the structure through the symmetry
/// module first if the higher-symmetry description is what is wanted.
void write_cif(const std::string& path, const Atoms& atoms, CifOptions options = {});
[[nodiscard]] std::string cif_to_string(const Atoms& atoms, CifOptions options = {});

}  // namespace fastatoms::io
