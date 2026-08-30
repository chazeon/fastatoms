#pragma once

/// VASP's own file formats.
///
/// POSCAR and CONTCAR hold one structure, XDATCAR a trajectory, OUTCAR the
/// results of a run. Between them they are how most plane-wave work arrives,
/// and reading them here means a VASP calculation can be picked up without a
/// conversion step - including the energies and forces, which come back as a
/// `SinglePointCalculator` the same way an extended XYZ frame's do.

#include <iosfwd>
#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/io/extxyz.hpp"

namespace fastatoms::io {

/// How to write a POSCAR.
struct PoscarOptions {
    /// Positions as fractions of the cell vectors rather than in ångström.
    ///
    /// Direct is what VASP itself writes and what most tooling expects, so it
    /// is the default; Cartesian is available because some workflows diff
    /// against it.
    bool direct{true};

    /// A comment for the first line. Empty means the chemical formula, which
    /// is more use than "POSCAR" when a directory holds a dozen of them.
    std::string comment{};

    /// Write selective dynamics flags from the structure's constraints, so a
    /// `FixAtoms` set up here is still fixed when VASP runs.
    bool selective_dynamics{true};

    int precision{16};
};

/// Read a POSCAR or CONTCAR.
///
/// VASP 5 and later name the species on line 6; VASP 4 did not, and left them
/// to be guessed from the POTCAR. A version-4 file is read, with the species
/// taken from the comment line when that parses as a list of symbols - which
/// is what every code that writes them does - and rejected otherwise rather
/// than guessed at.
[[nodiscard]] Atoms read_poscar(const std::string& path);
[[nodiscard]] Atoms parse_poscar(std::istream& stream, const std::string& origin = "<string>");

/// Write a POSCAR.
void write_poscar(const std::string& path, const Atoms& atoms, PoscarOptions options = {});
[[nodiscard]] std::string poscar_to_string(const Atoms& atoms, PoscarOptions options = {});

/// Read every frame of an XDATCAR.
///
/// The header is written once and the frames follow it, so the cell and the
/// species come from the top of the file and only the positions vary - unless
/// the run was a variable-cell relaxation, which repeats the header per frame.
/// Both layouts are handled.
[[nodiscard]] std::vector<Atoms> read_xdatcar(const std::string& path);

/// Read the structures and results of an OUTCAR.
///
/// One frame per ionic step, each carrying the energy, the forces and the
/// stress that VASP reported for it, so a relaxation can be replayed and
/// plotted without re-running anything.
[[nodiscard]] std::vector<Atoms> read_outcar(const std::string& path);

}  // namespace fastatoms::io
