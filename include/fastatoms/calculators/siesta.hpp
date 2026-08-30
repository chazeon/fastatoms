// SPDX-License-Identifier: MIT
#pragma once

/// \file siesta.hpp
/// \brief SIESTA interface: an .fdf deck in, the text output and
///        .FORCE_STRESS file out.

#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/calculators/file_io_calculator.hpp"

namespace fastatoms {

/// Calculator that drives SIESTA.
///
/// Parameters, beyond those of `FileIOCalculator`:
///   - `label`       - SystemLabel and the stem of the input/output files.
///                     Default "siesta".
///   - `xc`          - "pbe" (default), "lda", "pbesol", "revpbe", "blyp", ...
///                     Sets XC.functional and XC.authors.
///   - `mesh_cutoff` - MeshCutoff in Ry.
///   - `kpts`        - `[n1, n2, n3]` or a single integer; absent means the
///                     gamma point only.
///   - `kpts_shift`  - mesh shift, default `[0, 0, 0]`.
///   - `pp_path`     - directory holding the pseudopotential files; falls back
///                     to `SIESTA_PP_PATH`. Files are copied into the run
///                     directory.
///   - `fractional`  - `true` (default) writes fractional coordinates.
///   - `fdf`         - a mapping written verbatim, applied last.
///
/// Any other parameter becomes an fdf directive: `dm_tolerance: 1.0e-4` writes
/// `dm.tolerance 1.0e-4`. Underscores become dots and the letters are left as
/// written - fdf label matching ignores case and separators, so
/// `DM.Tolerance` works just as well if you prefer to write it that way. List
/// values become `%block` sections.
class SiestaCalculator : public FileIOCalculator {
public:
    explicit SiestaCalculator(Value parameters = {});

    [[nodiscard]] std::string name() const override { return "siesta"; }
    [[nodiscard]] std::vector<Property> implemented_properties() const override;

    /// The complete .fdf deck for `atoms`.
    [[nodiscard]] std::string fdf_string(const Atoms& atoms) const;

    /// Parse SIESTA's text output. Energies come from here; forces and stress
    /// are taken from it only when no .FORCE_STRESS file is available.
    [[nodiscard]] static Results parse_output_text(std::string_view text);

    /// Parse a .FORCE_STRESS file, which carries forces and stress to full
    /// precision in Ry/bohr and Ry/bohr^3.
    static void parse_force_stress(std::string_view text, Results& results);

protected:
    void write_input(const std::filesystem::path& directory, const Atoms& atoms,
                     const std::vector<Property>& properties) override;
    void parse_output(const std::filesystem::path& directory) override;

    [[nodiscard]] std::string default_command() const override;
    [[nodiscard]] std::vector<std::string> command_environment_names() const override;
    [[nodiscard]] std::string default_log_name() const override { return "siesta.log"; }

private:
    [[nodiscard]] std::string label() const;
    void copy_pseudopotentials(const std::filesystem::path& directory,
                               const std::vector<std::string>& symbols) const;
};

}  // namespace fastatoms
