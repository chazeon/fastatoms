// SPDX-License-Identifier: MIT
#pragma once

/// \file vasp.hpp
/// \brief VASP interface: POSCAR/INCAR/KPOINTS/POTCAR in, OUTCAR out.

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/calculators/file_io_calculator.hpp"

namespace fastatoms {

/// How the atoms are laid out in a POSCAR.
///
/// VASP requires atoms to be grouped by species, which generally reorders
/// them, so `order` records where each POSCAR row came from in the original
/// structure. Forces read back from OUTCAR are permuted through it.
struct VaspSpeciesLayout {
    std::vector<std::string> symbols{};    ///< unique species, first-appearance order
    std::vector<std::size_t> counts{};     ///< atoms of each species
    std::vector<std::size_t> order{};      ///< POSCAR row -> index in the Atoms object
};

[[nodiscard]] VaspSpeciesLayout vasp_species_layout(const Atoms& atoms);

/// Calculator that drives VASP.
///
/// Parameters, beyond those of `FileIOCalculator`:
///   - `xc`           - "pbe", "pbesol", "rpbe", "lda", ... Sets the GGA tag
///                      and picks the POTCAR flavour.
///   - `kpts`         - `[n1, n2, n3]`, a single integer, or absent for a
///                      gamma-point calculation.
///   - `kpts_shift`   - mesh shift, default `[0, 0, 0]`.
///   - `gamma`        - `true` (default) for a Gamma-centred mesh,
///                      `false` for Monkhorst-Pack.
///   - `incar`        - a mapping written verbatim into INCAR, applied last.
///   - `pp_path`      - root of the POTCAR library; falls back to
///                      `VASP_PP_PATH`.
///   - `pp_flavour`   - subdirectory of `pp_path`, e.g. "potpaw_PBE".
///   - `setups`       - per-element POTCAR choice, e.g. `{Li: Li_sv}` or
///                      `{Li: _sv}`.
///   - `write_potcar` - set to `false` to manage POTCAR yourself.
///   - `direct`       - `true` (default) writes fractional coordinates.
///
/// Any other parameter becomes an INCAR tag: `encut: 400` writes
/// `ENCUT = 400`, matching how ASE's VASP calculator behaves.
class VaspCalculator : public FileIOCalculator {
public:
    explicit VaspCalculator(Value parameters = {});

    [[nodiscard]] std::string name() const override { return "vasp"; }
    [[nodiscard]] std::vector<Property> implemented_properties() const override;

    /// \name Input generation
    /// Exposed so that inputs can be inspected, diffed or tested without a
    /// VASP installation.
    /// @{
    [[nodiscard]] std::string poscar_string(const Atoms& atoms,
                                            const VaspSpeciesLayout& layout) const;
    [[nodiscard]] std::string incar_string(const std::vector<Property>& properties) const;
    /// Empty when KSPACING is in use, in which case no KPOINTS file is written.
    [[nodiscard]] std::string kpoints_string() const;
    /// @}

    /// Parse the contents of an OUTCAR.
    ///
    /// Forces come back in POSCAR (species-grouped) order; `parse_output`
    /// permutes them back to the structure's own order.
    [[nodiscard]] static Results parse_outcar(std::string_view text);

protected:
    void write_input(const std::filesystem::path& directory, const Atoms& atoms,
                     const std::vector<Property>& properties) override;
    void parse_output(const std::filesystem::path& directory) override;

    [[nodiscard]] std::string default_command() const override { return "vasp_std"; }
    [[nodiscard]] std::vector<std::string> command_environment_names() const override;
    [[nodiscard]] std::string default_log_name() const override { return "vasp.log"; }

private:
    void write_potcar(const std::filesystem::path& directory,
                      const std::vector<std::string>& symbols) const;

    /// POSCAR row -> index in the structure, from the most recent input write.
    std::vector<std::size_t> order_{};
};

}  // namespace fastatoms
