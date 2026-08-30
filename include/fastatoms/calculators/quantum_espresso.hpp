// SPDX-License-Identifier: MIT
#pragma once

/// \file quantum_espresso.hpp
/// \brief Quantum ESPRESSO (pw.x) interface.

#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/calculators/file_io_calculator.hpp"

namespace fastatoms {

/// Calculator that drives Quantum ESPRESSO's `pw.x`.
///
/// Parameters, beyond those of `FileIOCalculator`:
///   - `pseudopotentials` - required: a mapping of element to UPF file name,
///                          e.g. `{Si: Si.pbe-n-kjpaw_psl.1.0.0.UPF}`.
///   - `pseudo_dir`       - where those files live; falls back to
///                          `ESPRESSO_PSEUDO`.
///   - `kpts`             - `[n1, n2, n3]`, a single integer, or absent for
///                          the gamma point only.
///   - `kpts_shift`       - `[s1, s2, s3]`, each 0 or 1.
///   - `input_file`       - default "pw.in"; `output_file` default "pw.out".
///   - `crystal_coordinates` - `true` (default) writes fractional positions.
///   - `control`, `system`, `electrons`, `ions`, `cell` - mappings written
///                          verbatim into the corresponding namelist.
///
/// Any other parameter is routed to the namelist that owns it - `ecutwfc` to
/// &SYSTEM, `conv_thr` to &ELECTRONS, `calculation` to &CONTROL - so flat
/// input files work as well as grouped ones. Unrecognised names fall through
/// to &SYSTEM, matching ASE.
class QuantumEspressoCalculator : public FileIOCalculator {
public:
    explicit QuantumEspressoCalculator(Value parameters = {});

    [[nodiscard]] std::string name() const override { return "espresso"; }
    [[nodiscard]] std::vector<Property> implemented_properties() const override;

    /// The complete `pw.x` input deck for `atoms`.
    [[nodiscard]] std::string input_string(const Atoms& atoms,
                                           const std::vector<Property>& properties) const;

    /// Parse the contents of a `pw.x` output file.
    [[nodiscard]] static Results parse_output_text(std::string_view text);

    /// Which namelist a pw.x keyword belongs to ("control", "system",
    /// "electrons", "ions" or "cell"). Unknown keywords map to "system".
    [[nodiscard]] static std::string namelist_for(std::string_view keyword);

protected:
    void write_input(const std::filesystem::path& directory, const Atoms& atoms,
                     const std::vector<Property>& properties) override;
    void parse_output(const std::filesystem::path& directory) override;

    [[nodiscard]] std::string default_command() const override;
    [[nodiscard]] std::vector<std::string> command_environment_names() const override;
    [[nodiscard]] std::string default_log_name() const override { return "espresso.log"; }

private:
    [[nodiscard]] std::string input_file_name() const;
    [[nodiscard]] std::string output_file_name() const;
};

}  // namespace fastatoms
