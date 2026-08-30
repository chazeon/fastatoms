// SPDX-License-Identifier: MIT
#pragma once

/// \file file_io_calculator.hpp
/// \brief Base class for calculators that talk to an external code through
///        files on disk.

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "fastatoms/calculators/calculator.hpp"
#include "fastatoms/util/process.hpp"

namespace fastatoms {

/// A calculator that runs an external program in a working directory.
///
/// Every evaluation follows the same three steps, which subclasses fill in:
///
///   1. `write_input()`  - turn the `Atoms` object into the code's input files
///   2. `run_engine()`   - launch the program and wait for it
///   3. `parse_output()` - read energy, forces and stress back out
///
/// The two halves are usable on their own: `write_inputs()` prepares a
/// directory for a job you will submit to a queue yourself, and
/// `read_results()` parses a run that has already finished. That covers the
/// common HPC workflow where the calculation does not run on the machine
/// driving it.
///
/// Parameters recognised here (subclasses add their own):
///   - `directory`   - where to run; created if missing. Default: the
///                     calculator name.
///   - `command`     - the shell command to run. Falls back to the code's
///                     environment variables, then to a built-in default.
///   - `log_file`    - combined stdout/stderr destination inside `directory`.
///   - `timeout`     - seconds before the job is killed. 0 (default) waits.
///   - `environment` - a mapping of extra environment variables for the child.
class FileIOCalculator : public Calculator {
public:
    /// The working directory, as configured.
    [[nodiscard]] std::filesystem::path directory() const;
    void set_directory(std::filesystem::path directory);

    /// The command that will be run, after resolving the parameter, then the
    /// code's environment variables, then the built-in default.
    [[nodiscard]] std::string command() const;
    void set_command(std::string command);

    /// Write the input files for `atoms` without running anything.
    ///
    /// Use this to stage a job for a batch queue. `directory` defaults to the
    /// configured one.
    void write_inputs(const Atoms& atoms, const std::filesystem::path& directory = {},
                      const std::vector<Property>& properties = default_properties());

    /// Parse a finished run without executing anything.
    ///
    /// Results become available through `results()`. The structure cache is
    /// left invalid on purpose: these numbers were not produced for any
    /// particular `Atoms` object this calculator knows about, so a later
    /// `calculate()` will still run the engine.
    void read_results(const std::filesystem::path& directory = {});

    /// Outcome of the most recent engine invocation.
    [[nodiscard]] const util::ProcessResult& last_process() const noexcept { return process_; }

    [[nodiscard]] std::string to_string() const override;

protected:
    explicit FileIOCalculator(Value parameters);

    void compute(const Atoms& atoms, const std::vector<Property>& properties) final;

    /// Write every file the engine needs into `directory`.
    virtual void write_input(const std::filesystem::path& directory, const Atoms& atoms,
                             const std::vector<Property>& properties) = 0;

    /// Read `directory` and fill `results_`.
    virtual void parse_output(const std::filesystem::path& directory) = 0;

    /// Command used when neither a parameter nor an environment variable says
    /// otherwise.
    [[nodiscard]] virtual std::string default_command() const = 0;

    /// Environment variables consulted for the run command, most specific
    /// first, e.g. {"FASTATOMS_VASP_COMMAND", "ASE_VASP_COMMAND", "VASP_COMMAND"}.
    [[nodiscard]] virtual std::vector<std::string> command_environment_names() const = 0;

    /// Default name of the combined stdout/stderr log inside the directory.
    [[nodiscard]] virtual std::string default_log_name() const;

    /// Launch the engine in `directory`.
    /// \throws CalculatorError on a non-zero exit, a signal, or a timeout.
    void run_engine(const std::filesystem::path& directory);

    /// Create `directory` if needed and return it.
    [[nodiscard]] std::filesystem::path prepare_directory(
        const std::filesystem::path& directory) const;

    /// Extra environment variables for the child process, from the
    /// `environment` parameter plus anything a subclass adds.
    [[nodiscard]] virtual std::vector<std::pair<std::string, std::string>> child_environment()
        const;

private:
    util::ProcessResult process_{};
};

}  // namespace fastatoms
