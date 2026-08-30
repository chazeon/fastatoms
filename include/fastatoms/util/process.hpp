// SPDX-License-Identifier: MIT
#pragma once

/// \file process.hpp
/// \brief Running an external program - the part of a file-based calculator
///        that actually launches the DFT engine.

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace fastatoms::util {

/// How to launch a command.
struct ProcessOptions {
    /// Directory the command runs in. Empty means the current directory.
    std::filesystem::path working_directory{};

    /// Variables to add to (or override in) the child's environment.
    std::vector<std::pair<std::string, std::string>> environment{};

    /// Where the child's stdout and stderr go. When empty a temporary file is
    /// used and deleted afterwards; either way the tail is returned.
    std::filesystem::path log_path{};

    /// Give up after this many seconds. Zero (the default) waits forever.
    /// On timeout the child is sent SIGTERM, then SIGKILL after a short grace
    /// period.
    double timeout_seconds{0.0};
};

/// What happened.
struct ProcessResult {
    int exit_code{-1};
    bool timed_out{false};
    bool signalled{false};
    int signal_number{0};
    std::filesystem::path log_path{};

    /// The last few kilobytes of the combined output, for diagnostics. The
    /// full text stays in `log_path` when one was given.
    std::string log_tail{};

    [[nodiscard]] bool ok() const noexcept {
        return exit_code == 0 && !timed_out && !signalled;
    }

    /// One-line summary such as "exit code 2" or "timed out after 60 s".
    [[nodiscard]] std::string status_string() const;
};

/// Run `command` through the system shell.
///
/// A shell is used deliberately: DFT run commands are shell expressions in
/// practice (`mpirun -np 8 pw.x -in pw.in > pw.out`), and every code's
/// documentation is written that way. The command therefore has the full
/// authority of the calling user - it comes from a configuration file the user
/// controls, never from untrusted input.
///
/// stdout and stderr are merged into the log file rather than piped, so a long
/// run cannot deadlock on a full pipe buffer.
[[nodiscard]] ProcessResult run_command(const std::string& command,
                                        const ProcessOptions& options = {});

/// A child process kept alive across many requests, with pipes on its standard
/// input and output.
///
/// `run_command` launches a program, waits, and collects its output; that is
/// the right shape for a DFT run that takes minutes. A machine-learning
/// potential is the opposite case - thousands of evaluations, each of them
/// milliseconds - so the interpreter must be started once and then fed. This
/// class is that: one `fork`/`exec`, then a binary request/response loop over
/// pipes, with no shell, no temporary files and no per-call startup.
///
/// The child's standard error is redirected to a log file rather than a pipe,
/// so a chatty program cannot deadlock by filling a buffer nobody drains.
class Subprocess {
public:
    /// Launch `program` with `args` (argv[0] is added automatically).
    ///
    /// \throws std::runtime_error if the child cannot be started.
    Subprocess(std::string program, std::vector<std::string> args,
               const ProcessOptions& options = {});

    ~Subprocess();

    Subprocess(const Subprocess&) = delete;
    Subprocess& operator=(const Subprocess&) = delete;

    /// True while the child is believed to be alive and the pipes are open.
    [[nodiscard]] bool running() const noexcept { return running_; }

    [[nodiscard]] int pid() const noexcept { return pid_; }

    /// Write exactly `bytes` to the child, retrying on partial writes.
    /// \throws std::runtime_error if the child has gone away.
    void write_exact(const void* data, std::size_t bytes);

    /// Read exactly `bytes` from the child, retrying on partial reads.
    /// \throws std::runtime_error on end of file or error.
    void read_exact(void* data, std::size_t bytes);

    /// Close the child's standard input, which is how a well-behaved worker is
    /// asked to exit.
    void close_input();

    /// Wait for the child, returning its exit status. Idempotent.
    int wait();

    /// SIGTERM, then SIGKILL after a short grace period.
    void terminate();

    /// The tail of whatever the child wrote to standard error - the first
    /// thing to look at when a request fails.
    [[nodiscard]] std::string error_tail(std::size_t max_bytes = 4096) const;

private:
    std::string program_{};
    std::filesystem::path log_path_{};
    bool owns_log_{false};
    int pid_{-1};
    int to_child_{-1};
    int from_child_{-1};
    bool running_{false};
    int status_{-1};
    bool reaped_{false};
};

/// Value of an environment variable, or `fallback` when it is unset or empty.
[[nodiscard]] std::string environment_value(const char* name, std::string fallback = {});

/// First of `names` that is set to a non-empty value, else `fallback`.
[[nodiscard]] std::string first_environment_value(const std::vector<std::string>& names,
                                                  std::string fallback = {});

}  // namespace fastatoms::util
