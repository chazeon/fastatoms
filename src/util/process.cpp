// SPDX-License-Identifier: MIT
#include "fastatoms/util/process.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include "fastatoms/util/text.hpp"

#if defined(_WIN32)
#  include <cstdio>
#else
#  include <fcntl.h>
#  include <signal.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <time.h>
#  include <unistd.h>
extern char** environ;
#endif

namespace fastatoms::util {
namespace {

/// How much of the log to keep for diagnostics.
constexpr std::size_t kMaxTailBytes = 64u * 1024u;

std::filesystem::path make_temporary_log_path() {
    static std::mt19937_64 rng(std::random_device{}());
    std::ostringstream name;
    name << "fastatoms-" << std::hex << rng() << ".log";
    return std::filesystem::temp_directory_path() / name.str();
}

/// Last `kMaxTailBytes` of a file, or an empty string if it cannot be read.
std::string read_tail(const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error) {
        return {};
    }
    std::string text;
    try {
        text = read_text_file(path);
    } catch (const std::exception&) {
        return {};
    }
    if (size > kMaxTailBytes) {
        text = "... (" + std::to_string(size - kMaxTailBytes) + " earlier bytes omitted)\n" +
               text.substr(text.size() - kMaxTailBytes);
    }
    return text;
}

#if !defined(_WIN32)

/// Build the child's environment before forking, so that nothing between fork
/// and exec has to allocate.
std::vector<std::string> build_environment(
    const std::vector<std::pair<std::string, std::string>>& overrides) {
    std::vector<std::string> entries;
    for (char** it = environ; it != nullptr && *it != nullptr; ++it) {
        const std::string entry(*it);
        const std::size_t equals = entry.find('=');
        const std::string key = equals == std::string::npos ? entry : entry.substr(0, equals);
        const bool overridden =
            std::any_of(overrides.begin(), overrides.end(),
                        [&key](const auto& pair) { return pair.first == key; });
        if (!overridden) {
            entries.push_back(entry);
        }
    }
    for (const auto& [key, value] : overrides) {
        entries.push_back(key + "=" + value);
    }
    return entries;
}

void sleep_briefly(double seconds) {
    struct timespec request {};
    request.tv_sec = static_cast<time_t>(seconds);
    request.tv_nsec = static_cast<long>((seconds - static_cast<double>(request.tv_sec)) * 1e9);
    nanosleep(&request, nullptr);
}

double monotonic_seconds() {
    struct timespec now {};
    clock_gettime(CLOCK_MONOTONIC, &now);
    return static_cast<double>(now.tv_sec) + static_cast<double>(now.tv_nsec) * 1e-9;
}

#endif  // !_WIN32

}  // namespace

std::string ProcessResult::status_string() const {
    if (timed_out) {
        return "timed out";
    }
    if (signalled) {
        return "killed by signal " + std::to_string(signal_number);
    }
    return "exit code " + std::to_string(exit_code);
}

std::string environment_value(const char* name, std::string fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    return std::string(value);
}

std::string first_environment_value(const std::vector<std::string>& names, std::string fallback) {
    for (const std::string& name : names) {
        std::string value = environment_value(name.c_str());
        if (!value.empty()) {
            return value;
        }
    }
    return fallback;
}

ProcessResult run_command(const std::string& command, const ProcessOptions& options) {
    if (util::trim(command).empty()) {
        throw std::invalid_argument("run_command: the command is empty");
    }

    ProcessResult result;
    const bool temporary_log = options.log_path.empty();
    result.log_path = temporary_log ? make_temporary_log_path() : options.log_path;

    if (!options.working_directory.empty()) {
        std::error_code error;
        std::filesystem::create_directories(options.working_directory, error);
        if (!std::filesystem::is_directory(options.working_directory)) {
            throw std::runtime_error("run_command: working directory '" +
                                     options.working_directory.string() + "' does not exist");
        }
    }
    if (result.log_path.has_parent_path() && !result.log_path.parent_path().empty()) {
        std::error_code error;
        std::filesystem::create_directories(result.log_path.parent_path(), error);
    }

#if defined(_WIN32)
    // Fallback path: no fork/exec, so the working directory is set by the shell.
    std::string shell_command;
    if (!options.working_directory.empty()) {
        shell_command += "cd /d \"" + options.working_directory.string() + "\" && ";
    }
    for (const auto& [key, value] : options.environment) {
        shell_command += "set \"" + key + "=" + value + "\" && ";
    }
    shell_command += "(" + command + ") > \"" + result.log_path.string() + "\" 2>&1";
    result.exit_code = std::system(shell_command.c_str());
    (void)options.timeout_seconds;  // not supported on this path
#else
    const std::vector<std::string> environment_storage = build_environment(options.environment);
    std::vector<char*> environment_pointers;
    environment_pointers.reserve(environment_storage.size() + 1);
    for (const std::string& entry : environment_storage) {
        environment_pointers.push_back(const_cast<char*>(entry.c_str()));
    }
    environment_pointers.push_back(nullptr);

    const std::string working_directory =
        options.working_directory.empty() ? std::string(".") : options.working_directory.string();
    const std::string log_path = result.log_path.string();

    const pid_t pid = fork();
    if (pid < 0) {
        throw std::system_error(errno, std::generic_category(), "run_command: fork failed");
    }

    if (pid == 0) {
        // Child. Only async-signal-safe calls from here to execve.
        if (chdir(working_directory.c_str()) != 0) {
            _exit(126);
        }
        const int fd = open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            _exit(126);
        }
        if (dup2(fd, STDOUT_FILENO) < 0 || dup2(fd, STDERR_FILENO) < 0) {
            _exit(126);
        }
        if (fd > STDERR_FILENO) {
            close(fd);
        }
        // Start a new process group so that a timeout can signal the whole
        // job tree, not just the shell.
        setpgid(0, 0);

        const char* argv[] = {"sh", "-c", command.c_str(), nullptr};
        execve("/bin/sh", const_cast<char* const*>(argv), environment_pointers.data());
        _exit(127);  // execve only returns on failure
    }

    // Parent.
    int status = 0;
    if (options.timeout_seconds > 0.0) {
        const double deadline = monotonic_seconds() + options.timeout_seconds;
        bool reaped = false;
        while (true) {
            const pid_t done = waitpid(pid, &status, WNOHANG);
            if (done == pid) {
                reaped = true;
                break;
            }
            if (done < 0 && errno != EINTR) {
                break;
            }
            if (monotonic_seconds() >= deadline) {
                result.timed_out = true;
                kill(-pid, SIGTERM);
                // Give the job a couple of seconds to shut down cleanly.
                const double grace = monotonic_seconds() + 2.0;
                while (monotonic_seconds() < grace) {
                    if (waitpid(pid, &status, WNOHANG) == pid) {
                        reaped = true;
                        break;
                    }
                    sleep_briefly(0.02);
                }
                if (!reaped) {
                    kill(-pid, SIGKILL);
                    waitpid(pid, &status, 0);
                }
                break;
            }
            sleep_briefly(0.02);
        }
    } else {
        while (waitpid(pid, &status, 0) < 0) {
            if (errno != EINTR) {
                throw std::system_error(errno, std::generic_category(),
                                        "run_command: waitpid failed");
            }
        }
    }

    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result.signalled = true;
        result.signal_number = WTERMSIG(status);
        result.exit_code = 128 + result.signal_number;
    }
#endif

    result.log_tail = read_tail(result.log_path);
    if (temporary_log) {
        std::error_code error;
        std::filesystem::remove(result.log_path, error);
        result.log_path.clear();
    }
    return result;
}

}  // namespace fastatoms::util
