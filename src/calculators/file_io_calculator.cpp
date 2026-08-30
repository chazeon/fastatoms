// SPDX-License-Identifier: MIT
#include "fastatoms/calculators/file_io_calculator.hpp"

#include <system_error>

#include "fastatoms/util/text.hpp"

namespace fastatoms {

FileIOCalculator::FileIOCalculator(Value parameters) : Calculator(std::move(parameters)) {}

std::filesystem::path FileIOCalculator::directory() const {
    const std::string configured = parameter("directory").value_or(std::string_view{});
    return configured.empty() ? std::filesystem::path(name()) : std::filesystem::path(configured);
}

void FileIOCalculator::set_directory(std::filesystem::path directory) {
    set_parameter("directory", Value(directory.string()));
}

std::string FileIOCalculator::command() const {
    const std::string configured = parameter("command").value_or(std::string_view{});
    if (!configured.empty()) {
        return configured;
    }
    return util::first_environment_value(command_environment_names(), default_command());
}

void FileIOCalculator::set_command(std::string command) {
    set_parameter("command", Value(std::move(command)));
}

std::string FileIOCalculator::default_log_name() const { return name() + ".log"; }

std::filesystem::path FileIOCalculator::prepare_directory(
    const std::filesystem::path& requested) const {
    const std::filesystem::path target = requested.empty() ? directory() : requested;
    std::error_code error;
    std::filesystem::create_directories(target, error);
    if (!std::filesystem::is_directory(target)) {
        fail("cannot create the working directory '" + target.string() +
             (error ? "': " + error.message() : "'"));
    }
    return target;
}

std::vector<std::pair<std::string, std::string>> FileIOCalculator::child_environment() const {
    std::vector<std::pair<std::string, std::string>> variables;
    const Value& environment = parameter("environment");
    if (environment.is_null()) {
        return variables;
    }
    if (!environment.is_map()) {
        fail("the 'environment' parameter must be a mapping of variable names to values");
    }
    for (const auto& [key, value] : environment.as_map()) {
        variables.emplace_back(key, value.is_string() ? value.as_string() : value.to_string());
    }
    return variables;
}

void FileIOCalculator::run_engine(const std::filesystem::path& directory) {
    const std::string to_run = command();
    if (util::trim(to_run).empty()) {
        fail("no run command is configured; set the 'command' parameter or one of " +
             [this] {
                 std::string names;
                 for (const std::string& variable : command_environment_names()) {
                     names += (names.empty() ? "" : ", ") + variable;
                 }
                 return names;
             }());
    }

    util::ProcessOptions options;
    options.working_directory = directory;
    options.environment = child_environment();
    options.timeout_seconds = parameter("timeout").value_or(0.0);

    const std::string log_name =
        parameter("log_file").value_or(std::string_view(default_log_name()));
    if (!log_name.empty()) {
        options.log_path = directory / log_name;
    }

    process_ = util::run_command(to_run, options);

    if (!process_.ok()) {
        std::string message = "the run command failed (" + process_.status_string() + ")\n";
        message += "  command   : " + to_run + "\n";
        message += "  directory : " + std::filesystem::absolute(directory).string() + "\n";
        if (!process_.log_path.empty()) {
            message += "  log       : " + std::filesystem::absolute(process_.log_path).string() +
                       "\n";
        }
        const std::string excerpt = util::tail(process_.log_tail, 20);
        if (!util::trim(excerpt).empty()) {
            message += "--- last lines of the log ---\n" + excerpt;
        }
        fail(message);
    }
}

void FileIOCalculator::write_inputs(const Atoms& atoms, const std::filesystem::path& target,
                                    const std::vector<Property>& properties) {
    if (atoms.empty()) {
        fail("cannot write input files for an empty structure");
    }
    write_input(prepare_directory(target), atoms, properties);
}

void FileIOCalculator::read_results(const std::filesystem::path& target) {
    const std::filesystem::path source = target.empty() ? directory() : target;
    if (!std::filesystem::is_directory(source)) {
        fail("'" + source.string() + "' is not a directory");
    }
    results_.clear();
    parse_output(source);
}

void FileIOCalculator::compute(const Atoms& atoms, const std::vector<Property>& properties) {
    if (atoms.empty()) {
        fail("cannot run a calculation on an empty structure");
    }
    const std::filesystem::path target = prepare_directory({});
    write_input(target, atoms, properties);
    run_engine(target);
    parse_output(target);
}

std::string FileIOCalculator::to_string() const {
    return name() + "(directory='" + directory().string() + "', command='" + command() + "')";
}

}  // namespace fastatoms
