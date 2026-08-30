// SPDX-License-Identifier: MIT
#include "fastatoms/calculators/mlip_worker.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <utility>
#include <vector>

#include "fastatoms/io/yaml_input.hpp"
#include "fastatoms/util/process.hpp"

namespace fastatoms::mlip {
namespace {

namespace fs = std::filesystem;

constexpr char kRequestMagic[4] = {'F', 'A', 'M', 'R'};
constexpr char kReplyMagic[4] = {'F', 'A', 'M', 'S'};
constexpr char kHelloMagic[4] = {'F', 'A', 'M', 'H'};
constexpr std::int32_t kProtocolVersion = 1;

/// Candidate roots for a conda installation, most specific first.
std::vector<fs::path> conda_roots() {
    std::vector<fs::path> roots;
    const std::string prefix = util::environment_value("CONDA_PREFIX");
    if (!prefix.empty()) {
        const fs::path path(prefix);
        // $CONDA_PREFIX is either the root itself or <root>/envs/<name>.
        if (path.parent_path().filename() == "envs") {
            roots.push_back(path.parent_path().parent_path());
        } else {
            roots.push_back(path);
        }
    }
    const std::string exe = util::environment_value("CONDA_EXE");
    if (!exe.empty()) {
        roots.push_back(fs::path(exe).parent_path().parent_path());
    }
    const std::string home = util::environment_value("HOME");
    if (!home.empty()) {
        for (const char* name : {"miniconda3", "anaconda3", "miniforge3", "mambaforge",
                                 "micromamba", "conda", "opt/miniconda3", "opt/anaconda3"}) {
            roots.push_back(fs::path(home) / name);
        }
    }
    roots.emplace_back("/opt/conda");
    roots.emplace_back("/usr/local/conda");
    return roots;
}

std::string first_existing(const std::vector<fs::path>& candidates) {
    std::error_code ignored;
    for (const fs::path& candidate : candidates) {
        if (fs::exists(candidate, ignored) && !fs::is_directory(candidate, ignored)) {
            return candidate.string();
        }
    }
    return {};
}

}  // namespace

std::string conda_python(const std::string& environment) {
    if (environment.empty()) {
        return {};
    }
    std::error_code ignored;

    // An absolute path is taken at face value: either the environment itself
    // or the interpreter inside it.
    const fs::path given(environment);
    if (given.is_absolute() || environment.find('/') != std::string::npos) {
        if (fs::is_directory(given, ignored)) {
            const std::string inside = first_existing({given / "bin" / "python",
                                                       given / "bin" / "python3",
                                                       given / "python.exe"});
            if (!inside.empty()) {
                return inside;
            }
        }
        if (fs::exists(given, ignored)) {
            return given.string();
        }
    }

    std::vector<fs::path> candidates;
    for (const fs::path& root : conda_roots()) {
        candidates.push_back(root / "envs" / environment / "bin" / "python");
        candidates.push_back(root / "envs" / environment / "bin" / "python3");
        candidates.push_back(root / "envs" / environment / "python.exe");
    }
    return first_existing(candidates);
}

std::string worker_script_path() {
    const std::string configured = util::environment_value("FASTATOMS_MLIP_WORKER");
    if (!configured.empty()) {
        return configured;
    }
    std::vector<fs::path> candidates;
#if defined(FASTATOMS_MLIP_WORKER_SOURCE)
    candidates.emplace_back(FASTATOMS_MLIP_WORKER_SOURCE);
#endif
#if defined(FASTATOMS_MLIP_WORKER_INSTALLED)
    candidates.emplace_back(FASTATOMS_MLIP_WORKER_INSTALLED);
#endif
    return first_existing(candidates);
}

namespace {

/// A model driven by a Python process that stays alive between calls.
class WorkerBackend final : public Backend {
public:
    WorkerBackend(const std::string& family, const std::string& model, const Value& parameters) {
        info_.family = family;
        info_.path = model;
        info_.backend = "worker";
        if (const Value* value = parameters.find("device"); value && value->is_string()) {
            info_.device = value->as_string();
        }
        if (const Value* value = parameters.find("dtype"); value && value->is_string()) {
            info_.dtype = value->as_string();
        }

        const std::string interpreter = find_interpreter(parameters);
        const std::string script = find_script(parameters);

        std::vector<std::string> args{"-u", script, "--family", family, "--model", model,
                                      "--device", info_.device, "--dtype", info_.dtype};
        if (const Value* value = parameters.find("model_options");
            value && (value->is_map() || value->is_string())) {
            args.emplace_back("--options");
            args.emplace_back(value->is_string() ? value->as_string() : value->to_string());
        }

        util::ProcessOptions options;
        // Torch's own threading is fine, but the worker must never try to open
        // a display or a GPU it was not asked for.
        options.environment.emplace_back("PYTHONUNBUFFERED", "1");
        options.environment.emplace_back("MPLBACKEND", "Agg");
        if (const Value* value = parameters.find("threads"); value && value->is_number()) {
            options.environment.emplace_back("OMP_NUM_THREADS",
                                             std::to_string(value->as_int()));
        }

        try {
            process_ = std::make_unique<util::Subprocess>(interpreter, args, options);
        } catch (const std::exception& error) {
            throw MLIPError("mlip: could not start the worker interpreter '" + interpreter +
                            "': " + error.what());
        }
        handshake(interpreter);
    }

    ~WorkerBackend() override { unload(); }

    [[nodiscard]] const ModelInfo& info() const override { return info_; }

    /// The worker's own package builds the graph, so C++ must not.
    [[nodiscard]] bool needs_edges() const override { return false; }

    void unload() override {
        if (process_) {
            process_->close_input();
            process_->terminate();
            process_.reset();
        }
    }

    [[nodiscard]] Prediction evaluate(const Graph& graph, bool need_forces,
                                      bool need_stress) override {
        if (!process_ || !process_->running()) {
            throw MLIPError("mlip: the worker process is no longer running" + diagnostics());
        }

        const auto natoms = static_cast<std::int32_t>(graph.natoms);
        std::string request;
        request.reserve(4 + 4 + 4 + 5 + graph.positions.size() * 8 + graph.numbers.size() * 8 +
                        72);
        request.append(kRequestMagic, 4);
        append_pod(request, kProtocolVersion);
        append_pod(request, natoms);
        request.push_back(static_cast<char>(need_forces ? 1 : 0));
        request.push_back(static_cast<char>(need_stress ? 1 : 0));
        for (int axis = 0; axis < 3; ++axis) {
            request.push_back(static_cast<char>(graph.pbc[static_cast<std::size_t>(axis)] ? 1 : 0));
        }
        append_array(request, graph.positions.data(), graph.positions.size());
        append_array(request, graph.numbers.data(), graph.numbers.size());
        append_array(request, graph.cell.m.data(), 9);

        try {
            process_->write_exact(request.data(), request.size());
        } catch (const std::exception& error) {
            throw MLIPError(std::string("mlip: sending the structure to the worker failed: ") +
                            error.what() + diagnostics());
        }

        char magic[4] = {};
        std::int32_t status = 0;
        try {
            process_->read_exact(magic, 4);
            process_->read_exact(&status, sizeof(status));
        } catch (const std::exception& error) {
            throw MLIPError(std::string("mlip: the worker did not answer: ") + error.what() +
                            diagnostics());
        }
        if (std::memcmp(magic, kReplyMagic, 4) != 0) {
            throw MLIPError("mlip: the worker sent a malformed reply" + diagnostics());
        }
        if (status != 0) {
            std::int32_t length = 0;
            process_->read_exact(&length, sizeof(length));
            std::string message(static_cast<std::size_t>(std::max(0, length)), '\0');
            if (length > 0) {
                process_->read_exact(message.data(), message.size());
            }
            throw MLIPError("mlip: the model failed: " + message);
        }

        std::uint8_t flags[2] = {0, 0};
        process_->read_exact(flags, 2);
        Prediction prediction;
        process_->read_exact(&prediction.energy, sizeof(double));
        if (flags[0] != 0) {
            prediction.forces.resize(3 * graph.natoms);
            process_->read_exact(prediction.forces.data(), prediction.forces.size() * 8);
            prediction.has_forces = true;
        }
        if (flags[1] != 0) {
            process_->read_exact(prediction.stress.data(), 6 * 8);
            prediction.has_stress = true;
        }
        return prediction;
    }

private:
    template <typename T>
    static void append_pod(std::string& buffer, const T& value) {
        buffer.append(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template <typename T>
    static void append_array(std::string& buffer, const T* data, std::size_t count) {
        buffer.append(reinterpret_cast<const char*>(data), count * sizeof(T));
    }

    [[nodiscard]] std::string diagnostics() const {
        if (!process_) {
            return {};
        }
        const std::string tail = process_->error_tail();
        return tail.empty() ? std::string{} : "\nThe worker said:\n" + tail;
    }

    [[nodiscard]] static std::string find_interpreter(const Value& parameters) {
        if (const Value* value = parameters.find("python"); value && value->is_string()) {
            const std::string given = value->as_string();
            const std::string resolved = conda_python(given);
            if (!resolved.empty()) {
                return resolved;
            }
            return given;  // a bare name is left for the PATH search
        }
        if (const Value* value = parameters.find("conda_env"); value && value->is_string()) {
            const std::string environment = value->as_string();
            const std::string resolved = conda_python(environment);
            if (resolved.empty()) {
                throw MLIPError("mlip: could not find the conda environment '" + environment +
                                "'. Create it, or point the calculator at an interpreter "
                                "directly with python='/path/to/bin/python'.");
            }
            return resolved;
        }
        return "python3";
    }

    [[nodiscard]] static std::string find_script(const Value& parameters) {
        if (const Value* value = parameters.find("worker"); value && value->is_string()) {
            return value->as_string();
        }
        const std::string found = worker_script_path();
        if (found.empty()) {
            throw MLIPError(
                "mlip: could not find the worker script. Set $FASTATOMS_MLIP_WORKER or pass "
                "worker='/path/to/fastatoms/mlip/worker.py'.");
        }
        return found;
    }

    /// Read the worker's opening message: what it loaded, and what it can do.
    void handshake(const std::string& interpreter) {
        char magic[4] = {};
        std::int32_t version = 0;
        std::int32_t length = 0;
        try {
            process_->read_exact(magic, 4);
            process_->read_exact(&version, sizeof(version));
            process_->read_exact(&length, sizeof(length));
        } catch (const std::exception& error) {
            throw MLIPError("mlip: the worker (" + interpreter +
                            ") did not start correctly: " + error.what() + diagnostics());
        }
        if (std::memcmp(magic, kHelloMagic, 4) != 0) {
            throw MLIPError("mlip: the worker sent an unexpected greeting" + diagnostics());
        }
        if (version != kProtocolVersion) {
            throw MLIPError("mlip: the worker speaks protocol version " +
                            std::to_string(version) + ", this build speaks " +
                            std::to_string(kProtocolVersion));
        }
        std::string payload(static_cast<std::size_t>(std::max(0, length)), '\0');
        if (length > 0) {
            process_->read_exact(payload.data(), payload.size());
        }

        Value description;
        try {
            description = io::parse_yaml_string(payload);
        } catch (const std::exception& error) {
            throw MLIPError(std::string("mlip: the worker's description could not be read: ") +
                            error.what());
        }
        if (const Value* value = description.find("error"); value && value->is_string()) {
            throw MLIPError("mlip: the worker could not load the model: " + value->as_string() +
                            diagnostics());
        }
        if (const Value* value = description.find("cutoff"); value && value->is_number()) {
            info_.cutoff = value->as_double();
        }
        if (const Value* value = description.find("version"); value && value->is_string()) {
            info_.version = value->as_string();
        }
        if (const Value* value = description.find("path"); value && value->is_string()) {
            info_.path = value->as_string();
        }
        if (const Value* value = description.find("provides_forces"); value && value->is_bool()) {
            info_.provides_forces = value->as_bool();
        }
        if (const Value* value = description.find("provides_stress"); value && value->is_bool()) {
            info_.provides_stress = value->as_bool();
        }
        if (const Value* value = description.find("species"); value && value->is_list()) {
            for (const Value& entry : value->as_list()) {
                if (entry.is_number()) {
                    info_.species.push_back(static_cast<int>(entry.as_int()));
                }
            }
        }
    }

    std::unique_ptr<util::Subprocess> process_{};
    ModelInfo info_{};
};

}  // namespace

std::shared_ptr<Backend> make_worker_backend(const std::string& family,
                                             const std::string& model,
                                             const Value& parameters) {
    return std::make_shared<WorkerBackend>(family, model, parameters);
}

}  // namespace fastatoms::mlip
