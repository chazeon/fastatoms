// SPDX-License-Identifier: MIT
#pragma once

/// \file mlip.hpp
/// \brief Machine-learning interatomic potentials: the shared calculator, the
///        graph they all consume, and the backend that evaluates them.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "fastatoms/calculators/calculator.hpp"
#include "fastatoms/core/neighbor_list.hpp"

namespace fastatoms::mlip {

/// Raised when a model cannot be found, loaded, or evaluated.
class MLIPError : public CalculatorError {
public:
    explicit MLIPError(const std::string& message) : CalculatorError(message) {}
};

/// What a backend reports about the model it has loaded.
struct ModelInfo {
    std::string family{};       ///< "mace", "mattersim", "chgnet", ...
    std::string path{};         ///< the file or checkpoint name it came from
    std::string backend{};      ///< "torchscript" or "worker"
    std::string device{"cpu"};
    std::string dtype{"float64"};
    std::string version{};      ///< whatever the model or package reports

    /// Interaction cutoff in angstrom - the radius the neighbour list must use.
    double cutoff{0.0};

    /// Atomic numbers the model was trained on. Empty means "no restriction
    /// declared", not "none supported".
    std::vector<int> species{};

    /// Whether the model returns these itself. What it does not return is
    /// obtained by differentiating the energy, which is the backend's job.
    bool provides_forces{false};
    bool provides_stress{false};

    [[nodiscard]] bool supports(int atomic_number) const;
    [[nodiscard]] std::string to_string() const;
};

/// An `Atoms` object in the form every message-passing potential consumes.
///
/// The edge list is the same object the rest of FastAtoms calls a neighbour
/// list, transposed into the layout these models expect: `edge_index` is
/// `2 * nedges`, receivers first then senders, and `unit_shifts` is the
/// integer cell offset that was applied to the sender, so that
///
///     r_ij = r[sender] + unit_shifts * cell - r[receiver]
///
/// which is the convention MACE, MatterSim, CHGNet and NequIP all use (some of
/// them spell `unit_shifts` as `pbc_offsets`).
struct Graph {
    std::size_t natoms{0};
    std::vector<double> positions{};        ///< 3N, atom-major, angstrom
    std::vector<std::int64_t> numbers{};    ///< N atomic numbers
    Mat3 cell{};                            ///< rows are the lattice vectors
    std::array<bool, 3> pbc{{false, false, false}};
    std::vector<std::int64_t> edge_index{}; ///< 2 * nedges: receivers, then senders
    std::vector<double> unit_shifts{};      ///< 3 * nedges, integer valued
    double cutoff{0.0};

    [[nodiscard]] std::size_t nedges() const noexcept { return edge_index.size() / 2; }
    [[nodiscard]] const std::int64_t* receivers() const noexcept { return edge_index.data(); }
    [[nodiscard]] const std::int64_t* senders() const noexcept {
        return edge_index.data() + nedges();
    }
};

/// Convert `atoms` into the graph a model of the given cutoff needs.
///
/// This is step 3 of the module: positions, species and cell in, the tensors a
/// neural potential wants out. It reuses the linear-scaling neighbour list, so
/// the graph costs O(N) rather than the O(N^2) an ASE-side implementation
/// would.
///
/// With `with_edges` false the edge list is skipped and `cutoff` is ignored,
/// which is what a backend that builds its own graph wants.
///
/// \throws MLIPError for an empty structure or a non-positive cutoff.
[[nodiscard]] Graph build_graph(const Atoms& atoms, double cutoff, bool with_edges = true);

/// What one evaluation produced. Units are ASE's: eV, eV/angstrom, and
/// eV/angstrom^3 in Voigt order.
struct Prediction {
    double energy{0.0};
    std::vector<double> forces{};
    std::array<double, 6> stress{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    bool has_forces{false};
    bool has_stress{false};
};

/// Evaluates a loaded model on a graph.
///
/// Two implementations ship with FastAtoms: `TorchBackend` runs a TorchScript
/// model in-process through LibTorch, and `WorkerBackend` drives a persistent
/// Python process. Both are addressed only through this interface, so a
/// calculator does not know or care which one it has.
class Backend {
public:
    virtual ~Backend() = default;

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    [[nodiscard]] virtual const ModelInfo& info() const = 0;

    /// Evaluate the model. `need_forces` and `need_stress` are requests, not
    /// guarantees: a backend may return more than was asked for when the model
    /// produces it anyway, and `Prediction::has_*` says what is actually there.
    [[nodiscard]] virtual Prediction evaluate(const Graph& graph, bool need_forces,
                                              bool need_stress) = 0;

    /// Whether `evaluate` needs `Graph::edge_index` filled in.
    ///
    /// The in-process backend does - that is the whole point of building the
    /// neighbour list in C++. A backend that hands the structure to a package's
    /// own ASE calculator does not, and building a list it will throw away
    /// would double the cost of every step.
    [[nodiscard]] virtual bool needs_edges() const { return true; }

    /// Free whatever the model holds. The next evaluation reloads it.
    virtual void unload() {}

protected:
    Backend() = default;
};

/// A calculator backed by a machine-learning interatomic potential.
///
/// The class is concrete: `MACECalculator` and the rest are thin subclasses
/// that preset a family name, a default model and a default Python
/// environment. Everything that is common - resolving the model, choosing a
/// backend, building the graph, checking that the structure only contains
/// species the model knows, converting units and filling `Results` - lives
/// here.
///
/// Parameters:
///   - `model`       - path to a TorchScript archive or a checkpoint, or a
///                     short name the family understands ("small", "medium",
///                     "MatterSim-v1.0.0-1M.pth", ...)
///   - `backend`     - "auto" (default), "torchscript" or "worker"
///   - `device`      - "cpu" (default), "cuda", "mps"
///   - `dtype`       - "float64" (default) or "float32"
///   - `cutoff`      - override the model's own cutoff, in angstrom
///   - `python`      - interpreter for the worker backend
///   - `conda_env`   - conda environment to take that interpreter from
///   - `worker`      - path to the worker script, when it is not alongside the
///                     installed package
///   - `threads`     - torch intra-op threads (0 leaves the default alone)
///   - `skin`        - extra neighbour-list radius in angstrom (default 0)
class MLIPCalculator : public Calculator {
public:
    explicit MLIPCalculator(Value parameters = {});
    ~MLIPCalculator() override;

    [[nodiscard]] std::string name() const override;
    [[nodiscard]] std::vector<Property> implemented_properties() const override;
    [[nodiscard]] std::string to_string() const override;

    /// Load the model now rather than on the first evaluation. Useful when the
    /// cost of loading should not be charged to the first MD step.
    void load();

    /// True once the backend exists.
    [[nodiscard]] bool loaded() const noexcept { return backend_ != nullptr; }

    /// Metadata about the loaded model; loads it if that has not happened yet.
    [[nodiscard]] const ModelInfo& model();

    /// The neighbour-list radius actually in use: the model's cutoff plus the
    /// configured skin, unless `cutoff` was set explicitly.
    [[nodiscard]] double cutoff();

    /// The graph that would be sent to the model for this structure.
    [[nodiscard]] Graph graph(const Atoms& atoms);

    /// Drop the loaded model, ending a worker process if there is one.
    void unload();

    /// Use a backend built elsewhere, instead of one made from the parameters.
    ///
    /// This is how a program supplies its own evaluator - a model already
    /// resident in another process, a hardware accelerator, a stub - without
    /// subclassing anything. Passing `nullptr` restores the normal behaviour.
    void set_backend(std::shared_ptr<Backend> backend);

    /// The family this calculator represents, e.g. "mace".
    [[nodiscard]] const std::string& family() const noexcept { return family_; }

protected:
    /// Subclasses call this to preset their identity before parameters are read.
    MLIPCalculator(std::string family, Value parameters);

    void compute(const Atoms& atoms, const std::vector<Property>& properties) override;

    /// Resolve the `model` parameter into something a backend can open. The
    /// base implementation returns it unchanged; families that understand
    /// short names override it.
    [[nodiscard]] virtual std::string resolve_model(const std::string& requested) const;

    /// Default Python environment for the worker backend. Empty means "use
    /// whatever `python` resolves to".
    [[nodiscard]] virtual std::string default_conda_env() const { return {}; }

    /// Model used when none was configured.
    [[nodiscard]] virtual std::string default_model() const { return {}; }

    std::string family_{"mlip"};
    std::shared_ptr<Backend> backend_{};
};

/// Build a backend for a family and parameter set, choosing between the
/// in-process and worker paths.
///
/// `backend`: "torchscript" insists on LibTorch, "worker" insists on Python,
/// and "auto" (the default) picks TorchScript when the model is a `.pt`
/// archive and LibTorch is compiled in, and the worker otherwise.
[[nodiscard]] std::shared_ptr<Backend> make_backend(const std::string& family,
                                                    const std::string& model,
                                                    const Value& parameters);

/// True when FastAtoms was built with LibTorch, so the in-process backend is
/// available at all.
[[nodiscard]] bool torch_available() noexcept;

/// The LibTorch version FastAtoms was built against, or an empty string.
[[nodiscard]] std::string torch_version();

}  // namespace fastatoms::mlip
