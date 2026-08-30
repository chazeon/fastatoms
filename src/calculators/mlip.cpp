// SPDX-License-Identifier: MIT
#include "fastatoms/calculators/mlip.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

#include "fastatoms/calculators/mlip_torch.hpp"
#include "fastatoms/calculators/mlip_worker.hpp"
#include "fastatoms/core/elements.hpp"
#include "fastatoms/util/text.hpp"

namespace fastatoms::mlip {
namespace {

/// Lower-case, with hyphens and underscores removed, for tolerant matching.
std::string canonical(std::string_view text) {
    std::string key;
    key.reserve(text.size());
    for (const char c : text) {
        if (c == '-' || c == '_' || c == ' ') {
            continue;
        }
        key.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return key;
}

bool ends_with(const std::string& text, std::string_view suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// ModelInfo
// ---------------------------------------------------------------------------
bool ModelInfo::supports(int atomic_number) const {
    if (species.empty()) {
        return true;
    }
    return std::find(species.begin(), species.end(), atomic_number) != species.end();
}

std::string ModelInfo::to_string() const {
    std::ostringstream out;
    out << (family.empty() ? "mlip" : family);
    if (!path.empty()) {
        out << "('" << path << "'";
        out << ", backend=" << backend;
        if (cutoff > 0.0) {
            out << ", cutoff=" << cutoff << " A";
        }
        if (!species.empty()) {
            out << ", " << species.size() << " species";
        }
        out << ")";
    }
    return out.str();
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------
Graph build_graph(const Atoms& atoms, double cutoff, bool with_edges) {
    if (atoms.empty()) {
        throw MLIPError("mlip: cannot evaluate an empty structure");
    }
    if (with_edges && !(cutoff > 0.0)) {
        throw MLIPError("mlip: the model cutoff must be positive, got " +
                        std::to_string(cutoff));
    }

    Graph graph;
    graph.natoms = atoms.size();
    graph.cell = atoms.cell().matrix();
    graph.pbc = atoms.pbc();
    graph.cutoff = cutoff;

    graph.positions.assign(atoms.positions().begin(), atoms.positions().end());
    graph.numbers.reserve(atoms.size());
    for (const int z : atoms.numbers()) {
        graph.numbers.push_back(static_cast<std::int64_t>(z));
    }

    if (!with_edges) {
        graph.cutoff = 0.0;
        return graph;
    }

    // A full list: these models want both (i, j) and (j, i), because messages
    // travel along every edge in one direction only.
    NeighborListOptions options;
    options.cutoff = cutoff;
    options.full = true;
    options.self_interaction = false;
    options.store_vectors = false;
    options.store_distances = false;
    const NeighborList pairs = build_neighbor_list(atoms, options);

    const std::size_t nedges = pairs.size();
    graph.edge_index.resize(2 * nedges);
    graph.unit_shifts.resize(3 * nedges);
    for (std::size_t e = 0; e < nedges; ++e) {
        graph.edge_index[e] = static_cast<std::int64_t>(pairs.i[e]);
        graph.edge_index[nedges + e] = static_cast<std::int64_t>(pairs.j[e]);
        graph.unit_shifts[3 * e + 0] = static_cast<double>(pairs.shifts[3 * e + 0]);
        graph.unit_shifts[3 * e + 1] = static_cast<double>(pairs.shifts[3 * e + 1]);
        graph.unit_shifts[3 * e + 2] = static_cast<double>(pairs.shifts[3 * e + 2]);
    }
    return graph;
}

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------
std::shared_ptr<Backend> make_backend(const std::string& family, const std::string& model,
                                      const Value& parameters) {
    std::string choice = "auto";
    if (const Value* requested = parameters.find("backend"); requested && requested->is_string()) {
        choice = canonical(requested->as_string());
    }

    const bool looks_compiled = ends_with(model, ".pt") || ends_with(model, ".pt2") ||
                                ends_with(model, ".ts") || ends_with(model, ".torchscript");

    if (choice == "torchscript" || choice == "torch" || choice == "libtorch") {
        if (!torch_available()) {
            throw MLIPError(
                "mlip: backend 'torchscript' was requested but FastAtoms was built without "
                "LibTorch. Reconfigure with -DFASTATOMS_WITH_TORCH=ON "
                "-DFASTATOMS_TORCH_PYTHON=<interpreter that has torch>, or use "
                "backend='worker'.");
        }
        return make_torch_backend(family, model, parameters);
    }
    if (choice == "worker" || choice == "python" || choice == "subprocess") {
        return make_worker_backend(family, model, parameters);
    }
    if (choice != "auto") {
        throw MLIPError("mlip: unknown backend '" + choice +
                        "'; expected 'auto', 'torchscript' or 'worker'");
    }

    // Automatic: an exported archive runs in-process when LibTorch is there,
    // and anything else - a raw checkpoint, a model that cannot be scripted -
    // goes to the Python worker.
    if (looks_compiled && torch_available()) {
        return make_torch_backend(family, model, parameters);
    }
    return make_worker_backend(family, model, parameters);
}

// ---------------------------------------------------------------------------
// MLIPCalculator
// ---------------------------------------------------------------------------
MLIPCalculator::MLIPCalculator(Value parameters) : Calculator(std::move(parameters)) {}

MLIPCalculator::MLIPCalculator(std::string family, Value parameters)
    : Calculator(std::move(parameters)), family_(std::move(family)) {}

MLIPCalculator::~MLIPCalculator() = default;

std::string MLIPCalculator::name() const { return family_; }

std::vector<Property> MLIPCalculator::implemented_properties() const {
    return {Property::Energy, Property::FreeEnergy, Property::Forces, Property::Stress};
}

std::string MLIPCalculator::resolve_model(const std::string& requested) const {
    return requested;
}

void MLIPCalculator::load() {
    if (backend_) {
        return;
    }
    std::string requested;
    if (const Value* value = parameters_.find("model"); value && value->is_string()) {
        requested = value->as_string();
    } else if (const Value* path = parameters_.find("path"); path && path->is_string()) {
        requested = path->as_string();
    }
    if (requested.empty()) {
        requested = default_model();
    }
    if (requested.empty()) {
        fail("no model was given; set the 'model' parameter to a TorchScript archive, a "
             "checkpoint, or a name this family recognises");
    }

    // Give the backend the family's default environment unless one was set.
    Value settings = parameters_.is_map() ? parameters_ : Value(Value::Map{});
    if (!settings.contains("conda_env") && !settings.contains("python")) {
        const std::string env = default_conda_env();
        if (!env.empty()) {
            settings.insert("conda_env", Value(env));
        }
    }

    backend_ = make_backend(family_, resolve_model(requested), settings);
}

void MLIPCalculator::unload() {
    backend_.reset();
    reset();
}

void MLIPCalculator::set_backend(std::shared_ptr<Backend> backend) {
    backend_ = std::move(backend);
    reset();
}

const ModelInfo& MLIPCalculator::model() {
    load();
    return backend_->info();
}

double MLIPCalculator::cutoff() {
    if (const Value* value = parameters_.find("cutoff"); value && value->is_number()) {
        const double explicit_cutoff = value->as_double();
        if (!(explicit_cutoff > 0.0)) {
            fail("cutoff must be positive");
        }
        return explicit_cutoff;
    }
    const ModelInfo& info = model();
    double skin = 0.0;
    if (const Value* value = parameters_.find("skin"); value && value->is_number()) {
        skin = std::max(0.0, value->as_double());
    }
    if (!(info.cutoff > 0.0)) {
        fail("the model did not report a cutoff; set the 'cutoff' parameter explicitly");
    }
    return info.cutoff + skin;
}

Graph MLIPCalculator::graph(const Atoms& atoms) {
    load();
    const bool with_edges = backend_->needs_edges();
    return build_graph(atoms, with_edges ? cutoff() : 0.0, with_edges);
}

void MLIPCalculator::compute(const Atoms& atoms, const std::vector<Property>& properties) {
    load();
    const ModelInfo& info = backend_->info();

    // A neural potential extrapolates silently outside its training set, so an
    // element it has never seen is worth an error rather than a number.
    if (!info.species.empty()) {
        std::vector<int> unknown;
        for (const int z : atoms.numbers()) {
            if (!info.supports(z) &&
                std::find(unknown.begin(), unknown.end(), z) == unknown.end()) {
                unknown.push_back(z);
            }
        }
        if (!unknown.empty()) {
            std::ostringstream message;
            message << "the model does not cover ";
            for (std::size_t k = 0; k < unknown.size(); ++k) {
                message << (k ? ", " : "") << symbol_from_number(unknown[k]);
            }
            message << "; it was trained on " << info.species.size() << " elements";
            fail(message.str());
        }
    }

    // One model call produces energy and forces together - for a network that
    // differentiates its own energy they come out of the same backward pass -
    // so asking only for what was requested would mean evaluating the model
    // twice for the energy and then the forces of the same structure. Every MD
    // step and every optimizer step does exactly that, so always take both.
    (void)properties;
    const bool want_forces = true;

    // Stress is only meaningful for a periodic cell, and asking a model to
    // differentiate with respect to a strain that cannot act is a waste.
    const bool periodic = atoms.cell().any_pbc() && atoms.cell().is_invertible();
    const bool want_stress = periodic;

    const Graph g = graph(atoms);
    Prediction prediction = backend_->evaluate(g, want_forces, want_stress);

    results_.clear();
    results_.energy = prediction.energy;
    results_.free_energy = prediction.energy;
    if (prediction.has_forces) {
        if (prediction.forces.size() != 3 * atoms.size()) {
            fail("the model returned " + std::to_string(prediction.forces.size() / 3) +
                 " force vectors for " + std::to_string(atoms.size()) + " atoms");
        }
        results_.forces = std::move(prediction.forces);
    }
    if (prediction.has_stress && periodic) {
        results_.stress = prediction.stress;
    }

    Value extra(Value::Map{});
    extra.insert("model", Value(info.path));
    extra.insert("backend", Value(info.backend));
    extra.insert("family", Value(info.family));
    extra.insert("cutoff", Value(g.cutoff));
    extra.insert("edges", Value(static_cast<std::int64_t>(g.nedges())));
    if (!info.version.empty()) {
        extra.insert("version", Value(info.version));
    }
    results_.extra = std::move(extra);
}

std::string MLIPCalculator::to_string() const {
    std::ostringstream out;
    out << family_ << "(";
    if (backend_) {
        const ModelInfo& info = backend_->info();
        out << "'" << info.path << "', backend=" << info.backend << ", device=" << info.device;
        if (info.cutoff > 0.0) {
            out << ", cutoff=" << info.cutoff;
        }
    } else {
        const Value* model = parameters_.find("model");
        out << (model && model->is_string() ? "'" + model->as_string() + "'" : "not loaded");
    }
    out << ")";
    return out.str();
}

}  // namespace fastatoms::mlip
