// SPDX-License-Identifier: MIT
#include "fastatoms/calculators/mlip_torch.hpp"

#include <string>

namespace fastatoms::mlip {

#if !defined(FASTATOMS_WITH_TORCH)

// ---------------------------------------------------------------------------
// Built without LibTorch: the entry point exists so that callers get a useful
// message instead of a link error.
// ---------------------------------------------------------------------------
bool torch_available() noexcept { return false; }

std::string torch_version() { return {}; }

std::shared_ptr<Backend> make_torch_backend(const std::string&, const std::string&,
                                            const Value&) {
    throw MLIPError(
        "mlip: this build has no LibTorch, so TorchScript models cannot be evaluated "
        "in process. Reconfigure with\n"
        "    cmake -S . -B build -DFASTATOMS_WITH_TORCH=ON "
        "-DFASTATOMS_TORCH_PYTHON=$HOME/miniconda3/envs/mace_env/bin/python\n"
        "or use the Python worker backend instead (backend='worker').");
}

#else

}  // namespace fastatoms::mlip

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <vector>

#include "fastatoms/io/yaml_input.hpp"

namespace fastatoms::mlip {
namespace {

torch::Device parse_device(const std::string& name) {
    if (name.empty() || name == "cpu") {
        return torch::Device(torch::kCPU);
    }
    if (name == "cuda" || name.rfind("cuda:", 0) == 0) {
        if (!torch::cuda::is_available()) {
            throw MLIPError("mlip: device '" + name +
                            "' was requested but this LibTorch build sees no CUDA device");
        }
        return torch::Device(name);
    }
    if (name == "mps") {
        if (!torch::hasMPS()) {
            throw MLIPError("mlip: device 'mps' was requested but this LibTorch build has "
                            "no Metal backend");
        }
        return torch::Device(torch::kMPS);
    }
    return torch::Device(name);
}

/// Read a module attribute, returning an empty optional when it is absent.
/// TorchScript throws for a missing attribute, and "absent" is a normal answer
/// here - the ABI's metadata is all optional.
c10::optional<c10::IValue> optional_attr(const torch::jit::Module& module,
                                         const std::string& name) {
    try {
        return module.attr(name);
    } catch (const std::exception&) {
        return c10::nullopt;
    }
}

double attr_double(const torch::jit::Module& module, const std::vector<std::string>& names,
                   double fallback) {
    for (const std::string& name : names) {
        if (const auto value = optional_attr(module, name)) {
            if (value->isTensor()) {
                const at::Tensor tensor = value->toTensor();
                if (tensor.numel() >= 1) {
                    return tensor.flatten()[0].item<double>();
                }
            } else if (value->isDouble()) {
                return value->toDouble();
            } else if (value->isInt()) {
                return static_cast<double>(value->toInt());
            }
        }
    }
    return fallback;
}

bool attr_bool(const torch::jit::Module& module, const std::string& name, bool fallback) {
    if (const auto value = optional_attr(module, name)) {
        if (value->isBool()) {
            return value->toBool();
        }
        if (value->isTensor()) {
            const at::Tensor tensor = value->toTensor();
            if (tensor.numel() >= 1) {
                return tensor.flatten()[0].item<double>() != 0.0;
            }
        }
        if (value->isInt()) {
            return value->toInt() != 0;
        }
    }
    return fallback;
}

std::vector<int> attr_species(const torch::jit::Module& module) {
    for (const char* name : {"atomic_numbers", "species", "supported_species"}) {
        if (const auto value = optional_attr(module, name)) {
            if (value->isTensor()) {
                const at::Tensor tensor = value->toTensor().to(torch::kCPU).flatten();
                std::vector<int> species;
                species.reserve(static_cast<std::size_t>(tensor.numel()));
                for (int64_t k = 0; k < tensor.numel(); ++k) {
                    species.push_back(static_cast<int>(tensor[k].item<int64_t>()));
                }
                return species;
            }
        }
    }
    return {};
}

/// Torch's global thread count is process-wide, so set it once.
void set_threads_once(int threads) {
    if (threads <= 0) {
        return;
    }
    static std::once_flag flag;
    std::call_once(flag, [threads] { torch::set_num_threads(threads); });
}

// -------------------------------------------------------------------------
// The backend
// -------------------------------------------------------------------------
class TorchBackend final : public Backend {
public:
    TorchBackend(const std::string& family, const std::string& model, const Value& parameters) {
        info_.family = family;
        info_.path = model;
        info_.backend = "torchscript";

        if (const Value* value = parameters.find("device"); value && value->is_string()) {
            info_.device = value->as_string();
        }
        if (const Value* value = parameters.find("dtype"); value && value->is_string()) {
            info_.dtype = value->as_string();
        }
        int threads = 0;
        if (const Value* value = parameters.find("threads"); value && value->is_number()) {
            threads = static_cast<int>(value->as_int());
        }
        set_threads_once(threads);

        device_ = parse_device(info_.device);
        scalar_type_ = info_.dtype == "float32" || info_.dtype == "float"
                           ? torch::kFloat32
                           : torch::kFloat64;

        // The exporter writes a small JSON description alongside the code;
        // asking for it here means the archive describes itself and the C++
        // side needs no per-family knowledge at all.
        std::error_code ignored;
        if (!std::filesystem::exists(model, ignored)) {
            throw MLIPError("mlip: no such model file: '" + model + "'");
        }

        torch::jit::ExtraFilesMap extra_files;
        extra_files["fastatoms_mlip.json"] = "";
        try {
            module_ = torch::jit::load(model, device_, extra_files);
        } catch (const std::exception& error) {
            // LibTorch appends its own C++ backtrace to every message. The
            // first line is the part a user can act on; the rest is noise in a
            // calculator's error report.
            std::string reason(error.what());
            if (const std::size_t newline = reason.find('\n'); newline != std::string::npos) {
                reason = reason.substr(0, newline);
            }
            std::ostringstream message;
            message << "mlip: could not load '" << model << "' as a TorchScript model.\n"
                    << "LibTorch said: " << reason << "\n"
                    << "A raw MACE .model or MatterSim .pth checkpoint is a pickled Python "
                       "object, not a TorchScript archive. Export it first:\n"
                    << "    python -m fastatoms.mlip.export " << model << " -o model.pt";
            throw MLIPError(message.str());
        }
        module_.eval();

        // Buffers first - every adapter registers its cutoff and species that
        // way - then the JSON, which wins where the two disagree because it is
        // what the exporter deliberately recorded.
        info_.cutoff = attr_double(module_, {"cutoff", "r_max", "two_body_cutoff"}, 0.0);
        info_.species = attr_species(module_);
        info_.provides_forces = attr_bool(module_, "provides_forces", false);
        info_.provides_stress = attr_bool(module_, "provides_stress", false);
        read_metadata(extra_files["fastatoms_mlip.json"], family);
    }

    [[nodiscard]] const ModelInfo& info() const override { return info_; }

    [[nodiscard]] Prediction evaluate(const Graph& graph, bool need_forces,
                                      bool need_stress) override {
        // Grad mode stays on: models that return their own forces compute them
        // by differentiating internally, so switching it off would break them.
        torch::AutoGradMode grad_mode(true);

        const auto n = static_cast<int64_t>(graph.natoms);
        const auto m = static_cast<int64_t>(graph.nedges());
        const auto options = torch::TensorOptions().dtype(scalar_type_);

        at::Tensor positions =
            torch::from_blob(const_cast<double*>(graph.positions.data()), {n, 3},
                             torch::kFloat64)
                .to(scalar_type_)
                .to(device_);
        at::Tensor cell = torch::from_blob(const_cast<double*>(graph.cell.m.data()), {3, 3},
                                           torch::kFloat64)
                              .to(scalar_type_)
                              .to(device_);
        at::Tensor numbers = torch::from_blob(const_cast<std::int64_t*>(graph.numbers.data()),
                                              {n}, torch::kInt64)
                                 .clone()
                                 .to(device_);
        at::Tensor edge_index =
            torch::from_blob(const_cast<std::int64_t*>(graph.edge_index.data()), {2, m},
                             torch::kInt64)
                .clone()
                .to(device_);
        at::Tensor unit_shifts =
            torch::from_blob(const_cast<double*>(graph.unit_shifts.data()), {m, 3},
                             torch::kFloat64)
                .to(scalar_type_)
                .to(device_);

        // What the model does not give us, we differentiate for. Forces come
        // from the gradient with respect to the positions; stress from the
        // gradient with respect to a strain applied to positions and cell
        // alike, which is the standard trick and the reason the cell has to
        // enter the graph as a tensor rather than as a constant.
        const bool autodiff_forces = need_forces && !info_.provides_forces;
        const bool autodiff_stress = need_stress && !info_.provides_stress;

        at::Tensor strain;
        if (autodiff_forces) {
            positions = positions.detach().requires_grad_(true);
        }
        if (autodiff_stress) {
            strain = torch::zeros({3, 3}, options.device(device_)).requires_grad_(true);
            // Symmetrised, so the gradient is a symmetric stress tensor and a
            // rigid rotation of the cell costs nothing.
            const at::Tensor symmetric = 0.5 * (strain + strain.transpose(0, 1));
            const at::Tensor deformation =
                torch::eye(3, options.device(device_)) + symmetric;
            positions = positions.matmul(deformation);
            cell = cell.matmul(deformation);
        }

        c10::Dict<std::string, at::Tensor> input;
        input.insert("positions", positions);
        input.insert("atomic_numbers", numbers);
        input.insert("cell", cell);
        input.insert("edge_index", edge_index);
        input.insert("unit_shifts", unit_shifts);

        c10::IValue raw;
        try {
            raw = module_.forward({input});
        } catch (const std::exception& error) {
            throw MLIPError(std::string("mlip: the model failed during evaluation: ") +
                            error.what());
        }
        if (!raw.isGenericDict()) {
            throw MLIPError(
                "mlip: the model returned " + std::string(raw.tagKind()) +
                " where the FastAtoms ABI expects a Dict[str, Tensor] with an 'energy' key");
        }
        const c10::Dict<c10::IValue, c10::IValue> output = raw.toGenericDict();

        const auto lookup = [&output](const char* key) -> at::Tensor {
            const auto found = output.find(key);
            if (found == output.end() || !found->value().isTensor()) {
                return {};
            }
            return found->value().toTensor();
        };

        at::Tensor energy = lookup("energy");
        if (!energy.defined()) {
            energy = lookup("total_energy");
        }
        if (!energy.defined()) {
            throw MLIPError("mlip: the model returned no 'energy' entry");
        }
        energy = energy.flatten();
        if (energy.numel() != 1) {
            energy = energy.sum();
        }
        energy = energy.reshape({});

        Prediction prediction;
        prediction.energy = energy.detach().to(torch::kCPU).item<double>();

        at::Tensor forces = lookup("forces");
        at::Tensor stress = lookup("stress");

        if (autodiff_forces || autodiff_stress) {
            std::vector<at::Tensor> inputs;
            if (autodiff_forces) {
                inputs.push_back(positions);
            }
            if (autodiff_stress) {
                inputs.push_back(strain);
            }
            std::vector<at::Tensor> gradients;
            try {
                gradients = torch::autograd::grad({energy}, inputs, /*grad_outputs=*/{},
                                                  /*retain_graph=*/c10::nullopt,
                                                  /*create_graph=*/false,
                                                  /*allow_unused=*/true);
            } catch (const std::exception& error) {
                std::string reason(error.what());
                if (const std::size_t newline = reason.find('\n');
                    newline != std::string::npos) {
                    reason = reason.substr(0, newline);
                }
                std::ostringstream message;
                message << "mlip: differentiating the model's energy failed: " << reason;
                if (reason.find("backward through the graph a second time") !=
                    std::string::npos) {
                    // The usual cause: the model already differentiated its own
                    // energy inside forward, which frees the graph on the way out.
                    message << "\nThis model computes its own derivatives, which "
                               "consumes the graph before FastAtoms can use it. Export "
                               "it declaring provides_forces and provides_stress (the "
                               "default for MACE), so its own forces and stress are used "
                               "instead of being differentiated again.";
                } else {
                    message << "\nA model that cannot be differentiated must return its "
                               "own forces and stress and declare provides_forces / "
                               "provides_stress.";
                }
                throw MLIPError(message.str());
            }
            std::size_t next = 0;
            if (autodiff_forces) {
                const at::Tensor gradient = gradients[next++];
                if (!gradient.defined()) {
                    throw MLIPError(
                        "mlip: the energy does not depend on the positions, so no forces "
                        "could be differentiated out of it");
                }
                forces = -gradient;
            }
            if (autodiff_stress) {
                const at::Tensor gradient = gradients[next++];
                if (gradient.defined()) {
                    stress = gradient;
                }
            }
        }

        if (forces.defined() && need_forces) {
            const at::Tensor host = forces.detach().to(torch::kCPU).to(torch::kFloat64).contiguous();
            prediction.forces.assign(host.data_ptr<double>(),
                                     host.data_ptr<double>() + host.numel());
            prediction.has_forces = true;
        }

        if (stress.defined() && need_stress) {
            at::Tensor host =
                stress.detach().to(torch::kCPU).to(torch::kFloat64).contiguous().flatten();
            const double volume = std::abs(graph.cell.determinant());
            std::array<double, 6> voigt{};
            if (host.numel() == 9) {
                const double* s = host.data_ptr<double>();
                // The model returned dE/dstrain when we injected the strain
                // ourselves; a model that reports its own stress has already
                // divided by the volume.
                const double scale = info_.provides_stress ? 1.0 : (volume > 0.0 ? 1.0 / volume : 0.0);
                voigt = {scale * s[0], scale * s[4], scale * s[8],
                         scale * 0.5 * (s[5] + s[7]), scale * 0.5 * (s[2] + s[6]),
                         scale * 0.5 * (s[1] + s[3])};
            } else if (host.numel() == 6) {
                const double* s = host.data_ptr<double>();
                voigt = {s[0], s[1], s[2], s[3], s[4], s[5]};
            } else {
                throw MLIPError("mlip: the model returned a stress with " +
                                std::to_string(host.numel()) +
                                " entries; expected 9 (3x3) or 6 (Voigt)");
            }
            prediction.stress = voigt;
            prediction.has_stress = true;
        }

        return prediction;
    }

private:
    /// Apply the exporter's JSON description, if the archive carries one.
    void read_metadata(const std::string& payload, const std::string& family) {
        if (payload.empty()) {
            return;
        }
        Value description;
        try {
            description = io::parse_yaml_string(payload);
        } catch (const std::exception&) {
            return;  // an unreadable description is not worth failing a load over
        }
        if (const Value* value = description.find("cutoff"); value && value->is_number()) {
            info_.cutoff = value->as_double();
        }
        if (const Value* value = description.find("provides_forces");
            value && value->is_bool()) {
            info_.provides_forces = value->as_bool();
        }
        if (const Value* value = description.find("provides_stress");
            value && value->is_bool()) {
            info_.provides_stress = value->as_bool();
        }
        if (const Value* value = description.find("model_version"); value && value->is_string()) {
            info_.version = value->as_string();
        }
        if (const Value* value = description.find("dtype"); value && value->is_string()) {
            info_.dtype = value->as_string();
        }
        if (const Value* value = description.find("family");
            value && value->is_string() && family.empty()) {
            info_.family = value->as_string();
        }
        if (const Value* value = description.find("species");
            value && value->is_list() && info_.species.empty()) {
            for (const Value& entry : value->as_list()) {
                if (entry.is_number()) {
                    info_.species.push_back(static_cast<int>(entry.as_int()));
                }
            }
        }
    }

    torch::jit::Module module_{};
    torch::Device device_{torch::kCPU};
    torch::ScalarType scalar_type_{torch::kFloat64};
    ModelInfo info_{};
};

}  // namespace

bool torch_available() noexcept { return true; }

std::string torch_version() { return TORCH_VERSION; }

std::shared_ptr<Backend> make_torch_backend(const std::string& family, const std::string& model,
                                            const Value& parameters) {
    return std::make_shared<TorchBackend>(family, model, parameters);
}

#endif  // FASTATOMS_WITH_TORCH

}  // namespace fastatoms::mlip
