// SPDX-License-Identifier: MIT
#pragma once

/// \file mlip_models.hpp
/// \brief The individual machine-learning potentials.
///
/// Each class is a thin specialisation of `MLIPCalculator`: it names its
/// family, its default checkpoint and the Python environment that has the
/// package installed. Everything else - graph construction, backend choice,
/// autodiff, caching - is shared.

#include <string>

#include "fastatoms/calculators/mlip.hpp"

namespace fastatoms::mlip {

/// MACE: equivariant message passing, and the one family that exports cleanly
/// to TorchScript, so it can run entirely in C++ through LibTorch.
///
/// `model` accepts an exported archive (`*.pt`, run in process), a `.model`
/// checkpoint, or one of the published names - "small", "medium", "large" for
/// MACE-MP, "off-small" and friends for MACE-OFF. Names and checkpoints need
/// the Python worker, because loading them means unpickling a Python object.
///
///     atoms.calc = fa.MACE(model="mace-off23-small.pt")   # in-process
///     atoms.calc = fa.MACE(model="medium")                # worker, mace_env
class MACECalculator : public MLIPCalculator {
public:
    explicit MACECalculator(Value parameters = {});

    [[nodiscard]] std::string name() const override { return "mace"; }

protected:
    [[nodiscard]] std::string default_conda_env() const override;
    [[nodiscard]] std::string default_model() const override { return "small"; }
    [[nodiscard]] std::string resolve_model(const std::string& requested) const override;
};

/// MatterSim: Microsoft's M3GNet-based universal potential.
///
/// This one only runs through the worker. Its graph convolutions are assembled
/// with `functools.partial`, which `torch.jit.script` cannot compile, so there
/// is no TorchScript archive to hand to LibTorch. The worker keeps one
/// interpreter alive for the whole run, so the cost is a pipe round trip per
/// step rather than an interpreter start.
///
///     atoms.calc = fa.MatterSim()                            # 1M checkpoint
///     atoms.calc = fa.MatterSim(model="MatterSim-v1.0.0-5M.pth")
class MatterSimCalculator : public MLIPCalculator {
public:
    explicit MatterSimCalculator(Value parameters = {});

    [[nodiscard]] std::string name() const override { return "mattersim"; }

protected:
    [[nodiscard]] std::string default_conda_env() const override;
    [[nodiscard]] std::string default_model() const override {
        return "MatterSim-v1.0.0-1M.pth";
    }
};

/// CHGNet: charge-informed universal potential, driven through the worker.
class CHGNetCalculator : public MLIPCalculator {
public:
    explicit CHGNetCalculator(Value parameters = {});

    [[nodiscard]] std::string name() const override { return "chgnet"; }

protected:
    [[nodiscard]] std::string default_conda_env() const override;
    [[nodiscard]] std::string default_model() const override { return "chgnet"; }
};

/// NequIP, and Allegro which deploys through it. A model compiled with
/// `nequip-compile` is TorchScript and runs in process; anything else goes
/// through the worker.
class NequIPCalculator : public MLIPCalculator {
public:
    /// \param allegro report the family as "allegro" rather than "nequip".
    explicit NequIPCalculator(Value parameters = {}, bool allegro = false);

    [[nodiscard]] std::string name() const override { return family_; }

protected:
    [[nodiscard]] std::string default_conda_env() const override;
};

/// The conda environment a family defaults to, before any parameter overrides
/// it: `mace_env` for MACE, `mlip_env` for everything else.
///
/// Both can be pointed elsewhere with `$FASTATOMS_MACE_ENV` and
/// `$FASTATOMS_MLIP_ENV`, which is how a machine with different environment
/// names avoids editing any code.
[[nodiscard]] std::string default_environment_for(const std::string& family);

/// Re-register `mace`, `mattersim`, `chgnet`, `nequip`, `allegro` and the
/// generic `mlip` with their built-in factories.
///
/// The registry lists them already. This is for putting a default back after
/// something has replaced it, and it is safe to call more than once.
void register_mlip_calculators();

}  // namespace fastatoms::mlip
