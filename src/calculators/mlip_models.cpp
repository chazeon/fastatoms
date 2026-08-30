// SPDX-License-Identifier: MIT
#include "fastatoms/calculators/mlip_models.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <utility>

#include "fastatoms/calculators/registry.hpp"
#include "fastatoms/util/process.hpp"

namespace fastatoms::mlip {
namespace {

namespace fs = std::filesystem;

std::string lowercase(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

}  // namespace

std::string default_environment_for(const std::string& family) {
    if (family == "mace") {
        return util::environment_value("FASTATOMS_MACE_ENV", "mace_env");
    }
    return util::environment_value("FASTATOMS_MLIP_ENV", "mlip_env");
}

// ---------------------------------------------------------------------------
// MACE
// ---------------------------------------------------------------------------
MACECalculator::MACECalculator(Value parameters)
    : MLIPCalculator("mace", std::move(parameters)) {}

std::string MACECalculator::default_conda_env() const { return default_environment_for("mace"); }

std::string MACECalculator::resolve_model(const std::string& requested) const {
    // An existing file - archive or checkpoint - is used as it stands.
    std::error_code ignored;
    if (fs::exists(requested, ignored)) {
        return requested;
    }

    // MACE's own downloader caches under ~/.cache/mace with the punctuation
    // stripped out of the name, so a short name usually resolves to a file
    // that is already there. Finding it locally keeps a run offline, and keeps
    // the same model in use when the network is not available.
    const std::string home = util::environment_value("HOME");
    if (home.empty()) {
        return requested;
    }
    const fs::path cache = fs::path(home) / ".cache" / "mace";
    if (!fs::is_directory(cache, ignored)) {
        return requested;
    }

    std::string wanted;
    for (const char c : lowercase(requested)) {
        if (std::isalnum(static_cast<unsigned char>(c)) != 0) {
            wanted.push_back(c);
        }
    }
    if (wanted.empty()) {
        return requested;
    }

    for (const fs::directory_entry& entry : fs::directory_iterator(cache, ignored)) {
        if (!entry.is_regular_file(ignored)) {
            continue;
        }
        std::string candidate;
        for (const char c : lowercase(entry.path().filename().string())) {
            if (std::isalnum(static_cast<unsigned char>(c)) != 0) {
                candidate.push_back(c);
            }
        }
        if (candidate.find(wanted) != std::string::npos) {
            return entry.path().string();
        }
    }
    return requested;
}

// ---------------------------------------------------------------------------
// The families that run through the worker
// ---------------------------------------------------------------------------
MatterSimCalculator::MatterSimCalculator(Value parameters)
    : MLIPCalculator("mattersim", std::move(parameters)) {}

std::string MatterSimCalculator::default_conda_env() const {
    return default_environment_for("mattersim");
}

CHGNetCalculator::CHGNetCalculator(Value parameters)
    : MLIPCalculator("chgnet", std::move(parameters)) {}

std::string CHGNetCalculator::default_conda_env() const {
    return default_environment_for("chgnet");
}

NequIPCalculator::NequIPCalculator(Value parameters, bool allegro)
    : MLIPCalculator(allegro ? "allegro" : "nequip", std::move(parameters)) {}

std::string NequIPCalculator::default_conda_env() const {
    return default_environment_for(family_);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
void register_mlip_calculators() {
    // The built-in registry already lists these; this exists for a program
    // that has replaced one of the names and wants the defaults back, and for
    // a statically linked build that wants to be sure the symbols are pulled
    // in. Registering an existing name replaces it, so calling it twice is
    // harmless.
    static const bool once = [] {
        register_calculator("mace", [](const Value& parameters) {
            return std::make_shared<MACECalculator>(parameters);
        });
        register_calculator("mattersim", [](const Value& parameters) {
            return std::make_shared<MatterSimCalculator>(parameters);
        });
        register_calculator("chgnet", [](const Value& parameters) {
            return std::make_shared<CHGNetCalculator>(parameters);
        });
        register_calculator("nequip", [](const Value& parameters) {
            return std::make_shared<NequIPCalculator>(parameters, false);
        });
        register_calculator("allegro", [](const Value& parameters) {
            return std::make_shared<NequIPCalculator>(parameters, true);
        });
        // A generic entry, for a family the build does not know by name: the
        // worker's `ase` loader can drive any ASE calculator, and the
        // TorchScript backend any archive that follows the ABI.
        register_calculator("mlip", [](const Value& parameters) {
            return std::make_shared<MLIPCalculator>(parameters);
        });
        return true;
    }();
    (void)once;
}

}  // namespace fastatoms::mlip
