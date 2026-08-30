// SPDX-License-Identifier: MIT
#include "fastatoms/optimize/registry.hpp"

#include <cctype>
#include <map>
#include <mutex>

#include "fastatoms/optimize/bfgs.hpp"
#include "fastatoms/optimize/conjugate_gradient.hpp"
#include "fastatoms/optimize/fire.hpp"
#include "fastatoms/optimize/lbfgs.hpp"
#include "fastatoms/optimize/steepest_descent.hpp"

namespace fastatoms::optimize {
namespace {

std::string canonical(std::string_view name) {
    std::string key;
    for (const char c : name) {
        if (c != '-' && c != '_' && c != ' ') {
            key += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    return key;
}

struct Registry {
    std::mutex mutex;
    std::map<std::string, OptimizerFactory> factories;
    std::map<std::string, std::string> display_names;
};

Registry& registry() {
    static Registry instance;
    static const bool initialised = [] {
        const auto add = [](const char* name, OptimizerFactory factory) {
            instance.factories[canonical(name)] = std::move(factory);
            instance.display_names[canonical(name)] = name;
        };

        add("sd", [](const Value& p) { return std::make_shared<SteepestDescent>(p); });
        add("cg", [](const Value& p) { return std::make_shared<ConjugateGradient>(p); });
        add("bfgs", [](const Value& p) { return std::make_shared<BFGS>(p); });
        add("lbfgs", [](const Value& p) { return std::make_shared<LBFGS>(p); });
        add("fire", [](const Value& p) { return std::make_shared<Fire>(p); });

        // Aliases share a factory but are not listed separately.
        for (const char* alias : {"steepest_descent", "steepestdescent", "gd", "gradient_descent"}) {
            instance.factories[canonical(alias)] = instance.factories[canonical("sd")];
        }
        for (const char* alias : {"conjugate_gradient", "conjugategradient"}) {
            instance.factories[canonical(alias)] = instance.factories[canonical("cg")];
        }
        for (const char* alias : {"l-bfgs", "limited_memory_bfgs"}) {
            instance.factories[canonical(alias)] = instance.factories[canonical("lbfgs")];
        }
        return true;
    }();
    (void)initialised;
    return instance;
}

}  // namespace

void register_optimizer(const std::string& name, OptimizerFactory factory) {
    if (!factory) {
        throw OptimizerError("register_optimizer: the factory for '" + name + "' is empty");
    }
    Registry& store = registry();
    const std::lock_guard<std::mutex> lock(store.mutex);
    store.factories[canonical(name)] = std::move(factory);
    store.display_names[canonical(name)] = name;
}

void unregister_optimizer(const std::string& name) {
    Registry& store = registry();
    const std::lock_guard<std::mutex> lock(store.mutex);
    store.factories.erase(canonical(name));
    store.display_names.erase(canonical(name));
}

std::shared_ptr<Optimizer> make_optimizer(std::string_view name, const Value& parameters) {
    Registry& store = registry();
    OptimizerFactory factory;
    {
        const std::lock_guard<std::mutex> lock(store.mutex);
        const auto it = store.factories.find(canonical(name));
        if (it != store.factories.end()) {
            factory = it->second;
        }
    }
    if (!factory) {
        std::string known;
        for (const std::string& available : registered_optimizers()) {
            known += (known.empty() ? "" : ", ") + available;
        }
        throw OptimizerError("unknown optimizer '" + std::string(name) + "'; available: " + known);
    }
    return factory(parameters);
}

std::shared_ptr<Optimizer> make_optimizer(const OptimizerParameters& spec) {
    // The parsed subtree carries every key the file set, including the ones
    // the typed struct does not know about; the typed values fill in whatever
    // the file left out.
    Value parameters = spec.parameters.is_map() ? spec.parameters : Value(Value::Map{});
    if (!parameters.contains("fmax")) {
        parameters.insert("fmax", Value(spec.fmax));
    }
    if (!parameters.contains("max_steps")) {
        parameters.insert("max_steps", Value(spec.max_steps));
    }
    if (!parameters.contains("relax_cell")) {
        parameters.insert("relax_cell", Value(spec.relax_cell));
    }
    return make_optimizer(spec.algorithm.empty() ? "bfgs" : spec.algorithm, parameters);
}

std::vector<std::string> registered_optimizers() {
    Registry& store = registry();
    const std::lock_guard<std::mutex> lock(store.mutex);
    std::vector<std::string> names;
    names.reserve(store.display_names.size());
    for (const auto& [key, display] : store.display_names) {
        names.push_back(display);
    }
    return names;
}

}  // namespace fastatoms::optimize
