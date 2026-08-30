// SPDX-License-Identifier: MIT
#include "fastatoms/calculators/registry.hpp"

#include <map>
#include <mutex>

#include "fastatoms/calculators/lennard_jones.hpp"
#include "fastatoms/calculators/quantum_espresso.hpp"
#include "fastatoms/calculators/siesta.hpp"
#include "fastatoms/calculators/vasp.hpp"
#include "fastatoms/util/text.hpp"

namespace fastatoms {
namespace {

/// "Quantum-Espresso" and "quantum_espresso" should find the same entry.
std::string canonical(std::string_view name) {
    std::string key = util::lowercase(name);
    std::string out;
    for (const char c : key) {
        if (c != '-' && c != '_' && c != ' ') {
            out += c;
        }
    }
    return out;
}

struct Registry {
    std::mutex mutex;
    std::map<std::string, CalculatorFactory> factories;  ///< keyed by canonical name
    std::map<std::string, std::string> display_names;    ///< canonical -> as registered
};

Registry& registry() {
    // Registry holds a mutex, so it cannot be copied or moved out of an
    // initialiser lambda; populate it in place instead.
    static Registry instance;
    static const bool initialised = [] {
        const auto add = [](const char* name, CalculatorFactory factory) {
            instance.factories[canonical(name)] = std::move(factory);
            instance.display_names[canonical(name)] = name;
        };

        add("vasp", [](const Value& p) { return std::make_shared<VaspCalculator>(p); });
        add("espresso",
            [](const Value& p) { return std::make_shared<QuantumEspressoCalculator>(p); });
        add("siesta", [](const Value& p) { return std::make_shared<SiestaCalculator>(p); });
        add("lennard_jones",
            [](const Value& p) { return std::make_shared<LennardJonesCalculator>(p); });

        // Aliases share a factory but are not listed separately.
        for (const char* alias : {"quantum_espresso", "qe", "pw", "pwscf"}) {
            instance.factories[canonical(alias)] = instance.factories[canonical("espresso")];
        }
        for (const char* alias : {"lj", "lennardjones"}) {
            instance.factories[canonical(alias)] = instance.factories[canonical("lennard_jones")];
        }
        return true;
    }();
    (void)initialised;
    return instance;
}

}  // namespace

void register_calculator(const std::string& name, CalculatorFactory factory) {
    if (!factory) {
        throw CalculatorError("register_calculator: the factory for '" + name + "' is empty");
    }
    Registry& store = registry();
    const std::lock_guard<std::mutex> lock(store.mutex);
    store.factories[canonical(name)] = std::move(factory);
    store.display_names[canonical(name)] = name;
}

void unregister_calculator(const std::string& name) {
    Registry& store = registry();
    const std::lock_guard<std::mutex> lock(store.mutex);
    store.factories.erase(canonical(name));
    store.display_names.erase(canonical(name));
}

std::shared_ptr<Calculator> make_calculator(std::string_view name, const Value& parameters) {
    Registry& store = registry();
    CalculatorFactory factory;
    {
        const std::lock_guard<std::mutex> lock(store.mutex);
        const auto it = store.factories.find(canonical(name));
        if (it != store.factories.end()) {
            factory = it->second;
        }
    }
    if (!factory) {
        std::string known;
        for (const std::string& available : registered_calculators()) {
            known += (known.empty() ? "" : ", ") + available;
        }
        throw CalculatorError("unknown calculator '" + std::string(name) + "'; available: " +
                              known);
    }
    return factory(parameters);
}

std::shared_ptr<Calculator> make_calculator(const CalculatorParameters& spec) {
    if (spec.name.empty()) {
        throw CalculatorError(
            "the input file's calculator section has no name; add e.g. calculator: {name: vasp}");
    }
    return make_calculator(spec.name, spec.parameters);
}

std::vector<std::string> registered_calculators() {
    Registry& store = registry();
    const std::lock_guard<std::mutex> lock(store.mutex);
    std::vector<std::string> names;
    names.reserve(store.display_names.size());
    for (const auto& [key, display] : store.display_names) {
        names.push_back(display);
    }
    return names;
}

}  // namespace fastatoms
