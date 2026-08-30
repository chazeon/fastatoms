// SPDX-License-Identifier: MIT
#include "fastatoms/phonon/registry.hpp"

#include <utility>

namespace fastatoms::phonon {
namespace {

/// Copy `key` from `spec` into `target` unless the file already set it.
void fill(Value& target, const std::string& key, Value value) {
    if (!target.contains(key)) {
        target.insert(key, std::move(value));
    }
}

}  // namespace

std::shared_ptr<Phonons> make_phonons(const Atoms& unit_cell, const Value& parameters) {
    return std::make_shared<Phonons>(unit_cell, parameters);
}

std::shared_ptr<Phonons> make_phonons(const Atoms& unit_cell, const PhononParameters& spec) {
    // The unparsed subtree carries every key the file set, including the ones
    // the typed struct does not know about; the typed values fill the gaps.
    Value parameters = spec.parameters.is_map() ? spec.parameters : Value(Value::Map{});

    Value::List supercell;
    supercell.reserve(spec.supercell.size());
    for (const double n : spec.supercell) {
        supercell.emplace_back(static_cast<std::int64_t>(n));
    }
    fill(parameters, "supercell", Value(std::move(supercell)));
    fill(parameters, "displacement", Value(spec.displacement));
    fill(parameters, "symmetry", Value(spec.symmetry));
    fill(parameters, "symmetry_tolerance", Value(spec.symmetry_tolerance));

    return make_phonons(unit_cell, parameters);
}

std::shared_ptr<Phonons> make_phonons(const Atoms& unit_cell,
                                      const SimulationParameters& simulation) {
    return make_phonons(unit_cell, simulation.phonon);
}

}  // namespace fastatoms::phonon
