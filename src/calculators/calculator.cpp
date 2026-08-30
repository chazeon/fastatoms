// SPDX-License-Identifier: MIT
#include "fastatoms/calculators/calculator.hpp"

#include <algorithm>

#include "fastatoms/util/text.hpp"

namespace fastatoms {

// ---------------------------------------------------------------------------
// Properties and results
// ---------------------------------------------------------------------------

std::string_view property_name(Property property) {
    switch (property) {
        case Property::Energy:     return "energy";
        case Property::FreeEnergy: return "free_energy";
        case Property::Forces:     return "forces";
        case Property::Stress:     return "stress";
        case Property::Charges:    return "charges";
        case Property::Magmom:     return "magmom";
    }
    return "unknown";
}

Property property_from_string(std::string_view name) {
    const std::string key = util::lowercase(name);
    if (key == "energy") {
        return Property::Energy;
    }
    if (key == "free_energy" || key == "free-energy") {
        return Property::FreeEnergy;
    }
    if (key == "forces" || key == "force") {
        return Property::Forces;
    }
    if (key == "stress") {
        return Property::Stress;
    }
    if (key == "charges" || key == "charge") {
        return Property::Charges;
    }
    if (key == "magmom" || key == "magnetic_moment") {
        return Property::Magmom;
    }
    throw CalculatorError("unknown property '" + std::string(name) + "'");
}

const std::vector<Property>& default_properties() {
    static const std::vector<Property> properties{Property::Energy, Property::Forces,
                                                  Property::Stress};
    return properties;
}

void Results::clear() {
    energy.reset();
    free_energy.reset();
    forces.clear();
    stress.reset();
    charges.clear();
    magmom.reset();
    extra = Value{};
}

bool Results::has(Property property) const {
    switch (property) {
        case Property::Energy:     return energy.has_value();
        case Property::FreeEnergy: return free_energy.has_value();
        case Property::Forces:     return !forces.empty();
        case Property::Stress:     return stress.has_value();
        case Property::Charges:    return !charges.empty();
        case Property::Magmom:     return magmom.has_value();
    }
    return false;
}

Mat3 voigt_to_matrix(const std::array<double, 6>& v) {
    return {v[0], v[5], v[4],
            v[5], v[1], v[3],
            v[4], v[3], v[2]};
}

std::array<double, 6> matrix_to_voigt(const Mat3& m) {
    return {m(0, 0), m(1, 1), m(2, 2),
            0.5 * (m(1, 2) + m(2, 1)),
            0.5 * (m(0, 2) + m(2, 0)),
            0.5 * (m(0, 1) + m(1, 0))};
}

double pressure_from_stress(const std::array<double, 6>& stress) {
    return -(stress[0] + stress[1] + stress[2]) / 3.0;
}

// ---------------------------------------------------------------------------
// Calculator
// ---------------------------------------------------------------------------

Calculator::Calculator(Value parameters) : parameters_(std::move(parameters)) {
    if (parameters_.is_null()) {
        parameters_ = Value(Value::Map{});
    }
    if (!parameters_.is_map()) {
        throw CalculatorError("calculator parameters must be a mapping, got " +
                              std::string(parameters_.type_name()));
    }
}

bool Calculator::implements(Property property) const {
    const std::vector<Property> supported = implemented_properties();
    return std::find(supported.begin(), supported.end(), property) != supported.end();
}

void Calculator::set_parameters(Value parameters) {
    if (parameters.is_null()) {
        parameters = Value(Value::Map{});
    }
    if (!parameters.is_map()) {
        throw CalculatorError("calculator parameters must be a mapping, got " +
                              std::string(parameters.type_name()));
    }
    parameters_ = std::move(parameters);
    reset();  // settings changed, so the cached numbers no longer apply
}

void Calculator::set_parameter(const std::string& key, Value value) {
    parameters_.insert(key, std::move(value));
    reset();
}

const Value& Calculator::parameter(std::string_view key) const {
    if (const Value* found = parameters_.find(key)) {
        return *found;
    }
    return Value::null();
}

void Calculator::fail(const std::string& message) const {
    throw CalculatorError(name() + ": " + message);
}

void Calculator::reset() {
    results_.clear();
    state_ = SystemState{};
}

bool Calculator::SystemState::matches(const Atoms& atoms) const {
    return valid && numbers == atoms.numbers() && positions == atoms.positions() &&
           cell == atoms.cell().matrix() && pbc == atoms.pbc();
}

void Calculator::SystemState::capture(const Atoms& atoms) {
    valid = true;
    numbers = atoms.numbers();
    positions = atoms.positions();
    cell = atoms.cell().matrix();
    pbc = atoms.pbc();
}

bool Calculator::needs_calculation(const Atoms& atoms,
                                   const std::vector<Property>& properties) const {
    if (!state_.matches(atoms)) {
        return true;
    }
    return std::any_of(properties.begin(), properties.end(),
                       [this](Property p) { return !results_.has(p); });
}

void Calculator::calculate(Atoms& atoms, const std::vector<Property>& properties) {
    for (const Property property : properties) {
        if (!implements(property)) {
            fail("does not implement the property '" + std::string(property_name(property)) + "'");
        }
    }
    if (!needs_calculation(atoms, properties)) {
        return;
    }

    // A structure change invalidates everything; a request for an extra
    // property on an unchanged structure does not.
    if (!state_.matches(atoms)) {
        results_.clear();
    }

    compute(atoms, properties);
    ++call_count_;
    state_.capture(atoms);
    store_on_atoms(atoms);

    for (const Property property : properties) {
        if (!results_.has(property)) {
            fail("the calculation finished but produced no '" +
                 std::string(property_name(property)) + "'");
        }
    }
}

void Calculator::store_on_atoms(Atoms& atoms) const {
    // Per-atom results live on the structure so that they follow it through
    // slicing, `extend` and writing. Scalars stay on the calculator, which is
    // the single source of truth; `Atoms::get_potential_energy()` reads them
    // back from there.
    if (results_.forces.size() == 3 * atoms.size()) {
        atoms.set_array("forces", results_.forces, 3);
    }
    if (results_.charges.size() == atoms.size()) {
        atoms.set_charges(results_.charges);
    }
}

double Calculator::get_potential_energy(Atoms& atoms, bool force_consistent) {
    const Property wanted = force_consistent ? Property::FreeEnergy : Property::Energy;
    calculate(atoms, {wanted});
    return force_consistent ? *results_.free_energy : *results_.energy;
}

std::vector<double> Calculator::get_forces(Atoms& atoms) {
    calculate(atoms, {Property::Forces});
    return results_.forces;
}

std::array<double, 6> Calculator::get_stress(Atoms& atoms) {
    calculate(atoms, {Property::Stress});
    return *results_.stress;
}

std::vector<double> Calculator::get_charges(Atoms& atoms) {
    calculate(atoms, {Property::Charges});
    return results_.charges;
}

double Calculator::get_magnetic_moment(Atoms& atoms) {
    calculate(atoms, {Property::Magmom});
    return *results_.magmom;
}

std::string Calculator::to_string() const { return name() + "()"; }

}  // namespace fastatoms
