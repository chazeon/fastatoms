// SPDX-License-Identifier: MIT
#pragma once

/// \file calculator.hpp
/// \brief The abstract calculator interface.

#include <memory>
#include <string>
#include <vector>

#include "fastatoms/calculators/results.hpp"
#include "fastatoms/core/atoms.hpp"
#include "fastatoms/core/constraint.hpp"

namespace fastatoms {

/// Base class for anything that can evaluate energies, forces and stresses for
/// an `Atoms` object.
///
/// The contract mirrors ASE's: a calculator holds the results of its most
/// recent evaluation and only recomputes when the structure it was given has
/// actually changed. Subclasses implement `compute()`; everything else -
/// caching, property dispatch, writing per-atom results back onto the
/// structure - is handled here.
///
/// Configuration is a `Value` map, the same type the YAML parser produces, so
/// a calculator can be built straight from an input file or from a Python dict
/// with no translation layer.
class Calculator {
public:
    virtual ~Calculator() = default;

    Calculator(const Calculator&) = delete;
    Calculator& operator=(const Calculator&) = delete;

    /// Short identifier, e.g. "vasp".
    [[nodiscard]] virtual std::string name() const = 0;

    /// Which properties this calculator can produce.
    [[nodiscard]] virtual std::vector<Property> implemented_properties() const = 0;

    [[nodiscard]] bool implements(Property property) const;

    /// \name Evaluation
    ///
    /// Each of these runs a calculation if the cached results do not already
    /// cover the request for this exact structure, then returns the value.
    /// Forces are also stored on `atoms` as the per-atom array "forces", and
    /// charges as the atoms' charges, so results travel with the structure.
    /// @{
    double get_potential_energy(Atoms& atoms, bool force_consistent = false);
    std::vector<double> get_forces(Atoms& atoms);
    std::array<double, 6> get_stress(Atoms& atoms);
    std::vector<double> get_charges(Atoms& atoms);
    double get_magnetic_moment(Atoms& atoms);
    /// @}

    /// Run a calculation for the given properties, unconditionally reusing
    /// cached results when they already cover the request.
    void calculate(Atoms& atoms, const std::vector<Property>& properties = default_properties());

    /// True when `calculate` would actually invoke the underlying engine.
    [[nodiscard]] bool needs_calculation(const Atoms& atoms,
                                         const std::vector<Property>& properties) const;

    /// Discard cached results, forcing the next request to recompute.
    void reset();

    [[nodiscard]] const Results& results() const noexcept { return results_; }

    /// Number of times the underlying engine has actually been invoked.
    [[nodiscard]] std::size_t call_count() const noexcept { return call_count_; }

    /// \name Configuration
    /// @{
    [[nodiscard]] const Value& parameters() const noexcept { return parameters_; }
    void set_parameters(Value parameters);
    void set_parameter(const std::string& key, Value value);
    /// @}

    /// The rigid-geometry constraints this model implies, or null when it
    /// implies none.
    ///
    /// A rigid water model fixes its own bond lengths and angle, and the
    /// dynamics has to know that to keep them fixed and to count the degrees
    /// of freedom correctly. Rather than have the user restate it, the
    /// calculator that knows the model says so, and `Dynamics` picks it up.
    [[nodiscard]] virtual const std::vector<DistanceConstraint>* rigid_constraints() const {
        return nullptr;
    }

    /// Human-readable summary, used by the Python `repr`.
    [[nodiscard]] virtual std::string to_string() const;

protected:
    explicit Calculator(Value parameters = {});

    /// Evaluate `properties` for `atoms` and fill `results_`.
    ///
    /// Implementations may compute more than was asked for - most engines
    /// return energy, forces and stress together anyway.
    virtual void compute(const Atoms& atoms, const std::vector<Property>& properties) = 0;

    /// Throw a CalculatorError naming this calculator.
    [[noreturn]] void fail(const std::string& message) const;

    /// Look up `key` in the parameters, returning the shared null value when
    /// it is absent.
    [[nodiscard]] const Value& parameter(std::string_view key) const;

    Results results_{};
    Value parameters_{};

private:
    /// The structure a set of results belongs to. Any difference means the
    /// cached numbers no longer apply.
    struct SystemState {
        bool valid{false};
        std::vector<int> numbers{};
        std::vector<double> positions{};
        Mat3 cell{};
        std::array<bool, 3> pbc{{false, false, false}};

        [[nodiscard]] bool matches(const Atoms& atoms) const;
        void capture(const Atoms& atoms);
    };

    void store_on_atoms(Atoms& atoms) const;

    SystemState state_{};
    std::size_t call_count_{0};
};

}  // namespace fastatoms
