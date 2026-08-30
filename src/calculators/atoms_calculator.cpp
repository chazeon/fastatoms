// SPDX-License-Identifier: MIT
///
/// The `Atoms` side of the calculator link. It lives in its own translation
/// unit so that `atoms.cpp` - the core container - stays free of any
/// dependency on the calculator hierarchy; `Atoms` only forward-declares it.

#include "fastatoms/calculators/calculator.hpp"
#include "fastatoms/core/atoms.hpp"

namespace fastatoms {

void Atoms::set_calculator(std::shared_ptr<Calculator> calculator) {
    calculator_ = std::move(calculator);
}

namespace {

Calculator& require_calculator(const std::shared_ptr<Calculator>& calculator) {
    if (!calculator) {
        throw CalculatorError(
            "no calculator is attached to this Atoms object; assign one with "
            "set_calculator() (Python: atoms.calc = ...)");
    }
    return *calculator;
}

}  // namespace

double Atoms::get_potential_energy(bool force_consistent) {
    return require_calculator(calculator_).get_potential_energy(*this, force_consistent);
}

std::vector<double> Atoms::get_forces() {
    std::vector<double> forces = require_calculator(calculator_).get_forces(*this);
    // Constraints are applied here, on the way out, rather than inside the
    // calculator. That is what makes them universal: every optimizer, every
    // integrator and every band image goes through this one function, so none
    // of them has to remember. The calculator's own cached result stays the
    // unconstrained force the model actually produced, which is what you want
    // when reading `atoms.calc.results`.
    adjust_forces(forces);
    return forces;
}

std::array<double, 6> Atoms::get_stress() {
    return require_calculator(calculator_).get_stress(*this);
}

}  // namespace fastatoms
