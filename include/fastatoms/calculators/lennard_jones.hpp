// SPDX-License-Identifier: MIT
#pragma once

/// \file lennard_jones.hpp
/// \brief A built-in analytic calculator.

#include "fastatoms/calculators/calculator.hpp"

namespace fastatoms {

/// Cut-and-shifted Lennard-Jones, evaluated in C++ on the linear-scaling
/// neighbour list.
///
/// This is the reference implementation of the `Calculator` interface: it
/// needs no external program, so it is what the test suite uses to check the
/// caching behaviour and the stress sign convention against finite
/// differences. It is also genuinely useful for smoke-testing a workflow
/// before pointing it at a DFT code.
///
/// Parameters:
///   - `epsilon` - well depth in eV (default 0.0103, argon)
///   - `sigma`   - length scale in angstrom (default 3.40)
///   - `cutoff`  - interaction cutoff in angstrom (default 3 * sigma)
///   - `shift`   - `true` (default) shifts the potential to zero at the cutoff
class LennardJonesCalculator : public Calculator {
public:
    explicit LennardJonesCalculator(Value parameters = {});

    [[nodiscard]] std::string name() const override { return "lennard_jones"; }
    [[nodiscard]] std::vector<Property> implemented_properties() const override;
    [[nodiscard]] std::string to_string() const override;

protected:
    void compute(const Atoms& atoms, const std::vector<Property>& properties) override;
};

}  // namespace fastatoms
