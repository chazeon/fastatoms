// SPDX-License-Identifier: MIT
#pragma once

/// \file single_point.hpp
/// \brief A calculator that holds results someone else computed.
///
/// A frame read from a trajectory carries an energy and forces that were
/// worked out by a DFT code hours ago. There is nothing left to evaluate, but
/// `atoms.get_potential_energy()` should still answer - so the numbers are
/// wrapped in a calculator that returns them and nothing else.
///
/// It is deliberately brittle about geometry: the results belong to the
/// structure they were computed for, so moving an atom invalidates them rather
/// than silently returning a number that is no longer true.

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "fastatoms/calculators/calculator.hpp"

namespace fastatoms {

/// Results attached to one fixed geometry.
class SinglePointCalculator : public Calculator {
public:
    /// \param atoms the structure the results belong to; its geometry is
    ///        remembered so that a later move can be detected
    explicit SinglePointCalculator(const Atoms& atoms, Results results = {});

    [[nodiscard]] std::string name() const override { return "single_point"; }
    [[nodiscard]] std::vector<Property> implemented_properties() const override;

    /// The results as they were handed over.
    [[nodiscard]] const Results& stored() const noexcept { return stored_; }

    [[nodiscard]] std::string to_string() const override;

protected:
    void compute(const Atoms& atoms, const std::vector<Property>& properties) override;

private:
    Results stored_{};
    std::vector<double> reference_positions_{};
    Mat3 reference_cell_{};
    std::vector<int> reference_numbers_{};
};

/// Wrap whatever a frame carries - `info["energy"]`, an `arrays["forces"]`
/// column, a `stress` entry - in a `SinglePointCalculator` and attach it.
///
/// Returns false when the frame carries nothing worth attaching, in which case
/// the structure is left alone rather than given an empty calculator.
bool attach_stored_results(Atoms& atoms);

}  // namespace fastatoms
