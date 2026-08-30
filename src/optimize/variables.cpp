// SPDX-License-Identifier: MIT
#include "fastatoms/optimize/variables.hpp"

#include <algorithm>
#include <cmath>

#include "fastatoms/calculators/calculator.hpp"

namespace fastatoms::optimize {
namespace {

Calculator& require_calculator(Atoms& atoms) {
    if (!atoms.has_calculator()) {
        throw OptimizerError(
            "the structure has no calculator; attach one before relaxing it "
            "(Python: atoms.calc = ...)");
    }
    return *atoms.calculator();
}

/// Run the calculator and hand back the energy plus the raw results.
const Results& evaluate_with(Atoms& atoms, const std::vector<Property>& properties) {
    Calculator& calculator = require_calculator(atoms);
    calculator.calculate(atoms, properties);
    return calculator.results();
}

}  // namespace

// ---------------------------------------------------------------------------
// Positions only
// ---------------------------------------------------------------------------

std::vector<double> AtomicVariables::pack(const Atoms& atoms) const {
    if (atoms.size() != natoms_) {
        throw OptimizerError("the structure changed size during the relaxation");
    }
    return atoms.positions();
}

void AtomicVariables::unpack(const std::vector<double>& x, Atoms& atoms) const {
    atoms.set_positions(x);
}

Evaluation AtomicVariables::evaluate(Atoms& atoms) const {
    // Only energy and forces are requested, so a calculator that cannot
    // produce a stress is still usable for a fixed-cell relaxation.
    const Results& results = evaluate_with(atoms, {Property::Energy, Property::Forces});

    Evaluation out;
    out.energy = *results.energy;
    out.gradient = scaled(-1.0, results.forces);
    return out;
}

// ---------------------------------------------------------------------------
// Positions and cell
// ---------------------------------------------------------------------------

CellVariables::CellVariables(const Atoms& atoms, CellVariablesOptions options)
    : natoms_(atoms.size()),
      reference_cell_(atoms.cell().matrix()),
      pressure_(options.pressure) {
    if (!atoms.cell().is_invertible()) {
        throw OptimizerError(
            "relaxing the cell needs a structure with a full 3x3 cell; give it one first");
    }
    if (!atoms.cell().any_pbc()) {
        throw OptimizerError(
            "relaxing the cell needs periodic boundary conditions; set atoms.pbc = True");
    }
    reference_inverse_ = reference_cell_.inverse();

    cell_factor_ = options.cell_factor > 0.0 ? options.cell_factor
                                             : static_cast<double>(std::max<std::size_t>(1, natoms_));

    // The Voigt mask becomes a symmetric 3x3 multiplier on the strain gradient.
    mask_ = voigt_to_matrix(options.mask);
    for (const double value : options.mask) {
        if (value < 0.0 || value > 1.0) {
            throw OptimizerError("the cell mask entries must be between 0 and 1");
        }
    }
}

Mat3 CellVariables::deformation(const Atoms& atoms) const {
    // cell = reference_cell * S, so S = reference_cell^-1 * cell.
    return reference_inverse_ * atoms.cell().matrix();
}

std::vector<double> CellVariables::pack(const Atoms& atoms) const {
    if (atoms.size() != natoms_) {
        throw OptimizerError("the structure changed size during the relaxation");
    }
    const Mat3 S = deformation(atoms);
    const Mat3 inverse_S = S.inverse();

    std::vector<double> x(size(), 0.0);
    for (std::size_t i = 0; i < natoms_; ++i) {
        // r_i = y_i S, so y_i = r_i S^-1.
        (atoms.position(i) * inverse_S).store(x.data() + 3 * i);
    }
    for (std::size_t a = 0; a < 3; ++a) {
        for (std::size_t b = 0; b < 3; ++b) {
            const double identity = a == b ? 1.0 : 0.0;
            x[3 * natoms_ + 3 * a + b] = (S(a, b) - identity) * cell_factor_;
        }
    }
    return x;
}

void CellVariables::unpack(const std::vector<double>& x, Atoms& atoms) const {
    if (x.size() != size()) {
        throw OptimizerError("unpack: the coordinate vector has the wrong length");
    }

    Mat3 S;
    for (std::size_t a = 0; a < 3; ++a) {
        for (std::size_t b = 0; b < 3; ++b) {
            const double identity = a == b ? 1.0 : 0.0;
            S(a, b) = identity + x[3 * natoms_ + 3 * a + b] / cell_factor_;
        }
    }

    // Set the cell without dragging the atoms: the positions below already
    // carry the deformation.
    atoms.set_cell(reference_cell_ * S, false);

    std::vector<double> positions(3 * natoms_);
    for (std::size_t i = 0; i < natoms_; ++i) {
        (Vec3::load(x.data() + 3 * i) * S).store(positions.data() + 3 * i);
    }
    atoms.set_positions(positions);
}

Evaluation CellVariables::evaluate(Atoms& atoms) const {
    const Results& results =
        evaluate_with(atoms, {Property::Energy, Property::Forces, Property::Stress});

    const Mat3 S = deformation(atoms);
    const double volume = atoms.volume();

    Evaluation out;
    // Under an external pressure the quantity being minimised is the enthalpy.
    out.energy = *results.energy + pressure_ * volume;
    out.gradient.assign(size(), 0.0);

    // dE/dy_i = -F_i S^T
    const Mat3 transposed = S.transposed();
    for (std::size_t i = 0; i < natoms_; ++i) {
        const Vec3 force = Vec3::load(results.forces.data() + 3 * i);
        ((-force) * transposed).store(out.gradient.data() + 3 * i);
    }

    // dE/dS = V S^-T sigma, with the external pressure folded into the stress
    // because d(pV)/deps_ab = p V delta_ab.
    Mat3 stress = voigt_to_matrix(*results.stress);
    for (std::size_t k = 0; k < 3; ++k) {
        stress(k, k) += pressure_;
    }
    const Mat3 cell_gradient = S.inverse().transposed() * stress;

    for (std::size_t a = 0; a < 3; ++a) {
        for (std::size_t b = 0; b < 3; ++b) {
            out.gradient[3 * natoms_ + 3 * a + b] =
                cell_gradient(a, b) * volume / cell_factor_ * mask_(a, b);
        }
    }
    return out;
}

}  // namespace fastatoms::optimize
