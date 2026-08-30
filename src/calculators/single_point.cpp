// SPDX-License-Identifier: MIT

#include "fastatoms/calculators/single_point.hpp"

#include <cmath>
#include <sstream>
#include <istream>

namespace fastatoms {
namespace {

/// Did the structure move, change shape, or change composition?
///
/// Compared exactly rather than with a tolerance: these results were computed
/// for one geometry, and "close enough" is a judgement the caller has to make
/// for themselves rather than one this class should make silently.
bool differs(const Atoms& atoms, const std::vector<double>& positions, const Mat3& cell,
             const std::vector<int>& numbers) {
    if (atoms.size() * 3 != positions.size() || atoms.numbers() != numbers) {
        return true;
    }
    const std::vector<double>& current = atoms.positions();
    for (std::size_t k = 0; k < positions.size(); ++k) {
        if (current[k] != positions[k]) {
            return true;
        }
    }
    const Mat3 now = atoms.cell().matrix();
    for (std::size_t k = 0; k < 9; ++k) {
        if (now.m[k] != cell.m[k]) {
            return true;
        }
    }
    return false;
}

}  // namespace

SinglePointCalculator::SinglePointCalculator(const Atoms& atoms, Results results)
    : stored_(std::move(results)),
      reference_positions_(atoms.positions()),
      reference_cell_(atoms.cell().matrix()),
      reference_numbers_(atoms.numbers()) {}

std::vector<Property> SinglePointCalculator::implemented_properties() const {
    std::vector<Property> out;
    if (stored_.energy) out.push_back(Property::Energy);
    if (stored_.free_energy) out.push_back(Property::FreeEnergy);
    if (!stored_.forces.empty()) out.push_back(Property::Forces);
    if (stored_.stress) out.push_back(Property::Stress);
    if (!stored_.charges.empty()) out.push_back(Property::Charges);
    if (stored_.magmom) out.push_back(Property::Magmom);
    return out;
}

void SinglePointCalculator::compute(const Atoms& atoms,
                                    const std::vector<Property>& properties) {
    if (differs(atoms, reference_positions_, reference_cell_, reference_numbers_)) {
        fail("these results were computed for a different geometry; they belong to the "
             "structure they were read with, and moving an atom does not move them with it");
    }
    for (const Property property : properties) {
        if (!stored_.has(property)) {
            fail("no stored " + std::string(property_name(property)) +
                 ": this structure carried only what its file recorded");
        }
    }
    results_ = stored_;
}

std::string SinglePointCalculator::to_string() const {
    std::ostringstream out;
    out << "SinglePointCalculator(";
    bool first = true;
    for (const Property property : implemented_properties()) {
        out << (first ? "" : ", ") << property_name(property);
        first = false;
    }
    if (first) {
        out << "nothing stored";
    }
    out << ")";
    return out.str();
}

bool attach_stored_results(Atoms& atoms) {
    Results results;
    const Value& info = atoms.info();

    if (const Value& energy = info["energy"]; energy.is_number()) {
        results.energy = energy.as_double();
    }
    if (const Value& free_energy = info["free_energy"]; free_energy.is_number()) {
        results.free_energy = free_energy.as_double();
    }
    // An energy with no free energy is its own free energy, which is what
    // every code that reports only one of them means.
    if (results.energy && !results.free_energy) {
        results.free_energy = results.energy;
    }

    if (atoms.has_array("forces")) {
        const Atoms::Array& forces = atoms.array("forces");
        if (forces.ncols == 3 && forces.data.size() == 3 * atoms.size()) {
            results.forces = forces.data;
        }
    }

    // Extended XYZ writes the stress as nine numbers in row-major order, or as
    // six in Voigt; both are accepted.
    for (const char* key : {"stress", "virial"}) {
        const Value& entry = info[key];
        if (entry.is_null()) {
            continue;
        }
        // Written as a quoted `stress="xx xy xz ..."`, which the comment-line
        // parser keeps as one string rather than splitting - so both shapes
        // have to be accepted here.
        std::vector<double> values;
        if (entry.is_string()) {
            std::istringstream stream(entry.as_string());
            double number = 0.0;
            while (stream >> number) {
                values.push_back(number);
            }
        } else {
            try {
                values = entry.as_doubles();
            } catch (const InputError&) {
                continue;
            }
        }
        if (values.size() != 6 && values.size() != 9) {
            continue;
        }
        const double volume = atoms.volume();
        if (values.size() == 9) {
            Mat3 matrix;
            for (std::size_t k = 0; k < 9; ++k) {
                matrix.m[k] = values[k];
            }
            // A virial is the stress times the volume, with the opposite sign.
            if (std::string(key) == "virial" && volume > 0.0) {
                for (std::size_t k = 0; k < 9; ++k) {
                    matrix.m[k] /= -volume;
                }
            }
            results.stress = matrix_to_voigt(matrix);
            break;
        }
        if (values.size() == 6) {
            std::array<double, 6> voigt{};
            for (std::size_t k = 0; k < 6; ++k) {
                voigt[k] = values[k];
                if (std::string(key) == "virial" && volume > 0.0) {
                    voigt[k] /= -volume;
                }
            }
            results.stress = voigt;
            break;
        }
    }

    if (atoms.has_array("initial_charges") || atoms.has_charges()) {
        // Only calculator-produced charges belong here; the initial ones are
        // an input and stay where they are.
    }

    if (!results.energy && results.forces.empty() && !results.stress) {
        return false;
    }
    atoms.set_calculator(std::make_shared<SinglePointCalculator>(atoms, std::move(results)));
    return true;
}

}  // namespace fastatoms
