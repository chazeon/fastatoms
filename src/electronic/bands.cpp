// SPDX-License-Identifier: MIT

#include "fastatoms/electronic/bands.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "fastatoms/calculators/calculator.hpp"

namespace fastatoms::electronic {

BandStructure::BandStructure(std::vector<Vec3> kpoints, std::vector<double> energies,
                             std::size_t bands, std::size_t spins, double fermi_level,
                             std::vector<double> weights)
    : kpoints_(std::move(kpoints)),
      energies_(std::move(energies)),
      weights_(std::move(weights)),
      bands_(bands),
      spins_(spins),
      fermi_level_(fermi_level) {
    if (spins_ != 1 && spins_ != 2) {
        throw ElectronicError("spins must be 1 or 2, not " + std::to_string(spins_));
    }
    if (kpoints_.empty()) {
        throw ElectronicError("a band structure needs at least one k-point");
    }
    if (bands_ == 0) {
        throw ElectronicError("a band structure needs at least one band");
    }
    const std::size_t expected = spins_ * kpoints_.size() * bands_;
    if (energies_.size() != expected) {
        throw ElectronicError("expected " + std::to_string(expected) + " energies for " +
                              std::to_string(spins_) + " spin(s), " +
                              std::to_string(kpoints_.size()) + " k-points and " +
                              std::to_string(bands_) + " bands, got " +
                              std::to_string(energies_.size()));
    }
    if (weights_.empty()) {
        weights_.assign(kpoints_.size(), 1.0 / static_cast<double>(kpoints_.size()));
    } else if (weights_.size() != kpoints_.size()) {
        throw ElectronicError("there are " + std::to_string(weights_.size()) +
                              " weights for " + std::to_string(kpoints_.size()) + " k-points");
    } else {
        double total = 0.0;
        for (const double weight : weights_) {
            if (weight < 0.0) {
                throw ElectronicError("k-point weights must not be negative");
            }
            total += weight;
        }
        if (!(total > 0.0)) {
            throw ElectronicError("the k-point weights sum to zero");
        }
        for (double& weight : weights_) {
            weight /= total;
        }
    }
}

double BandStructure::energy(std::size_t spin, std::size_t kpoint, std::size_t band) const {
    if (spin >= spins_ || kpoint >= kpoints_.size() || band >= bands_) {
        throw ElectronicError("energy(" + std::to_string(spin) + ", " +
                              std::to_string(kpoint) + ", " + std::to_string(band) +
                              ") is outside a band structure of " + std::to_string(spins_) +
                              " x " + std::to_string(kpoints_.size()) + " x " +
                              std::to_string(bands_));
    }
    return energies_[(spin * kpoints_.size() + kpoint) * bands_ + band];
}

std::vector<double> BandStructure::band(std::size_t spin, std::size_t index) const {
    std::vector<double> out;
    out.reserve(kpoints_.size());
    for (std::size_t k = 0; k < kpoints_.size(); ++k) {
        out.push_back(energy(spin, k, index));
    }
    return out;
}

double BandStructure::minimum() const {
    if (energies_.empty()) {
        throw ElectronicError("an empty band structure has no minimum");
    }
    return *std::min_element(energies_.begin(), energies_.end());
}

double BandStructure::maximum() const {
    if (energies_.empty()) {
        throw ElectronicError("an empty band structure has no maximum");
    }
    return *std::max_element(energies_.begin(), energies_.end());
}

bool BandStructure::is_metal(double tolerance) const {
    // A band that is neither wholly below nor wholly above the Fermi level
    // crosses it.
    for (std::size_t spin = 0; spin < spins_; ++spin) {
        for (std::size_t band = 0; band < bands_; ++band) {
            bool below = false;
            bool above = false;
            for (std::size_t k = 0; k < kpoints_.size(); ++k) {
                const double e = energy(spin, k, band);
                below = below || e < fermi_level_ - tolerance;
                above = above || e > fermi_level_ + tolerance;
            }
            if (below && above) {
                return true;
            }
        }
    }
    return false;
}

BandGap BandStructure::band_gap() const {
    BandGap gap;
    if (empty()) {
        throw ElectronicError("an empty band structure has no gap");
    }
    if (is_metal()) {
        gap.metallic = true;
        return gap;
    }

    // The fundamental gap is between the highest occupied state anywhere and
    // the lowest empty state anywhere - the two need not share a k-point,
    // which is the whole distinction between an indirect gap and a direct one.
    double valence = -std::numeric_limits<double>::infinity();
    double conduction = std::numeric_limits<double>::infinity();
    bool any_occupied = false;
    bool any_empty = false;

    for (std::size_t spin = 0; spin < spins_; ++spin) {
        for (std::size_t k = 0; k < kpoints_.size(); ++k) {
            for (std::size_t band = 0; band < bands_; ++band) {
                const double e = energy(spin, k, band);
                if (e <= fermi_level_) {
                    if (e > valence) {
                        valence = e;
                        gap.valence_kpoint = k;
                        gap.valence_band = band;
                    }
                    any_occupied = true;
                } else {
                    if (e < conduction) {
                        conduction = e;
                        gap.conduction_kpoint = k;
                        gap.conduction_band = band;
                    }
                    any_empty = true;
                }
            }
        }
    }
    if (!any_occupied || !any_empty) {
        throw ElectronicError("every state is on the same side of the Fermi level; the level "
                              "is probably not set");
    }

    gap.valence_maximum = valence;
    gap.conduction_minimum = conduction;
    gap.gap = conduction - valence;

    // The direct gap is the smallest separation at a single k-point.
    gap.direct_gap = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < kpoints_.size(); ++k) {
        double highest = -std::numeric_limits<double>::infinity();
        double lowest = std::numeric_limits<double>::infinity();
        for (std::size_t spin = 0; spin < spins_; ++spin) {
            for (std::size_t band = 0; band < bands_; ++band) {
                const double e = energy(spin, k, band);
                if (e <= fermi_level_) {
                    highest = std::max(highest, e);
                } else {
                    lowest = std::min(lowest, e);
                }
            }
        }
        if (highest > -std::numeric_limits<double>::infinity() &&
            lowest < std::numeric_limits<double>::infinity() &&
            lowest - highest < gap.direct_gap) {
            gap.direct_gap = lowest - highest;
            gap.direct_kpoint = k;
        }
    }
    gap.direct = std::abs(gap.direct_gap - gap.gap) < 1e-9;
    return gap;
}

BandStructure BandStructure::shifted() const {
    std::vector<double> moved = energies_;
    for (double& value : moved) {
        value -= fermi_level_;
    }
    return BandStructure(kpoints_, std::move(moved), bands_, spins_, 0.0, weights_);
}

BandStructure BandStructure::from_calculator(Atoms& atoms) {
    if (!atoms.has_calculator()) {
        throw ElectronicError("no calculator is attached to read eigenvalues from");
    }
    (void)atoms.get_potential_energy();   // make sure something has been computed
    const Value& extra = atoms.calculator()->results().extra;

    const Value& eigenvalues = extra["eigenvalues"];
    if (eigenvalues.is_null()) {
        throw ElectronicError(
            "this calculator reported no eigenvalues; none of the calculators shipped with "
            "FastAtoms produce them yet, so build the band structure from arrays instead");
    }
    const std::vector<double> flat = eigenvalues.as_doubles();

    std::vector<Vec3> kpoints;
    const Value& points = extra["kpoints"];
    if (points.is_null()) {
        throw ElectronicError("eigenvalues were reported without the k-points they belong to");
    }
    const std::vector<double> coordinates = points.as_doubles();
    if (coordinates.size() % 3 != 0) {
        throw ElectronicError("the k-point list is not a multiple of three");
    }
    for (std::size_t k = 0; k + 2 < coordinates.size(); k += 3) {
        kpoints.push_back(Vec3{coordinates[k], coordinates[k + 1], coordinates[k + 2]});
    }

    const auto spins = static_cast<std::size_t>(extra["spins"].value_or(static_cast<std::int64_t>(1)));
    if (kpoints.empty() || spins == 0 || flat.size() % (kpoints.size() * spins) != 0) {
        throw ElectronicError("the eigenvalue array does not divide evenly among the k-points");
    }
    const std::size_t bands = flat.size() / (kpoints.size() * spins);
    return BandStructure(std::move(kpoints), flat, bands, spins,
                         extra["fermi_level"].value_or(0.0));
}

std::string BandGap::to_string() const {
    std::ostringstream out;
    if (metallic) {
        return "BandGap(metallic)";
    }
    out << "BandGap(" << gap << " eV " << (direct ? "direct" : "indirect") << ", direct "
        << direct_gap << " eV)";
    return out.str();
}

std::string BandStructure::to_string() const {
    std::ostringstream out;
    out << "BandStructure(" << kpoints_.size() << " k-points, " << bands_ << " bands";
    if (spins_ == 2) {
        out << ", spin-polarised";
    }
    out << ", E_F = " << fermi_level_ << " eV)";
    return out.str();
}

}  // namespace fastatoms::electronic
