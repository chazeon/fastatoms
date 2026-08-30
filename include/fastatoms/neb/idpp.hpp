// SPDX-License-Identifier: MIT
#pragma once

/// \file idpp.hpp
/// \brief Image-dependent pair potential: a usable starting band where a
///        straight line is not one.

#include <cstddef>
#include <string>
#include <vector>

#include "fastatoms/calculators/calculator.hpp"
#include "fastatoms/core/atoms.hpp"

namespace fastatoms::neb {

/// The IDPP objective for one image (Smidstrup, Pedersen, Hobolth and
/// Jonsson, J. Chem. Phys. 140, 214106 (2014)).
///
/// Interpolating coordinates in a straight line is fine for a molecule that
/// rotates a little, and useless in anything dense: two atoms swapping places
/// pass through each other, and the first force call on the resulting band
/// lands somewhere with an enormous repulsion that no optimizer recovers from
/// gracefully.
///
/// The fix is to interpolate the *pair distances* instead - a quantity with no
/// such pathology - and then find the coordinates that best reproduce them:
///
///     S_i = sum over j<k  w(t_jk) (d_jk - t_jk)^2,    w(t) = 1/t^4
///
/// where `t_jk` is the linear interpolation between the two endpoints' `j-k`
/// distances, and `d_jk` is the distance in the image being placed. The 1/t^4
/// weight is what makes it work: without it the objective is dominated by the
/// many large distances, which are also the ones that matter least.
///
/// This is an objective, not a physical potential, so its "energy" is in no
/// unit worth naming and its "forces" are the negative gradient of it.
class IDPPCalculator : public Calculator {
public:
    /// \param targets the target distance matrix, N*N row-major
    /// \param mic measure distances with the minimum-image convention
    IDPPCalculator(std::vector<double> targets, std::size_t natoms, bool mic);

    [[nodiscard]] std::string name() const override { return "idpp"; }
    [[nodiscard]] std::vector<Property> implemented_properties() const override;

    /// The target distances this image is trying to reproduce.
    [[nodiscard]] const std::vector<double>& targets() const noexcept { return targets_; }

protected:
    void compute(const Atoms& atoms, const std::vector<Property>& properties) override;

private:
    std::vector<double> targets_;
    std::size_t natoms_;
    bool mic_;
    /// 1 / sum of the weights, so the objective does not scale as the fourth
    /// power of the system's typical interatomic distance.
    double scale_{1.0};
};

/// How hard to work at the interpolation.
struct IDPPOptions {
    /// Force tolerance for the band on the IDPP objective. This is a starting
    /// guess, not an answer, so it does not want to be tight.
    double fmax{0.1};
    int max_steps{200};
    /// Spring constant for the band that relaxes on the objective.
    double spring_constant{1.0};
    bool minimum_image{true};
};

/// A band between two endpoints, placed by IDPP rather than by a straight line.
///
/// `images` counts the endpoints, which are returned unchanged. The endpoints
/// need no calculator: the objective is built from their geometry alone.
///
/// \throws NEBError if the endpoints do not match, or if `images` is under 3.
[[nodiscard]] std::vector<Atoms> idpp_interpolate(const Atoms& initial, const Atoms& final,
                                                  std::size_t images,
                                                  const IDPPOptions& options = {});

/// The closest approach between any two atoms in `atoms`, in angstrom.
///
/// The one number that says whether an interpolated band is usable: a straight
/// line through a dense system routinely produces images with atoms 0.2 A
/// apart, and no calculator will survive that.
[[nodiscard]] double closest_approach(const Atoms& atoms, bool mic = true);

}  // namespace fastatoms::neb
