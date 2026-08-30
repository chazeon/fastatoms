// SPDX-License-Identifier: MIT
#pragma once

/// \file surface_tension.hpp
/// \brief Interfacial tension from the pressure tensor of a trajectory.
///
/// A slab with two flat interfaces perpendicular to one axis has a pressure
/// tensor that is not isotropic: normal to the surface it is the bulk
/// pressure, in the plane of the surface it is not, and the difference
/// integrated across the box is the tension holding the interface together.
///
///     gamma = (L_n / 2) [ P_nn - (P_t1t1 + P_t2t2) / 2 ]
///
/// The half is not a normalisation of the average - it is there because a
/// periodic slab has *two* interfaces, and the integral counts both.

#include <string>
#include <vector>

#include "fastatoms/analysis/rdf.hpp"
#include "fastatoms/core/atoms.hpp"
#include "fastatoms/core/vec3.hpp"

namespace fastatoms::analysis {

/// eV per square angstrom, in millinewtons per metre - the unit surface
/// tension is normally quoted in. Water is about 72 mN/m at room temperature.
inline constexpr double kMilliNewtonPerMetre = 16021.766208;

/// The result of averaging the tension over a trajectory.
struct SurfaceTension {
    /// The tension, in eV per square angstrom.
    double gamma{0.0};
    /// The same, in millinewtons per metre.
    double gamma_mN_per_m{0.0};

    /// Standard error of the mean over frames. Surface tension converges
    /// slowly, and a value without this next to it says very little.
    double standard_error{0.0};
    double standard_error_mN_per_m{0.0};

    /// The per-frame values, in eV per square angstrom, so the run can be
    /// blocked or plotted.
    std::vector<double> per_frame{};

    /// The trajectory-averaged pressure tensor, in eV per cubic angstrom.
    Mat3 mean_pressure{};

    /// Which axis is normal to the interface: 0, 1 or 2.
    int normal{2};
    int frames{0};

    [[nodiscard]] std::string to_string() const;
};

/// Surface tension from a trajectory.
///
/// Each frame needs a calculator that can produce a stress; the kinetic
/// contribution is included when the frame carries momenta. The frames are
/// taken by non-const reference because evaluating the stress is what a
/// calculator does to a structure.
///
/// \param normal the axis perpendicular to the interface, 2 (z) by default
/// \throws AnalysisError for an empty trajectory or an axis that is not 0, 1
///         or 2
[[nodiscard]] SurfaceTension surface_tension(std::vector<Atoms>& frames, int normal = 2);

/// Surface tension from pressure tensors that have already been computed.
///
/// The route to take when the stresses came from somewhere else - a log file,
/// another code - or when the trajectory is too large to hold.
///
/// \param pressures one tensor per frame, in eV per cubic angstrom
/// \param lengths   the cell length along the normal, per frame, in angstrom
[[nodiscard]] SurfaceTension surface_tension_from_pressures(
    const std::vector<Mat3>& pressures, const std::vector<double>& lengths, int normal = 2);

/// The pressure tensor of one frame, in eV per cubic angstrom.
///
/// The negative of the stress, with the kinetic term included when the frame
/// has momenta - the sign convention that makes a compressed solid have a
/// positive pressure.
[[nodiscard]] Mat3 pressure_tensor(Atoms& atoms);

}  // namespace fastatoms::analysis
