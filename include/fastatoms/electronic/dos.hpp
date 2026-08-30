// SPDX-License-Identifier: MIT
#pragma once

/// \file dos.hpp
/// \brief Density of states, by the linear tetrahedron method and by smearing.
///
/// Both answer the same question - how many states there are per unit energy -
/// and differ in how they interpolate between the k-points that were actually
/// computed. Smearing replaces each eigenvalue with a bump of finite width and
/// adds them up; the tetrahedron method fills the zone with tetrahedra, takes
/// the bands to be linear inside each, and integrates the isosurface exactly.
///
/// The tetrahedron method needs a regular grid and gives sharp van Hove
/// singularities; smearing works on any set of k-points and rounds them off.

#include <array>
#include <string>
#include <vector>

#include "fastatoms/electronic/bands.hpp"

namespace fastatoms::electronic {

/// Which bump to put at each eigenvalue.
enum class Smearing {
    Gaussian,          ///< the obvious one, and always non-negative
    MethfesselPaxton,  ///< converges faster in the width, at the cost of going negative
    FermiDirac,        ///< the physical one, when the width is a real temperature
};

[[nodiscard]] std::string_view smearing_name(Smearing smearing);
[[nodiscard]] Smearing smearing_from_string(std::string_view name);

/// What to compute.
struct DOSOptions {
    /// Energy window. Leaving both at zero takes the range from the data with
    /// a margin of a few smearing widths.
    double emin{0.0};
    double emax{0.0};
    int points{600};

    /// Smearing width in eV. Ignored by the tetrahedron method.
    double width{0.1};
    Smearing smearing{Smearing::Gaussian};
    /// Order of the Methfessel-Paxton expansion. Zero is a plain Gaussian.
    int order{1};

    /// States per band. Two for a non-spin-polarised calculation, where each
    /// band holds a spin-up and a spin-down electron; one when the two spins
    /// were computed separately. Zero asks for the sensible default.
    int spin_degeneracy{0};
};

/// A computed density of states.
struct DensityOfStates {
    /// The energy grid, in eV.
    std::vector<double> energy{};
    /// States per unit cell per eV, summed over bands and spins.
    std::vector<double> dos{};
    /// The running integral: how many states lie below each energy. At the
    /// Fermi level this is the number of electrons per cell.
    std::vector<double> integrated{};
    /// Per spin channel, laid out `[spin][point]`. Empty for one spin.
    std::vector<double> spin_dos{};

    double fermi_level{0.0};
    std::size_t spins{1};
    /// "tetrahedron", "gaussian", "methfessel-paxton" or "fermi-dirac".
    std::string method{};

    [[nodiscard]] std::size_t size() const noexcept { return energy.size(); }
    /// The density of states at the Fermi level, interpolated.
    [[nodiscard]] double at_fermi() const;
    [[nodiscard]] std::string to_string() const;
};

/// The k-points of a regular grid, in the order the tetrahedron method expects.
///
/// Index `(i * n2 + j) * n3 + k`, with fractional coordinates `i / n1` and so
/// on. Gamma-centred by default; the shifted form is the original Monkhorst
/// and Pack one, which misses Gamma and is better for metals.
[[nodiscard]] std::vector<Vec3> monkhorst_pack(int n1, int n2, int n3,
                                               bool gamma_centred = true);

/// Density of states by smearing.
///
/// Works on any set of k-points, including a band path, though a path is a
/// poor sample of the zone and the result should not be read as a real DOS.
[[nodiscard]] DensityOfStates density_of_states(const BandStructure& bands,
                                                DOSOptions options = {});

/// Density of states by the linear tetrahedron method.
///
/// The bands must be on the regular grid `monkhorst_pack(n1, n2, n3)` returns,
/// in that order. Each microcell of the grid is cut into six tetrahedra along
/// its shortest body diagonal, the bands are taken to vary linearly inside
/// each, and the resulting isosurface integral is done in closed form - which
/// is what makes the van Hove singularities come out sharp instead of smeared.
///
/// \param reciprocal the reciprocal lattice vectors as rows, used only to
///        decide which body diagonal is shortest; identity is a fine default
///        for a cubic grid
/// \throws ElectronicError if the number of k-points does not match the grid
[[nodiscard]] DensityOfStates density_of_states_tetrahedron(const BandStructure& bands,
                                                            std::array<int, 3> grid,
                                                            DOSOptions options = {},
                                                            Mat3 reciprocal = Mat3::identity());

/// The smearing function itself, normalised to integrate to one.
///
/// Exposed because it is the piece worth checking on its own: a
/// Methfessel-Paxton function of order one integrates to one and goes
/// negative, which is correct and surprising.
[[nodiscard]] double smearing_delta(double x, Smearing smearing, int order);

/// The occupation a smearing implies at `x = (E - mu) / width`: one well below
/// the Fermi level, zero well above.
[[nodiscard]] double smearing_occupation(double x, Smearing smearing, int order);

}  // namespace fastatoms::electronic
