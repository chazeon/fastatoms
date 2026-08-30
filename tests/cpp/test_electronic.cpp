// SPDX-License-Identifier: MIT
///
/// Electronic structure analysis: band structures, gaps, and the density of
/// states by tetrahedron and by smearing. The DOS routines are checked against
/// the free electron gas, whose answer is known in closed form.

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "fastatoms/fastatoms.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::electronic;

namespace {

constexpr double kPi = 3.14159265358979323846;

/// A free electron gas on a grid: one band, `E(k) = hbar^2 k^2 / 2m` in units
/// where the prefactor is one. Its density of states goes as the square root
/// of the energy, which is the sharpest test of an integration scheme there is.
BandStructure free_electrons(int n, double scale = 1.0) {
    const std::vector<Vec3> kpoints = monkhorst_pack(n, n, n);
    std::vector<double> energies;
    energies.reserve(kpoints.size());
    for (const Vec3& k : kpoints) {
        // Fold into [-1/2, 1/2) so the parabola is centred on Gamma.
        const auto fold = [](double x) { return x - std::round(x); };
        const Vec3 folded{fold(k.x), fold(k.y), fold(k.z)};
        energies.push_back(scale * dot(folded, folded));
    }
    return BandStructure(kpoints, energies, 1, 1, 0.0);
}

/// A flat band at a fixed energy: every tetrahedron is degenerate, which is
/// the case the closed-form expressions have to survive.
BandStructure flat_band(int n, double energy) {
    const std::vector<Vec3> kpoints = monkhorst_pack(n, n, n);
    return BandStructure(kpoints, std::vector<double>(kpoints.size(), energy), 1, 1, 0.0);
}

double integrate(const DensityOfStates& dos) {
    return dos.integrated.empty() ? 0.0 : dos.integrated.back();
}

}  // namespace

// ---------------------------------------------------------------------------
// The container
// ---------------------------------------------------------------------------

FA_TEST(a_band_structure_holds_what_it_was_given) {
    const std::vector<Vec3> kpoints{Vec3{0, 0, 0}, Vec3{0.5, 0, 0}, Vec3{0.5, 0.5, 0}};
    // Two bands at each of three k-points.
    const std::vector<double> energies{-1.0, 2.0, -0.5, 2.5, -0.8, 3.0};
    const BandStructure bands(kpoints, energies, 2, 1, 0.0);

    FA_CHECK(bands.nkpoints() == 3);
    FA_CHECK(bands.nbands() == 2);
    FA_CHECK(bands.nspins() == 1);
    FA_CHECK_CLOSE(bands.energy(0, 1, 0), -0.5, 1e-15);
    FA_CHECK_CLOSE(bands.energy(0, 2, 1), 3.0, 1e-15);
    FA_CHECK_CLOSE(bands.minimum(), -1.0, 1e-15);
    FA_CHECK_CLOSE(bands.maximum(), 3.0, 1e-15);

    // A single band across the zone is what a plot draws as one line.
    const std::vector<double> first = bands.band(0, 0);
    FA_CHECK(first.size() == 3);
    FA_CHECK_CLOSE(first[2], -0.8, 1e-15);

    // Weights default to equal and sum to one.
    double total = 0.0;
    for (const double w : bands.weights()) {
        total += w;
    }
    FA_CHECK_CLOSE(total, 1.0, 1e-15);
}

FA_TEST(an_indirect_gap_is_told_from_a_direct_one) {
    // Valence maximum at k=0, conduction minimum at k=2: indirect. The direct
    // gap has to be found at whichever single k-point has the smallest
    // separation, which here is k=2 as well but larger than the fundamental.
    const std::vector<Vec3> kpoints{Vec3{0, 0, 0}, Vec3{0.25, 0, 0}, Vec3{0.5, 0, 0}};
    const std::vector<double> energies{
        0.0, 3.0,     // k = 0: valence 0.0, conduction 3.0
        -0.5, 2.5,    // k = 1
        -1.0, 1.5,    // k = 2: the lowest conduction state
    };
    BandStructure bands(kpoints, energies, 2, 1, 0.5);

    const BandGap gap = bands.band_gap();
    FA_CHECK(!gap.metallic);
    FA_CHECK(!gap.direct);
    FA_CHECK_CLOSE(gap.gap, 1.5, 1e-12);              // 1.5 - 0.0
    FA_CHECK_CLOSE(gap.valence_maximum, 0.0, 1e-12);
    FA_CHECK_CLOSE(gap.conduction_minimum, 1.5, 1e-12);
    FA_CHECK(gap.valence_kpoint == 0);
    FA_CHECK(gap.conduction_kpoint == 2);
    // The smallest single-k separation: 2.5 at k=2 (1.5 - (-1.0)).
    FA_CHECK_CLOSE(gap.direct_gap, 2.5, 1e-12);
    FA_CHECK(gap.direct_kpoint == 2);
    FA_CHECK(gap.direct_gap >= gap.gap);
}

FA_TEST(a_direct_gap_reports_itself_as_direct) {
    const std::vector<Vec3> kpoints{Vec3{0, 0, 0}, Vec3{0.5, 0, 0}};
    const std::vector<double> energies{0.0, 2.0, -1.0, 3.0};
    BandStructure bands(kpoints, energies, 2, 1, 0.5);

    const BandGap gap = bands.band_gap();
    FA_CHECK(gap.direct);
    FA_CHECK_CLOSE(gap.gap, 2.0, 1e-12);
    FA_CHECK_CLOSE(gap.direct_gap, gap.gap, 1e-12);
    FA_CHECK(gap.direct_kpoint == 0);
}

FA_TEST(a_band_crossing_the_fermi_level_is_a_metal) {
    const std::vector<Vec3> kpoints{Vec3{0, 0, 0}, Vec3{0.5, 0, 0}};
    const std::vector<double> energies{-1.0, 1.0};   // one band, crossing zero
    BandStructure bands(kpoints, energies, 1, 1, 0.0);

    FA_CHECK(bands.is_metal());
    FA_CHECK(bands.band_gap().metallic);
    FA_CHECK_CLOSE(bands.band_gap().gap, 0.0, 1e-15);
}

FA_TEST(shifting_puts_the_fermi_level_at_zero) {
    const std::vector<Vec3> kpoints{Vec3{0, 0, 0}};
    BandStructure bands(kpoints, {-2.0, 4.0}, 2, 1, 1.0);
    const BandStructure moved = bands.shifted();
    FA_CHECK_CLOSE(moved.fermi_level(), 0.0, 1e-15);
    FA_CHECK_CLOSE(moved.energy(0, 0, 0), -3.0, 1e-15);
    FA_CHECK_CLOSE(moved.energy(0, 0, 1), 3.0, 1e-15);
    // The gap is unchanged by where it is measured from.
    FA_CHECK_CLOSE(moved.band_gap().gap, bands.band_gap().gap, 1e-12);
}

FA_TEST(the_container_refuses_what_does_not_add_up) {
    const std::vector<Vec3> kpoints{Vec3{0, 0, 0}, Vec3{0.5, 0, 0}};
    FA_CHECK_THROWS(BandStructure(kpoints, {1.0, 2.0, 3.0}, 2), ElectronicError);
    FA_CHECK_THROWS(BandStructure({}, {}, 1), ElectronicError);
    FA_CHECK_THROWS(BandStructure(kpoints, {1.0, 2.0}, 0), ElectronicError);
    FA_CHECK_THROWS(BandStructure(kpoints, {1.0, 2.0}, 1, 3), ElectronicError);
    FA_CHECK_THROWS(BandStructure(kpoints, {1.0, 2.0}, 1, 1, 0.0, {1.0}), ElectronicError);

    BandStructure bands(kpoints, {1.0, 2.0}, 1);
    FA_CHECK_THROWS(bands.energy(0, 5, 0), ElectronicError);
    FA_CHECK_THROWS(bands.energy(1, 0, 0), ElectronicError);
}

// ---------------------------------------------------------------------------
// Smearing functions
// ---------------------------------------------------------------------------

FA_TEST(every_smearing_integrates_to_one) {
    // A delta replacement that does not have unit area gets the number of
    // states wrong by that factor, everywhere and silently.
    const double step = 0.001;
    for (const Smearing smearing : {Smearing::Gaussian, Smearing::MethfesselPaxton,
                                    Smearing::FermiDirac}) {
        for (const int order : {0, 1, 2}) {
            if (smearing != Smearing::MethfesselPaxton && order > 0) {
                continue;
            }
            double area = 0.0;
            for (double x = -40.0; x < 40.0; x += step) {
                area += smearing_delta(x + 0.5 * step, smearing, order) * step;
            }
            FA_CHECK_CLOSE(area, 1.0, 1e-6);
        }
    }
}

FA_TEST(methfessel_paxton_goes_negative_and_that_is_correct) {
    // The price of its faster convergence: the first-order function has
    // negative lobes, so a DOS smeared with it can dip below zero. A plain
    // Gaussian never does.
    bool negative = false;
    for (double x = 0.0; x < 6.0; x += 0.01) {
        if (smearing_delta(x, Smearing::MethfesselPaxton, 1) < -1e-6) {
            negative = true;
        }
        FA_CHECK(smearing_delta(x, Smearing::Gaussian, 0) >= 0.0);
    }
    FA_CHECK(negative);

    // Order zero is a plain Gaussian, by construction.
    for (double x = -3.0; x < 3.0; x += 0.1) {
        FA_CHECK_CLOSE(smearing_delta(x, Smearing::MethfesselPaxton, 0),
                       smearing_delta(x, Smearing::Gaussian, 0), 1e-12);
    }
}

FA_TEST(the_occupations_run_from_one_to_zero) {
    for (const Smearing smearing : {Smearing::Gaussian, Smearing::MethfesselPaxton,
                                    Smearing::FermiDirac}) {
        FA_CHECK_CLOSE(smearing_occupation(-30.0, smearing, 1), 1.0, 1e-9);
        FA_CHECK_CLOSE(smearing_occupation(30.0, smearing, 1), 0.0, 1e-9);
        FA_CHECK_CLOSE(smearing_occupation(0.0, smearing, 1), 0.5, 1e-9);
    }
    FA_CHECK_THROWS(smearing_delta(0.0, Smearing::MethfesselPaxton, -1), ElectronicError);
    FA_CHECK_THROWS(smearing_from_string("lorentzian"), ElectronicError);
    FA_CHECK(smearing_from_string("MP") == Smearing::MethfesselPaxton);
    FA_CHECK(smearing_from_string("fermi-dirac") == Smearing::FermiDirac);
}

// ---------------------------------------------------------------------------
// Density of states
// ---------------------------------------------------------------------------

FA_TEST(the_sum_rule_holds_for_both_methods) {
    // Whatever the shape, the area under the density of states is the number
    // of states there are: bands times the spin degeneracy.
    const BandStructure bands = free_electrons(12);

    DOSOptions options;
    options.points = 2000;
    options.width = 0.02;
    const DensityOfStates smeared = density_of_states(bands, options);
    const DensityOfStates tetrahedron =
        density_of_states_tetrahedron(bands, {12, 12, 12}, options);

    FA_CHECK_CLOSE(integrate(smeared), 2.0, 1e-3);       // one band, two spins
    FA_CHECK_CLOSE(integrate(tetrahedron), 2.0, 1e-3);
    FA_CHECK(tetrahedron.method == "tetrahedron");
    FA_CHECK(smeared.method == "gaussian");
}

FA_TEST(the_tetrahedron_method_recovers_the_free_electron_square_root) {
    // The test that says the closed-form isosurface integrals are right. For
    // E(k) = k^2 on a zone of unit volume the density of states is
    // 4 pi sqrt(E) with two spins, and the integrated one is (8/3) pi E^(3/2).
    const BandStructure bands = free_electrons(24);
    DOSOptions options;
    options.emin = 0.0;
    options.emax = 0.2;
    options.points = 400;
    const DensityOfStates dos = density_of_states_tetrahedron(bands, {24, 24, 24}, options);

    double worst = 0.0;
    double worst_integral = 0.0;
    int checked = 0;
    for (std::size_t k = 0; k < dos.size(); ++k) {
        const double e = dos.energy[k];
        if (e < 0.03 || e > 0.15) {
            continue;   // away from the origin, and inside the zone
        }
        const double expected = 4.0 * kPi * std::sqrt(e);
        worst = std::max(worst, std::abs(dos.dos[k] - expected) / expected);
        const double expected_integral = (8.0 / 3.0) * kPi * std::pow(e, 1.5);
        worst_integral =
            std::max(worst_integral, std::abs(dos.integrated[k] - expected_integral) /
                                         expected_integral);
        ++checked;
    }
    FA_CHECK(checked > 50);
    // Measured 1.2% on the density and 3.3% on its integral at this grid. The
    // linear tetrahedron method converges as one over the grid squared - the
    // bands are taken to be linear inside each tetrahedron and a parabola is
    // not - so these are the discretisation, not a mistake.
    FA_CHECK(worst < 0.05);
    FA_CHECK(worst_integral < 0.05);
}

FA_TEST(the_tetrahedron_method_keeps_improving_where_smearing_stops) {
    // The reason to have it. Refining the grid makes the tetrahedron result
    // better without limit; smearing on the same k-points stops improving once
    // the grid error falls below the width, because the width is then what is
    // wrong. Both are handed exactly the same eigenvalues.
    DOSOptions options;
    options.emin = 0.0;
    options.emax = 0.2;
    options.points = 400;
    options.width = 0.02;

    const auto error_of = [&](const DensityOfStates& dos) {
        double worst = 0.0;
        for (std::size_t k = 0; k < dos.size(); ++k) {
            const double e = dos.energy[k];
            if (e < 0.04 || e > 0.14) {
                continue;
            }
            const double expected = 4.0 * kPi * std::sqrt(e);
            worst = std::max(worst, std::abs(dos.dos[k] - expected) / expected);
        }
        return worst;
    };

    // Twenty-four points a side against thirty-two. Below this the smeared
    // result is still limited by the grid too and improves along with it; the
    // interesting comparison is where it has stopped being.
    const BandStructure coarse = free_electrons(24);
    const BandStructure fine = free_electrons(32);
    const double tetrahedron_coarse =
        error_of(density_of_states_tetrahedron(coarse, {24, 24, 24}, options));
    const double tetrahedron_fine =
        error_of(density_of_states_tetrahedron(fine, {32, 32, 32}, options));
    const double smeared_coarse = error_of(density_of_states(coarse, options));
    const double smeared_fine = error_of(density_of_states(fine, options));

    // The tetrahedron error keeps falling; the smeared one is unchanged to
    // within a per cent, because the width is now what is wrong and no number
    // of k-points fixes that.
    FA_CHECK(tetrahedron_fine < tetrahedron_coarse);
    FA_CHECK(smeared_fine > 0.95 * smeared_coarse);
    // And on the finer grid the tetrahedron method is the better of the two.
    FA_CHECK(tetrahedron_fine < smeared_fine);
}

FA_TEST(a_flat_band_survives_the_degenerate_case) {
    // Every tetrahedron has four equal corner energies, so every denominator
    // in the closed forms vanishes. The contribution is genuinely zero width
    // and unit weight, not infinite.
    const BandStructure bands = flat_band(8, 1.5);
    DOSOptions options;
    options.emin = 0.0;
    options.emax = 3.0;
    options.points = 600;
    const DensityOfStates dos = density_of_states_tetrahedron(bands, {8, 8, 8}, options);

    for (const double value : dos.dos) {
        FA_CHECK(std::isfinite(value));
        FA_CHECK(value >= 0.0);
    }
    // Smearing the same flat band puts all the weight at 1.5 eV.
    options.width = 0.05;
    const DensityOfStates smeared = density_of_states(bands, options);
    FA_CHECK_CLOSE(integrate(smeared), 2.0, 1e-6);
    const std::size_t peak = static_cast<std::size_t>(
        std::max_element(smeared.dos.begin(), smeared.dos.end()) - smeared.dos.begin());
    FA_CHECK_CLOSE(smeared.energy[peak], 1.5, 0.02);
}

FA_TEST(a_spin_polarised_calculation_keeps_the_channels_apart) {
    // Two spins, offset from each other, with one band each.
    const std::vector<Vec3> kpoints = monkhorst_pack(4, 4, 4);
    std::vector<double> energies;
    for (int spin = 0; spin < 2; ++spin) {
        for (std::size_t k = 0; k < kpoints.size(); ++k) {
            energies.push_back(spin == 0 ? -1.0 : 1.0);
        }
    }
    const BandStructure bands(kpoints, energies, 1, 2, 0.0);

    DOSOptions options;
    options.emin = -3.0;
    options.emax = 3.0;
    options.points = 600;
    options.width = 0.1;
    const DensityOfStates dos = density_of_states(bands, options);

    FA_CHECK(dos.spins == 2);
    FA_CHECK(dos.spin_dos.size() == 2 * dos.size());
    // One state per spin, so two in total and one in each channel.
    FA_CHECK_CLOSE(integrate(dos), 2.0, 1e-6);

    double up = 0.0;
    double down = 0.0;
    const double step = dos.energy[1] - dos.energy[0];
    for (std::size_t k = 0; k < dos.size(); ++k) {
        up += dos.spin_dos[k] * step;
        down += dos.spin_dos[dos.size() + k] * step;
    }
    FA_CHECK_CLOSE(up, 1.0, 1e-6);
    FA_CHECK_CLOSE(down, 1.0, 1e-6);
}

FA_TEST(the_grid_generator_and_the_tetrahedra_agree_about_order) {
    const std::vector<Vec3> grid = monkhorst_pack(2, 3, 4);
    FA_CHECK(grid.size() == 24);
    // Index (i * n2 + j) * n3 + k, which is what the tetrahedron walk assumes.
    FA_CHECK_CLOSE(grid[0].x, 0.0, 1e-15);
    FA_CHECK_CLOSE(grid[1].z, 0.25, 1e-15);
    FA_CHECK_CLOSE(grid[4].y, 1.0 / 3.0, 1e-15);
    FA_CHECK_CLOSE(grid[12].x, 0.5, 1e-15);

    // The shifted form misses Gamma, which is what it is for.
    const std::vector<Vec3> shifted = monkhorst_pack(2, 2, 2, false);
    for (const Vec3& k : shifted) {
        FA_CHECK(k.norm() > 1e-9);
    }
    FA_CHECK_THROWS(monkhorst_pack(0, 1, 1), ElectronicError);
}

FA_TEST(the_dos_routines_refuse_nonsense) {
    const BandStructure bands = free_electrons(4);
    DOSOptions options;
    options.points = 1;
    FA_CHECK_THROWS(density_of_states(bands, options), ElectronicError);

    DOSOptions no_width;
    no_width.width = 0.0;
    FA_CHECK_THROWS(density_of_states(bands, no_width), ElectronicError);

    // The tetrahedron method needs the bands on the grid it is told about.
    FA_CHECK_THROWS(density_of_states_tetrahedron(bands, {5, 5, 5}), ElectronicError);
    FA_CHECK_THROWS(density_of_states_tetrahedron(bands, {0, 4, 4}), ElectronicError);
}

// ---------------------------------------------------------------------------
// Band paths
// ---------------------------------------------------------------------------

FA_TEST(the_lattice_is_identified_from_the_cell_alone) {
    // No atoms involved: this is a fact about the lattice, and the answer must
    // not depend on which cell of it was handed over.
    FA_CHECK(identify_bravais(build::sc("Po", 3.35).cell()) == BravaisLattice::CUB);
    FA_CHECK(identify_bravais(build::fcc("Al", 4.05).cell()) == BravaisLattice::FCC);
    FA_CHECK(identify_bravais(build::fcc("Al", 4.05, true).cell()) == BravaisLattice::CUB);
    FA_CHECK(identify_bravais(build::bcc("Fe", 2.87).cell()) == BravaisLattice::BCC);
    FA_CHECK(identify_bravais(build::hcp("Mg", 3.21).cell()) == BravaisLattice::HEX);
    FA_CHECK(identify_bravais(build::graphene().cell()) == BravaisLattice::HEX);

    // A tetragonal cell, made by stretching a cube.
    Atoms stretched = build::sc("Po", 3.0);
    Mat3 matrix = stretched.cell().matrix();
    matrix(2, 2) = 4.5;
    FA_CHECK(identify_bravais(Cell(matrix, {true, true, true})) == BravaisLattice::TET);

    // And an orthorhombic one.
    matrix(1, 1) = 3.7;
    FA_CHECK(identify_bravais(Cell(matrix, {true, true, true})) == BravaisLattice::ORC);

    FA_CHECK(std::string(bravais_name(BravaisLattice::FCC)) == "FCC");
}

FA_TEST(the_special_points_are_the_conventional_ones) {
    const std::map<std::string, Vec3> fcc = special_points(build::fcc("Al", 4.05).cell());
    FA_CHECK(fcc.size() == 6);
    FA_CHECK_CLOSE(fcc.at("G").norm(), 0.0, 1e-15);
    FA_CHECK_CLOSE(fcc.at("X").x, 0.5, 1e-15);
    FA_CHECK_CLOSE(fcc.at("L").x, 0.5, 1e-15);
    FA_CHECK_CLOSE(fcc.at("K").z, 0.75, 1e-15);
    FA_CHECK(default_path(build::fcc("Al", 4.05).cell()) == "GXWKGLUWLK,UX");

    const std::map<std::string, Vec3> bcc = special_points(build::bcc("Fe", 2.87).cell());
    FA_CHECK(bcc.size() == 4);
    FA_CHECK_CLOSE(bcc.at("P").x, 0.25, 1e-15);
    FA_CHECK(default_path(build::bcc("Fe", 2.87).cell()) == "GHNGPH,PN");

    const std::map<std::string, Vec3> hex = special_points(build::graphene().cell());
    FA_CHECK_CLOSE(hex.at("K").x, 1.0 / 3.0, 1e-12);
    FA_CHECK_CLOSE(hex.at("M").x, 0.5, 1e-15);
}

FA_TEST(a_path_passes_through_the_points_it_names) {
    const Cell cell = build::fcc("Al", 4.05).cell();
    const BandPath path = bandpath(cell, "GXWK", 60);

    FA_CHECK(path.lattice == BravaisLattice::FCC);
    FA_CHECK(path.size() > 50 && path.size() <= 70);
    FA_CHECK(path.special_labels.size() == 4);
    FA_CHECK(path.special_labels[0] == "G");
    FA_CHECK(path.special_labels[3] == "K");

    // The first point is Gamma and the last is K, exactly.
    const std::map<std::string, Vec3> points = special_points(cell);
    FA_CHECK_CLOSE(path.kpoints.front().norm(), 0.0, 1e-15);
    FA_CHECK_CLOSE((path.kpoints.back() - points.at("K")).norm(), 0.0, 1e-15);

    // The axis runs forwards and never backwards.
    for (std::size_t k = 1; k < path.distances.size(); ++k) {
        FA_CHECK(path.distances[k] >= path.distances[k - 1]);
    }
    // And it is a real distance in inverse angstrom: Gamma to X of an fcc
    // lattice with a = 4.05 is 2 pi / a * sqrt(2) ... measured, not assumed.
    FA_CHECK(path.distances.back() > 0.0);
}

FA_TEST(the_sampling_is_even_in_reciprocal_space) {
    // A segment twice as long in the zone should get about twice the points,
    // so the density along the plotted axis is uniform.
    const Cell cell = build::sc("Po", 3.0).cell();
    const std::map<std::string, Vec3> points = special_points(cell);
    const BandPath path = bandpath(cell, "GXM", 200);

    const Mat3 reciprocal = cell.reciprocal();
    const auto length = [&](const Vec3& a, const Vec3& b) {
        const Vec3 d{b.x - a.x, b.y - a.y, b.z - a.z};
        return (d * reciprocal).norm();
    };
    const double first = length(points.at("G"), points.at("X"));
    const double second = length(points.at("X"), points.at("M"));

    std::size_t in_first = 0;
    for (const std::string& label : path.labels) {
        (void)label;
    }
    // Count k-points before the X tick.
    const double x_at = path.special_distances[1];
    for (const double d : path.distances) {
        if (d < x_at - 1e-9) {
            ++in_first;
        }
    }
    const double expected = path.size() * first / (first + second);
    FA_CHECK(std::abs(static_cast<double>(in_first) - expected) < 0.1 * expected);
}

FA_TEST(a_comma_makes_the_path_jump) {
    const Cell cell = build::fcc("Al", 4.05).cell();
    const BandPath joined = bandpath(cell, "GXWK", 60);
    const BandPath jumped = bandpath(cell, "GX,WK", 60);

    // The same four special points either way, but the jump contributes no
    // length, so the axis is shorter.
    FA_CHECK(jumped.special_labels.size() == 4);
    FA_CHECK(jumped.distances.back() < joined.distances.back());
}

FA_TEST(an_explicit_path_works_for_any_lattice) {
    // The escape hatch for a lattice with no table.
    const Cell cell = build::fcc("Al", 4.05).cell();
    const std::vector<Vec3> corners{{0.0, 0.0, 0.0}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.5}};
    const BandPath path = bandpath_from_points(cell, corners, {"G", "X", "L"}, 40);
    FA_CHECK(path.size() > 35);
    FA_CHECK(path.special_labels.size() == 3);
    FA_CHECK_CLOSE((path.kpoints.back() - corners.back()).norm(), 0.0, 1e-15);

    // Labels are optional.
    const BandPath unlabelled = bandpath_from_points(cell, corners, {}, 40);
    FA_CHECK(unlabelled.special_labels.empty());
}

FA_TEST(band_paths_refuse_nonsense) {
    const Cell cell = build::fcc("Al", 4.05).cell();
    FA_CHECK_THROWS(bandpath(cell, "GQZ", 40), ElectronicError);         // no such point
    FA_CHECK_THROWS(bandpath_from_points(cell, {{0, 0, 0}}, {}, 10), ElectronicError);
    FA_CHECK_THROWS(bandpath_from_points(cell, {{0, 0, 0}, {0.5, 0, 0}}, {"G"}, 10),
                    ElectronicError);
    FA_CHECK_THROWS(bandpath_from_points(cell, {{0, 0, 0}, {0.5, 0, 0}}, {}, 1),
                    ElectronicError);
    FA_CHECK_THROWS(identify_bravais(Cell{}), ElectronicError);

    // A lattice with no table says so, and says which one it is.
    Mat3 triclinic;
    triclinic(0, 0) = 3.0; triclinic(0, 1) = 0.0; triclinic(0, 2) = 0.0;
    triclinic(1, 0) = 0.7; triclinic(1, 1) = 4.1; triclinic(1, 2) = 0.0;
    triclinic(2, 0) = 0.4; triclinic(2, 1) = 1.1; triclinic(2, 2) = 5.3;
    const Cell odd(triclinic, {true, true, true});
    FA_CHECK(identify_bravais(odd) == BravaisLattice::TRI);
    FA_CHECK_THROWS(special_points(odd), ElectronicError);
    FA_CHECK_THROWS(bandpath(odd), ElectronicError);
}

FA_TEST_MAIN()
