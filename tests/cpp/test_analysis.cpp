// SPDX-License-Identifier: MIT
///
/// Trajectory analysis: g(r), S(q) and surface tension. Each is checked
/// against something known independently - the coordination shells of a
/// crystal, an ideal gas, the agreement of three different routes to the same
/// structure factor.

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "fastatoms/fastatoms.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::analysis;

namespace {

/// The coordination number at a radius, read off the running integral.
double coordination_at(const RadialDistribution& rdf, double radius) {
    double value = 0.0;
    for (std::size_t k = 0; k < rdf.size(); ++k) {
        if (rdf.r[k] <= radius) {
            value = rdf.coordination[k];
        }
    }
    return value;
}

/// Where g(r) has its peaks, above a threshold.
std::vector<double> peaks(const RadialDistribution& rdf, double threshold) {
    std::vector<double> out;
    for (std::size_t k = 1; k + 1 < rdf.size(); ++k) {
        if (rdf.g[k] > threshold && rdf.g[k] >= rdf.g[k - 1] && rdf.g[k] > rdf.g[k + 1]) {
            out.push_back(rdf.r[k]);
        }
    }
    return out;
}

/// Atoms at random positions in a cube: an ideal gas, whose g(r) is one.
Atoms ideal_gas(std::size_t count, double side, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, side);
    std::vector<double> positions;
    for (std::size_t k = 0; k < 3 * count; ++k) {
        positions.push_back(uniform(rng));
    }
    Atoms atoms(std::vector<std::string>(count, "Ar"), positions, Cell::cubic(side));
    atoms.set_pbc({true, true, true});
    return atoms;
}

}  // namespace

// ---------------------------------------------------------------------------
// g(r)
// ---------------------------------------------------------------------------

FA_TEST(the_crystal_shells_come_out_where_they_should) {
    const double a = 4.05;
    const Atoms crystal = build::fcc("Al", a, true).repeat(4, 4, 4);

    RDFOptions options;
    options.rmax = 7.0;
    options.bins = 700;
    const RadialDistribution rdf = radial_distribution(crystal, options);

    // The first five fcc shells, at a/sqrt(2), a, a*sqrt(1.5), a*sqrt(2) and
    // a*sqrt(2.5). Nothing about the histogram knows these.
    const std::vector<double> expected{a / std::sqrt(2.0), a, a * std::sqrt(1.5),
                                       a * std::sqrt(2.0), a * std::sqrt(2.5)};
    const std::vector<double> found = peaks(rdf, 1.5);
    FA_CHECK(found.size() >= expected.size());
    for (std::size_t k = 0; k < expected.size(); ++k) {
        FA_CHECK_CLOSE(found[k], expected[k], 0.02);
    }

    // And the shells hold the numbers of atoms they are supposed to: 12, then
    // 6 more, then 24 more.
    FA_CHECK_CLOSE(coordination_at(rdf, 3.2), 12.0, 1e-9);
    FA_CHECK_CLOSE(coordination_at(rdf, 4.4), 18.0, 1e-9);
    FA_CHECK_CLOSE(coordination_at(rdf, 5.3), 42.0, 1e-9);
}

FA_TEST(an_ideal_gas_has_a_flat_g_of_r) {
    // The normalisation, checked against the only case where the answer is
    // known exactly. Averaged over frames to beat the noise down.
    RDFOptions options;
    options.rmax = 8.0;
    options.bins = 40;
    RadialDistributionAccumulator accumulator(options);
    for (unsigned frame = 0; frame < 60; ++frame) {
        accumulator.add(ideal_gas(400, 25.0, frame + 1));
    }
    const RadialDistribution rdf = accumulator.result();

    // Skip the first bins, where the count per frame is tiny and the scatter
    // is large; from 2 A out the average must sit on one.
    double worst = 0.0;
    for (std::size_t k = 0; k < rdf.size(); ++k) {
        if (rdf.r[k] > 2.0) {
            worst = std::max(worst, std::abs(rdf.g[k] - 1.0));
        }
    }
    FA_CHECK(worst < 0.05);
}

FA_TEST(species_pairs_are_selected_independently) {
    // Rock salt: the sodium-chlorine correlation starts at a/2, while
    // sodium-sodium and chlorine-chlorine start at a/sqrt(2).
    const double a = 5.64;
    const Atoms salt = build::rocksalt("Na", "Cl", a, true).repeat(3, 3, 3);

    RDFOptions options;
    options.rmax = 6.0;
    options.bins = 600;

    RDFOptions cross = options;
    cross.first = {"Na"};
    cross.second = {"Cl"};
    const RadialDistribution na_cl = radial_distribution(salt, cross);

    RDFOptions like = options;
    like.first = {"Na"};
    like.second = {"Na"};
    const RadialDistribution na_na = radial_distribution(salt, like);

    FA_CHECK(na_cl.label == "Na-Cl");
    FA_CHECK(na_na.label == "Na-Na");
    FA_CHECK_CLOSE(peaks(na_cl, 1.5).front(), a / 2, 0.02);
    FA_CHECK_CLOSE(peaks(na_na, 1.5).front(), a / std::sqrt(2.0), 0.02);

    // Six chlorines around every sodium, and twelve sodiums.
    FA_CHECK_CLOSE(coordination_at(na_cl, 3.5), 6.0, 1e-9);
    FA_CHECK_CLOSE(coordination_at(na_na, 4.5), 12.0, 1e-9);

    // The cross correlation is symmetric in its two selections.
    RDFOptions reversed = options;
    reversed.first = {"Cl"};
    reversed.second = {"Na"};
    const RadialDistribution cl_na = radial_distribution(salt, reversed);
    for (std::size_t k = 0; k < na_cl.size(); ++k) {
        FA_CHECK_CLOSE(na_cl.g[k], cl_na.g[k], 1e-9);
    }
}

FA_TEST(the_accumulator_and_the_whole_trajectory_agree) {
    const std::vector<Atoms> frames{ideal_gas(200, 20.0, 1), ideal_gas(200, 20.0, 2),
                                    ideal_gas(200, 20.0, 3)};
    RDFOptions options;
    options.rmax = 6.0;
    options.bins = 60;

    const RadialDistribution all_at_once = radial_distribution(frames, options);
    RadialDistributionAccumulator accumulator(options);
    for (const Atoms& frame : frames) {
        accumulator.add(frame);
    }
    const RadialDistribution streamed = accumulator.result();

    FA_CHECK(streamed.frames == 3);
    for (std::size_t k = 0; k < all_at_once.size(); ++k) {
        FA_CHECK_CLOSE(all_at_once.g[k], streamed.g[k], 1e-12);
    }
}

FA_TEST(g_of_r_says_where_it_stops_being_trustworthy) {
    // Beyond half the smallest cell width a shell no longer fits in the box.
    // The value is reported rather than the request being refused, because
    // sometimes you do want to look out there.
    const Atoms small = build::fcc("Al", 4.05, true);
    RDFOptions options;
    options.rmax = 12.0;
    const RadialDistribution rdf = radial_distribution(small, options);
    FA_CHECK_CLOSE(rdf.reliable_rmax, 4.05 / 2, 1e-12);
    FA_CHECK(rdf.rmax > rdf.reliable_rmax);
}

FA_TEST(bad_requests_are_refused) {
    const Atoms atoms = build::fcc("Al", 4.05);
    RDFOptions bad;
    bad.rmax = -1.0;
    FA_CHECK_THROWS(radial_distribution(atoms, bad), AnalysisError);

    RDFOptions no_bins;
    no_bins.bins = 0;
    FA_CHECK_THROWS(radial_distribution(atoms, no_bins), AnalysisError);

    FA_CHECK_THROWS(radial_distribution(std::vector<Atoms>{}), AnalysisError);

    // Frames of different sizes cannot be averaged.
    RadialDistributionAccumulator accumulator{};
    accumulator.add(ideal_gas(50, 15.0, 1));
    FA_CHECK_THROWS(accumulator.add(ideal_gas(51, 15.0, 2)), AnalysisError);

    // No cell means no density to normalise against.
    Atoms cluster(std::vector<std::string>{"Ar", "Ar"}, {0.0, 0.0, 0.0, 3.8, 0.0, 0.0});
    FA_CHECK_THROWS(radial_distribution(cluster), AnalysisError);
}

// ---------------------------------------------------------------------------
// S(q)
// ---------------------------------------------------------------------------

namespace {

/// S(q) at the bin nearest a given q.
double structure_at(const StructureFactor& sq, double q) {
    std::size_t best = 0;
    for (std::size_t k = 1; k < sq.size(); ++k) {
        if (std::abs(sq.q[k] - q) < std::abs(sq.q[best] - q)) {
            best = k;
        }
    }
    return sq.s[best];
}

}  // namespace

FA_TEST(the_reciprocal_lattice_finds_the_bragg_peaks) {
    // The conventional fcc cell, whose reflections are allowed only when h, k
    // and l are all even or all odd. That extinction rule is a fact about the
    // structure and nothing in the code knows it.
    const double a = 4.05;
    const StructureFactor sq =
        structure_factor_reciprocal({build::fcc("Al", a, true)}, 5.0, 500);

    const double two_pi = 2.0 * 3.14159265358979323846;
    // Allowed: (111) and (200) and (220), each with all four atoms in phase,
    // so |sum|^2 / N = 16 / 4 = 4.
    FA_CHECK_CLOSE(structure_at(sq, two_pi * std::sqrt(3.0) / a), 4.0, 1e-6);
    FA_CHECK_CLOSE(structure_at(sq, two_pi * 2.0 / a), 4.0, 1e-6);
    FA_CHECK_CLOSE(structure_at(sq, two_pi * 2.0 * std::sqrt(2.0) / a), 4.0, 1e-6);
    // Forbidden: (100) and (110), where the four atoms cancel exactly.
    FA_CHECK_CLOSE(structure_at(sq, two_pi / a), 0.0, 1e-9);
    FA_CHECK_CLOSE(structure_at(sq, two_pi * std::sqrt(2.0) / a), 0.0, 1e-9);
}

FA_TEST(an_ideal_gas_has_a_structure_factor_of_one) {
    // Every route must give the same flat answer, which is what says the
    // normalisations agree.
    std::vector<Atoms> frames;
    for (unsigned k = 0; k < 20; ++k) {
        frames.push_back(ideal_gas(500, 28.0, k + 1));
    }

    StructureFactorOptions options;
    options.qmin = 1.0;
    options.qmax = 8.0;
    options.points = 30;

    const StructureFactor debye = structure_factor_debye(frames, options);
    RDFOptions rdf_options;
    rdf_options.rmax = 12.0;
    rdf_options.bins = 240;
    const StructureFactor transformed =
        structure_factor_from_rdf(radial_distribution(frames, rdf_options), options);
    const StructureFactor reciprocal = structure_factor_reciprocal(frames, 8.0, 40);

    for (std::size_t k = 0; k < debye.size(); ++k) {
        FA_CHECK(std::abs(debye.s[k] - 1.0) < 0.15);
        FA_CHECK(std::abs(transformed.s[k] - 1.0) < 0.15);
    }
    for (std::size_t k = 0; k < reciprocal.size(); ++k) {
        if (reciprocal.weight[k] > 20.0 && reciprocal.q[k] > 1.0) {
            FA_CHECK(std::abs(reciprocal.s[k] - 1.0) < 0.25);
        }
    }
    FA_CHECK(debye.method == "debye");
    FA_CHECK(transformed.method == "rdf");
    FA_CHECK(reciprocal.method == "reciprocal");
}

FA_TEST(the_debye_sum_and_the_transform_of_g_of_r_agree) {
    // Two independent routes to the same quantity on a real, structured
    // system. They are not expected to agree to machine precision - one bins
    // distances and one does not - but they must trace the same curve.
    std::vector<Atoms> frames;
    for (unsigned k = 0; k < 12; ++k) {
        frames.push_back(ideal_gas(300, 22.0, 100 + k));
    }
    StructureFactorOptions options;
    options.qmin = 1.5;
    options.qmax = 9.0;
    options.points = 25;

    RDFOptions rdf_options;
    rdf_options.rmax = 11.0;
    rdf_options.bins = 220;
    const RadialDistribution rdf = radial_distribution(frames, rdf_options);

    StructureFactorOptions matched = options;
    matched.rmax = rdf.rmax;
    const StructureFactor debye = structure_factor_debye(frames, matched);
    const StructureFactor transformed = structure_factor_from_rdf(rdf, matched);

    for (std::size_t k = 0; k < debye.size(); ++k) {
        FA_CHECK(std::abs(debye.s[k] - transformed.s[k]) < 0.1);
    }
}

FA_TEST(structure_factors_refuse_nonsense) {
    const std::vector<Atoms> frames{build::fcc("Al", 4.05)};
    StructureFactorOptions backwards;
    backwards.qmin = 5.0;
    backwards.qmax = 1.0;
    FA_CHECK_THROWS(structure_factor_debye(frames, backwards), AnalysisError);

    StructureFactorOptions no_points;
    no_points.points = 0;
    FA_CHECK_THROWS(structure_factor_debye(frames, no_points), AnalysisError);

    FA_CHECK_THROWS(structure_factor_debye({}), AnalysisError);
    FA_CHECK_THROWS(structure_factor_reciprocal(frames, -1.0), AnalysisError);
    FA_CHECK_THROWS(structure_factor_reciprocal({}), AnalysisError);
}

// ---------------------------------------------------------------------------
// Surface tension
// ---------------------------------------------------------------------------

FA_TEST(the_tension_formula_is_the_formula) {
    // Arithmetic first, against tensors chosen so the answer can be worked out
    // by hand: gamma = (L/2)[P_zz - (P_xx + P_yy)/2].
    Mat3 pressure = Mat3::diagonal(1.0, 3.0, 10.0);
    const SurfaceTension out = surface_tension_from_pressures({pressure}, {20.0}, 2);
    // (20/2) * (10 - (1 + 3)/2) = 10 * 8 = 80
    FA_CHECK_CLOSE(out.gamma, 80.0, 1e-12);
    FA_CHECK_CLOSE(out.gamma_mN_per_m, 80.0 * kMilliNewtonPerMetre, 1e-6);
    FA_CHECK(out.frames == 1);
    FA_CHECK(out.normal == 2);
    FA_CHECK_CLOSE(out.standard_error, 0.0, 1e-15);

    // The same tensor read along x instead: (20/2) * (1 - (3 + 10)/2) = -55.
    const SurfaceTension along_x = surface_tension_from_pressures({pressure}, {20.0}, 0);
    FA_CHECK_CLOSE(along_x.gamma, -55.0, 1e-12);
    FA_CHECK(along_x.normal == 0);
}

FA_TEST(an_isotropic_pressure_gives_no_tension) {
    // Whatever the pressure is, if it is the same in every direction there is
    // no interface and no tension. This is the null result the sign convention
    // has to respect.
    for (const double p : {-0.05, 0.0, 0.05, 2.0}) {
        const SurfaceTension out =
            surface_tension_from_pressures({Mat3::diagonal(p, p, p)}, {30.0}, 2);
        FA_CHECK_CLOSE(out.gamma, 0.0, 1e-14);
    }
}

FA_TEST(the_average_and_its_error_come_from_the_frames) {
    std::vector<Mat3> pressures;
    std::vector<double> lengths;
    // Tensions of 10, 20 and 30 by construction, with L = 2 so the half and
    // the length cancel.
    for (const double zz : {10.0, 20.0, 30.0}) {
        pressures.push_back(Mat3::diagonal(0.0, 0.0, zz));
        lengths.push_back(2.0);
    }
    const SurfaceTension out = surface_tension_from_pressures(pressures, lengths, 2);
    FA_CHECK(out.per_frame.size() == 3);
    FA_CHECK_CLOSE(out.per_frame[0], 10.0, 1e-12);
    FA_CHECK_CLOSE(out.gamma, 20.0, 1e-12);
    // Standard error of the mean of {10, 20, 30}: sd is 10, over sqrt(3).
    FA_CHECK_CLOSE(out.standard_error, 10.0 / std::sqrt(3.0), 1e-12);
    // And the mean pressure tensor is the mean of the tensors.
    FA_CHECK_CLOSE(out.mean_pressure(2, 2), 20.0, 1e-12);
}

FA_TEST(a_bulk_crystal_has_no_interface) {
    // The physical null test, all the way through a calculator: an unstrained
    // cubic crystal has an isotropic pressure and so no tension.
    Atoms crystal = build::fcc("Ar", 5.26, true).repeat(3, 3, 3);
    crystal.set_calculator(std::make_shared<LennardJonesCalculator>(
        Value(Value::Map{{"epsilon", Value(0.0103)}, {"sigma", Value(3.40)},
                         {"cutoff", Value(8.0)}})));
    std::vector<Atoms> frames{crystal};

    const SurfaceTension out = surface_tension(frames, 2);
    FA_CHECK(std::abs(out.gamma_mN_per_m) < 1e-6);

    // The pressure tensor really is isotropic and really is a pressure: this
    // lattice constant is under the equilibrium one, so the crystal pushes out.
    const Mat3 pressure = out.mean_pressure;
    FA_CHECK_CLOSE(pressure(0, 0), pressure(2, 2), 1e-12);
    FA_CHECK(std::abs(pressure(0, 1)) < 1e-12);
}

FA_TEST(a_stretched_cell_shows_the_anisotropy) {
    // Pull the crystal along z and the normal component of the pressure falls
    // below the in-plane ones, which is a negative tension by this convention.
    Atoms crystal = build::fcc("Ar", 5.26, true).repeat(3, 3, 3);
    Mat3 stretched = crystal.cell().matrix();
    stretched(2, 2) *= 1.06;
    crystal.set_cell(Cell(stretched, {true, true, true}), true);
    crystal.set_calculator(std::make_shared<LennardJonesCalculator>(
        Value(Value::Map{{"epsilon", Value(0.0103)}, {"sigma", Value(3.40)},
                         {"cutoff", Value(8.0)}})));

    std::vector<Atoms> frames{crystal};
    const SurfaceTension out = surface_tension(frames, 2);
    FA_CHECK(out.gamma < 0.0);
    FA_CHECK(out.mean_pressure(2, 2) < out.mean_pressure(0, 0));
}

FA_TEST(surface_tension_refuses_nonsense) {
    std::vector<Atoms> empty;
    FA_CHECK_THROWS(surface_tension(empty), AnalysisError);
    FA_CHECK_THROWS(surface_tension_from_pressures({}, {}, 2), AnalysisError);
    FA_CHECK_THROWS(surface_tension_from_pressures({Mat3::identity()}, {1.0}, 3),
                    AnalysisError);
    // One length per tensor, or the averaging is meaningless.
    FA_CHECK_THROWS(
        surface_tension_from_pressures({Mat3::identity(), Mat3::identity()}, {1.0}, 2),
        AnalysisError);
}

FA_TEST_MAIN()
