// SPDX-License-Identifier: MIT
//
// The phonon calculation itself, checked against dispersions that are known
// analytically and against the unreduced calculation.

#include <cmath>
#include <memory>
#include <vector>

#include "fastatoms/calculators/calculator.hpp"
#include "fastatoms/calculators/lennard_jones.hpp"
#include "fastatoms/core/neighbor_list.hpp"
#include "fastatoms/phonon/phonons.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::phonon;

namespace {

constexpr double kTwoPi = 6.283185307179586;
/// sqrt(eV / (angstrom^2 amu)) expressed in THz, written out independently of
/// the library so the conversion itself is under test.
constexpr double kToTerahertz = 0.09822694788464063 * 1000.0 / kTwoPi;

Value settings(std::initializer_list<std::pair<const char*, Value>> entries) {
    Value map{Value::Map{}};
    for (const auto& [key, value] : entries) {
        map.insert(key, value);
    }
    return map;
}

/// Independent harmonic wells, `E = (k/2) sum |r_i - r0_i|^2`.
///
/// Every atom is bound to its own site with the same stiffness, so the force
/// constants are `k` times the identity on the diagonal and zero elsewhere,
/// and every mode has `omega = sqrt(k/m)` at every q. An Einstein solid: the
/// one phonon problem with an exact answer at every step.
class EinsteinCalculator : public Calculator {
public:
    EinsteinCalculator(double stiffness, std::vector<double> centres)
        : Calculator(Value(Value::Map{})),
          stiffness_(stiffness),
          centres_(std::move(centres)) {}

    [[nodiscard]] std::string name() const override { return "einstein"; }
    [[nodiscard]] std::vector<Property> implemented_properties() const override {
        return {Property::Energy, Property::FreeEnergy, Property::Forces};
    }

protected:
    void compute(const Atoms& atoms, const std::vector<Property>&) override {
        const std::vector<double>& positions = atoms.positions();
        double energy = 0.0;
        std::vector<double> forces(positions.size());
        for (std::size_t k = 0; k < positions.size(); ++k) {
            const double d = positions[k] - centres_[k];
            energy += 0.5 * stiffness_ * d * d;
            forces[k] = -stiffness_ * d;
        }
        results_.energy = energy;
        results_.free_energy = energy;
        results_.forces = std::move(forces);
    }

private:
    double stiffness_;
    std::vector<double> centres_;
};

/// Central springs between neighbours, `E = (k/2) sum (r_ij - r0)^2`.
///
/// At the natural length the transverse restoring force vanishes, so a simple
/// cubic lattice with nearest-neighbour springs has the textbook dispersion
/// `omega_L(q)^2 = (2k/m)(1 - cos(2 pi q))` along [100] with two flat zero
/// branches - an analytic answer for a genuinely q-dependent problem.
class SpringCalculator : public Calculator {
public:
    SpringCalculator(double stiffness, double rest_length, double cutoff)
        : Calculator(Value(Value::Map{})),
          stiffness_(stiffness),
          rest_length_(rest_length),
          cutoff_(cutoff) {}

    [[nodiscard]] std::string name() const override { return "springs"; }
    [[nodiscard]] std::vector<Property> implemented_properties() const override {
        return {Property::Energy, Property::FreeEnergy, Property::Forces};
    }

protected:
    void compute(const Atoms& atoms, const std::vector<Property>&) override {
        NeighborListOptions options;
        options.cutoff = cutoff_;
        options.full = false;
        const NeighborList pairs = build_neighbor_list(atoms, options);

        double energy = 0.0;
        std::vector<double> forces(3 * atoms.size(), 0.0);
        for (std::size_t k = 0; k < pairs.size(); ++k) {
            const Vec3 dr = Vec3::load(pairs.vectors.data() + 3 * k);
            const double r = dr.norm();
            if (r <= 0.0) {
                continue;
            }
            const double stretch = r - rest_length_;
            energy += 0.5 * stiffness_ * stretch * stretch;

            // F_i = k (r - r0) dr / r, and the partner gets the opposite.
            const Vec3 force = dr * (stiffness_ * stretch / r);
            const auto i = static_cast<std::size_t>(pairs.i[k]);
            const auto j = static_cast<std::size_t>(pairs.j[k]);
            for (std::size_t a = 0; a < 3; ++a) {
                forces[3 * i + a] += force[a];
                forces[3 * j + a] -= force[a];
            }
        }
        results_.energy = energy;
        results_.free_energy = energy;
        results_.forces = std::move(forces);
    }

private:
    double stiffness_;
    double rest_length_;
    double cutoff_;
};

Atoms simple_cubic(double a, double mass) {
    Atoms atoms(std::vector<int>{18}, {0.0, 0.0, 0.0}, Cell::cubic(a));
    atoms.set_masses({mass});
    return atoms;
}

Atoms fcc_argon(double a = 5.26) {
    const Vec3 a1{0.0, 0.5 * a, 0.5 * a};
    const Vec3 a2{0.5 * a, 0.0, 0.5 * a};
    const Vec3 a3{0.5 * a, 0.5 * a, 0.0};
    Atoms atoms(std::vector<int>{18}, {0.0, 0.0, 0.0}, Cell(a1, a2, a3, {true, true, true}));
    atoms.set_calculator(std::make_shared<LennardJonesCalculator>(settings(
        {{"epsilon", Value(0.0103)}, {"sigma", Value(3.40)}, {"cutoff", Value(8.0)}})));
    return atoms;
}

Atoms diamond_silicon(double a = 5.43) {
    const Vec3 a1{0.0, 0.5 * a, 0.5 * a};
    const Vec3 a2{0.5 * a, 0.0, 0.5 * a};
    const Vec3 a3{0.5 * a, 0.5 * a, 0.0};
    Atoms atoms(std::vector<int>{14, 14}, std::vector<double>(6, 0.0),
                Cell(a1, a2, a3, {true, true, true}));
    atoms.set_scaled_positions({0.0, 0.0, 0.0, 0.25, 0.25, 0.25});
    return atoms;
}

}  // namespace

// ---------------------------------------------------------------------------
// An exactly solvable crystal
// ---------------------------------------------------------------------------

FA_TEST(einstein_solid_has_the_analytic_frequency_everywhere) {
    const double stiffness = 2.0;   // eV/angstrom^2
    const double mass = 8.0;        // amu
    Atoms unit = simple_cubic(4.0, mass);

    Phonons phonons(unit, settings({{"supercell", Value(2)},
                                    {"displacement", Value(0.01)},
                                    // Each atom is bound to its own site, so
                                    // the rows do not sum to zero and imposing
                                    // that would be wrong here.
                                    {"acoustic_sum_rule", Value(false)}}));
    phonons.run(std::make_shared<EinsteinCalculator>(
        stiffness, phonons.supercell().atoms.positions()));

    // Phi(i,j) = k delta_ij, exactly.
    const ForceConstants& constants = phonons.force_constants();
    for (std::size_t j = 0; j < constants.columns(); ++j) {
        const Mat3& block = constants.block(0, j);
        for (std::size_t a = 0; a < 3; ++a) {
            for (std::size_t b = 0; b < 3; ++b) {
                const double expected = (j == 0 && a == b) ? stiffness : 0.0;
                FA_CHECK_CLOSE(block(a, b), expected, 1e-9);
            }
        }
    }

    // omega = sqrt(k/m) at every q, threefold degenerate.
    const double expected = std::sqrt(stiffness / mass) * kToTerahertz;
    for (const Vec3& q : {Vec3{0, 0, 0}, Vec3{0.5, 0, 0}, Vec3{0.25, 0.5, 0.125},
                          Vec3{0.5, 0.5, 0.5}}) {
        const std::vector<double> frequencies = phonons.frequencies(q);
        FA_CHECK(frequencies.size() == 3);
        for (const double value : frequencies) {
            FA_CHECK_CLOSE(value, expected, 1e-6);
        }
    }
}

FA_TEST(a_spring_lattice_reproduces_its_analytic_dispersion) {
    const double a = 3.0;
    const double stiffness = 1.5;   // eV/angstrom^2
    const double mass = 4.0;        // amu

    Atoms unit = simple_cubic(a, mass);
    // A cutoff between a and a*sqrt(2) keeps only the six nearest neighbours.
    auto springs = std::make_shared<SpringCalculator>(stiffness, a, 1.2 * a);

    Phonons phonons(unit, settings({{"supercell", Value(4)},
                                    {"displacement", Value(0.002)}}));
    phonons.run(springs);

    // omega_L(q)^2 = (2k/m)(1 - cos(2 pi q)) along [100]; the two transverse
    // branches are flat at zero because a central spring at its rest length
    // resists no transverse motion.
    //
    // "At its rest length" is the catch, and it is worth being precise about.
    // Displacing an atom by d transverse to a bond stretches that bond by
    // d^2/2a, leaving a spurious transverse force constant k d^2 / 2a^2. Its
    // ratio to the real longitudinal one is d^2/2a^2, so the spurious branch
    // tracks the real one scaled by d/(a sqrt(2)) - an artifact of finite
    // displacements rather than a bug, and one whose size is predictable
    // exactly. Asserting that prediction is a stronger statement than any
    // loose bound would be.
    const double displacement = 0.002;
    for (const double q : {0.0, 0.125, 0.25, 0.375, 0.5}) {
        const std::vector<double> frequencies = phonons.frequencies(Vec3{q, 0.0, 0.0});
        FA_CHECK(frequencies.size() == 3);

        const double squared = (2.0 * stiffness / mass) * (1.0 - std::cos(kTwoPi * q));
        const double longitudinal = std::sqrt(squared) * kToTerahertz;
        const double artifact = longitudinal * displacement / (a * std::sqrt(2.0));

        // Sorted ascending: the two artifact branches, then the real one.
        FA_CHECK_CLOSE(frequencies[0], artifact, 1e-4);
        FA_CHECK_CLOSE(frequencies[1], artifact, 1e-4);
        FA_CHECK_CLOSE(frequencies[2], longitudinal, 2e-4);
    }

    // Along [111] all three branches are degenerate at the zone boundary of
    // this lattice; check one interior point against the same formula, which
    // separates into independent Cartesian directions for central springs.
    const double q = 0.25;
    const std::vector<double> frequencies = phonons.frequencies(Vec3{q, q, q});
    const double squared = (2.0 * stiffness / mass) * (1.0 - std::cos(kTwoPi * q));
    for (const double value : frequencies) {
        FA_CHECK_CLOSE(value, std::sqrt(squared) * kToTerahertz, 2e-4);
    }
}

// ---------------------------------------------------------------------------
// Symmetry reduction
// ---------------------------------------------------------------------------

FA_TEST(symmetry_reduction_reproduces_the_full_calculation) {
    // The test the whole symmetry machinery stands on: displacing only the
    // independent atoms along the independent directions, then rebuilding the
    // rest from the space group, has to give the same force constants as
    // moving everything.
    Atoms unit = diamond_silicon();
    auto springs = std::make_shared<SpringCalculator>(1.2, 5.43 * std::sqrt(3.0) / 4.0, 3.0);

    Phonons reduced(unit, settings({{"supercell", Value(2)}, {"symmetry", Value(true)}}));
    Phonons full(unit, settings({{"supercell", Value(2)}, {"symmetry", Value(false)}}));

    // Symmetry has to actually save something, or the test proves nothing.
    FA_CHECK(reduced.calculation_count() < full.calculation_count());
    FA_CHECK(full.calculation_count() == full.calculation_count_without_symmetry());

    reduced.run(springs);
    full.run(springs);

    const ForceConstants& a = reduced.force_constants();
    const ForceConstants& b = full.force_constants();
    FA_CHECK(a.rows() == b.rows());
    FA_CHECK(a.columns() == b.columns());

    double largest = 0.0;
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.columns(); ++j) {
            for (std::size_t k = 0; k < 9; ++k) {
                largest = std::max(largest, std::abs(a.block(i, j).m[k] - b.block(i, j).m[k]));
            }
        }
    }
    FA_CHECK(largest < 1e-8);

    // And so the frequencies agree wherever they are asked for.
    for (const Vec3& q : {Vec3{0, 0, 0}, Vec3{0.5, 0, 0}, Vec3{0.3, 0.1, 0.25}}) {
        const std::vector<double> from_reduced = reduced.frequencies(q);
        const std::vector<double> from_full = full.frequencies(q);
        FA_CHECK(from_reduced.size() == from_full.size());
        for (std::size_t k = 0; k < from_reduced.size(); ++k) {
            FA_CHECK_CLOSE(from_reduced[k], from_full[k], 1e-6);
        }
    }
}

FA_TEST(symmetry_reduces_the_work_by_the_expected_amount) {
    // Simple cubic: one atom, site symmetry m-3m. One direction is enough,
    // and the other two follow from the cubic axes being equivalent.
    Phonons cubic(simple_cubic(3.0, 10.0), settings({{"supercell", Value(2)}}));
    FA_CHECK(cubic.displacements().size() == 2);  // +d and -d along one axis
    FA_CHECK(cubic.calculation_count_without_symmetry() == 6);
    FA_CHECK(cubic.symmetry().independent_atoms.size() == 1);

    // Diamond silicon: two atoms in one orbit, site symmetry of order 24, so
    // again a single direction on a single atom.
    Phonons silicon(diamond_silicon(), settings({{"supercell", Value(2)}}));
    FA_CHECK(silicon.symmetry().independent_atoms.size() == 1);
    FA_CHECK(silicon.displacements().size() == 2);
    FA_CHECK(silicon.calculation_count_without_symmetry() == 12);

    // One-sided differences halve the count again.
    Phonons one_sided(diamond_silicon(),
                      settings({{"supercell", Value(2)}, {"central", Value(false)}}));
    FA_CHECK(one_sided.displacements().size() == 1);

    // With symmetry switched off, every atom and direction is displaced.
    Phonons unreduced(diamond_silicon(),
                      settings({{"supercell", Value(2)}, {"symmetry", Value(false)}}));
    FA_CHECK(unreduced.displacements().size() == 12);
}

FA_TEST(lowering_the_symmetry_costs_more_displacements) {
    // Zincblende rather than diamond: the two sites now hold different
    // elements, so no operation can exchange them and both have to be
    // displaced. (Merely moving an atom would not do it - inversion through
    // the midpoint keeps any two like atoms in a two-atom cell equivalent.)
    Atoms zincblende = diamond_silicon();
    zincblende.set_number(1, 32);

    Phonons phonons(zincblende, settings({{"supercell", Value(2)}}));
    FA_CHECK(phonons.symmetry().independent_atoms.size() == 2);
    FA_CHECK(phonons.displacements().size() == 4);  // one direction each, both signs

    // A structure with no symmetry left at all needs every displacement.
    Atoms distorted = diamond_silicon();
    std::vector<double> positions = distorted.positions();
    positions[0] += 0.21;
    positions[1] -= 0.13;
    positions[5] += 0.07;
    distorted.set_positions(positions);
    Phonons awkward(distorted, settings({{"supercell", Value(2)}}));
    FA_CHECK(awkward.displacements().size() > 4);
}

// ---------------------------------------------------------------------------
// Physical properties
// ---------------------------------------------------------------------------

FA_TEST(acoustic_modes_vanish_at_the_zone_centre) {
    // Translational invariance means three modes go to zero at Gamma. That is
    // exactly what the acoustic sum rule enforces, so this is its test.
    Atoms unit = fcc_argon();
    Phonons phonons(unit, settings({{"supercell", Value(2)}, {"displacement", Value(0.01)}}));
    phonons.run();

    FA_CHECK_CLOSE(phonons.force_constants().acoustic_sum_violation(), 0.0, 1e-12);

    const std::vector<double> gamma = phonons.frequencies(Vec3{0, 0, 0});
    FA_CHECK(gamma.size() == 3);
    for (const double value : gamma) {
        FA_CHECK(std::abs(value) < 1e-6);
    }

    // Away from Gamma the crystal is stable: real frequencies, and they grow.
    const std::vector<double> boundary = phonons.frequencies(Vec3{0.5, 0.0, 0.0});
    for (const double value : boundary) {
        FA_CHECK(value > 0.0);
    }
    FA_CHECK(boundary.back() > 0.1);
}

FA_TEST(the_dynamical_matrix_is_hermitian_and_time_reversal_symmetric) {
    Phonons phonons(fcc_argon(), settings({{"supercell", Value(2)}}));
    phonons.run();

    for (const Vec3& q : {Vec3{0.0, 0.0, 0.0}, Vec3{0.5, 0.0, 0.0}, Vec3{0.13, -0.4, 0.22}}) {
        const ComplexMatrix matrix = phonons.dynamical_matrix(q);
        FA_CHECK(matrix.size() == 3);
        FA_CHECK(matrix.hermiticity_error() < 1e-12);

        // omega(q) = omega(-q): the force constants are real.
        const std::vector<double> forward = phonons.frequencies(q);
        const std::vector<double> backward = phonons.frequencies(q * -1.0);
        for (std::size_t k = 0; k < forward.size(); ++k) {
            FA_CHECK_CLOSE(forward[k], backward[k], 1e-9);
        }
    }
}

FA_TEST(frequencies_are_periodic_in_the_reciprocal_lattice) {
    Phonons phonons(diamond_silicon(), settings({{"supercell", Value(2)}}));
    phonons.run(std::make_shared<SpringCalculator>(1.2, 5.43 * std::sqrt(3.0) / 4.0, 3.0));

    const Vec3 q{0.3, 0.1, -0.2};
    const std::vector<double> here = phonons.frequencies(q);
    const std::vector<double> shifted = phonons.frequencies(q + Vec3{1.0, -2.0, 3.0});
    FA_CHECK(here.size() == 6);
    for (std::size_t k = 0; k < here.size(); ++k) {
        FA_CHECK_CLOSE(here[k], shifted[k], 1e-8);
    }
}

FA_TEST(an_unstable_crystal_reports_imaginary_frequencies) {
    // Stretch argon until its neighbours sit beyond the inflection point of
    // the pair potential, where the bond stiffness turns negative and the
    // lattice stops being mechanically stable. The convention is to report the
    // resulting imaginary frequencies as negative numbers.
    Atoms stretched = fcc_argon(7.0);
    Phonons phonons(stretched, settings({{"supercell", Value(2)}}));
    phonons.run();

    bool unstable = false;
    for (const Vec3& q : {Vec3{0.5, 0.0, 0.0}, Vec3{0.5, 0.5, 0.0}, Vec3{0.5, 0.5, 0.5},
                          Vec3{0.25, 0.25, 0.0}}) {
        if (phonons.frequencies(q).front() < -1e-3) {
            unstable = true;
        }
    }
    FA_CHECK(unstable);

    // Argon at its equilibrium spacing, by contrast, is stable everywhere.
    Phonons stable(fcc_argon(5.31), settings({{"supercell", Value(2)}}));
    stable.run();
    for (const Vec3& q : {Vec3{0.5, 0.0, 0.0}, Vec3{0.5, 0.5, 0.5}}) {
        FA_CHECK(stable.frequencies(q).front() > -1e-6);
    }
}

FA_TEST(band_structures_and_meshes) {
    Phonons phonons(fcc_argon(), settings({{"supercell", Value(2)}}));
    phonons.run();

    const std::vector<Vec3> path{{0, 0, 0}, {0.25, 0, 0}, {0.5, 0, 0}};
    const std::vector<std::vector<double>> bands = phonons.band_structure(path);
    FA_CHECK(bands.size() == 3);
    FA_CHECK(bands[0].size() == 3);
    FA_CHECK(bands[0][2] < bands[2][2]);  // the branch rises away from Gamma

    const std::vector<double> mesh = phonons.mesh_frequencies(4, 4, 4);
    FA_CHECK(mesh.size() == 64 * 3);
    FA_CHECK_THROWS(phonons.mesh_frequencies(0, 1, 1), PhononError);
}

// ---------------------------------------------------------------------------
// Driving it by hand
// ---------------------------------------------------------------------------

FA_TEST(forces_can_be_supplied_from_outside) {
    // The workflow for running the displacements on a cluster: take the
    // supercells, compute them elsewhere, feed the forces back.
    Atoms unit = fcc_argon();
    auto calculator = unit.calculator();

    Phonons offline(unit, settings({{"supercell", Value(2)}}));
    const std::vector<Atoms> configurations = offline.displaced_supercells();
    FA_CHECK(configurations.size() == offline.displacements().size());

    std::vector<std::vector<double>> forces;
    for (Atoms configuration : configurations) {
        configuration.set_calculator(calculator);
        forces.push_back(configuration.get_forces());
    }
    offline.set_forces(forces);

    Phonons direct(unit, settings({{"supercell", Value(2)}}));
    direct.run();

    for (const Vec3& q : {Vec3{0, 0, 0}, Vec3{0.5, 0.25, 0}}) {
        const std::vector<double> offline_frequencies = offline.frequencies(q);
        const std::vector<double> direct_frequencies = direct.frequencies(q);
        for (std::size_t k = 0; k < direct_frequencies.size(); ++k) {
            FA_CHECK_CLOSE(offline_frequencies[k], direct_frequencies[k], 1e-9);
        }
    }
}

FA_TEST(phonon_errors_are_reported) {
    Atoms molecule(std::vector<int>{18, 18}, {0.0, 0.0, 0.0, 3.0, 0.0, 0.0});
    FA_CHECK_THROWS(Phonons(molecule), PhononError);

    Atoms empty;
    FA_CHECK_THROWS(Phonons(empty), PhononError);

    Atoms unit = fcc_argon();
    unit.set_calculator(nullptr);
    Phonons phonons(unit, settings({{"supercell", Value(2)}}));
    FA_CHECK_THROWS(phonons.run(), PhononError);
    FA_CHECK_THROWS(phonons.force_constants(), PhononError);
    FA_CHECK_THROWS(phonons.frequencies(Vec3{0, 0, 0}), PhononError);

    FA_CHECK_THROWS(Phonons(fcc_argon(), settings({{"displacement", Value(0.0)}}))
                        .displacement_distance(),
                    PhononError);
    FA_CHECK_THROWS(phonons.set_forces({}), PhononError);
}

FA_TEST_MAIN()
