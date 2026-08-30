// SPDX-License-Identifier: MIT
//
// The nudged elastic band, checked against surfaces whose saddle points are
// known in closed form.
//
// A band on a real system can only be compared with another implementation.
// A single particle on an analytic two-dimensional surface can be compared
// with the answer: the saddle of E = (x^2-1)^2 + y^2 + a x is a root of a
// cubic, so the barrier is exact to machine precision and "the climbing image
// found the saddle" is a statement with a number attached.
//
// The tilted surface matters more than the symmetric one. On a symmetric
// barrier with an odd number of images, one image sits on the saddle whether
// it climbs or not, so a broken climbing image still looks right. Tilting it
// moves the saddle off the midpoint and the difference becomes visible.

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fastatoms/calculators/calculator.hpp"
#include "fastatoms/neb/autoneb.hpp"
#include "fastatoms/neb/idpp.hpp"
#include "fastatoms/neb/neb.hpp"
#include "fastatoms/neb/registry.hpp"
#include "fastatoms/optimize/fire.hpp"
#include "fastatoms/optimize/lbfgs.hpp"
#include "fastatoms/calculators/registry.hpp"
#include "fastatoms/io/yaml_input.hpp"
#include "fastatoms/optimize/registry.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::neb;

namespace {

constexpr double kTilt = 0.30;

/// One particle on E(x, y) = (x^2 - 1)^2 + y^2 + tilt * x.
class TiltedWell : public Calculator {
public:
    explicit TiltedWell(double tilt = kTilt) : tilt_(tilt) {}

    [[nodiscard]] std::string name() const override { return "tilted_well"; }
    [[nodiscard]] std::vector<Property> implemented_properties() const override {
        return {Property::Energy, Property::FreeEnergy, Property::Forces};
    }

    [[nodiscard]] static double energy_at(double x, double tilt = kTilt) {
        return (x * x - 1.0) * (x * x - 1.0) + tilt * x;
    }

    /// Newton on dE/dx = 4x^3 - 4x + tilt, from a starting guess.
    [[nodiscard]] static double stationary_point(double guess, double tilt = kTilt) {
        double x = guess;
        for (int step = 0; step < 200; ++step) {
            const double slope = 4 * x * x * x - 4 * x + tilt;
            const double curvature = 12 * x * x - 4;
            x -= slope / curvature;
        }
        return x;
    }

protected:
    void compute(const Atoms& atoms, const std::vector<Property>&) override {
        const double x = atoms.position(0).x;
        const double y = atoms.position(0).y;
        results_.clear();
        results_.energy = energy_at(x, tilt_);
        results_.free_energy = results_.energy;
        results_.forces = {-(4 * x * (x * x - 1) + tilt_), -2 * y, 0.0};
    }

private:
    double tilt_;
};

Atoms particle(double x, double y, double tilt = kTilt) {
    Atoms atoms;
    atoms.append(1, Vec3{x, y, 0.0});
    atoms.set_calculator(std::make_shared<TiltedWell>(tilt));
    return atoms;
}

std::shared_ptr<optimize::Optimizer> band_optimizer(double fmax = 1e-8, int steps = 5000) {
    return optimize::make_optimizer(
        "fire", Value(Value::Map{{"fmax", Value(fmax)},
                                 {"max_steps", Value(static_cast<std::int64_t>(steps))},
                                 {"line_search", Value(std::string("none"))}}));
}

}  // namespace

// ---------------------------------------------------------------------------
// Building a band
// ---------------------------------------------------------------------------
FA_TEST(interpolation_places_images_evenly) {
    const std::vector<Atoms> band =
        NEB::interpolate(particle(-1.0, 0.0), particle(1.0, 0.0), 3);
    FA_CHECK(band.size() == 5);
    for (std::size_t k = 0; k < band.size(); ++k) {
        const double expected = -1.0 + 2.0 * static_cast<double>(k) / 4.0;
        FA_CHECK_CLOSE(band[k].position(0).x, expected, 1e-12);
    }
}

FA_TEST(a_band_refuses_what_it_cannot_relax) {
    Atoms bare;
    bare.append(1, Vec3{0.0, 0.0, 0.0});

    // Fewer than three images: nothing to move.
    FA_CHECK_THROWS(NEB(std::vector<Atoms>{particle(-1, 0), particle(1, 0)}), NEBError);
    // An image with no calculator.
    FA_CHECK_THROWS(NEB(std::vector<Atoms>{particle(-1, 0), bare, particle(1, 0)}), NEBError);
    // Endpoints that are not the same system.
    Atoms other;
    other.append(2, Vec3{0.0, 0.0, 0.0});
    other.set_calculator(std::make_shared<TiltedWell>());
    FA_CHECK_THROWS(NEB(std::vector<Atoms>{particle(-1, 0), other, particle(1, 0)}), NEBError);
    FA_CHECK_THROWS(NEB::interpolate(particle(-1, 0), particle(1, 0), 0), NEBError);
}

FA_TEST(the_coordinate_vector_is_the_movable_images) {
    NEB band = NEB::from_endpoints(particle(-1, 0), particle(1, 0), 3);
    FA_CHECK(band.size() == 5);
    FA_CHECK(band.movable() == 3);
    FA_CHECK(band.ndof() == 3 * 1 * 3);

    std::vector<double> x = band.positions();
    FA_CHECK(x.size() == band.ndof());
    // The endpoints are not in it, and are never moved.
    x[0] = 0.5;
    band.set_positions(x);
    FA_CHECK_CLOSE(band.image(1).position(0).x, 0.5, 1e-12);
    FA_CHECK_CLOSE(band.image(0).position(0).x, -1.0, 1e-12);
    FA_CHECK_CLOSE(band.image(4).position(0).x, 1.0, 1e-12);

    FA_CHECK_THROWS(band.set_positions(std::vector<double>(5, 0.0)), NEBError);
    FA_CHECK_THROWS(band.tangent(0), NEBError);
    FA_CHECK_THROWS(band.image(99), NEBError);
}

// ---------------------------------------------------------------------------
// The projection
// ---------------------------------------------------------------------------
FA_TEST(the_tangent_follows_the_uphill_neighbour) {
    // On a rising stretch the tangent points forward; on a falling one, back.
    // Getting this wrong - falling through to the mixed case - leaves a band
    // that finds the right saddle and never converges, because the projection
    // disagrees with itself between neighbouring images.
    NEB band = NEB::from_endpoints(particle(-1, 0), particle(1, 0), 5);
    const BandProfile profile = band.profile();

    for (std::size_t k = 1; k + 1 < band.size(); ++k) {
        const std::vector<double> tau = band.tangent(k);
        double length = 0.0;
        for (const double c : tau) {
            length += c * c;
        }
        FA_CHECK_CLOSE(std::sqrt(length), 1.0, 1e-12);   // always a unit vector

        const bool rising = profile.energies[k + 1] > profile.energies[k] &&
                            profile.energies[k] > profile.energies[k - 1];
        const bool falling = profile.energies[k + 1] < profile.energies[k] &&
                             profile.energies[k] < profile.energies[k - 1];
        if (rising) {
            FA_CHECK(tau[0] > 0.0);      // towards the higher neighbour, +x
        } else if (falling) {
            FA_CHECK(tau[0] > 0.0);      // the path still runs +x
        }
    }
}

FA_TEST(the_true_force_survives_only_perpendicular_to_the_tangent) {
    // The defining property of the nudging. On a band with no springs acting -
    // evenly spaced, so every spring term cancels - what is left of the true
    // force must be exactly perpendicular to the local tangent, at every
    // image. If any parallel component leaked through, the images would slide
    // downhill into the endpoints and the path would collapse.
    NEBParameters parameters;
    parameters.spring_constant = 1.0;
    NEB band = NEB::from_endpoints(particle(-1.0, 0.4), particle(1.0, -0.3), 7, parameters);

    const std::vector<double> forces = band.forces();
    for (std::size_t k = 1; k + 1 < band.size(); ++k) {
        const std::vector<double> tau = band.tangent(k);
        double parallel = 0.0;
        for (std::size_t c = 0; c < tau.size(); ++c) {
            parallel += forces[(k - 1) * tau.size() + c] * tau[c];
        }
        FA_CHECK_CLOSE(parallel, 0.0, 1e-12);
    }
}

FA_TEST(the_climbing_image_keeps_a_parallel_component_and_the_others_do_not) {
    // The exception that proves it: the climber's parallel component is not
    // removed but reversed, which is the whole of the climbing algorithm.
    NEBParameters parameters;
    parameters.spring_constant = 1.0;
    parameters.climbing = true;
    NEB band = NEB::from_endpoints(particle(-1.0, 0.4), particle(1.0, -0.3), 7, parameters);

    const std::vector<double> forces = band.forces();
    const std::size_t top = band.highest_image();
    std::size_t with_parallel = 0;
    for (std::size_t k = 1; k + 1 < band.size(); ++k) {
        const std::vector<double> tau = band.tangent(k);
        double parallel = 0.0;
        for (std::size_t c = 0; c < tau.size(); ++c) {
            parallel += forces[(k - 1) * tau.size() + c] * tau[c];
        }
        if (std::abs(parallel) > 1e-10) {
            ++with_parallel;
            FA_CHECK(k == top);
        }
    }
    FA_CHECK(with_parallel == 1);
}

FA_TEST(the_spring_force_acts_only_along_the_band) {
    // Bunch the images towards one end and check the springs push back along
    // the path and nowhere else. A spring with a perpendicular component is
    // the corner-cutting the nudging exists to prevent.
    NEBParameters parameters;
    parameters.spring_constant = 1.0;
    NEB band = NEB::from_endpoints(particle(-1, 0), particle(1, 0), 3, parameters);
    band.image(1).set_position(0, Vec3{-0.9, 0.0, 0.0});
    band.image(2).set_position(0, Vec3{-0.8, 0.0, 0.0});
    band.image(3).set_position(0, Vec3{-0.7, 0.0, 0.0});

    const std::vector<double> forces = band.forces();
    for (std::size_t k = 0; k < band.movable(); ++k) {
        // The path lies along x, so nothing should be pushed along y or z.
        FA_CHECK_CLOSE(forces[3 * k + 1], 0.0, 1e-12);
        FA_CHECK_CLOSE(forces[3 * k + 2], 0.0, 1e-12);
    }
    // The bunched images are pulled forwards, towards the empty stretch.
    FA_CHECK(forces[3 * 2 + 0] > 0.0);
}

// ---------------------------------------------------------------------------
// Saddle points
// ---------------------------------------------------------------------------
FA_TEST(a_symmetric_barrier_relaxes_to_a_symmetric_profile) {
    NEBParameters parameters;
    parameters.spring_constant = 1.0;
    NEB band = NEB::from_endpoints(particle(-1, 0, 0.0), particle(1, 0, 0.0), 7, parameters);
    for (std::size_t k = 1; k + 1 < band.size(); ++k) {
        band.image(k).set_position(0, band.image(k).position(0) + Vec3{0.0, 0.35, 0.0});
    }

    auto optimizer = band_optimizer();
    const optimize::OptimizationResult result = band.run(*optimizer);
    FA_CHECK(result.converged);

    const BandProfile profile = band.profile();
    // The band was pushed off the path and came back onto it.
    for (std::size_t k = 0; k < band.size(); ++k) {
        FA_CHECK_CLOSE(band.image(k).position(0).y, 0.0, 1e-6);
    }
    // A symmetric surface must give a symmetric profile.
    for (std::size_t k = 0; k < profile.energies.size(); ++k) {
        FA_CHECK_CLOSE(profile.energies[k],
                       profile.energies[profile.energies.size() - 1 - k], 1e-6);
    }
    FA_CHECK_CLOSE(profile.forward_barrier, 1.0, 1e-6);
}

FA_TEST(the_climbing_image_finds_the_saddle_a_plain_band_misses) {
    // The whole point of CI-NEB, with the difference measured.
    const double saddle_x = TiltedWell::stationary_point(0.0);
    const double minimum_x = TiltedWell::stationary_point(-1.0);
    const double exact = TiltedWell::energy_at(saddle_x) - TiltedWell::energy_at(minimum_x);

    double plain_error = 0.0;
    double climbing_error = 0.0;
    for (const bool climbing : {false, true}) {
        NEBParameters parameters;
        parameters.spring_constant = 1.0;
        parameters.climbing = climbing;
        NEB band = NEB::from_endpoints(particle(minimum_x, 0),
                                       particle(TiltedWell::stationary_point(1.0), 0), 5,
                                       parameters);
        auto optimizer = band_optimizer();
        FA_CHECK(band.run(*optimizer).converged);

        const BandProfile profile = band.profile();
        const double error = std::abs(profile.forward_barrier - exact);
        (climbing ? climbing_error : plain_error) = error;
        FA_CHECK(profile.climbing == climbing);

        if (climbing) {
            // The climbing image sits on the saddle itself, so the barrier is
            // exact rather than the best of a handful of samples.
            FA_CHECK_CLOSE(band.image(profile.highest).position(0).x, saddle_x, 1e-6);
            FA_CHECK(error < 1e-9);
        }
    }
    std::printf("    plain band %.2e, climbing %.2e (exact barrier %.9f)\n", plain_error,
                climbing_error, exact);
    // The plain band samples the barrier; it does not sit on it.
    FA_CHECK(plain_error > 1e-3);
    FA_CHECK(climbing_error < plain_error);
}

FA_TEST(climbing_can_wait_until_the_band_has_settled) {
    NEBParameters parameters;
    parameters.spring_constant = 1.0;
    parameters.climbing = true;
    parameters.climb_threshold = 0.1;
    NEB band = NEB::from_endpoints(particle(-1, 0), particle(1, 0), 5, parameters);
    // Far from the path, so the threshold is not met at the start.
    for (std::size_t k = 1; k + 1 < band.size(); ++k) {
        band.image(k).set_position(0, band.image(k).position(0) + Vec3{0.0, 1.5, 0.0});
    }
    (void)band.forces();
    FA_CHECK(!band.is_climbing());

    auto optimizer = band_optimizer();
    FA_CHECK(band.run(*optimizer).converged);
    FA_CHECK(band.is_climbing());
}

FA_TEST(a_line_search_is_refused_rather_than_tolerated) {
    // The band force is not the gradient of anything, so there is nothing for
    // a line search to decrease. Saying so beats converging badly.
    NEB band = NEB::from_endpoints(particle(-1, 0), particle(1, 0), 3);
    optimize::LBFGS searching(Value(Value::Map{{"line_search", Value(std::string("wolfe"))}}));
    FA_CHECK_THROWS(band.run(searching), NEBError);

    // The same optimizer with the search off is accepted.
    optimize::LBFGS plain(Value(Value::Map{{"line_search", Value(std::string("none"))},
                                           {"max_steps", Value(std::int64_t{50})}}));
    (void)band.run(plain);
}

FA_TEST(any_optimizer_without_a_line_search_can_drive_a_band) {
    const double saddle_x = TiltedWell::stationary_point(0.0);
    const double minimum_x = TiltedWell::stationary_point(-1.0);
    const double exact = TiltedWell::energy_at(saddle_x) - TiltedWell::energy_at(minimum_x);

    for (const char* algorithm : {"fire", "lbfgs", "sd"}) {
        NEBParameters parameters;
        parameters.spring_constant = 1.0;
        parameters.climbing = true;
        NEB band = NEB::from_endpoints(particle(minimum_x, 0),
                                       particle(TiltedWell::stationary_point(1.0), 0), 5,
                                       parameters);
        auto optimizer = optimize::make_optimizer(
            algorithm, Value(Value::Map{{"fmax", Value(1e-6)},
                                        {"max_steps", Value(std::int64_t{20000})},
                                        {"line_search", Value(std::string("none"))}}));
        const optimize::OptimizationResult result = band.run(*optimizer);
        const BandProfile profile = band.profile();
        std::printf("    %-6s converged %d in %5d steps, barrier error %.2e\n", algorithm,
                    static_cast<int>(result.converged), result.steps,
                    std::abs(profile.forward_barrier - exact));
        FA_CHECK(result.converged);
        FA_CHECK_CLOSE(profile.forward_barrier, exact, 1e-5);
    }
}

// ---------------------------------------------------------------------------
// AutoNEB
// ---------------------------------------------------------------------------
FA_TEST(autoneb_grows_the_band_and_finds_the_same_saddle) {
    const double saddle_x = TiltedWell::stationary_point(0.0);
    const double minimum_x = TiltedWell::stationary_point(-1.0);
    const double exact = TiltedWell::energy_at(saddle_x) - TiltedWell::energy_at(minimum_x);

    AutoNEBParameters parameters;
    parameters.initial_images = 3;
    parameters.max_images = 9;
    parameters.fmax = 1e-8;
    parameters.coarse_fmax = 1e-2;
    parameters.max_steps = 5000;
    NEBParameters band_parameters;
    band_parameters.spring_constant = 1.0;

    AutoNEB automatic(particle(minimum_x, 0), particle(TiltedWell::stationary_point(1.0), 0),
                      parameters, band_parameters);
    const AutoNEBResult result = automatic.run("fire");

    std::printf("    %s\n", result.to_string().c_str());
    FA_CHECK(result.converged);
    FA_CHECK(automatic.band().size() == 9);
    // It grew one image at a time, relaxing between each.
    FA_CHECK(result.history.front() == 3);
    FA_CHECK(result.history.back() == 9);
    FA_CHECK(result.stages == static_cast<int>(result.history.size()));

    FA_CHECK_CLOSE(result.profile.forward_barrier, exact, 1e-8);
    FA_CHECK_CLOSE(automatic.band().image(result.profile.highest).position(0).x, saddle_x, 1e-6);
}

FA_TEST(autoneb_puts_new_images_where_the_band_is_least_resolved) {
    AutoNEBParameters parameters;
    parameters.initial_images = 5;
    parameters.max_images = 6;
    NEBParameters band_parameters;
    AutoNEB automatic(particle(-1, 0), particle(1, 0), parameters, band_parameters);

    // Bunch four images at one end so the widest gap is unambiguous.
    for (std::size_t k = 1; k + 1 < automatic.band().size(); ++k) {
        automatic.band().image(k).set_position(0, Vec3{-0.95 + 0.02 * static_cast<double>(k),
                                                       0.0, 0.0});
    }
    const std::size_t before = automatic.band().size();
    const std::size_t where = automatic.insert_image();
    FA_CHECK(automatic.band().size() == before + 1);
    // The gap between the last bunched image and the far endpoint.
    FA_CHECK(where == before - 1);

    // At max_images nothing more is added.
    FA_CHECK(automatic.insert_image() == automatic.band().size());
}

FA_TEST(autoneb_can_remove_and_respace_images) {
    AutoNEBParameters parameters;
    parameters.initial_images = 7;
    parameters.max_images = 7;
    parameters.prune_tolerance = 1.0;   // anything is redundant enough
    NEBParameters band_parameters;
    AutoNEB automatic(particle(-1, 0), particle(1, 0), parameters, band_parameters);

    const std::size_t before = automatic.band().size();
    const std::size_t removed = automatic.prune_image();
    FA_CHECK(removed < before);
    FA_CHECK(automatic.band().size() == before - 1);
    // Never the endpoints.
    FA_CHECK(removed > 0);

    // Respacing leaves the endpoints alone and makes the spacing even.
    for (std::size_t k = 1; k + 1 < automatic.band().size(); ++k) {
        automatic.band().image(k).set_position(0, Vec3{-0.95, 0.0, 0.0});
    }
    automatic.redistribute();
    FA_CHECK_CLOSE(automatic.band().image(0).position(0).x, -1.0, 1e-12);
    FA_CHECK_CLOSE(automatic.band().image(automatic.band().size() - 1).position(0).x, 1.0,
                   1e-12);
    const BandProfile profile = automatic.band().profile();
    const std::size_t last = profile.distances.size() - 1;
    const double spacing = profile.distances[last] / static_cast<double>(last);
    for (std::size_t k = 1; k <= last; ++k) {
        FA_CHECK_CLOSE(profile.distances[k] - profile.distances[k - 1], spacing, 1e-6);
    }
}

FA_TEST(autoneb_rejects_a_configuration_that_cannot_grow) {
    AutoNEBParameters parameters;
    parameters.initial_images = 2;
    FA_CHECK_THROWS(AutoNEB(particle(-1, 0), particle(1, 0), parameters), NEBError);

    parameters.initial_images = 7;
    parameters.max_images = 5;
    FA_CHECK_THROWS(AutoNEB(particle(-1, 0), particle(1, 0), parameters), NEBError);
}

FA_TEST_MAIN()

// ---------------------------------------------------------------------------
// IDPP
// ---------------------------------------------------------------------------
namespace {

/// Two atoms swapping places past a third. A straight line drives the pair
/// through each other; the closest approach at the midpoint is exactly zero.
std::pair<Atoms, Atoms> exchange() {
    const std::vector<std::string> symbols{"Ar", "Ar", "Ar"};
    Atoms initial(symbols, {0.0, 0.0, 0.0, 3.8, 0.0, 0.0, 1.9, 3.6, 0.0});
    Atoms final(symbols, {3.8, 0.0, 0.0, 0.0, 0.0, 0.0, 1.9, 3.6, 0.0});
    for (Atoms* atoms : {&initial, &final}) {
        atoms->set_cell(Mat3::diagonal(20.0, 20.0, 20.0));
        atoms->set_pbc({false, false, false});
    }
    return {initial, final};
}

double worst_approach(const std::vector<Atoms>& band) {
    double worst = std::numeric_limits<double>::infinity();
    for (const Atoms& image : band) {
        worst = std::min(worst, closest_approach(image, false));
    }
    return worst;
}

}  // namespace

FA_TEST(a_straight_line_puts_exchanging_atoms_on_top_of_each_other) {
    // The premise of IDPP, asserted rather than assumed: if this ever stops
    // being true the test below stops testing anything.
    const auto [initial, final] = exchange();
    const std::vector<Atoms> band = NEB::interpolate(initial, final, 5, false);
    FA_CHECK_CLOSE(worst_approach(band), 0.0, 1e-12);
}

FA_TEST(idpp_takes_the_atoms_around_each_other_instead) {
    const auto [initial, final] = exchange();
    const std::vector<Atoms> band = idpp_interpolate(initial, final, 7);

    // From nothing at all to very nearly the separation the endpoints have.
    FA_CHECK(worst_approach(band) > 3.0);
    FA_CHECK(band.size() == 7);
}

FA_TEST(idpp_leaves_the_endpoints_exactly_where_they_were) {
    const auto [initial, final] = exchange();
    const std::vector<Atoms> band = idpp_interpolate(initial, final, 7);

    for (std::size_t a = 0; a < initial.size(); ++a) {
        FA_CHECK_CLOSE((band.front().position(a) - initial.position(a)).norm(), 0.0, 1e-14);
        FA_CHECK_CLOSE((band.back().position(a) - final.position(a)).norm(), 0.0, 1e-14);
    }
}

FA_TEST(idpp_is_reproducible) {
    // The nudge that breaks the symmetric saddle is deterministic, so two runs
    // of the same input must agree bit for bit. A random one would not.
    const auto [initial, final] = exchange();
    const std::vector<Atoms> first = idpp_interpolate(initial, final, 7);
    const std::vector<Atoms> second = idpp_interpolate(initial, final, 7);

    for (std::size_t k = 0; k < first.size(); ++k) {
        for (std::size_t a = 0; a < first[k].size(); ++a) {
            FA_CHECK_CLOSE((first[k].position(a) - second[k].position(a)).norm(), 0.0, 1e-15);
        }
    }
}

FA_TEST(the_idpp_objective_matches_its_own_gradient) {
    // Central differences against the analytic force, which is the only way to
    // know the weighting and the chain rule are both right.
    const auto [initial, final] = exchange();
    std::vector<Atoms> band = NEB::interpolate(initial, final, 3, false);
    Atoms image = band[2];

    std::vector<double> targets(image.size() * image.size(), 0.0);
    for (std::size_t i = 0; i < image.size(); ++i) {
        for (std::size_t j = 0; j < image.size(); ++j) {
            targets[i * image.size() + j] = i == j ? 0.0 : 2.5 + 0.3 * static_cast<double>(i + j);
        }
    }
    auto calculator = std::make_shared<IDPPCalculator>(targets, image.size(), false);
    image.set_calculator(calculator);
    image.set_position(0, Vec3{0.31, -0.22, 0.14});   // off any symmetry

    const std::vector<double> analytic = image.get_forces();
    const double h = 1e-6;
    for (std::size_t c = 0; c < 3 * image.size(); ++c) {
        std::vector<double> positions = image.positions();
        positions[c] += h;
        image.set_positions(positions);
        const double plus = image.get_potential_energy();
        positions[c] -= 2.0 * h;
        image.set_positions(positions);
        const double minus = image.get_potential_energy();
        positions[c] += h;
        image.set_positions(positions);

        FA_CHECK_CLOSE(analytic[c], -(plus - minus) / (2.0 * h), 1e-7);
    }
}

FA_TEST(idpp_rejects_a_band_too_short_to_be_one) {
    const auto [initial, final] = exchange();
    FA_CHECK_THROWS(idpp_interpolate(initial, final, 2), NEBError);
}

// ---------------------------------------------------------------------------
// The input deck
// ---------------------------------------------------------------------------
namespace {

const char* const kDeck = R"(
structure:
  cell: [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]
  atoms:
    - [Ar, 0.0, 0.0, 0.0]
    - [Ar, 3.8, 0.0, 0.0]

simulation:
  task: neb
  calculator:
    name: lennard_jones
    parameters: {epsilon: 0.0103, sigma: 3.40, cutoff: 9.0}
  neb:
    final:
      cell: [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]
      atoms:
        - [Ar, 0.0, 0.0, 0.0]
        - [Ar, 7.6, 0.0, 0.0]
    images: 5
    spring_constant: 0.5
    climb: true
    tangent: bisect
    fix: [0]
  optimizer: {algorithm: fire, fmax: 0.01, max_steps: 100}
)";

}  // namespace

FA_TEST(the_deck_parses_a_neb_task) {
    const io::InputDeck deck = io::read_yaml_string(kDeck);
    FA_CHECK(deck.simulation.task == Task::NEB);
    FA_CHECK(deck.simulation.task_name == "neb");

    const BandParameters& spec = deck.simulation.neb;
    FA_CHECK(spec.images == 5);
    FA_CHECK_CLOSE(spec.spring_constant, 0.5, 1e-15);
    FA_CHECK(spec.climbing);
    FA_CHECK(spec.tangent == "bisect");
    FA_CHECK(spec.fixed_atoms.size() == 1 && spec.fixed_atoms[0] == 0);
    // The endpoint was written inline, so no path was recorded for it.
    FA_CHECK(spec.final.empty());
    FA_CHECK(!spec.final_structure.is_null());
}

FA_TEST(the_deck_builds_the_band_it_describes) {
    const io::InputDeck deck = io::read_yaml_string(kDeck);
    Atoms atoms = deck.atoms;
    atoms.set_calculator(make_calculator(deck.simulation.calculator));

    NEB band = make_band(atoms, deck.simulation);
    FA_CHECK(band.size() == 5);
    FA_CHECK(band.natoms() == 2);
    FA_CHECK(band.parameters().tangent == Tangent::Bisect);
    FA_CHECK(band.parameters().climbing);
    FA_CHECK(band.parameters().fixed_atoms.size() == 1);

    // The first endpoint came from `structure:`, the last from the inline one.
    FA_CHECK_CLOSE(band.image(0).position(1).x, 3.8, 1e-12);
    FA_CHECK_CLOSE(band.image(4).position(1).x, 7.6, 1e-12);
    // And the calculator reached every image.
    FA_CHECK(band.image(2).has_calculator());
}

FA_TEST(fixed_atoms_accept_ranges_as_well_as_indices) {
    const std::string deck = R"(
structure: {cell: 20.0, atoms: [[Ar, 0.0, 0.0, 0.0]]}
simulation:
  task: neb
  neb: {final: other.extxyz, fix: ["0-3", 7, "10-11"]}
)";
    const io::InputDeck parsed = io::read_yaml_string(deck);
    const std::vector<std::int64_t> expected{0, 1, 2, 3, 7, 10, 11};
    FA_CHECK(parsed.simulation.neb.fixed_atoms == expected);
}

FA_TEST(the_deck_rejects_a_band_it_cannot_build) {
    // Every one of these needs a structure, or it would be rejected for that
    // instead and the test would prove nothing.
    const std::string stub = "structure: {cell: 20.0, atoms: [[Ar, 0.0, 0.0, 0.0]]}\n";
    FA_CHECK(io::read_yaml_string(stub + "simulation: {task: neb, neb: {final: f.extxyz}}")
                 .simulation.task == Task::NEB);

    // No final endpoint and no band file: there is nothing to interpolate to.
    FA_CHECK_THROWS(io::read_yaml_string(stub + "simulation: {task: neb, neb: {images: 5}}"),
                    InputError);

    // Two images are the endpoints, and a band with nothing in between is not
    // a band.
    FA_CHECK_THROWS(
        io::read_yaml_string(stub + "simulation: {task: neb, neb: {final: f.extxyz, images: 2}}"),
        InputError);

    // Growing to fewer images than it starts with.
    FA_CHECK_THROWS(io::read_yaml_string(
                        stub +
                        "simulation: {task: neb, neb: {final: f.extxyz, images: 9, max_images: 5}}"),
                    InputError);

    // A range that runs backwards.
    FA_CHECK_THROWS(io::read_yaml_string(
                        stub + "simulation: {task: neb, neb: {final: f.extxyz, fix: [\"7-2\"]}}"),
                    InputError);
}

FA_TEST(autoneb_settings_come_from_the_deck_and_the_optimizer_section) {
    const io::InputDeck deck = io::read_yaml_string(R"(
structure: {cell: 20.0, atoms: [[Ar, 0.0, 0.0, 0.0]]}
simulation:
  task: neb
  neb:
    final: f.extxyz
    auto: true
    images: 5
    max_images: 11
    coarse_fmax: 0.3
    insert: distance
    prune: true
    redistribute: true
    climb: true
  optimizer: {algorithm: fire, fmax: 0.001, max_steps: 321}
)");
    FA_CHECK(wants_autoneb(deck.simulation));

    const AutoNEBParameters parameters =
        autoneb_parameters(deck.simulation.neb, deck.simulation.optimizer);
    FA_CHECK(parameters.initial_images == 5);
    FA_CHECK(parameters.max_images == 11);
    FA_CHECK(parameters.insert == Insertion::Distance);
    FA_CHECK(parameters.prune && parameters.redistribute && parameters.climb);
    FA_CHECK_CLOSE(parameters.coarse_fmax, 0.3, 1e-15);
    // fmax and the step ceiling belong to the optimizer, and are taken there.
    FA_CHECK_CLOSE(parameters.fmax, 0.001, 1e-15);
    FA_CHECK(parameters.max_steps == 321);
}
