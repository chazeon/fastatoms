// SPDX-License-Identifier: MIT
///
/// The motion constraints: the projections themselves, and - the part that
/// actually matters - that every consumer of a force in the codebase honours
/// them. A constraint that only the optimizer respects is worse than none,
/// because the run looks right.

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "fastatoms/fastatoms.hpp"
#include "testing.hpp"

using namespace fastatoms;

namespace {

/// Four argon atoms, none of them at a symmetric point, so every component of
/// every force is non-zero and a projection that does nothing is visible.
Atoms cluster() {
    const std::vector<std::string> symbols{"Ar", "Ar", "Ar", "Ar"};
    Atoms atoms(symbols, {0.0, 0.0, 0.0, 3.6, 0.0, 0.0, 0.0, 3.6, 0.0, 1.8, 1.8, 2.6});
    atoms.set_cell(Mat3::diagonal(20.0, 20.0, 20.0));
    atoms.set_pbc({false, false, false});
    atoms.set_calculator(std::make_shared<LennardJonesCalculator>(Value(Value::Map{
        {"epsilon", Value(0.0103)}, {"sigma", Value(3.40)}, {"cutoff", Value(9.0)}})));
    return atoms;
}

Vec3 row(const std::vector<double>& rows, std::size_t i) {
    return Vec3{rows[3 * i], rows[3 * i + 1], rows[3 * i + 2]};
}

}  // namespace

// ---------------------------------------------------------------------------
// The projections
// ---------------------------------------------------------------------------

FA_TEST(fix_atoms_zeroes_the_rows_it_names_and_no_others) {
    Atoms atoms = cluster();
    const std::vector<double> raw = atoms.get_forces();

    atoms.set_constraints({std::make_shared<FixAtoms>(std::vector<std::size_t>{0, 2})});
    const std::vector<double> held = atoms.get_forces();

    FA_CHECK_CLOSE(row(held, 0).norm(), 0.0, 1e-15);
    FA_CHECK_CLOSE(row(held, 2).norm(), 0.0, 1e-15);
    FA_CHECK_CLOSE((row(held, 1) - row(raw, 1)).norm(), 0.0, 1e-15);
    FA_CHECK_CLOSE((row(held, 3) - row(raw, 3)).norm(), 0.0, 1e-15);
    // The raw force on the held atoms was not zero to begin with, or this
    // would prove nothing.
    FA_CHECK(row(raw, 0).norm() > 1e-6);
}

FA_TEST(fix_line_keeps_only_the_component_along_its_direction) {
    Atoms atoms = cluster();
    const std::vector<double> raw = atoms.get_forces();

    const Vec3 axis{1.0, 1.0, 0.0};
    atoms.set_constraints({std::make_shared<FixLine>(std::vector<std::size_t>{3}, axis)});
    const std::vector<double> held = atoms.get_forces();

    const Vec3 unit{axis.x / axis.norm(), axis.y / axis.norm(), axis.z / axis.norm()};
    const Vec3 f = row(held, 3);
    const double along = f.x * unit.x + f.y * unit.y + f.z * unit.z;
    // What is left is exactly its own projection: nothing perpendicular.
    FA_CHECK_CLOSE((f - Vec3{along * unit.x, along * unit.y, along * unit.z}).norm(), 0.0, 1e-15);
    // And the component along the line is untouched.
    const Vec3 before = row(raw, 3);
    FA_CHECK_CLOSE(along, before.x * unit.x + before.y * unit.y + before.z * unit.z, 1e-15);
}

FA_TEST(fix_plane_removes_only_the_component_along_its_normal) {
    Atoms atoms = cluster();
    const std::vector<double> raw = atoms.get_forces();

    const Vec3 normal{0.0, 0.0, 1.0};
    atoms.set_constraints({std::make_shared<FixPlane>(std::vector<std::size_t>{3}, normal)});
    const std::vector<double> held = atoms.get_forces();

    FA_CHECK_CLOSE(held[3 * 3 + 2], 0.0, 1e-15);
    FA_CHECK_CLOSE(held[3 * 3 + 0], raw[3 * 3 + 0], 1e-15);
    FA_CHECK_CLOSE(held[3 * 3 + 1], raw[3 * 3 + 1], 1e-15);
    FA_CHECK(std::abs(raw[3 * 3 + 2]) > 1e-6);
}

FA_TEST(a_direction_needs_a_direction) {
    FA_CHECK_THROWS(FixLine(std::vector<std::size_t>{0}, Vec3{0.0, 0.0, 0.0}), ConstraintError);
    FA_CHECK_THROWS(FixPlane(std::vector<std::size_t>{0}, Vec3{0.0, 0.0, 0.0}), ConstraintError);
}

FA_TEST(a_constraint_must_fit_the_structure_it_is_attached_to) {
    Atoms atoms = cluster();
    FA_CHECK_THROWS(
        atoms.set_constraints({std::make_shared<FixAtoms>(std::vector<std::size_t>{0, 9})}),
        ConstraintError);
    // And the failed attach left nothing behind.
    FA_CHECK(!atoms.has_constraints());
}

FA_TEST(a_line_and_a_plane_pass_through_where_the_atom_already_is) {
    // The constraint is on the displacement, not the position, so an atom is
    // never teleported onto a plane through the origin when one is attached.
    Atoms atoms = cluster();
    const std::vector<double> start = atoms.positions();
    atoms.set_constraints({std::make_shared<FixPlane>(std::vector<std::size_t>{3}, Vec3{0, 0, 1})});

    std::vector<double> moved = start;
    moved[3 * 3 + 0] += 0.5;
    moved[3 * 3 + 2] += 0.5;      // forbidden
    atoms.move_to(moved);

    FA_CHECK_CLOSE(atoms.position(3).x, start[9] + 0.5, 1e-15);
    FA_CHECK_CLOSE(atoms.position(3).z, start[11], 1e-15);   // still 2.6, not 0
}

// ---------------------------------------------------------------------------
// Degrees of freedom
// ---------------------------------------------------------------------------

FA_TEST(each_constraint_removes_the_dimensions_it_should) {
    Atoms atoms = cluster();

    atoms.set_constraints({std::make_shared<FixAtoms>(std::vector<std::size_t>{0, 1})});
    FA_CHECK(atoms.constrained_degrees_of_freedom() == 6);

    atoms.set_constraints({std::make_shared<FixLine>(std::vector<std::size_t>{0}, Vec3{0, 0, 1})});
    FA_CHECK(atoms.constrained_degrees_of_freedom() == 2);

    atoms.set_constraints({std::make_shared<FixPlane>(std::vector<std::size_t>{0}, Vec3{0, 0, 1})});
    FA_CHECK(atoms.constrained_degrees_of_freedom() == 1);
}

FA_TEST(an_atom_named_twice_is_not_charged_twice) {
    Atoms atoms = cluster();

    // Three directions frozen, not four: the plane's normal is already inside
    // what FixAtoms removed. Summing what each constraint claims would say 4,
    // and the temperature computed from it would be wrong.
    atoms.set_constraints({
        std::make_shared<FixAtoms>(std::vector<std::size_t>{0}),
        std::make_shared<FixPlane>(std::vector<std::size_t>{0}, Vec3{0, 0, 1}),
    });
    FA_CHECK(atoms.constrained_degrees_of_freedom() == 3);

    // Two perpendicular planes leave a line: two removed, and here the naive
    // sum happens to agree.
    atoms.set_constraints({
        std::make_shared<FixPlane>(std::vector<std::size_t>{0}, Vec3{0, 0, 1}),
        std::make_shared<FixPlane>(std::vector<std::size_t>{0}, Vec3{1, 0, 0}),
    });
    FA_CHECK(atoms.constrained_degrees_of_freedom() == 2);

    // The same plane twice is still one direction.
    atoms.set_constraints({
        std::make_shared<FixPlane>(std::vector<std::size_t>{0}, Vec3{0, 0, 1}),
        std::make_shared<FixPlane>(std::vector<std::size_t>{0}, Vec3{0, 0, 1}),
    });
    FA_CHECK(atoms.constrained_degrees_of_freedom() == 1);

    // A line inside a plane leaves the line: two removed, not three.
    atoms.set_constraints({
        std::make_shared<FixPlane>(std::vector<std::size_t>{0}, Vec3{0, 0, 1}),
        std::make_shared<FixLine>(std::vector<std::size_t>{0}, Vec3{1, 0, 0}),
    });
    FA_CHECK(atoms.constrained_degrees_of_freedom() == 2);
}

FA_TEST(duplicate_indices_inside_one_constraint_are_collapsed) {
    Atoms atoms = cluster();
    atoms.set_constraints({std::make_shared<FixAtoms>(std::vector<std::size_t>{1, 1, 1, 0})});
    FA_CHECK(atoms.constraints().front()->indices().size() == 2);
    FA_CHECK(atoms.constrained_degrees_of_freedom() == 6);
}

// ---------------------------------------------------------------------------
// The constrained force is the gradient of the constrained energy
// ---------------------------------------------------------------------------

FA_TEST(the_constrained_force_matches_finite_differences_of_the_energy) {
    // The property that makes a constrained relaxation converge at all: what
    // `get_forces` returns has to be the true gradient of the energy over the
    // moves the constraints allow. If it is not, the optimizer is minimising
    // one function while measuring another and never agrees with itself.
    Atoms atoms = cluster();
    atoms.set_constraints({
        std::make_shared<FixAtoms>(std::vector<std::size_t>{0}),
        std::make_shared<FixPlane>(std::vector<std::size_t>{3}, Vec3{0, 0, 1}),
        std::make_shared<FixLine>(std::vector<std::size_t>{2}, Vec3{1, 0, 0}),
    });

    const std::vector<double> analytic = atoms.get_forces();
    const std::vector<double> start = atoms.positions();
    const double h = 1e-6;

    for (std::size_t c = 0; c < start.size(); ++c) {
        // Step along one Cartesian direction, let the constraints have their
        // say, and difference the energy over the distance actually travelled.
        std::vector<double> forward = start;
        forward[c] += h;
        atoms.set_positions(start);
        atoms.move_to(forward);
        const std::vector<double> moved = atoms.positions();
        const double travelled = moved[c] - start[c];
        if (std::abs(travelled) < 0.5 * h) {
            // The constraint forbade this direction; the force must be zero.
            FA_CHECK_CLOSE(analytic[c], 0.0, 1e-14);
            continue;
        }
        const double plus = atoms.get_potential_energy();

        std::vector<double> backward = start;
        backward[c] -= h;
        atoms.set_positions(start);
        atoms.move_to(backward);
        const double minus = atoms.get_potential_energy();

        atoms.set_positions(start);
        FA_CHECK_CLOSE(analytic[c], -(plus - minus) / (2.0 * travelled), 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Every consumer honours them
// ---------------------------------------------------------------------------

FA_TEST(every_optimizer_honours_them) {
    // The one that caught the real bug. The optimizers read the calculator's
    // `Results` rather than calling `get_forces`, so constraining only the
    // latter left them minimising the unconstrained gradient: the geometry
    // came out right, the reported force never fell, and the run failed to
    // converge while sitting exactly on the answer.
    for (const std::string& algorithm : {"sd", "cg", "bfgs", "lbfgs", "fire"}) {
        Atoms atoms = cluster();
        const std::vector<double> start = atoms.positions();
        atoms.set_constraints({
            std::make_shared<FixAtoms>(std::vector<std::size_t>{0, 1}),
            std::make_shared<FixPlane>(std::vector<std::size_t>{3}, Vec3{0, 0, 1}),
        });

        auto optimizer = optimize::make_optimizer(
            algorithm, Value(Value::Map{{"fmax", Value(1e-4)},
                                        {"max_steps", Value(static_cast<std::int64_t>(3000))}}));
        const optimize::OptimizationResult result = optimizer->run(atoms);

        FA_CHECK(result.converged);
        // Frozen atoms did not move at all.
        for (std::size_t c = 0; c < 6; ++c) {
            FA_CHECK_CLOSE(atoms.positions()[c], start[c], 1e-15);
        }
        // The one held to a plane kept its height exactly.
        FA_CHECK_CLOSE(atoms.position(3).z, start[11], 1e-12);
        // And something actually relaxed, or this proves nothing.
        FA_CHECK((atoms.position(2) - Vec3::load(start.data() + 6)).norm() > 1e-3);
    }
}

FA_TEST(molecular_dynamics_honours_them) {
    Atoms atoms = cluster();
    const std::vector<double> start = atoms.positions();
    atoms.set_constraints({
        std::make_shared<FixAtoms>(std::vector<std::size_t>{0, 1}),
        std::make_shared<FixLine>(std::vector<std::size_t>{3}, Vec3{0, 0, 1}),
    });
    md::set_maxwell_boltzmann_velocities(atoms, 300.0, 12345);

    // A frozen atom given a Maxwell-Boltzmann velocity would wander off, so
    // the draw has to be projected too.
    const std::vector<double> velocities = atoms.velocities();
    FA_CHECK_CLOSE(row(velocities, 0).norm(), 0.0, 1e-15);
    FA_CHECK_CLOSE(velocities[3 * 3 + 0], 0.0, 1e-15);
    FA_CHECK_CLOSE(velocities[3 * 3 + 1], 0.0, 1e-15);

    md::Dynamics dynamics(Value(Value::Map{{"integrator", Value(std::string("velocity_verlet"))},
                                           {"timestep", Value(1.0)}}));
    dynamics.run(atoms, 300);

    for (std::size_t c = 0; c < 6; ++c) {
        FA_CHECK_CLOSE(atoms.positions()[c], start[c], 1e-14);
    }
    FA_CHECK_CLOSE(atoms.position(3).x, start[9], 1e-12);
    FA_CHECK_CLOSE(atoms.position(3).y, start[10], 1e-12);
    FA_CHECK(std::abs(atoms.position(3).z - start[11]) > 1e-6);
}

FA_TEST(the_temperature_is_computed_with_the_constrained_count) {
    Atoms atoms = cluster();
    atoms.set_constraints({std::make_shared<FixAtoms>(std::vector<std::size_t>{0, 1})});
    md::set_maxwell_boltzmann_velocities(atoms, 300.0, 7);

    md::Dynamics dynamics(Value(Value::Map{{"integrator", Value(std::string("velocity_verlet"))},
                                           {"timestep", Value(1.0)}}));
    const md::DynamicsResult result = dynamics.run(atoms, 1);

    // Twelve to start, three removed with the centre of mass, six more by the
    // two frozen atoms. The count is not reported directly, so it is read back
    // out of the temperature it was used to compute.
    const md::DynamicsStep& first = result.history.front();
    const double dof = 2.0 * first.kinetic_energy / (first.temperature * units::kB);
    FA_CHECK_CLOSE(dof, 12.0 - 3.0 - 6.0, 1e-9);
}

FA_TEST(the_band_honours_them) {
    // The band adds a spring term along the tangent and reverses a component
    // on the climbing image, neither of which is obliged to respect anything,
    // so the projection has to be applied again afterwards.
    const std::vector<std::string> symbols{"Ar", "Ar"};
    Atoms initial(symbols, {-5.0, 0.0, 0.0, 0.0, 2.25, 2.25});
    Atoms final(symbols, {5.0, 0.0, 0.0, 0.0, 2.25, 2.25});
    for (Atoms* atoms : {&initial, &final}) {
        atoms->set_cell(Cell(Mat3::diagonal(20.0, 4.5, 4.5)));
        atoms->set_pbc({false, true, true});
        atoms->set_calculator(std::make_shared<LennardJonesCalculator>(Value(Value::Map{
            {"epsilon", Value(0.0103)}, {"sigma", Value(3.40)}, {"cutoff", Value(9.0)}})));
        atoms->set_constraints({std::make_shared<FixAtoms>(std::vector<std::size_t>{1})});
    }

    neb::NEBParameters parameters;
    parameters.spring_constant = 0.5;
    parameters.climbing = true;
    neb::NEB band = neb::NEB::from_endpoints(initial, final, 5, parameters);
    // Off the straight line, so the band has to relax rather than start there.
    for (std::size_t k = 1; k + 1 < band.size(); ++k) {
        band.image(k).set_position(0, band.image(k).position(0) + Vec3{0.4, 0.3, 0.2});
    }

    optimize::Fire optimizer(Value(Value::Map{
        {"fmax", Value(1e-5)}, {"max_steps", Value(static_cast<std::int64_t>(8000))}}));
    const optimize::OptimizationResult result = band.run(optimizer);
    FA_CHECK(result.converged);

    // The wall never moved, in any image, and the barrier is the known one.
    for (std::size_t k = 0; k < band.size(); ++k) {
        FA_CHECK_CLOSE((band.image(k).position(1) - Vec3{0.0, 2.25, 2.25}).norm(), 0.0, 1e-12);
    }
    FA_CHECK_CLOSE(band.profile().forward_barrier, 0.122725, 1e-4);
}

FA_TEST_MAIN()
