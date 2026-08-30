// SPDX-License-Identifier: MIT
#include "fastatoms/md/dynamics.hpp"

#include <cmath>
#include <sstream>

#include "fastatoms/calculators/calculator.hpp"
#include "fastatoms/core/units.hpp"
#include "fastatoms/md/velocity.hpp"

namespace fastatoms::md {
namespace {

/// Read a sub-setting that may be given either as a bare name or as a mapping
/// with a `name` key, so `thermostat: nose_hoover` and
/// `thermostat: {name: nose_hoover, tau: 50}` both work.
std::pair<std::string, Value> split_named_block(const Value& node, const Value& fallback) {
    if (node.is_string()) {
        return {node.as_string(), fallback};
    }
    if (node.is_map()) {
        std::string name;
        if (const Value* found = node.find("name"); found != nullptr && found->is_string()) {
            name = found->as_string();
        }
        // The block's own keys win, with the surrounding md settings behind them.
        Value merged = fallback.is_map() ? fallback : Value(Value::Map{});
        for (const auto& [key, value] : node.as_map()) {
            if (key != "name") {
                merged.insert(key, value);
            }
        }
        return {name, merged};
    }
    return {std::string{}, fallback};
}

bool is_finite(const std::vector<double>& values) {
    for (const double value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

Dynamics::Dynamics(Value parameters) : parameters_(std::move(parameters)) {
    if (parameters_.is_null()) {
        parameters_ = Value(Value::Map{});
    }
    if (!parameters_.is_map()) {
        throw DynamicsError("dynamics parameters must be a mapping, got " +
                            std::string(parameters_.type_name()));
    }
    // Resolved now rather than on the first run, so that `ensemble()` and the
    // repr describe what will actually happen, and an unknown name is caught
    // at once.
    build_pieces();
}

const Value& Dynamics::parameter(std::string_view key) const {
    if (const Value* found = parameters_.find(key)) {
        return *found;
    }
    return Value::null();
}

void Dynamics::fail(const std::string& message) const {
    throw DynamicsError("dynamics: " + message);
}

void Dynamics::set_parameters(Value parameters) {
    if (parameters.is_null()) {
        parameters = Value(Value::Map{});
    }
    if (!parameters.is_map()) {
        throw DynamicsError("dynamics parameters must be a mapping");
    }
    parameters_ = std::move(parameters);
    prepared_ = false;
    build_pieces();
}

void Dynamics::set_parameter(const std::string& key, Value value) {
    parameters_.insert(key, std::move(value));
    prepared_ = false;
    build_pieces();
}

double Dynamics::timestep_fs() const {
    const double value = parameter("timestep").value_or(parameter("dt").value_or(1.0));
    if (!(value > 0.0)) {
        fail("'timestep' must be positive");
    }
    return value;
}

double Dynamics::timestep() const { return timestep_fs() * units::fs; }

int Dynamics::configured_steps() const {
    const auto value = static_cast<int>(parameter("steps").value_or(std::int64_t{0}));
    if (value < 0) {
        fail("'steps' must not be negative");
    }
    return value;
}

double Dynamics::temperature() const {
    const double value = parameter("temperature").value_or(0.0);
    if (value < 0.0) {
        fail("'temperature' must not be negative");
    }
    return value;
}

double Dynamics::pressure() const {
    // Both spellings mean GPa; the internal unit is eV/angstrom^3.
    double gpa = parameter("pressure_GPa").value_or(0.0);
    if (gpa == 0.0) {
        gpa = parameter("pressure").value_or(0.0);
    }
    return gpa * units::GPa;
}

void Dynamics::set_integrator(std::shared_ptr<Integrator> integrator) {
    integrator_ = std::move(integrator);
    integrator_is_explicit_ = true;
    prepared_ = false;
}

void Dynamics::set_thermostat(std::shared_ptr<Thermostat> thermostat) {
    thermostat_ = std::move(thermostat);
    thermostat_is_explicit_ = true;
    prepared_ = false;
}

void Dynamics::set_barostat(std::shared_ptr<Barostat> barostat) {
    barostat_ = std::move(barostat);
    barostat_is_explicit_ = true;
    prepared_ = false;
}

void Dynamics::set_constraints(std::shared_ptr<Constraints> constraints) {
    constraints_ = std::move(constraints);
    constraints_are_explicit_ = true;
    prepared_ = false;
}

std::string Dynamics::ensemble() const {
    if (barostat_) {
        return "npt";
    }
    return thermostat_ ? "nvt" : "nve";
}

void Dynamics::attach(DynamicsCallback callback, int interval) {
    if (!callback) {
        return;
    }
    if (interval < 1) {
        fail("an observer interval must be at least 1");
    }
    observers_.push_back(Observer{std::move(callback), interval});
}

void Dynamics::clear_callbacks() { observers_.clear(); }

std::string Dynamics::to_string() const {
    std::ostringstream out;
    out.precision(4);
    out << "Dynamics(" << ensemble() << ", " << (integrator_ ? integrator_->name() : "unset")
        << ", dt=" << timestep_fs() << " fs";
    if (thermostat_) {
        out << ", " << thermostat_->to_string() << " at " << temperature() << " K";
    }
    if (barostat_) {
        out << ", " << barostat_->to_string();
    }
    out << ")";
    return out.str();
}

// ---------------------------------------------------------------------------
// Assembling the pieces
// ---------------------------------------------------------------------------

void Dynamics::build_pieces() {
    if (!integrator_is_explicit_) {
        integrator_ = make_integrator(parameter("integrator").value_or(std::string_view{}));
    }

    // `ensemble` is a shorthand: it only fills in what has not been named.
    const std::string ensemble_name = parameter("ensemble").value_or(std::string_view{});

    if (!thermostat_is_explicit_) {
        const auto [name, settings] = split_named_block(parameter("thermostat"), parameters_);
        std::string chosen = name;
        if (chosen.empty() && (ensemble_name == "nvt" || ensemble_name == "npt")) {
            chosen = "nose_hoover";
        }
        thermostat_ = make_thermostat(chosen, settings);
    }

    if (!barostat_is_explicit_) {
        const auto [name, settings] = split_named_block(parameter("barostat"), parameters_);
        std::string chosen = name;
        if (chosen.empty() && ensemble_name == "npt") {
            chosen = "parrinello_rahman";
        }
        barostat_ = make_barostat(chosen, settings);
    }
}

void Dynamics::prepare(Atoms& atoms) {
    if (atoms.empty()) {
        fail("cannot run dynamics on an empty structure");
    }
    if (!atoms.has_calculator()) {
        fail("the structure has no calculator; attach one before running dynamics "
             "(Python: atoms.calc = ...)");
    }
    if (!integrator_) {
        build_pieces();
    }

    // Rigid geometry. Unless it was set by hand, take whatever the calculator
    // declares: a water model knows its own bond lengths, and having the user
    // restate them would only be a chance to get them wrong.
    if (!constraints_are_explicit_) {
        constraints_.reset();
        const Value& wanted = parameter("constraints");
        const bool enabled = wanted.is_bool() ? wanted.as_bool() : true;
        if (enabled) {
            // The constraints follow from the molecules the calculator found,
            // so it needs to have seen the structure first. The result is
            // cached, so the first step does not pay for this again.
            (void)atoms.get_potential_energy();
            if (const std::vector<DistanceConstraint>* declared =
                    atoms.calculator()->rigid_constraints();
                declared != nullptr && !declared->empty()) {
                auto built = std::make_shared<Constraints>(*declared);
                if (const Value& tolerance = parameter("constraint_tolerance");
                    tolerance.is_number()) {
                    built->set_tolerance(tolerance.as_double());
                }
                constraints_ = std::move(built);
            }
        }
    }

    const bool fix_com = parameter("fix_center_of_mass").value_or(true);
    context_.timestep = timestep();
    context_.degrees_of_freedom = momentum_degrees_of_freedom(atoms, fix_com);
    context_.constraints = constraints_.get();

    if (constraints_ && !constraints_->empty()) {
        // Each constraint removes one degree of freedom. A rigid water box has
        // 6 per molecule rather than 9, and a temperature computed with 9
        // reads a third too low.
        context_.degrees_of_freedom -= constraints_->removed_degrees_of_freedom();
        if (context_.degrees_of_freedom < 1) {
            fail("the constraints remove every degree of freedom this structure has");
        }
        // Start on the constraint surface, whatever the input geometry was.
        constraints_->enforce(atoms);
    }

    // The other kind: the projections attached to the structure itself. Three
    // degrees of freedom go per frozen atom, two per atom held to a line, one
    // per atom held to a plane - counted from the projection rather than
    // summed, so an atom named by two constraints is not charged twice.
    if (atoms.has_constraints()) {
        context_.degrees_of_freedom -= atoms.constrained_degrees_of_freedom();
        if (context_.degrees_of_freedom < 1) {
            fail("the constraints remove every degree of freedom this structure has");
        }
    }
    context_.temperature = temperature();
    context_.pressure = pressure();
    context_.fix_center_of_mass = fix_com;
    context_.step = 0;
    time_fs_ = 0.0;

    if (thermostat_ && !(context_.temperature > 0.0)) {
        fail("a '" + thermostat_->name() +
             "' thermostat needs a target temperature; set 'temperature'");
    }
    if (barostat_) {
        barostat_->begin(atoms, context_);
    }

    // Velocities: draw them when asked, or when the structure has none and a
    // temperature was given - starting an NVT run from rest is never what
    // anyone means.
    const bool has_momenta = atoms.has_momenta();
    const bool initialize =
        parameter("initialize_velocities").value_or(!has_momenta && context_.temperature > 0.0);
    if (initialize) {
        const auto seed = parameter("seed").value_or(std::int64_t{0});
        set_maxwell_boltzmann_velocities(atoms, context_.temperature, seed, fix_com, true);
    } else if (fix_com && has_momenta) {
        atoms.zero_center_of_mass_momentum();
    }

    if (constraints_ && !constraints_->empty() && atoms.has_momenta()) {
        // The drawn velocities have components along the constrained bonds;
        // projecting them out now means the first reported temperature counts
        // the same degrees of freedom as every later one.
        std::vector<double> velocities = atoms.velocities();
        constraints_->rattle(atoms, velocities);
        atoms.set_velocities(velocities);
        if (fix_com) {
            atoms.zero_center_of_mass_momentum();
        }
    }

    integrator_->begin(atoms, context_);
    if (thermostat_) {
        thermostat_->begin(atoms, context_);
    }

    // The pressure is only meaningful for a periodic cell, and only available
    // if the calculator can produce a stress.
    track_pressure_ = atoms.cell().is_invertible() && atoms.cell().any_pbc() &&
                      atoms.calculator()->implements(Property::Stress);
    if (barostat_ && !track_pressure_) {
        fail("a barostat needs a periodic cell and a calculator that reports stress");
    }

    // Writers are rebuilt for each run so a second run does not append to the
    // first one's file unless asked.
    trajectory_.reset();
    log_.reset();

    if (const std::string path = parameter("trajectory").value_or(std::string_view{});
        !path.empty()) {
        TrajectoryOptions options;
        options.interval =
            static_cast<int>(parameter("trajectory_interval")
                                 .value_or(parameter("interval").value_or(std::int64_t{1})));
        options.velocities = parameter("trajectory_velocities").value_or(true);
        options.forces = parameter("trajectory_forces").value_or(true);
        options.append = parameter("trajectory_append").value_or(false);
        trajectory_ = std::make_unique<TrajectoryWriter>(path, options);
    }

    if (const Value& log = parameter("log"); !log.is_null()) {
        // `true` means standard output, a path means that file, and `false`
        // or an empty string means no log at all.
        std::string path;
        bool wanted = false;
        if (log.is_bool()) {
            wanted = log.as_bool();
        } else {
            path = log.as_string();
            wanted = !path.empty();
        }
        if (wanted) {
            const auto interval =
                static_cast<int>(parameter("log_interval")
                                     .value_or(parameter("interval").value_or(std::int64_t{1})));
            log_ = std::make_unique<LogWriter>(path, interval);
        }
    }

    prepared_ = true;
}

// ---------------------------------------------------------------------------
// One step
// ---------------------------------------------------------------------------

void Dynamics::apply_thermostat(Atoms& atoms, double dt) {
    const VelocityUpdate update = thermostat_->apply(atoms, dt, context_);
    // A scheme with its own copy of the velocities has to see the change -
    // including the exact noise a stochastic thermostat drew.
    integrator_->adjust_velocities(update);
}

void Dynamics::apply_barostat(Atoms& atoms, double dt) {
    const Mat3 transform = barostat_->deform_cell(atoms, dt, context_);
    integrator_->transform_velocities(transform);
}

void Dynamics::step(Atoms& atoms) {
    const double dt = context_.timestep;
    const double half = 0.5 * dt;

    // Order matters for cost as well as accuracy. The thermostat and the
    // barostat's kicks only read the stress and rescale velocities, so they
    // land where the forces are already cached; the cell is deformed once, in
    // front of the integrator, so the step needs one force evaluation without
    // a barostat and two with one - never three.
    if (thermostat_) {
        apply_thermostat(atoms, half);
    }
    if (barostat_) {
        barostat_->kick(atoms, half, context_);
        apply_barostat(atoms, dt);
    }

    integrator_->step(atoms, context_);

    if (barostat_) {
        barostat_->kick(atoms, half, context_);
    }
    if (thermostat_) {
        apply_thermostat(atoms, half);
    }

    ++context_.step;
}

// ---------------------------------------------------------------------------
// Sampling and running
// ---------------------------------------------------------------------------

DynamicsStep Dynamics::sample(Atoms& atoms) {
    DynamicsStep record;
    record.step = context_.step;
    record.time = time_fs_;
    record.potential_energy = atoms.get_potential_energy();
    record.kinetic_energy = atoms.kinetic_energy();
    record.total_energy = record.potential_energy + record.kinetic_energy;
    record.temperature = atoms.temperature(context_.degrees_of_freedom);
    record.volume = atoms.volume();
    record.pressure = track_pressure_ ? instantaneous_pressure(atoms) : 0.0;

    // The quantity that should stay flat: the physical energy plus whatever
    // the extended-system pieces are holding.
    record.conserved_energy = record.total_energy;
    if (thermostat_) {
        record.conserved_energy += thermostat_->conserved_energy(context_);
    }
    if (barostat_) {
        record.conserved_energy += barostat_->conserved_energy(atoms, context_);
    }
    return record;
}

DynamicsResult Dynamics::run(Atoms& atoms, int steps) {
    const int requested = steps >= 0 ? steps : configured_steps();

    // Counted before `prepare`, because some schemes evaluate forces while
    // setting up - leapfrog needs them to step its velocities half a step
    // back - and that work should show up in the total like any other.
    if (!atoms.has_calculator()) {
        fail("the structure has no calculator; attach one before running dynamics "
             "(Python: atoms.calc = ...)");
    }
    const std::size_t calls_before = atoms.calculator()->call_count();

    prepare(atoms);

    const auto interval =
        static_cast<int>(parameter("interval").value_or(std::int64_t{1}));
    if (interval < 1) {
        fail("'interval' must be at least 1");
    }

    DynamicsResult result;
    const auto record = [&](bool force) {
        const bool wanted = force || context_.step % interval == 0;
        if (!wanted && !trajectory_ && !log_) {
            return true;
        }
        const DynamicsStep frame = sample(atoms);
        if (wanted) {
            result.history.push_back(frame);
        }
        if (trajectory_ && (force || trajectory_->wants(context_.step))) {
            trajectory_->write(atoms, frame);
        }
        if (log_ && (force || log_->wants(context_.step))) {
            log_->write(frame);
        }
        for (const Observer& observer : observers_) {
            if (context_.step % observer.interval == 0 || force) {
                if (!observer.callback(atoms, frame)) {
                    return false;
                }
            }
        }
        return true;
    };

    const auto finish = [&](DynamicsStopReason reason, std::string message) {
        result.reason = reason;
        result.time = time_fs_;
        result.evaluations =
            static_cast<int>(atoms.calculator()->call_count() - calls_before);
        result.message = std::move(message);
        return result;
    };

    if (!record(true)) {
        return finish(DynamicsStopReason::Interrupted, "an observer stopped the run at step 0");
    }

    for (int k = 1; k <= requested; ++k) {
        step(atoms);
        time_fs_ += timestep_fs();
        ++result.steps;

        // A too-large timestep shows up as an atom flying to infinity; stop
        // and say so rather than filling the trajectory with NaNs. Velocities
        // usually overflow first, so check both.
        if (!is_finite(atoms.positions()) || !is_finite(atoms.momenta())) {
            record(true);
            return finish(DynamicsStopReason::Unstable,
                          "the trajectory stopped being finite at step " + std::to_string(k) +
                              "; the timestep is almost certainly too large");
        }

        const bool last = k == requested;
        if (!record(last)) {
            return finish(DynamicsStopReason::Interrupted,
                          "an observer stopped the run at step " + std::to_string(k));
        }
    }

    return finish(DynamicsStopReason::Completed,
                  "completed " + std::to_string(result.steps) + " steps (" +
                      std::to_string(time_fs_) + " fs)");
}

}  // namespace fastatoms::md
