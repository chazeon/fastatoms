// SPDX-License-Identifier: MIT
#pragma once

/// \file dynamics.hpp
/// \brief The molecular dynamics loop.

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/io/parameters.hpp"
#include "fastatoms/md/barostat.hpp"
#include "fastatoms/md/constraints.hpp"
#include "fastatoms/md/integrator.hpp"
#include "fastatoms/md/thermostat.hpp"
#include "fastatoms/md/trajectory.hpp"
#include "fastatoms/md/types.hpp"

namespace fastatoms::md {

/// Runs molecular dynamics on a structure.
///
/// The class owns the loop; the physics of a single step is composed from
/// three pluggable pieces:
///
///   - an `Integrator`, which does plain Newtonian motion,
///   - an optional `Thermostat`, for the NVT ensemble,
///   - an optional `Barostat`, for NPT.
///
/// Each step applies them in a symmetric Trotter splitting:
///
///     thermostat(dt/2)
///     barostat kick(dt/2), deform(dt)
///     integrate(dt)
///     barostat kick(dt/2)
///     thermostat(dt/2)
///
/// which is second-order accurate and keeps the extended-system energy
/// conserved. The thermostat and the barostat's kicks only read the stress and
/// rescale velocities, so they cost nothing: an NVE or NVT step needs exactly
/// one force evaluation, and NPT needs two, because deforming the cell in
/// front of the integrator invalidates the cached forces once.
///
/// `step()` is virtual, so a subclass can replace the whole scheme while
/// keeping the loop, the sampling and the trajectory writing.
///
/// A run stops early if a coordinate or momentum stops being finite. That is a
/// backstop against arithmetic that has overflowed, not a guarantee that the
/// physics is sound: watch the drift of `conserved_energy` for that.
///
/// Parameters (all optional):
///   - `timestep`    - femtoseconds. Default 1.0.
///   - `steps`       - how many to take when `run()` is called without a count.
///   - `temperature` - target in kelvin; required by every thermostat.
///   - `pressure`    - target in GPa (`pressure_GPa` is a synonym).
///   - `integrator`  - "velocity_verlet" (default), "leapfrog" or "euler".
///   - `thermostat`  - "nose_hoover", "nose", "berendsen", or absent for NVE.
///                     May also be a mapping, `{name: nose_hoover, tau: 50}`.
///   - `barostat`    - "parrinello_rahman", "berendsen", or absent.
///   - `ensemble`    - "nve", "nvt" or "npt". A shorthand that fills in a
///                     default thermostat and barostat when none is named.
///   - `tau`, `friction` - thermostat coupling, in fs and 1/fs respectively.
///   - `initialize_velocities`, `seed` - draw Maxwell-Boltzmann velocities.
///   - `fix_center_of_mass` - remove the net momentum at the start and count
///                     `3N - 3` degrees of freedom. Default true.
///   - `trajectory`, `trajectory_interval` - extended XYZ output.
///   - `log`, `log_interval` - a table of the per-step scalars. `true` writes
///                     to standard output, a path writes to that file, and
///                     `false` or an empty string turns it off.
///   - `interval`    - how often to sample into `result.history` and call the
///                     observers. Default 1.
class Dynamics {
public:
    explicit Dynamics(Value parameters = {});
    virtual ~Dynamics() = default;

    Dynamics(const Dynamics&) = delete;
    Dynamics& operator=(const Dynamics&) = delete;

    [[nodiscard]] virtual std::string name() const { return "dynamics"; }

    /// \name Composition
    /// @{
    void set_integrator(std::shared_ptr<Integrator> integrator);
    void set_thermostat(std::shared_ptr<Thermostat> thermostat);
    void set_barostat(std::shared_ptr<Barostat> barostat);

    [[nodiscard]] const std::shared_ptr<Integrator>& integrator() const noexcept {
        return integrator_;
    }
    [[nodiscard]] const std::shared_ptr<Thermostat>& thermostat() const noexcept {
        return thermostat_;
    }
    [[nodiscard]] const std::shared_ptr<Barostat>& barostat() const noexcept {
        return barostat_;
    }

    /// Constrain a set of distances during the run.
    ///
    /// Passing `nullptr` turns constraints off, including the ones the
    /// calculator would otherwise have supplied. Each constraint removes one
    /// degree of freedom from the temperature, which `prepare` accounts for.
    void set_constraints(std::shared_ptr<Constraints> constraints);

    [[nodiscard]] const std::shared_ptr<Constraints>& constraints() const noexcept {
        return constraints_;
    }

    /// "nve", "nvt" or "npt", from what is attached. Accurate as soon as the
    /// object is built: the pieces are resolved from the parameters up front,
    /// not on the first `run`.
    [[nodiscard]] std::string ensemble() const;
    /// @}

    /// Run `steps` steps, or the configured number when `steps` is negative.
    ///
    /// \throws DynamicsError if no calculator is attached, if a thermostat has
    ///         no target temperature, or if the structure cannot be moved.
    DynamicsResult run(Atoms& atoms, int steps = -1);

    /// Advance by a single timestep. Assumes `prepare()` has been called.
    virtual void step(Atoms& atoms);

    /// Set up the context, the pieces and the initial velocities. `run` does
    /// this itself; call it directly only when driving `step()` by hand.
    void prepare(Atoms& atoms);

    /// The current state as a log record, without advancing anything.
    [[nodiscard]] DynamicsStep sample(Atoms& atoms);

    /// \name Observers
    /// @{
    /// Call `callback(atoms, step)` every `interval` steps. Returning false
    /// stops the run.
    void attach(DynamicsCallback callback, int interval = 1);
    void clear_callbacks();
    /// @}

    /// \name Configuration
    /// @{
    [[nodiscard]] const Value& parameters() const noexcept { return parameters_; }
    void set_parameters(Value parameters);
    void set_parameter(const std::string& key, Value value);

    /// Timestep in femtoseconds.
    [[nodiscard]] double timestep_fs() const;
    /// Timestep in the internal time unit.
    [[nodiscard]] double timestep() const;
    [[nodiscard]] int configured_steps() const;
    /// Target temperature in kelvin, zero when unset.
    [[nodiscard]] double temperature() const;
    /// Target pressure in eV/angstrom^3.
    [[nodiscard]] double pressure() const;
    /// @}

    /// Femtoseconds simulated since `prepare()`.
    [[nodiscard]] double time_fs() const noexcept { return time_fs_; }

    [[nodiscard]] const DynamicsContext& context() const noexcept { return context_; }

    [[nodiscard]] virtual std::string to_string() const;

protected:
    /// Apply the thermostat and keep the integrator's own velocities in step.
    void apply_thermostat(Atoms& atoms, double dt);
    /// Apply the barostat and keep the integrator's own velocities in step.
    void apply_barostat(Atoms& atoms, double dt);

    [[nodiscard]] const Value& parameter(std::string_view key) const;
    [[noreturn]] void fail(const std::string& message) const;

    Value parameters_{};
    DynamicsContext context_{};

private:
    struct Observer {
        DynamicsCallback callback;
        int interval{1};
    };

    void build_pieces();

    std::shared_ptr<Integrator> integrator_{};
    std::shared_ptr<Thermostat> thermostat_{};
    std::shared_ptr<Barostat> barostat_{};
    std::shared_ptr<Constraints> constraints_{};

    // A piece set through `set_*` is left alone when the parameters change, so
    // that `set_thermostat(nullptr)` really does mean NVE even if the input
    // file asked for a thermostat.
    bool integrator_is_explicit_{false};
    bool thermostat_is_explicit_{false};
    bool barostat_is_explicit_{false};
    bool constraints_are_explicit_{false};

    std::vector<Observer> observers_{};
    std::unique_ptr<TrajectoryWriter> trajectory_{};
    std::unique_ptr<LogWriter> log_{};

    double time_fs_{0.0};
    bool track_pressure_{false};
    bool prepared_{false};
};

}  // namespace fastatoms::md
