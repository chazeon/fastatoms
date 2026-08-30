// SPDX-License-Identifier: MIT
#pragma once

/// \file thermostat.hpp
/// \brief Temperature control for the NVT ensemble.

#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/io/parameters.hpp"
#include "fastatoms/md/types.hpp"

namespace fastatoms::md {

/// Base class for temperature control.
///
/// A thermostat is applied as an operator around the integrator's step, for
/// half a timestep on each side. That Trotter splitting is what lets any
/// thermostat combine with any integrator: the integrator only ever does plain
/// Newtonian dynamics, and the thermostat only ever rescales velocities and
/// advances its own variables.
///
/// `apply` reports how it transformed the velocities, so that an integrator
/// holding its own copy of them - leapfrog does - can follow along.
class Thermostat {
public:
    virtual ~Thermostat() = default;

    Thermostat(const Thermostat&) = delete;
    Thermostat& operator=(const Thermostat&) = delete;

    [[nodiscard]] virtual std::string name() const = 0;

    /// Reset internal variables for a new run.
    virtual void begin(const Atoms& atoms, const DynamicsContext& context) {
        (void)atoms;
        (void)context;
    }

    /// Evolve the thermostat over a time `dt`, changing velocities in place.
    /// \returns the transformation that was applied to them.
    virtual VelocityUpdate apply(Atoms& atoms, double dt,
                                 const DynamicsContext& context) = 0;

    /// Energy of the thermostat's own degrees of freedom, in eV. Added to the
    /// conserved quantity so it can be checked for drift.
    [[nodiscard]] virtual double conserved_energy(const DynamicsContext& context) const {
        (void)context;
        return 0.0;
    }

    /// True when the thermostat is meant to sample the canonical distribution
    /// rather than merely hold the mean temperature.
    [[nodiscard]] virtual bool samples_canonical() const { return true; }

    [[nodiscard]] const Value& parameters() const noexcept { return parameters_; }
    [[nodiscard]] virtual std::string to_string() const;

    /// Coupling time constant in femtoseconds.
    [[nodiscard]] double time_constant_fs() const;

protected:
    explicit Thermostat(Value parameters);

    [[nodiscard]] const Value& parameter(std::string_view key) const;
    [[noreturn]] void fail(const std::string& message) const;

    Value parameters_{};
};

/// Berendsen weak coupling: rescale velocities towards the target.
///
///     lambda = sqrt(1 + (dt/tau) (T_target/T - 1))
///
/// Cheap, very stable, and the usual choice for pushing a structure to
/// temperature before a production run. It does **not** sample the canonical
/// distribution - it suppresses the fluctuations of the kinetic energy, and
/// over long runs it drains energy into the centre-of-mass motion, the "flying
/// ice cube". Use it to equilibrate, then switch to Nose-Hoover.
class BerendsenThermostat : public Thermostat {
public:
    explicit BerendsenThermostat(Value parameters = {});

    [[nodiscard]] std::string name() const override { return "berendsen"; }
    VelocityUpdate apply(Atoms& atoms, double dt, const DynamicsContext& context) override;
    [[nodiscard]] bool samples_canonical() const override { return false; }
};

/// Nose's original extended-system thermostat.
///
/// Nose (1984) couples the physical system to one extra degree of freedom `s`
/// with conjugate momentum `p_s` and mass `Q`, giving the Hamiltonian
///
///     H = sum p^2/(2 m s^2) + U(r) + p_s^2/(2Q) + g kB T ln s
///
/// whose microcanonical dynamics reproduce the canonical distribution for the
/// physical variables. Hoover's change of variables turns Nose's virtual-time
/// equations into the real-time form integrated here, with friction
/// `xi = p_s/Q`:
///
///     dv/dt = F/m - xi v,   ds/dt = s xi,   dxi/dt = (2K - g kB T)/Q
///
/// The difference from `NoseHooverThermostat` is Nose's own count of degrees
/// of freedom, `g = N_f + 1`: the extra one comes from the time scaling in the
/// original derivation. That count is right for sampling in virtual time and
/// slightly off in real time, which is precisely why Hoover's `g = N_f`
/// version superseded it. This class keeps `s` and reports Nose's conserved
/// Hamiltonian; prefer `NoseHooverThermostat` for production work.
class NoseThermostat : public Thermostat {
public:
    explicit NoseThermostat(Value parameters = {});

    [[nodiscard]] std::string name() const override { return "nose"; }

    void begin(const Atoms& atoms, const DynamicsContext& context) override;
    VelocityUpdate apply(Atoms& atoms, double dt, const DynamicsContext& context) override;
    [[nodiscard]] double conserved_energy(const DynamicsContext& context) const override;

    /// The extended coordinate `s`.
    [[nodiscard]] double s() const noexcept { return s_; }
    /// The friction variable `xi = p_s / Q`.
    [[nodiscard]] double friction() const noexcept { return friction_; }
    /// The thermostat mass `Q`, in eV * (internal time)^2.
    [[nodiscard]] double mass(const DynamicsContext& context) const;

private:
    double s_{1.0};
    double friction_{0.0};
};

/// Nose-Hoover, optionally as a chain.
///
///     dv/dt = F/m - xi_1 v
///     dxi_1/dt = (2K - N_f kB T)/Q_1
///
/// The real-time formulation with `g = N_f`, which is the count that makes the
/// physical variables sample the canonical distribution. Integrated with the
/// symmetric Trotter splitting of Martyna, Tuckerman and Klein, so the
/// extended-system energy is conserved to second order in the timestep.
///
/// A single Nose-Hoover variable is famously non-ergodic for small or stiff
/// systems - a lone harmonic oscillator never explores its canonical
/// distribution. Setting `chain_length` above one thermostats the thermostat,
/// and a chain of three or four fixes it. `chain_length: 1` is the plain
/// Nose-Hoover thermostat.
class NoseHooverThermostat : public Thermostat {
public:
    explicit NoseHooverThermostat(Value parameters = {});

    [[nodiscard]] std::string name() const override { return "nose_hoover"; }

    void begin(const Atoms& atoms, const DynamicsContext& context) override;
    VelocityUpdate apply(Atoms& atoms, double dt, const DynamicsContext& context) override;
    [[nodiscard]] double conserved_energy(const DynamicsContext& context) const override;

    /// Number of thermostats in the chain.
    [[nodiscard]] std::size_t chain_length() const;
    /// The friction variables, outermost last.
    [[nodiscard]] const std::vector<double>& friction() const noexcept { return friction_; }
    /// The thermostat masses `Q_j`, in eV * (internal time)^2.
    [[nodiscard]] std::vector<double> masses(const DynamicsContext& context) const;

private:
    std::vector<double> friction_{};
    std::vector<double> position_{};  ///< the integrals of the frictions, ln s_j
};

/// Langevin dynamics: friction plus a matching random force.
///
///     dv/dt = F/m - gamma v + sqrt(2 gamma kB T / m) eta(t)
///
/// The friction drains energy and the noise puts it back, in the ratio the
/// fluctuation-dissipation theorem fixes, so the stationary distribution is
/// canonical. Unlike a single Nose-Hoover variable it is ergodic by
/// construction: every degree of freedom is thermostatted independently, so
/// even a lone oscillator samples correctly.
///
/// The thermostat is the Ornstein-Uhlenbeck half of an OBABO splitting, and
/// that half is solved **exactly** rather than discretised:
///
///     v <- c1 v + c2 sqrt(kB T / m) xi,   c1 = exp(-gamma dt),
///                                          c2 = sqrt(1 - c1^2)
///
/// with `xi` standard normal. All the timestep error therefore comes from the
/// Verlet part, not from the thermostat.
///
/// Being stochastic, it has no conserved quantity of its own. What it does
/// track is the heat it has exchanged with the bath, so that
/// `potential + kinetic + heat` stays flat and can be checked for drift the
/// same way as for the deterministic thermostats.
///
/// Parameters:
///   - `friction` - gamma, in 1/fs. Defaults to the reciprocal of `tau`, which
///                  is itself 100 fs unless set. Around 0.01/fs is a good
///                  starting point: strong enough to equilibrate, weak enough
///                  not to swamp the dynamics.
///   - `seed`     - makes the noise reproducible. Zero draws a fresh one.
class LangevinThermostat : public Thermostat {
public:
    explicit LangevinThermostat(Value parameters = {});

    [[nodiscard]] std::string name() const override { return "langevin"; }

    void begin(const Atoms& atoms, const DynamicsContext& context) override;
    VelocityUpdate apply(Atoms& atoms, double dt, const DynamicsContext& context) override;
    [[nodiscard]] double conserved_energy(const DynamicsContext& context) const override;

    /// The friction coefficient in 1/fs.
    [[nodiscard]] double friction() const;

    /// Energy handed to the bath so far, in eV. Negative when the bath has
    /// been heating the system.
    [[nodiscard]] double heat() const noexcept { return heat_; }

    [[nodiscard]] std::string to_string() const override;

private:
    std::mt19937_64 generator_{};
    double heat_{0.0};
};

/// Build a thermostat by name: `berendsen`, `langevin`, `nose`, or `nose_hoover`
/// (aliases `nvt`, `nh`). Matching ignores case, hyphens and underscores.
/// An empty name gives no thermostat at all, i.e. NVE.
/// \throws DynamicsError if the name is not recognised.
[[nodiscard]] std::shared_ptr<Thermostat> make_thermostat(std::string_view name,
                                                          const Value& parameters = Value{});

/// Every thermostat name, in alphabetical order.
[[nodiscard]] std::vector<std::string> thermostat_names();

}  // namespace fastatoms::md
