// SPDX-License-Identifier: MIT
#pragma once

/// \file barostat.hpp
/// \brief Pressure control for the NPT ensemble.

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "fastatoms/io/parameters.hpp"
#include "fastatoms/md/types.hpp"

namespace fastatoms::md {

/// Base class for pressure control.
///
/// A barostat changes the simulation box and carries the atoms with it, so
/// that the internal stress relaxes towards the target. Like a thermostat it
/// is applied as a half-step operator on each side of the integrator's step,
/// which keeps the splitting symmetric.
///
/// Because the box changes between force evaluations, an NPT step costs two
/// force calls rather than one. That is inherent to deforming the cell around
/// the integrator; the alternative is folding the deformation into the
/// integrator's position update, at the price of the integrator and the
/// barostat no longer being independent.
///
/// The operator is split in two so that no force evaluation is wasted:
/// `kick` updates the barostat's own variables from the current stress and
/// leaves the geometry alone, while `deform_cell` changes the box. A step then
/// runs kick, deform, integrate, kick - the kicks bracket the motion
/// symmetrically, and both of them land where the forces are already cached.
///
/// `deform_cell` returns the matrix it applied to the velocities, so an
/// integrator keeping its own copy of them can follow along.
class Barostat {
public:
    virtual ~Barostat() = default;

    Barostat(const Barostat&) = delete;
    Barostat& operator=(const Barostat&) = delete;

    [[nodiscard]] virtual std::string name() const = 0;

    virtual void begin(const Atoms& atoms, const DynamicsContext& context) {
        (void)atoms;
        (void)context;
    }

    /// Advance the barostat's own variables by `dt` using the current stress,
    /// without touching the geometry. A relaxation scheme with no internal
    /// state has nothing to do here.
    virtual void kick(Atoms& atoms, double dt, const DynamicsContext& context) {
        (void)atoms;
        (void)dt;
        (void)context;
    }

    /// Deform the cell over a time `dt`, carrying the atoms with it.
    /// \returns the transform applied to the velocities, identity if none.
    virtual Mat3 deform_cell(Atoms& atoms, double dt, const DynamicsContext& context) = 0;

    /// Energy of the barostat's own degrees of freedom plus the `pV` term, in
    /// eV. Added to the conserved quantity.
    [[nodiscard]] virtual double conserved_energy(Atoms& atoms,
                                                  const DynamicsContext& context) const {
        (void)atoms;
        (void)context;
        return 0.0;
    }

    /// True when the barostat is meant to sample the isothermal-isobaric
    /// distribution rather than merely hold the mean pressure.
    [[nodiscard]] virtual bool samples_isobaric() const { return true; }

    [[nodiscard]] const Value& parameters() const noexcept { return parameters_; }
    [[nodiscard]] virtual std::string to_string() const;

    /// Coupling time constant in femtoseconds.
    [[nodiscard]] double time_constant_fs() const;

    /// Whether only the volume may change, leaving the cell shape fixed.
    [[nodiscard]] bool is_isotropic() const;

protected:
    explicit Barostat(Value parameters);

    [[nodiscard]] const Value& parameter(std::string_view key) const;
    [[noreturn]] void fail(const std::string& message) const;

    /// Check that the structure can have its cell changed at all.
    void require_periodic(const Atoms& atoms) const;

    /// Apply an affine deformation: `cell <- cell D` with the atoms carried
    /// along, so fractional coordinates are unchanged.
    static void deform(Atoms& atoms, const Mat3& deformation);

    Value parameters_{};
};

/// Berendsen weak coupling: scale the box towards the target pressure.
///
///     mu = [1 - kappa (dt/tau_P) (P_target - P)]^(1/3)
///
/// with `kappa` the isothermal compressibility. The box and the atomic
/// positions are both multiplied by `mu`; velocities are left alone.
///
/// Robust and forgiving, and the usual way to bring a cell to the right
/// density before a production run. Like the Berendsen thermostat it does
/// **not** sample the correct distribution - the volume fluctuations come out
/// too small - so switch to Parrinello-Rahman for production NPT.
class BerendsenBarostat : public Barostat {
public:
    explicit BerendsenBarostat(Value parameters = {});

    [[nodiscard]] std::string name() const override { return "berendsen"; }
    Mat3 deform_cell(Atoms& atoms, double dt, const DynamicsContext& context) override;
    [[nodiscard]] bool samples_isobaric() const override { return false; }

    /// Isothermal compressibility in 1/GPa. Roughly 0.45 for water, 0.01 for a
    /// stiff solid. It only sets the coupling strength, so an order-of-
    /// magnitude estimate is enough.
    [[nodiscard]] double compressibility() const;
};

/// Parrinello-Rahman, in the strain-rate formulation.
///
/// The cell carries a velocity of its own, a strain rate `eps_dot`, driven by
/// the imbalance between the internal and the applied stress:
///
///     d(eps_dot)/dt = V (P_internal - P_external) / W
///     cell  <- cell (I + eps_dot dt)
///     r     <- r (I + eps_dot dt)
///     v     <- v (I - eps_dot dt)
///
/// where the internal pressure tensor includes the kinetic term and `W` is the
/// barostat mass, `(N_f + 3) kB T tau_P^2` unless given. Atoms and their
/// velocities transform affinely with the box, which is what makes the volume
/// a genuine dynamical variable rather than a relaxation target - so the
/// volume fluctuations are physical and the ensemble is isothermal-isobaric.
///
/// `isotropic: true` (the default) keeps only the trace of the strain rate, so
/// the cell changes size but not shape. Set it to false for the full cell
/// dynamics, which lets a crystal find its own lattice angles.
///
/// A barostat alone has no way to control temperature, and the cell degree of
/// freedom will heat up. Always pair this with a thermostat.
class ParrinelloRahmanBarostat : public Barostat {
public:
    explicit ParrinelloRahmanBarostat(Value parameters = {});

    [[nodiscard]] std::string name() const override { return "parrinello_rahman"; }

    void begin(const Atoms& atoms, const DynamicsContext& context) override;
    void kick(Atoms& atoms, double dt, const DynamicsContext& context) override;
    Mat3 deform_cell(Atoms& atoms, double dt, const DynamicsContext& context) override;
    [[nodiscard]] double conserved_energy(Atoms& atoms,
                                          const DynamicsContext& context) const override;

    /// The current strain rate, in units of inverse internal time.
    [[nodiscard]] const Mat3& strain_rate() const noexcept { return strain_rate_; }

    /// The barostat mass `W`, in eV * (internal time)^2.
    [[nodiscard]] double mass(const DynamicsContext& context) const;

private:
    Mat3 strain_rate_{};
};

/// Build a barostat by name: `berendsen`, or `parrinello_rahman` (aliases
/// `pr`, `npt`). Matching ignores case, hyphens and underscores. An empty name
/// gives no barostat, i.e. a fixed cell.
/// \throws DynamicsError if the name is not recognised.
[[nodiscard]] std::shared_ptr<Barostat> make_barostat(std::string_view name,
                                                      const Value& parameters = Value{});

/// Every barostat name, in alphabetical order.
[[nodiscard]] std::vector<std::string> barostat_names();

}  // namespace fastatoms::md
