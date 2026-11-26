// SPDX-License-Identifier: MIT
#pragma once

/// \file parameters.hpp
/// \brief A small, order-preserving dynamic value tree plus the typed
///        simulation-parameter structs built on top of it.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fastatoms/core/value.hpp"

namespace fastatoms {

/// What the run should do.
enum class Task { SinglePoint, MolecularDynamics, Optimization, Phonon, NEB, Custom };

[[nodiscard]] Task task_from_string(std::string_view name);
[[nodiscard]] std::string_view task_to_string(Task task);

/// Which calculator (force model) to attach.
struct CalculatorParameters {
    std::string name{};   ///< e.g. "lennard_jones", "eam", "external"
    Value parameters{};   ///< free-form, calculator specific
};

/// Molecular-dynamics settings. Times are in femtoseconds as written in the
/// YAML file; `timestep_internal()` converts to the internal time unit.
struct MolecularDynamicsParameters {
    std::string ensemble{"nve"};      ///< nve, nvt, npt
    double timestep_fs{1.0};
    std::int64_t steps{0};
    double temperature_K{0.0};
    double pressure_GPa{0.0};
    std::string thermostat{};         ///< e.g. "langevin", "berendsen"
    double friction{0.01};            ///< thermostat coupling, 1/fs
    std::int64_t seed{0};
    bool initialize_velocities{false};

    /// The unparsed `md:` subtree, so that settings the typed fields do not
    /// cover - integrator, barostat, chain length, coupling times - reach the
    /// dynamics unchanged.
    Value parameters{};

    [[nodiscard]] double timestep_internal() const;
};

/// Geometry-optimisation settings.
struct OptimizerParameters {
    std::string algorithm{"bfgs"};
    double fmax{0.05};        ///< eV/angstrom
    std::int64_t max_steps{200};
    bool relax_cell{false};

    /// The unparsed `optimizer:` subtree, so that settings the typed fields do
    /// not cover - line search, trust radius, external pressure - reach the
    /// optimizer unchanged.
    Value parameters{};
};

/// Phonon settings. `supercell` is an integer transformation matrix written
/// either as three diagonal entries or as all nine.
struct PhononParameters {
    std::vector<double> supercell{2.0, 2.0, 2.0};  ///< 3 or 9 entries
    double displacement{0.01};                     ///< angstrom
    bool symmetry{true};                           ///< reduce the displacement set
    double symmetry_tolerance{1e-5};               ///< angstrom

    /// The unparsed `phonons:` subtree, so that settings the typed fields do
    /// not cover - central differences, the acoustic sum rule, a q-point path -
    /// reach the calculation unchanged.
    Value parameters{};
};

/// Nudged-elastic-band settings.
///
/// The endpoints are files, or the deck's own `structure:` for the initial
/// one. They are not resolved here: parsing stays free of file I/O, so a
/// document can be read and validated without the structures it names being
/// present. `neb::make_band` does the reading.
struct BandParameters {
    /// Path to the first endpoint. Empty means the deck's `structure:`.
    std::string initial{};
    /// Path to the last endpoint. Required unless `band` is set.
    std::string final{};
    /// The endpoints written inline, in the `structure:` layout, instead of
    /// named as files. Small systems are more legible in one document than
    /// split across three, and it keeps a self-contained deck possible.
    Value initial_structure{};
    Value final_structure{};
    /// A prepared multi-frame band - what `fastatoms-interpolate` writes, or a
    /// partly converged band being restarted. Overrides `initial` and `final`,
    /// and `images` then follows the file.
    std::string band{};

    /// Images including the two endpoints, when interpolating.
    std::int64_t images{7};

    double spring_constant{0.1};
    bool climbing{false};
    double climb_threshold{0.0};
    std::string tangent{"improved"};
    bool variable_springs{false};
    double spring_softening{0.5};
    bool minimum_image{true};
    std::vector<std::int64_t> fixed_atoms{};

    /// Grow the band with AutoNEB instead of relaxing a fixed one.
    bool automatic{false};
    std::int64_t max_images{11};
    double coarse_fmax{0.2};
    std::string insert{"energy"};
    bool prune{false};
    double prune_tolerance{0.01};
    bool redistribute{false};

    /// The unparsed `neb:` subtree.
    Value parameters{};
};

/// Where results should go.
struct OutputParameters {
    std::string trajectory{};
    std::string log{};
    std::int64_t interval{1};
};

/// The `simulation:` section of an input file.
struct SimulationParameters {
    Task task{Task::SinglePoint};
    std::string task_name{"single_point"};
    CalculatorParameters calculator{};
    MolecularDynamicsParameters md{};
    OptimizerParameters optimizer{};
    PhononParameters phonon{};
    BandParameters neb{};
    OutputParameters output{};

    /// The unparsed `simulation:` subtree, so that consumers can read keys the
    /// typed structs do not know about.
    Value raw{};
};

}  // namespace fastatoms
