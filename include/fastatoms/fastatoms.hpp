// SPDX-License-Identifier: MIT
#pragma once

/// \file fastatoms.hpp
/// \brief Umbrella header: includes the whole public API.

#include "fastatoms/calculators/calculator.hpp"
#include "fastatoms/calculators/file_io_calculator.hpp"
#include "fastatoms/calculators/lennard_jones.hpp"
#include "fastatoms/calculators/mlip.hpp"
#include "fastatoms/calculators/mlip_models.hpp"
#include "fastatoms/calculators/mlip_torch.hpp"
#include "fastatoms/calculators/mlip_worker.hpp"
#include "fastatoms/calculators/quantum_espresso.hpp"
#include "fastatoms/calculators/registry.hpp"
#include "fastatoms/calculators/results.hpp"
#include "fastatoms/calculators/siesta.hpp"
#include "fastatoms/calculators/vasp.hpp"
#include "fastatoms/core/atoms.hpp"
#include "fastatoms/core/cell.hpp"
#include "fastatoms/core/constraint.hpp"
#include "fastatoms/core/elements.hpp"
#include "fastatoms/core/neighbor_list.hpp"
#include "fastatoms/core/units.hpp"
#include "fastatoms/core/vec3.hpp"
#include "fastatoms/io/parameters.hpp"
#include "fastatoms/io/yaml_input.hpp"
#include "fastatoms/forcefield/ewald.hpp"
#include "fastatoms/forcefield/forcefield.hpp"
#include "fastatoms/forcefield/topology.hpp"
#include "fastatoms/forcefield/water.hpp"
#include "fastatoms/md/barostat.hpp"
#include "fastatoms/md/constraints.hpp"
#include "fastatoms/md/dynamics.hpp"
#include "fastatoms/md/integrator.hpp"
#include "fastatoms/md/registry.hpp"
#include "fastatoms/md/thermostat.hpp"
#include "fastatoms/md/trajectory.hpp"
#include "fastatoms/md/types.hpp"
#include "fastatoms/md/velocity.hpp"
#include "fastatoms/optimize/bfgs.hpp"
#include "fastatoms/optimize/conjugate_gradient.hpp"
#include "fastatoms/optimize/lbfgs.hpp"
#include "fastatoms/optimize/line_search.hpp"
#include "fastatoms/optimize/optimizer.hpp"
#include "fastatoms/optimize/registry.hpp"
#include "fastatoms/optimize/steepest_descent.hpp"
#include "fastatoms/optimize/types.hpp"
#include "fastatoms/optimize/variables.hpp"
#include "fastatoms/phonon/eigen.hpp"
#include "fastatoms/phonon/force_constants.hpp"
#include "fastatoms/phonon/phonons.hpp"
#include "fastatoms/phonon/registry.hpp"
#include "fastatoms/phonon/supercell.hpp"
#include "fastatoms/phonon/symmetry.hpp"
#include "fastatoms/util/process.hpp"
#include "fastatoms/util/text.hpp"

/// FastAtoms - a high-performance C++ core for atomistic simulations.
namespace fastatoms {

/// Library version string, e.g. "0.1.0".
[[nodiscard]] const char* version() noexcept;

}  // namespace fastatoms
