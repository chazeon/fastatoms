# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.20)

project(FastAtoms
    VERSION 0.1.3
    DESCRIPTION "High-performance C++ core for atomistic simulations, driven from Python"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Project-wide settings
# ---------------------------------------------------------------------------
set(FASTATOMS_IS_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(FASTATOMS_IS_TOP_LEVEL ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The core is linked into a shared Python module, so it must be position
# independent even when built as a static library.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

include(GNUInstallDirs)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(FASTATOMS_BUILD_PYTHON   "Build the pybind11 Python extension" ON)
option(FASTATOMS_BUILD_TESTS    "Build the C++ test suite" ${FASTATOMS_IS_TOP_LEVEL})
option(FASTATOMS_BUILD_EXAMPLES "Build the C++ example programs" ${FASTATOMS_IS_TOP_LEVEL})
option(FASTATOMS_ENABLE_OPENMP  "Use OpenMP to parallelise the hot loops" ON)
option(FASTATOMS_ENABLE_NATIVE  "Optimise for the building machine (-march=native)" OFF)
option(FASTATOMS_ENABLE_LTO     "Enable link-time optimisation in optimised builds" OFF)
option(FASTATOMS_ENABLE_WERROR  "Treat compiler warnings as errors" OFF)
option(FASTATOMS_SHARED_CORE    "Build the core as a shared library" OFF)
option(FASTATOMS_INSTALL        "Generate install rules for the C++ library" ON)
option(FASTATOMS_WITH_TORCH     "Evaluate machine-learning potentials in process with LibTorch" OFF)

# LibTorch ships inside the `torch` wheel, so the easiest way to find it is to
# ask an interpreter that has torch installed where its CMake files live:
#
#   cmake -S . -B build -DFASTATOMS_WITH_TORCH=ON \
#         -DFASTATOMS_TORCH_PYTHON=$HOME/miniconda3/envs/mace_env/bin/python
set(FASTATOMS_TORCH_PYTHON "" CACHE STRING
    "Interpreter whose torch installation supplies LibTorch")

include(cmake/FastAtomsDependencies.cmake)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
set(FASTATOMS_CORE_SOURCES
    src/core/atoms.cpp
    src/core/constraints.cpp
    src/core/cell.cpp
    src/core/elements.cpp
    src/core/neighbor_list.cpp
    src/io/extxyz.cpp
    src/io/formats.cpp
    src/io/parameters.cpp
    src/io/yaml_input.cpp
    src/calculators/atoms_calculator.cpp
    src/calculators/calculator.cpp
    src/calculators/file_io_calculator.cpp
    src/calculators/lennard_jones.cpp
    src/calculators/mlip.cpp
    src/calculators/mlip_models.cpp
    src/calculators/mlip_torch.cpp
    src/calculators/mlip_worker.cpp
    src/calculators/quantum_espresso.cpp
    src/calculators/registry.cpp
    src/calculators/siesta.cpp
    src/calculators/vasp.cpp
    src/md/barostat.cpp
    src/md/constraints.cpp
    src/md/dynamics.cpp
    src/md/integrator.cpp
    src/md/registry.cpp
    src/md/thermostat.cpp
    src/md/trajectory.cpp
    src/md/types.cpp
    src/md/velocity.cpp
    src/forcefield/ewald.cpp
    src/forcefield/forcefield.cpp
    src/forcefield/topology.cpp
    src/forcefield/water.cpp
    src/phonon/eigen.cpp
    src/phonon/force_constants.cpp
    src/phonon/phonons.cpp
    src/phonon/registry.cpp
    src/phonon/supercell.cpp
    src/phonon/symmetry.cpp
    src/optimize/bfgs.cpp
    src/optimize/conjugate_gradient.cpp
    src/neb/autoneb.cpp
    src/neb/idpp.cpp
    src/neb/neb.cpp
    src/neb/registry.cpp
    src/optimize/fire.cpp
    src/optimize/lbfgs.cpp
    src/optimize/line_search.cpp
    src/optimize/optimizer.cpp
    src/optimize/registry.cpp
    src/optimize/steepest_descent.cpp
    src/optimize/variables.cpp
    src/util/process.cpp
    src/util/text.cpp
    src/version.cpp)

set(FASTATOMS_PUBLIC_HEADERS
    include/fastatoms/fastatoms.hpp
    include/fastatoms/core/atoms.hpp
    include/fastatoms/core/cell.hpp
    include/fastatoms/core/constraint.hpp
    include/fastatoms/core/constraints.hpp
    include/fastatoms/core/elements.hpp
    include/fastatoms/core/neighbor_list.hpp
    include/fastatoms/core/units.hpp
    include/fastatoms/core/vec3.hpp
    include/fastatoms/io/extxyz.hpp
    include/fastatoms/io/formats.hpp
    include/fastatoms/io/parameters.hpp
    include/fastatoms/io/yaml_input.hpp
    include/fastatoms/calculators/calculator.hpp
    include/fastatoms/calculators/file_io_calculator.hpp
    include/fastatoms/calculators/lennard_jones.hpp
    include/fastatoms/calculators/mlip.hpp
    include/fastatoms/calculators/mlip_models.hpp
    include/fastatoms/calculators/mlip_torch.hpp
    include/fastatoms/calculators/mlip_worker.hpp
    include/fastatoms/calculators/quantum_espresso.hpp
    include/fastatoms/calculators/registry.hpp
    include/fastatoms/calculators/results.hpp
    include/fastatoms/calculators/siesta.hpp
    include/fastatoms/calculators/vasp.hpp
    include/fastatoms/md/barostat.hpp
    include/fastatoms/md/constraints.hpp
    include/fastatoms/md/dynamics.hpp
    include/fastatoms/md/integrator.hpp
    include/fastatoms/md/registry.hpp
    include/fastatoms/md/thermostat.hpp
    include/fastatoms/md/trajectory.hpp
    include/fastatoms/md/types.hpp
    include/fastatoms/md/velocity.hpp
    include/fastatoms/forcefield/ewald.hpp
    include/fastatoms/forcefield/forcefield.hpp
    include/fastatoms/forcefield/topology.hpp
    include/fastatoms/forcefield/water.hpp
    include/fastatoms/phonon/eigen.hpp
    include/fastatoms/phonon/force_constants.hpp
    include/fastatoms/phonon/phonons.hpp
    include/fastatoms/phonon/registry.hpp
    include/fastatoms/phonon/supercell.hpp
    include/fastatoms/phonon/symmetry.hpp
    include/fastatoms/optimize/bfgs.hpp
    include/fastatoms/optimize/conjugate_gradient.hpp
    include/fastatoms/neb/autoneb.hpp
    include/fastatoms/neb/idpp.hpp
    include/fastatoms/neb/neb.hpp
    include/fastatoms/neb/registry.hpp
    include/fastatoms/optimize/fire.hpp
    include/fastatoms/optimize/lbfgs.hpp
    include/fastatoms/optimize/line_search.hpp
    include/fastatoms/optimize/optimizer.hpp
    include/fastatoms/optimize/registry.hpp
    include/fastatoms/optimize/steepest_descent.hpp
    include/fastatoms/optimize/types.hpp
    include/fastatoms/optimize/variables.hpp
    include/fastatoms/optimize/vector_math.hpp
    include/fastatoms/util/process.hpp
    include/fastatoms/util/text.hpp)

if(FASTATOMS_SHARED_CORE)
    add_library(fastatoms_core SHARED ${FASTATOMS_CORE_SOURCES} ${FASTATOMS_PUBLIC_HEADERS})
else()
    add_library(fastatoms_core STATIC ${FASTATOMS_CORE_SOURCES} ${FASTATOMS_PUBLIC_HEADERS})
endif()
add_library(FastAtoms::core ALIAS fastatoms_core)

target_include_directories(fastatoms_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(fastatoms_core
    PRIVATE FASTATOMS_VERSION="${PROJECT_VERSION}")

# PRIVATE, but a static library still records it as a link-only interface
# dependency, because a consumer has to supply yaml-cpp's objects at link time.
# FastAtomsConfig.cmake resolves it again with find_dependency().
target_link_libraries(fastatoms_core PRIVATE ${FASTATOMS_YAML_CPP_TARGET})

# ---------------------------------------------------------------------------
# LibTorch, for running machine-learning potentials without an interpreter
# ---------------------------------------------------------------------------
set(FASTATOMS_TORCH_ORIGIN "disabled")
if(FASTATOMS_WITH_TORCH)
    if(FASTATOMS_TORCH_PYTHON)
        execute_process(
            COMMAND "${FASTATOMS_TORCH_PYTHON}" -c
                    "import torch, sys; sys.stdout.write(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE FASTATOMS_TORCH_CMAKE_PATH
            ERROR_VARIABLE FASTATOMS_TORCH_PROBE_ERROR
            RESULT_VARIABLE FASTATOMS_TORCH_PROBE_STATUS)
        if(NOT FASTATOMS_TORCH_PROBE_STATUS EQUAL 0)
            message(FATAL_ERROR
                "FASTATOMS_TORCH_PYTHON=${FASTATOMS_TORCH_PYTHON} cannot import torch:\n"
                "${FASTATOMS_TORCH_PROBE_ERROR}")
        endif()
        list(PREPEND CMAKE_PREFIX_PATH "${FASTATOMS_TORCH_CMAKE_PATH}")
    endif()

    find_package(Torch REQUIRED)
    target_link_libraries(fastatoms_core PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(fastatoms_core PRIVATE FASTATOMS_WITH_TORCH=1)
    if(Torch_VERSION)
        set(FASTATOMS_TORCH_ORIGIN "${Torch_VERSION}")
    elseif(TORCH_VERSION)
        set(FASTATOMS_TORCH_ORIGIN "${TORCH_VERSION}")
    else()
        set(FASTATOMS_TORCH_ORIGIN "found")
    endif()

    # LibTorch's headers trip -Wshadow and friends, and they are not ours to
    # fix, so they come in as system headers.
    get_target_property(FASTATOMS_TORCH_INCLUDES torch INTERFACE_INCLUDE_DIRECTORIES)
    if(FASTATOMS_TORCH_INCLUDES)
        target_include_directories(fastatoms_core SYSTEM PRIVATE ${FASTATOMS_TORCH_INCLUDES})
    endif()

    # LibTorch loads its own OpenMP runtime (libtorch_cpu depends on
    # @rpath/libomp.dylib). Linking a second one - the compiler's - aborts the
    # process before main with
    #
    #   OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
    #                   already initialized
    #
    # and the two cannot be merged: the copy torch ships records an absolute
    # install name that only exists on the machine that built it, so linking
    # that file directly produces a binary dyld cannot load. Rather than an
    # unsupported KMP_DUPLICATE_LIB_OK, FastAtoms simply stops adding a second
    # runtime and uses the one already in the process.
    #
    # The only thing that loses threads is the neighbour-list build, which is a
    # rounding error next to a neural network evaluation. A build that wants
    # both should leave FASTATOMS_WITH_TORCH off and use the worker backend.
    if(FASTATOMS_OPENMP_FOUND)
        set(FASTATOMS_OPENMP_FOUND OFF)
        set(FASTATOMS_OPENMP_NOTE " (off: LibTorch supplies the OpenMP runtime)")
        message(STATUS
            "LibTorch supplies OpenMP; FastAtoms will not link a second runtime")
    endif()
endif()

# Where the MLIP worker script lives, so a calculator can find it without the
# caller having to say. The source tree comes first for a build tree that is
# used in place, and the installed copy is the fallback.
target_compile_definitions(fastatoms_core PRIVATE
    FASTATOMS_MLIP_WORKER_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/python/fastatoms/mlip/worker.py"
    FASTATOMS_MLIP_WORKER_INSTALLED="${CMAKE_INSTALL_FULL_DATADIR}/fastatoms/mlip/worker.py")

set_target_properties(fastatoms_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME fastatoms
    # So the installed target is FastAtoms::core, matching the in-tree alias.
    EXPORT_NAME core
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

# --- Warnings ---------------------------------------------------------------
if(MSVC)
    target_compile_options(fastatoms_core PRIVATE /W4 /permissive-)
    if(FASTATOMS_ENABLE_WERROR)
        target_compile_options(fastatoms_core PRIVATE /WX)
    endif()
else()
    target_compile_options(fastatoms_core PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
    if(FASTATOMS_ENABLE_WERROR)
        target_compile_options(fastatoms_core PRIVATE -Werror)
    endif()
endif()

# --- Optimisation -----------------------------------------------------------
# -O3 already comes from the Release build type; these are the extras that are
# not safe to turn on unconditionally.
if(FASTATOMS_ENABLE_NATIVE AND NOT MSVC)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-march=native FASTATOMS_HAS_MARCH_NATIVE)
    if(FASTATOMS_HAS_MARCH_NATIVE)
        target_compile_options(fastatoms_core PUBLIC
            $<$<NOT:$<CONFIG:Debug>>:-march=native>)
    else()
        check_cxx_compiler_flag(-mcpu=native FASTATOMS_HAS_MCPU_NATIVE)
        if(FASTATOMS_HAS_MCPU_NATIVE)
            target_compile_options(fastatoms_core PUBLIC
                $<$<NOT:$<CONFIG:Debug>>:-mcpu=native>)
        endif()
    endif()
endif()

if(FASTATOMS_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT FASTATOMS_IPO_SUPPORTED OUTPUT FASTATOMS_IPO_MESSAGE)
    if(FASTATOMS_IPO_SUPPORTED)
        set_property(TARGET fastatoms_core PROPERTY
            INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    else()
        message(WARNING "LTO requested but not supported: ${FASTATOMS_IPO_MESSAGE}")
    endif()
endif()

if(FASTATOMS_OPENMP_FOUND)
    target_link_libraries(fastatoms_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
if(FASTATOMS_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(FASTATOMS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(FASTATOMS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------
# Installation
#
# Skipped when building a Python wheel, where only the extension module and the
# package sources belong in the archive.
# ---------------------------------------------------------------------------
if(FASTATOMS_INSTALL)
include(CMakePackageConfigHelpers)

install(TARGETS fastatoms_core
    EXPORT FastAtomsTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/fastatoms
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp")

# The MLIP worker runs under a different interpreter from the one that imports
# FastAtoms, so it is installed as data rather than only as part of the wheel.
install(FILES python/fastatoms/mlip/worker.py
    DESTINATION ${CMAKE_INSTALL_DATADIR}/fastatoms/mlip)

install(EXPORT FastAtomsTargets
    FILE FastAtomsTargets.cmake
    NAMESPACE FastAtoms::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FastAtoms)

configure_package_config_file(
    cmake/FastAtomsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/FastAtomsConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FastAtoms)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/FastAtomsConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/FastAtomsConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/FastAtomsConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FastAtoms)
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if(FASTATOMS_IS_TOP_LEVEL)
    message(STATUS "")
    message(STATUS "FastAtoms ${PROJECT_VERSION}")
    message(STATUS "  build type ......... ${CMAKE_BUILD_TYPE}")
    message(STATUS "  compiler ........... ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "  OpenMP ............. ${FASTATOMS_OPENMP_FOUND}${FASTATOMS_OPENMP_NOTE}")
    message(STATUS "  python bindings .... ${FASTATOMS_BUILD_PYTHON}")
    message(STATUS "  tests .............. ${FASTATOMS_BUILD_TESTS}")
    message(STATUS "  yaml-cpp ........... ${FASTATOMS_YAML_CPP_ORIGIN}")
    message(STATUS "  LibTorch (MLIP) .... ${FASTATOMS_TORCH_ORIGIN}")
    message(STATUS "")
endif()
