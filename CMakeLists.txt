# SPDX-License-Identifier: MIT
#
# Slim structure core: Atoms/Cell, structure-file IO, neighbour pairs.
# Builds natively (C++ tests, optional pybind11 test module) and with
# Emscripten (the WASM module behind the `fastatoms` npm package).
#
# The rest of the tree (MD, optimizers, phonons, calculators, YAML input) is
# kept but not built; its old build lives in cmake/legacy/CMakeLists.legacy.txt.
cmake_minimum_required(VERSION 3.20)

project(FastAtoms
    VERSION 0.3.0
    DESCRIPTION "Structure IO and neighbour lists for atomistic data, in C++ and WASM"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(FASTATOMS_IS_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(FASTATOMS_IS_TOP_LEVEL ON)
endif()

# ---------------------------------------------------------------------------
# Feature switches: cut the build (and the .wasm) down to what is needed.
# ---------------------------------------------------------------------------
option(FASTATOMS_WITH_VASP      "POSCAR/CONTCAR/XDATCAR"            ON)
option(FASTATOMS_WITH_XYZ       "XYZ and extended XYZ"              ON)
option(FASTATOMS_WITH_CIF       "CIF via gemmi"                     ON)
option(FASTATOMS_WITH_LAMMPS    "LAMMPS data and dump"              ON)
option(FASTATOMS_WITH_NEIGHBORS "Neighbour pairs via vesin"         ON)
option(FASTATOMS_WITH_SYMMETRY  "Space groups via spglib"           ON)

option(FASTATOMS_BUILD_TESTS  "Build the C++ tests" ${FASTATOMS_IS_TOP_LEVEL})
option(FASTATOMS_BUILD_PYTHON "Build the pybind11 test module"  OFF)
option(FASTATOMS_BUILD_WASM   "Build the embind module (Emscripten only)" ${EMSCRIPTEN})
option(FASTATOMS_ENABLE_WERROR "Treat warnings as errors" OFF)

include(cmake/Dependencies.cmake)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
set(FASTATOMS_SOURCES
    src/core/atoms.cpp
    src/core/cell.cpp
    src/core/elements.cpp
    src/core/value.cpp
    src/io/format.cpp)

set(FASTATOMS_FEATURE_DEFINES)
if(FASTATOMS_WITH_VASP)
    list(APPEND FASTATOMS_SOURCES src/io/vasp.cpp)
    list(APPEND FASTATOMS_FEATURE_DEFINES FASTATOMS_WITH_VASP)
endif()
if(FASTATOMS_WITH_XYZ)
    list(APPEND FASTATOMS_SOURCES src/io/xyz.cpp)
    list(APPEND FASTATOMS_FEATURE_DEFINES FASTATOMS_WITH_XYZ)
endif()
if(FASTATOMS_WITH_CIF)
    list(APPEND FASTATOMS_SOURCES src/io/cif.cpp)
    list(APPEND FASTATOMS_FEATURE_DEFINES FASTATOMS_WITH_CIF)
endif()
if(FASTATOMS_WITH_LAMMPS)
    list(APPEND FASTATOMS_SOURCES src/io/lammps_data.cpp src/io/lammps_dump.cpp)
    list(APPEND FASTATOMS_FEATURE_DEFINES FASTATOMS_WITH_LAMMPS)
endif()
if(FASTATOMS_WITH_NEIGHBORS)
    list(APPEND FASTATOMS_SOURCES src/core/neighbor_list.cpp)
    list(APPEND FASTATOMS_FEATURE_DEFINES FASTATOMS_WITH_NEIGHBORS)
endif()

if(FASTATOMS_WITH_SYMMETRY)
    file(GLOB _symmetry_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/symmetry/*.cpp)
    list(APPEND FASTATOMS_SOURCES ${_symmetry_sources})
    list(APPEND FASTATOMS_FEATURE_DEFINES FASTATOMS_WITH_SYMMETRY)
endif()

add_library(fastatoms STATIC ${FASTATOMS_SOURCES})
add_library(FastAtoms::fastatoms ALIAS fastatoms)
target_include_directories(fastatoms PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_definitions(fastatoms PUBLIC ${FASTATOMS_FEATURE_DEFINES})
target_compile_features(fastatoms PUBLIC cxx_std_20)
if(FASTATOMS_WITH_CIF)
    target_link_libraries(fastatoms PRIVATE fastatoms_gemmi)
endif()
if(FASTATOMS_WITH_NEIGHBORS)
    target_link_libraries(fastatoms PRIVATE fastatoms_vesin)
endif()
if(FASTATOMS_WITH_SYMMETRY)
    target_link_libraries(fastatoms PRIVATE fastatoms_spglib)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(fastatoms PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
    if(FASTATOMS_ENABLE_WERROR)
        target_compile_options(fastatoms PRIVATE -Werror)
    endif()
endif()

# ---------------------------------------------------------------------------
# Front ends
# ---------------------------------------------------------------------------
if(FASTATOMS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/core)
endif()

if(FASTATOMS_BUILD_PYTHON)
    add_subdirectory(bindings/python)
endif()

if(FASTATOMS_BUILD_WASM)
    if(NOT EMSCRIPTEN)
        message(FATAL_ERROR "FASTATOMS_BUILD_WASM needs the Emscripten toolchain (emcmake cmake ...)")
    endif()
    add_subdirectory(bindings/wasm)
endif()

message(STATUS "FastAtoms ${PROJECT_VERSION}: vasp=${FASTATOMS_WITH_VASP} xyz=${FASTATOMS_WITH_XYZ} "
               "cif=${FASTATOMS_WITH_CIF} lammps=${FASTATOMS_WITH_LAMMPS} "
               "neighbors=${FASTATOMS_WITH_NEIGHBORS} symmetry=${FASTATOMS_WITH_SYMMETRY}")
