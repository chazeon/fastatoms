# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.20)

project(FastAtoms
    VERSION 0.1.0
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

include(cmake/FastAtomsDependencies.cmake)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
set(FASTATOMS_CORE_SOURCES
    src/core/atoms.cpp
    src/core/cell.cpp
    src/core/elements.cpp
    src/core/neighbor_list.cpp
    src/io/parameters.cpp
    src/io/yaml_input.cpp
    src/version.cpp)

set(FASTATOMS_PUBLIC_HEADERS
    include/fastatoms/fastatoms.hpp
    include/fastatoms/core/atoms.hpp
    include/fastatoms/core/cell.hpp
    include/fastatoms/core/elements.hpp
    include/fastatoms/core/neighbor_list.hpp
    include/fastatoms/core/units.hpp
    include/fastatoms/core/vec3.hpp
    include/fastatoms/io/parameters.hpp
    include/fastatoms/io/yaml_input.hpp)

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
    message(STATUS "  OpenMP ............. ${FASTATOMS_OPENMP_FOUND}")
    message(STATUS "  python bindings .... ${FASTATOMS_BUILD_PYTHON}")
    message(STATUS "  tests .............. ${FASTATOMS_BUILD_TESTS}")
    message(STATUS "  yaml-cpp ........... ${FASTATOMS_YAML_CPP_ORIGIN}")
    message(STATUS "")
endif()
