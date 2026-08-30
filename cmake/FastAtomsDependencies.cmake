# SPDX-License-Identifier: MIT
#
# Dependency resolution for FastAtoms.
#
# Every dependency is looked for on the system first and downloaded only as a
# fallback, so a packaged build (conda, spack, a distro) never pulls from the
# network while a bare `cmake -B build` still works out of the box.
#
# Set FETCHCONTENT_FULLY_DISCONNECTED=ON to forbid downloads entirely.

include(FetchContent)

set(FASTATOMS_YAML_CPP_TAG "yaml-cpp-0.9.0" CACHE STRING
    "yaml-cpp git tag to download when it is not found on the system")
set(FASTATOMS_PYBIND11_TAG "v3.1.0" CACHE STRING
    "pybind11 git tag to download when it is not found on the system")

# ---------------------------------------------------------------------------
# OpenMP (optional)
# ---------------------------------------------------------------------------
set(FASTATOMS_OPENMP_FOUND OFF)
if(FASTATOMS_ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS CXX QUIET)
    if(OpenMP_CXX_FOUND)
        set(FASTATOMS_OPENMP_FOUND ON)
    else()
        message(STATUS "OpenMP not found; FastAtoms will build single-threaded")
    endif()
endif()

# ---------------------------------------------------------------------------
# yaml-cpp (required, for the input parser)
# ---------------------------------------------------------------------------
find_package(yaml-cpp QUIET CONFIG)

if(yaml-cpp_FOUND)
    set(FASTATOMS_YAML_CPP_ORIGIN "system (${yaml-cpp_DIR})")
    set(FASTATOMS_YAML_CPP_VENDORED OFF)
else()
    set(FASTATOMS_YAML_CPP_ORIGIN "downloaded ${FASTATOMS_YAML_CPP_TAG}")
    set(FASTATOMS_YAML_CPP_VENDORED ON)
    message(STATUS "yaml-cpp not found on the system; fetching ${FASTATOMS_YAML_CPP_TAG}")

    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
    set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    # A static libfastatoms does not absorb yaml-cpp's objects, so anything
    # linking against an installed FastAtoms still needs them. When we are the
    # ones providing yaml-cpp, install it alongside so the prefix is complete.
    set(YAML_CPP_INSTALL ${FASTATOMS_INSTALL} CACHE BOOL "" FORCE)

    # Older yaml-cpp releases declare a cmake_minimum_required that CMake 4
    # rejects outright; this keeps them configurable.
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

    FetchContent_Declare(yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG        ${FASTATOMS_YAML_CPP_TAG}
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(yaml-cpp)

    unset(CMAKE_POLICY_VERSION_MINIMUM)
endif()

# yaml-cpp exports `yaml-cpp::yaml-cpp` from 0.8 on and a bare `yaml-cpp`
# before that; normalise the two behind one variable. (An ALIAS is no use here
# because the modern name is itself an alias, and aliasing an alias is an
# error.)
if(TARGET yaml-cpp::yaml-cpp)
    set(FASTATOMS_YAML_CPP_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
    set(FASTATOMS_YAML_CPP_TARGET yaml-cpp)
else()
    message(FATAL_ERROR "yaml-cpp was resolved but exported no usable target")
endif()

# ---------------------------------------------------------------------------
# pybind11 (only for the Python extension)
# ---------------------------------------------------------------------------
if(FASTATOMS_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

    # A pip-installed pybind11 ships its own CMake package; ask the interpreter
    # where it is before falling back to a download.
    if(NOT pybind11_DIR AND Python3_EXECUTABLE)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c
                    "import pybind11, sys; sys.stdout.write(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE FASTATOMS_PYBIND11_HINT
            ERROR_QUIET
            RESULT_VARIABLE FASTATOMS_PYBIND11_QUERY)
        if(FASTATOMS_PYBIND11_QUERY EQUAL 0 AND EXISTS "${FASTATOMS_PYBIND11_HINT}")
            set(pybind11_DIR "${FASTATOMS_PYBIND11_HINT}" CACHE PATH "" FORCE)
        endif()
    endif()

    find_package(pybind11 QUIET CONFIG)

    if(pybind11_FOUND)
        set(FASTATOMS_PYBIND11_ORIGIN "system (${pybind11_DIR})")
    else()
        set(FASTATOMS_PYBIND11_ORIGIN "downloaded ${FASTATOMS_PYBIND11_TAG}")
        message(STATUS "pybind11 not found; fetching ${FASTATOMS_PYBIND11_TAG}")
        FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        ${FASTATOMS_PYBIND11_TAG}
            GIT_SHALLOW    TRUE)
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()
