#!/usr/bin/env bash
# Build FastAtoms into the conda prefix: the C++ core, the Python extension,
# the headers and the CMake package files, in one configure/build/install pass.
#
# Nothing here asks the user to run CMake. `conda build conda/` calls this, and
# the same script serves a plain `conda-build` and a conda-forge feedstock.
set -euxo pipefail

# ---------------------------------------------------------------------------
# Where things go
# ---------------------------------------------------------------------------
# $PREFIX is the environment being built into; $SP_DIR is its site-packages.
# The Python install directory is passed explicitly rather than letting CMake
# ask the interpreter for it, because when cross-compiling the host
# interpreter cannot be run at all.
declare -a CMAKE_FLAGS=(
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="${PREFIX}"
    # Where find_package looks for yaml-cpp and pybind11. They come from the
    # host environment; FETCHCONTENT_FULLY_DISCONNECTED below turns a failure
    # to find them into an error instead of a download.
    -DCMAKE_PREFIX_PATH="${PREFIX}"
    -DPython3_EXECUTABLE="${PYTHON}"
    -DPython3_FIND_STRATEGY=LOCATION
    -DFASTATOMS_PYTHON_INSTALL_DIR="${SP_DIR}/fastatoms"

    # Without these the installed `fastatoms-run` cannot find libyaml-cpp:
    # it is linked as @rpath/libyaml-cpp.dylib and the binary is installed
    # with no LC_RPATH at all. conda-build's relocation pass would probably
    # paper over it, but a prefix that is correct before post-processing is
    # one less thing depending on post-processing.
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
    -DCMAKE_INSTALL_RPATH="${PREFIX}/lib"

    # Install everything: this package carries the C++ library as well as the
    # bindings, so a downstream find_package(FastAtoms) works against the env.
    -DFASTATOMS_INSTALL=ON
    -DFASTATOMS_BUILD_PYTHON=ON
    -DFASTATOMS_BUILD_EXAMPLES=ON
    # Built so the suite can run below, then not installed.
    -DFASTATOMS_BUILD_TESTS=ON

    # A conda build has no network. Without this, a yaml-cpp that failed to be
    # found would be quietly fetched from GitHub - which fails on a sandboxed
    # builder, and on an unsandboxed one silently vendors a copy the package
    # then depends on invisibly. Failing at configure time is the better half
    # of that trade.
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON

    # LibTorch is not a host dependency: pytorch would multiply the build by an
    # order of magnitude for a feature that also works through the Python
    # worker. Machine-learning potentials still run, out of process.
    -DFASTATOMS_WITH_TORCH=OFF
)

# ---------------------------------------------------------------------------
# OpenMP, on macOS
# ---------------------------------------------------------------------------
# find_package(OpenMP) asks find_library for libomp, and on a machine with
# Homebrew installed that search reaches /opt/homebrew before the prefix. The
# result builds and tests clean and is still wrong: the package links an
# OpenMP runtime conda does not ship, so it fails on any machine without
# Homebrew, and on one with it two copies load and abort with
#
#     OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
#                     already initialized
#
# Naming the library outright is the fix. Linux needs none of this: conda's gcc
# brings its own libgomp and there is no second one to find.
if [[ "${target_platform:-}" == osx-* ]]; then
    CMAKE_FLAGS+=(
        -DOpenMP_CXX_FLAGS="-fopenmp"
        -DOpenMP_CXX_LIB_NAMES=omp
        -DOpenMP_omp_LIBRARY="${PREFIX}/lib/libomp.dylib"
    )
fi

# conda-forge passes the cross-compilation toolchain file and the deployment
# target through $CMAKE_ARGS. Appending it last lets it override anything here.
cmake -S "${SRC_DIR}" -B build "${CMAKE_FLAGS[@]}" ${CMAKE_ARGS:-}

cmake --build build --parallel "${CPU_COUNT:-1}"

# ---------------------------------------------------------------------------
# Check before packaging
# ---------------------------------------------------------------------------
# The C++ suites only: they need nothing that is not already here. The Python
# suite runs in the test: section of meta.yaml, against the installed package
# rather than the build tree, which is the more useful thing to check.
#
# Skipped when cross-compiling, where the binaries cannot be executed.
if [[ "${build_platform:-}" == "${target_platform:-}" ]]; then
    ctest --test-dir build --output-on-failure -E "python"
fi

cmake --install build
