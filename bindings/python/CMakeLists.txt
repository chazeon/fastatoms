# SPDX-License-Identifier: MIT
#
# pybind11 test module `fastatoms._core`. Built only with FASTATOMS_BUILD_PYTHON=ON
# (the top-level CMakeLists adds this directory); the scikit-build-core wheel in
# the repository root installs the extension into the `fastatoms` package.

pybind11_add_module(_core MODULE
    src/module.cpp
    src/common.cpp
    src/bind_atoms.cpp
    src/bind_formats.cpp
    src/bind_neighbors.cpp
    src/bind_symmetry.cpp)

target_link_libraries(_core PRIVATE FastAtoms::fastatoms)
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_core PRIVATE -Wall -Wextra -Wshadow)
endif()

install(TARGETS _core LIBRARY DESTINATION fastatoms)
