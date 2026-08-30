# SPDX-License-Identifier: MIT
#
# C++ unit tests plus, when the extension is built, the Python test suite.

set(FASTATOMS_CPP_TESTS
    test_cell
    test_atoms
    test_neighbor_list
    test_yaml
    test_extxyz
    test_calculators
    test_dft_calculators
    test_mlip
    test_forcefield
    test_optimizers
    test_relaxation
    test_neb
    test_constraints
    test_build
    test_symmetry
    test_analysis
    test_electronic
    test_adsorption
    test_globalopt
    test_md
    test_md_ensembles
    test_phonon_setup
    test_phonons)

foreach(test_name IN LISTS FASTATOMS_CPP_TESTS)
    add_executable(${test_name} cpp/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE fastatoms_core)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
    if(NOT MSVC)
        target_compile_options(${test_name} PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME cpp.${test_name} COMMAND ${test_name})
endforeach()

if(FASTATOMS_BUILD_PYTHON)
    add_test(NAME python.bindings
        COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                "${CMAKE_CURRENT_SOURCE_DIR}/python"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
    set_tests_properties(python.bindings PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python;PYTHONDONTWRITEBYTECODE=1"
        DEPENDS fastatoms_python)
endif()
