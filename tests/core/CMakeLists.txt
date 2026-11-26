# SPDX-License-Identifier: MIT
# One executable per test file; each registers with ctest. Test data lives
# in tests/data and is found through FASTATOMS_TEST_DATA.
set(FASTATOMS_CORE_TESTS test_atoms test_labels test_formats)
if(FASTATOMS_WITH_VASP)
    list(APPEND FASTATOMS_CORE_TESTS test_vasp)
endif()
if(FASTATOMS_WITH_XYZ)
    list(APPEND FASTATOMS_CORE_TESTS test_xyz)
endif()
if(FASTATOMS_WITH_CIF)
    list(APPEND FASTATOMS_CORE_TESTS test_cif)
endif()
if(FASTATOMS_WITH_LAMMPS)
    list(APPEND FASTATOMS_CORE_TESTS test_lammps)
endif()
if(FASTATOMS_WITH_NEIGHBORS)
    list(APPEND FASTATOMS_CORE_TESTS test_neighbors)
endif()

if(FASTATOMS_WITH_SYMMETRY)
    list(APPEND FASTATOMS_CORE_TESTS test_symmetry)
endif()

foreach(name IN LISTS FASTATOMS_CORE_TESTS)
    # util/text (file helpers) is not part of the library: the readers take
    # strings, and file streams would bloat the WASM build. Tests use it.
    add_executable(${name} ${name}.cpp ${CMAKE_SOURCE_DIR}/src/util/text.cpp)
    target_link_libraries(${name} PRIVATE fastatoms)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(${name} PRIVATE
        FASTATOMS_TEST_DATA="${CMAKE_SOURCE_DIR}/tests/data")
    add_test(NAME ${name} COMMAND ${name})
endforeach()
