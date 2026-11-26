# SPDX-License-Identifier: MIT
#
# External libraries for the slim core. Only the pieces we use are compiled:
#
#   gemmi  - header-only CIF parser + small-structure model; we compile
#            symmetry.cpp and sprintf.cpp ourselves (no zlib, no mmCIF).
#   vesin  - neighbour lists; we compile its CPU sources only (no CUDA,
#            no gpulite download).
#   spglib - space groups, standardisation (all C sources, no Fortran/Python).
#   pybind11 - only for the Python test module.
#
# Each is taken from FASTATOMS_<NAME>_SOURCE_DIR when set (an existing
# checkout), else downloaded with FetchContent. FETCHCONTENT_FULLY_DISCONNECTED=ON
# forbids downloads.

include(FetchContent)

set(FASTATOMS_GEMMI_TAG "v0.7.5" CACHE STRING "gemmi git tag")
set(FASTATOMS_VESIN_TAG "v0.6.2" CACHE STRING "vesin git tag")
set(FASTATOMS_PYBIND11_TAG "v3.1.0" CACHE STRING "pybind11 git tag")
set(FASTATOMS_SPGLIB_TAG "v2.8.0" CACHE STRING "spglib git tag")

# Fetch a source tree without running its own CMakeLists.
function(fastatoms_fetch_sources name repo tag out_var)
    string(TOUPPER "${name}" upper)
    if(FASTATOMS_${upper}_SOURCE_DIR)
        set(${out_var} "${FASTATOMS_${upper}_SOURCE_DIR}" PARENT_SCOPE)
        return()
    endif()
    FetchContent_Declare(${name}
        GIT_REPOSITORY ${repo}
        GIT_TAG        ${tag}
        GIT_SHALLOW    TRUE)
    FetchContent_GetProperties(${name})
    if(NOT ${name}_POPULATED)
        if(POLICY CMP0169)
            cmake_policy(SET CMP0169 OLD)  # FetchContent_Populate without add_subdirectory
        endif()
        FetchContent_Populate(${name})
    endif()
    set(${out_var} "${${name}_SOURCE_DIR}" PARENT_SCOPE)
endfunction()

if(FASTATOMS_WITH_CIF)
    fastatoms_fetch_sources(gemmi https://github.com/project-gemmi/gemmi.git
                            ${FASTATOMS_GEMMI_TAG} GEMMI_DIR)
    add_library(fastatoms_gemmi STATIC
        ${GEMMI_DIR}/src/symmetry.cpp
        ${GEMMI_DIR}/src/sprintf.cpp)
    target_include_directories(fastatoms_gemmi SYSTEM PUBLIC ${GEMMI_DIR}/include)
    target_compile_features(fastatoms_gemmi PUBLIC cxx_std_17)
    set_target_properties(fastatoms_gemmi PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(FASTATOMS_WITH_NEIGHBORS)
    fastatoms_fetch_sources(vesin https://github.com/Luthaf/vesin.git
                            ${FASTATOMS_VESIN_TAG} VESIN_DIR)
    set(_v ${VESIN_DIR}/vesin/src)
    # vesin.cpp reports errors in vesin_free through std::cerr; one reachable
    # stream object pulls the whole iostream/locale machinery (~45 KB) into
    # the WASM build. Compile a copy that uses fprintf instead. If upstream
    # changes and the patch no longer applies, the original is used.
    file(READ ${_v}/vesin.cpp _vesin_src)
    set(_vesin_patched "${_vesin_src}")
    string(REPLACE "#include <iostream>" "#include <cstdio>" _vesin_patched "${_vesin_patched}")
    string(REPLACE "std::cerr << \"error in vesin_free: \" << e.what() << std::endl;"
                   "std::fprintf(stderr, \"error in vesin_free: %s\\n\", e.what());"
                   _vesin_patched "${_vesin_patched}")
    string(REPLACE "std::cerr << \"fatal error in vesin_free, unknown type thrown as exception\" << std::endl;"
                   "std::fputs(\"fatal error in vesin_free, unknown type thrown as exception\\n\", stderr);"
                   _vesin_patched "${_vesin_patched}")
    if(_vesin_patched MATCHES "std::cerr")
        message(WARNING "vesin: could not patch out std::cerr; using the original vesin.cpp")
        set(_vesin_main ${_v}/vesin.cpp)
    else()
        set(_vesin_main ${CMAKE_CURRENT_BINARY_DIR}/vesin_patched/vesin.cpp)
        file(CONFIGURE OUTPUT ${_vesin_main} CONTENT "${_vesin_patched}" @ONLY)
    endif()
    add_library(fastatoms_vesin STATIC
        ${_vesin_main}
        ${_v}/threadpool.cpp
        ${_v}/cpu_cell_list.cpp
        ${_v}/verlet.cpp
        ${_v}/vesin_cuda_stub.cpp)
    target_include_directories(fastatoms_vesin SYSTEM PUBLIC ${VESIN_DIR}/vesin/include)
    target_include_directories(fastatoms_vesin PRIVATE ${_v})
    target_compile_features(fastatoms_vesin PUBLIC cxx_std_17)
    # Static use: no dllexport/visibility games.
    target_compile_definitions(fastatoms_vesin PUBLIC VESIN_STATIC_DEFINE)
    set_target_properties(fastatoms_vesin PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(NOT EMSCRIPTEN)
        find_package(Threads REQUIRED)
        target_link_libraries(fastatoms_vesin PRIVATE Threads::Threads)
    endif()
endif()

if(FASTATOMS_BUILD_PYTHON)
    find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE _pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _pybind11_rc ERROR_QUIET)
        if(_pybind11_rc EQUAL 0 AND _pybind11_dir)
            find_package(pybind11 CONFIG QUIET PATHS "${_pybind11_dir}" NO_DEFAULT_PATH)
        endif()
    endif()
    if(NOT pybind11_FOUND)
        FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        ${FASTATOMS_PYBIND11_TAG}
            GIT_SHALLOW    TRUE)
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

if(FASTATOMS_WITH_SYMMETRY)
    fastatoms_fetch_sources(spglib https://github.com/spglib/spglib.git
                            ${FASTATOMS_SPGLIB_TAG} SPGLIB_DIR)
    # spglib's own CMake pulls in git-hash and packaging machinery; its C
    # sources only need version.h, which we configure here.
    set(Spglib_VERSION_MAJOR 2)
    set(Spglib_VERSION_MINOR 8)
    set(Spglib_VERSION_PATCH 0)
    set(Spglib_VERSION 2.8.0)
    set(Spglib_VERSION_FULL 2.8.0)
    set(Spglib_COMMIT "${FASTATOMS_SPGLIB_TAG}")
    configure_file(${SPGLIB_DIR}/src/version.h.in
                   ${CMAKE_CURRENT_BINARY_DIR}/spglib_generated/version.h)
    file(GLOB _spglib_sources ${SPGLIB_DIR}/src/*.c)
    list(FILTER _spglib_sources EXCLUDE REGEX "spglib_f\\.c$")  # Fortran wrapper
    add_library(fastatoms_spglib STATIC ${_spglib_sources})
    target_include_directories(fastatoms_spglib SYSTEM PUBLIC ${SPGLIB_DIR}/include)
    target_include_directories(fastatoms_spglib PRIVATE
        ${SPGLIB_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/spglib_generated)
    target_compile_definitions(fastatoms_spglib PRIVATE SPG_BUILD)
    set_target_properties(fastatoms_spglib PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(NOT MSVC)
        target_link_libraries(fastatoms_spglib PUBLIC m)
    endif()
endif()
