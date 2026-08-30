# SPDX-License-Identifier: MIT
#
# The `fastatoms` Python package: a pybind11 extension plus the pure-Python
# wrapper that ships alongside it.

set(FASTATOMS_BINDING_SOURCES
    bindings/module.cpp
    bindings/helpers.cpp
    bindings/bind_atoms.cpp
    bindings/bind_cell.cpp
    bindings/bind_elements.cpp
    bindings/bind_io.cpp
    bindings/bind_neighbor_list.cpp)

pybind11_add_module(fastatoms_python MODULE ${FASTATOMS_BINDING_SOURCES})

target_link_libraries(fastatoms_python PRIVATE fastatoms_core)
target_include_directories(fastatoms_python PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bindings)

set_target_properties(fastatoms_python PROPERTIES
    OUTPUT_NAME _core
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/fastatoms)

# Multi-config generators append the configuration name to the output path.
foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER ${config} config_upper)
    set_target_properties(fastatoms_python PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_BINARY_DIR}/python/fastatoms)
endforeach()

if(NOT MSVC)
    target_compile_options(fastatoms_python PRIVATE -Wall -Wextra)
endif()

# Stage the pure-Python sources next to the extension so that the build tree is
# directly importable: PYTHONPATH=<build>/python python -c "import fastatoms".
set(FASTATOMS_PY_SOURCES
    fastatoms/__init__.py
    fastatoms/py.typed)

foreach(source IN LISTS FASTATOMS_PY_SOURCES)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${source}
                   ${CMAKE_BINARY_DIR}/python/${source} COPYONLY)
endforeach()

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
set(FASTATOMS_INSTALL_PY_SOURCES ON)

if(NOT DEFINED FASTATOMS_PYTHON_INSTALL_DIR)
    if(DEFINED SKBUILD_PROJECT_NAME)
        # Building a wheel: land inside the package directory. scikit-build-core
        # copies the .py sources itself, so only the extension is installed here.
        set(FASTATOMS_PYTHON_INSTALL_DIR "${SKBUILD_PROJECT_NAME}")
        set(FASTATOMS_INSTALL_PY_SOURCES OFF)
    else()
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c
                    "import sysconfig, sys; sys.stdout.write(sysconfig.get_path('purelib'))"
            OUTPUT_VARIABLE FASTATOMS_SITE_PACKAGES
            ERROR_QUIET)
        if(FASTATOMS_SITE_PACKAGES)
            set(FASTATOMS_PYTHON_INSTALL_DIR "${FASTATOMS_SITE_PACKAGES}/fastatoms")
        else()
            set(FASTATOMS_PYTHON_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/fastatoms")
        endif()
    endif()
endif()

install(TARGETS fastatoms_python
    LIBRARY DESTINATION ${FASTATOMS_PYTHON_INSTALL_DIR})

if(FASTATOMS_INSTALL_PY_SOURCES)
    install(FILES ${FASTATOMS_PY_SOURCES}
        DESTINATION ${FASTATOMS_PYTHON_INSTALL_DIR})
endif()
