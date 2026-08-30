# SPDX-License-Identifier: MIT
add_executable(fastatoms_run run_input.cpp)
target_link_libraries(fastatoms_run PRIVATE fastatoms_core)
set_target_properties(fastatoms_run PROPERTIES OUTPUT_NAME fastatoms-run)

add_executable(fastatoms_interpolate interpolate.cpp)
target_link_libraries(fastatoms_interpolate PRIVATE fastatoms_core)
set_target_properties(fastatoms_interpolate PROPERTIES OUTPUT_NAME fastatoms-interpolate)

if(NOT MSVC)
    target_compile_options(fastatoms_run PRIVATE -Wall -Wextra)
    target_compile_options(fastatoms_interpolate PRIVATE -Wall -Wextra)
endif()

if(FASTATOMS_INSTALL)
    install(TARGETS fastatoms_run fastatoms_interpolate
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
