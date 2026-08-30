# SPDX-License-Identifier: MIT
add_executable(fastatoms_run run_input.cpp)
target_link_libraries(fastatoms_run PRIVATE fastatoms_core)
set_target_properties(fastatoms_run PROPERTIES OUTPUT_NAME fastatoms-run)

if(NOT MSVC)
    target_compile_options(fastatoms_run PRIVATE -Wall -Wextra)
endif()

if(FASTATOMS_INSTALL)
    install(TARGETS fastatoms_run RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
