# SPDX-License-Identifier: MIT
#
# The embind module behind the `fastatoms` npm package (js/). Configured by
# `emcmake cmake -DFASTATOMS_BUILD_WASM=ON ...`; js/scripts/build-wasm.sh
# drives it. Output in the build directory: fastatoms.mjs + fastatoms.wasm
# (or only fastatoms.mjs with FASTATOMS_WASM_SINGLE_FILE=ON).

option(FASTATOMS_WASM_SINGLE_FILE "Embed the .wasm in the .mjs (-sSINGLE_FILE=1)" OFF)

if(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    set(FASTATOMS_WASM_OPT -Oz)
else()
    set(FASTATOMS_WASM_OPT -O3)
endif()

# The C++ layer throws; embind entry points catch and rethrow as JS Error.
# Wasm exceptions are smaller than the JS-based ones. Every target that can
# throw has to be compiled with them, so the dependencies are covered too.
set(FASTATOMS_WASM_EH -fwasm-exceptions)

# Link-time optimisation over the whole module (core, formats, vesin, spglib).
# Measured on the minimal preset: about -16 KiB of wasm, see js/README.md.
option(FASTATOMS_WASM_LTO "Link-time optimisation (-flto) for every wasm object" ON)
set(FASTATOMS_WASM_LTO_FLAGS "")
if(FASTATOMS_WASM_LTO)
    set(FASTATOMS_WASM_LTO_FLAGS -flto)
endif()

target_compile_options(fastatoms PRIVATE ${FASTATOMS_WASM_OPT} ${FASTATOMS_WASM_EH} ${FASTATOMS_WASM_LTO_FLAGS})
if(TARGET fastatoms_gemmi)
    target_compile_options(fastatoms_gemmi PRIVATE ${FASTATOMS_WASM_OPT} ${FASTATOMS_WASM_EH} ${FASTATOMS_WASM_LTO_FLAGS})
endif()
if(TARGET fastatoms_vesin)
    target_compile_options(fastatoms_vesin PRIVATE ${FASTATOMS_WASM_OPT} ${FASTATOMS_WASM_EH} ${FASTATOMS_WASM_LTO_FLAGS})
endif()
# spglib is plain C and never throws, so -fwasm-exceptions is harmless here; it
# is applied anyway so that every object in the link uses the same EH model.
if(TARGET fastatoms_spglib)
    target_compile_options(fastatoms_spglib PRIVATE ${FASTATOMS_WASM_OPT} ${FASTATOMS_WASM_EH} ${FASTATOMS_WASM_LTO_FLAGS})
endif()

add_executable(fastatoms_wasm bindings.cpp)
target_link_libraries(fastatoms_wasm PRIVATE fastatoms)
target_compile_features(fastatoms_wasm PRIVATE cxx_std_20)
target_compile_options(fastatoms_wasm PRIVATE ${FASTATOMS_WASM_OPT} ${FASTATOMS_WASM_EH} ${FASTATOMS_WASM_LTO_FLAGS})

set_target_properties(fastatoms_wasm PROPERTIES
    OUTPUT_NAME "fastatoms"
    SUFFIX ".mjs"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# The module needs no file system, and emmalloc is smaller than the default
# allocator (about -6 KiB on the minimal preset).
target_link_options(fastatoms_wasm PRIVATE
    ${FASTATOMS_WASM_OPT}
    ${FASTATOMS_WASM_EH}
    ${FASTATOMS_WASM_LTO_FLAGS}
    -sMALLOC=emmalloc
    -sFILESYSTEM=0
    --bind
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=createFastAtoms
    -sALLOW_MEMORY_GROWTH=1
    -sENVIRONMENT=web,node,worker
    --closure=1)

if(FASTATOMS_WASM_SINGLE_FILE)
    target_link_options(fastatoms_wasm PRIVATE -sSINGLE_FILE=1)
endif()
