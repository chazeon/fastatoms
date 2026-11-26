#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Build the embind module and copy it into js/dist/wasm/.
#
#   scripts/build-wasm.sh [--full | --minimal | --gemmy] [-DOPTION=VALUE ...]
#
# Presets (feature switches of the top-level CMakeLists.txt):
#   --full     (default) every format, the neighbour list and symmetry (spglib)
#   --minimal  VASP + XYZ + neighbour list; no CIF (gemmi), LAMMPS or symmetry
#   --gemmy    VASP + XYZ + CIF + neighbour list; no LAMMPS or symmetry
# Any -D argument is passed to CMake after the preset, so it wins.
#
# Environment:
#   EMSDK                          emsdk checkout (default ~/emsdk); its
#                                  emsdk_env.sh is sourced when emcmake is not
#                                  on PATH.
#   FASTATOMS_GEMMI_SOURCE_DIR     existing gemmi checkout (skips the download)
#   FASTATOMS_VESIN_SOURCE_DIR     existing vesin checkout (skips the download)
#   FASTATOMS_SPGLIB_SOURCE_DIR    existing spglib checkout (skips the download)
#
# Output: js/dist/wasm/fastatoms.mjs, its types fastatoms.d.mts (copied from
# js/src/wasm/) and fastatoms.wasm (only fastatoms.mjs with
# -DFASTATOMS_WASM_SINGLE_FILE=ON). The CMake build tree is
# <repo>/build/wasm-<preset>. The sizes of the .wasm are printed at the end.

set -euo pipefail

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//' | sed '/^set -euo/,$d'
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$(cd "$JS_DIR/.." && pwd)"

PRESET="full"
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --full) PRESET="full" ;;
        --minimal) PRESET="minimal" ;;
        --gemmy) PRESET="gemmy" ;;
        -h|--help) usage; exit 0 ;;
        -D*) EXTRA+=("$1") ;;
        *)
            echo "build-wasm.sh: unknown argument '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

# Emscripten toolchain.
if ! command -v emcmake >/dev/null 2>&1; then
    EMSDK_DIR="${EMSDK:-$HOME/emsdk}"
    if [[ ! -f "$EMSDK_DIR/emsdk_env.sh" ]]; then
        echo "build-wasm.sh: emcmake not on PATH and no emsdk at $EMSDK_DIR" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$EMSDK_DIR/emsdk_env.sh" >/dev/null
fi

# Feature switches per preset.
case "$PRESET" in
    full)
        FEATURES=(
            -DFASTATOMS_WITH_VASP=ON
            -DFASTATOMS_WITH_XYZ=ON
            -DFASTATOMS_WITH_CIF=ON
            -DFASTATOMS_WITH_LAMMPS=ON
            -DFASTATOMS_WITH_NEIGHBORS=ON
            -DFASTATOMS_WITH_SYMMETRY=ON
        )
        ;;
    minimal)
        FEATURES=(
            -DFASTATOMS_WITH_VASP=ON
            -DFASTATOMS_WITH_XYZ=ON
            -DFASTATOMS_WITH_CIF=OFF
            -DFASTATOMS_WITH_LAMMPS=OFF
            -DFASTATOMS_WITH_NEIGHBORS=ON
            -DFASTATOMS_WITH_SYMMETRY=OFF
        )
        ;;
    gemmy)
        FEATURES=(
            -DFASTATOMS_WITH_VASP=ON
            -DFASTATOMS_WITH_XYZ=ON
            -DFASTATOMS_WITH_CIF=ON
            -DFASTATOMS_WITH_LAMMPS=OFF
            -DFASTATOMS_WITH_NEIGHBORS=ON
            -DFASTATOMS_WITH_SYMMETRY=OFF
        )
        ;;
esac

# Existing checkouts of the external libraries, to avoid a download. Unset,
# CMake fetches the pinned tags (cmake/Dependencies.cmake).
GEMMI_DIR="${FASTATOMS_GEMMI_SOURCE_DIR:-}"
VESIN_DIR="${FASTATOMS_VESIN_SOURCE_DIR:-}"
SPGLIB_DIR="${FASTATOMS_SPGLIB_SOURCE_DIR:-}"
DEPS=()
[[ -n "$GEMMI_DIR" ]] && DEPS+=("-DFASTATOMS_GEMMI_SOURCE_DIR=$GEMMI_DIR")
[[ -n "$VESIN_DIR" ]] && DEPS+=("-DFASTATOMS_VESIN_SOURCE_DIR=$VESIN_DIR")
[[ -n "$SPGLIB_DIR" ]] && DEPS+=("-DFASTATOMS_SPGLIB_SOURCE_DIR=$SPGLIB_DIR")

BUILD_DIR="$REPO/build/wasm-$PRESET"

echo "build-wasm.sh: preset=$PRESET build=$BUILD_DIR"
# Later -D arguments override earlier ones, so the user's -D come last.
emcmake cmake -S "$REPO" -B "$BUILD_DIR" -G Ninja \
    -DFASTATOMS_BUILD_TESTS=OFF \
    -DFASTATOMS_BUILD_WASM=ON \
    -DCMAKE_BUILD_TYPE=MinSizeRel \
    -DFASTATOMS_WASM_SINGLE_FILE=OFF \
    ${FEATURES[@]+"${FEATURES[@]}"} \
    ${DEPS[@]+"${DEPS[@]}"} \
    ${EXTRA[@]+"${EXTRA[@]}"}

cmake --build "$BUILD_DIR" --target fastatoms_wasm --parallel

# The build tree is reused between runs, so a fastatoms.wasm from an earlier
# two-file build can still sit in it. Decide what to package from the option
# CMake resolved for this tree, not from which files happen to exist.
SINGLE_FILE_RAW=$(sed -n 's/^FASTATOMS_WASM_SINGLE_FILE:[A-Za-z]*=//p' "$BUILD_DIR/CMakeCache.txt" | head -n 1)
case "$(printf '%s' "$SINGLE_FILE_RAW" | tr '[:lower:]' '[:upper:]')" in
    ON|TRUE|YES|1) SINGLE_FILE=1 ;;
    OFF|FALSE|NO|0|"") SINGLE_FILE=0 ;;
    *)
        echo "build-wasm.sh: cannot read FASTATOMS_WASM_SINGLE_FILE from $BUILD_DIR/CMakeCache.txt" >&2
        exit 1
        ;;
esac

OUT_DIR="$JS_DIR/dist/wasm"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cp "$BUILD_DIR/fastatoms.mjs" "$OUT_DIR/"
cp "$JS_DIR/src/wasm/fastatoms.d.mts" "$OUT_DIR/"
SIZE_NOTE=""
if [[ "$SINGLE_FILE" == 1 ]]; then
    SIZE_NOTE="single-file build: no separate fastatoms.wasm"
else
    if [[ ! -f "$BUILD_DIR/fastatoms.wasm" ]]; then
        echo "build-wasm.sh: expected $BUILD_DIR/fastatoms.wasm for a two-file build" >&2
        exit 1
    fi
    cp "$BUILD_DIR/fastatoms.wasm" "$OUT_DIR/"
    WASM="$OUT_DIR/fastatoms.wasm"
    RAW=$(wc -c < "$WASM" | tr -d ' ')
    GZ=$(gzip -9c "$WASM" | wc -c | tr -d ' ')
    SIZE_NOTE="fastatoms.wasm: $RAW bytes ($(( (RAW + 512) / 1024 )) KiB), gzip -9: $GZ bytes"
fi

echo "build-wasm.sh: copied to $OUT_DIR"
echo "build-wasm.sh: $SIZE_NOTE"
