#!/bin/sh
# Run ASE's I/O tests against fastatoms.
#
#   tests/ase_compat/run.sh            # all three modes, one summary each
#   tests/ase_compat/run.sh read       # one mode: read | write | both
#   tests/ase_compat/run.sh none       # ASE's own code, no patches (baseline)
#   tests/ase_compat/run.sh read -x    # extra arguments go to pytest
#
# ASE's tests are collected from the installed package (--pyargs). No ASE file
# is copied into the repo. The plugin only swaps functions at runtime.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

MODULES="
ase.test.fio.test_vasp_structure
ase.test.fio.vasp.test_poscar
ase.test.fio.vasp.test_vasp_poscar
ase.test.fio.test_cif
ase.test.fio.test_extxyz
ase.test.fio.test_xyz
ase.test.lammpsdata.test_lammpsdata_read
ase.test.lammpsdata.test_lammpsdata_write
ase.test.lammpsdata.test_write_and_read
ase.test.lammpsdata.test_read_connectivity
ase.test.fio.test_lammpsdump
"

run_mode() {
    mode=$1
    shift
    # shellcheck disable=SC2086 # MODULES is a deliberate word list
    # PYTHONPATH makes the repo root importable so `-p tests.ase_compat.plugin` resolves.
    PYTHONPATH="$ROOT" uv run pytest -p tests.ase_compat.plugin --fa-mode="$mode" \
        --pyargs $MODULES -q "$@"
}

case "${1:-all}" in
    all)
        shift 2>/dev/null || true
        for m in read write both; do run_mode "$m" "$@" || true; done
        ;;
    read|write|both|none)
        mode=$1
        shift
        run_mode "$mode" "$@"
        ;;
    *)
        echo "usage: $0 [all|read|write|both|none] [pytest args]" >&2
        exit 2
        ;;
esac
