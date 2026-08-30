# SPDX-License-Identifier: MIT
"""Tests for the calculator module.

The DFT engines themselves are not available here, so the parsers run against
representative excerpts of each code's output and the full write/execute/parse
pipeline runs against a stand-in engine: a shell script that drops a canned
output file in place.
"""

from __future__ import annotations

import os
import stat

import numpy as np
import pytest

import fastatoms as fa

RYDBERG_IN_EV = 13.605693122994
BOHR_IN_ANGSTROM = 0.529177210903
KBAR_TO_EV_PER_A3 = 0.1 / 160.21766208


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
OUTCAR = """
 vasp.6.3.2 18Mar2022 complex

--------------------------------------- Iteration      1(   9)  --------------

 POSITION                                       TOTAL-FORCE (eV/Angst)
 ------------------------------------------------------------------------------
      0.00000      0.00000      0.00000        -0.012345      0.023456   -0.034567
      1.35750      1.35750      1.35750         0.012345     -0.023456    0.034567
 ------------------------------------------------------------------------------
    total drift:                               -0.000000     -0.000000    0.000000

  in kB     -12.34567   -23.45678   -34.56789     1.23456     2.34567     3.45678
  external pressure =      -23.46 kB  Pullay stress =        0.00 kB

  free  energy   TOTEN  =       -10.84567890 eV

  energy  without entropy=      -10.83456789  energy(sigma->0) =      -10.84012340

 E-fermi :   5.67890     XC(G=0):  -9.12340     alpha+bet : -0.56780
 number of electron       8.0000000 magnetization       0.0000000

 reached required accuracy - stopping structural energy minimisation
"""

PW_OUTPUT = """
     Program PWSCF v.7.1 starts

     the Fermi energy is     6.2345678 ev

!    total energy              =     -22.79981270 Ry
     smearing contrib. (-TS)   =      -0.00123000 Ry

     convergence has been achieved in   8 iterations

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.00100000    0.00200000    0.00300000
     atom    2 type  1   force =    -0.00100000   -0.00200000   -0.00300000

     Total force =     0.005292     Total SCF correction =     0.000001

          total   stress  (Ry/bohr**3)                   (kbar)     P=      -12.34
   0.00010000   0.00002000   0.00003000           14.71        2.94        4.41
   0.00002000   0.00020000   0.00004000            2.94       29.42        5.88
   0.00003000   0.00004000   0.00030000            4.41        5.88       44.13

     JOB DONE.
"""

SIESTA_OUTPUT = """
siesta: Atomic forces (eV/Ang):
     1    0.010000    0.020000    0.030000
     2   -0.010000   -0.020000   -0.030000
----------------------------------------

siesta: Stress tensor (static) (eV/Ang**3):
        -0.001000     0.000100     0.000200
         0.000100    -0.002000     0.000300
         0.000200     0.000300    -0.003000

siesta: Etot    =      -215.245972
siesta: FreeEng =      -215.246500

siesta: Fermi =        -3.123456
siesta: Job completed
"""


def silicon() -> fa.Atoms:
    return fa.Atoms("Si2", [[0.0, 0.0, 0.0], [1.3575] * 3], cell=5.43, pbc=True)


def write_stub_engine(directory, script_name, output_name, content):
    """A shell script that stands in for a DFT binary."""
    reference = directory / (output_name + ".reference")
    reference.write_text(content)
    script = directory / script_name
    script.write_text(
        "#!/bin/sh\n"
        f'echo "stub engine in $(pwd)"\n'
        f'cp "{output_name}.reference" "{output_name}"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


# --------------------------------------------------------------------------
# The interface, via the analytic reference calculator
# --------------------------------------------------------------------------
def test_calculator_attaches_to_atoms():
    atoms = fa.Atoms("Ar2", [[0, 0, 0], [4.0, 0, 0]], cell=15.0, pbc=True)
    assert atoms.calc is None

    atoms.calc = fa.LennardJones(epsilon=0.01, sigma=3.0, cutoff=10.0, shift=False)
    assert atoms.calc is not None
    assert atoms.calc.name == "lennard_jones"

    energy = atoms.get_potential_energy()
    ratio6 = (3.0 / 4.0) ** 6
    assert energy == pytest.approx(4 * 0.01 * (ratio6**2 - ratio6))

    forces = atoms.get_forces()
    assert forces.shape == (2, 3)
    assert forces[0, 0] == pytest.approx(-forces[1, 0])

    stress = atoms.get_stress()
    assert stress.shape == (6,)


def test_calculator_can_be_named():
    atoms = silicon()
    atoms.calc = "lennard_jones"
    assert atoms.calc.name == "lennard_jones"
    atoms.calc = None
    assert atoms.calc is None

    with pytest.raises(TypeError):
        atoms.calc = 42


def test_results_are_cached_until_the_structure_changes():
    atoms = silicon()
    atoms.calc = fa.LennardJones()

    assert atoms.calc.call_count == 0
    atoms.get_potential_energy()
    atoms.get_forces()
    atoms.get_stress()
    assert atoms.calc.call_count == 1

    atoms.positions[0, 0] += 0.05
    atoms.get_potential_energy()
    assert atoms.calc.call_count == 2

    atoms.calc.set("epsilon", 0.02)
    atoms.get_potential_energy()
    assert atoms.calc.call_count == 3


def test_results_object():
    atoms = silicon()
    atoms.calc = fa.LennardJones()
    atoms.get_potential_energy()

    results = atoms.calc.results
    assert results.energy is not None
    assert results.free_energy == pytest.approx(results.energy)
    assert results.forces.shape == (2, 3)
    assert results.stress.shape == (6,)
    assert results.stress_matrix.shape == (3, 3)
    assert results.stress_matrix == pytest.approx(fa.voigt_to_matrix(results.stress))
    assert results.pressure == pytest.approx(fa.pressure_from_stress(results.stress))
    assert results.charges is None
    assert results.magmom is None
    assert "Results(" in repr(results)


def test_forces_are_stored_on_the_structure():
    atoms = silicon()
    atoms.calc = fa.LennardJones()
    forces = atoms.get_forces()

    assert atoms.has_array("forces")
    assert atoms.get_array("forces") == pytest.approx(forces)
    # Per-atom results follow the structure through slicing.
    assert atoms[[0]].get_array("forces") == pytest.approx(forces[:1])


def test_copy_drops_the_calculator():
    atoms = silicon()
    atoms.calc = fa.LennardJones()
    assert atoms.copy().calc is None


def test_missing_calculator_and_property_errors():
    atoms = silicon()
    with pytest.raises(fa.CalculatorError):
        atoms.get_potential_energy()
    assert issubclass(fa.CalculatorError, RuntimeError)

    atoms.calc = fa.LennardJones()
    with pytest.raises(fa.CalculatorError):
        atoms.calc.calculate(atoms, ["charges"])
    with pytest.raises(fa.CalculatorError):
        atoms.calc.calculate(atoms, ["not_a_property"])


def test_properties_accept_strings_and_enums():
    atoms = silicon()
    atoms.calc = fa.LennardJones()
    atoms.calc.calculate(atoms, "energy")
    atoms.calc.calculate(atoms, [fa.Property.forces, "stress"])
    assert atoms.calc.call_count == 1
    assert atoms.calc.implemented_properties == [
        "energy",
        "free_energy",
        "forces",
        "stress",
    ]


# --------------------------------------------------------------------------
# VASP
# --------------------------------------------------------------------------
def test_vasp_input_generation():
    calc = fa.Vasp(xc="pbe", encut=420, ismear=0, sigma=0.05, lwave=False, kpts=[4, 4, 2])

    incar = calc.incar_string()
    assert "GGA = PE" in incar
    assert "ENCUT = 420" in incar
    assert "LWAVE = .FALSE." in incar
    assert "KPTS" not in incar

    kpoints = calc.kpoints_string()
    assert "Gamma" in kpoints
    assert "4 4 2" in kpoints

    poscar = calc.poscar_string(silicon())
    lines = poscar.splitlines()
    assert lines[0].startswith("Si2")
    assert "Si" in lines[5]
    assert lines[7].strip() == "Direct"
    assert len(lines) == 10


def test_vasp_species_grouping():
    atoms = fa.Atoms(["O", "H", "O", "H", "H"], np.zeros((5, 3)), cell=10.0)
    layout = fa.vasp_species_layout(atoms)
    assert layout["symbols"] == ["O", "H"]
    assert layout["counts"] == [2, 3]
    assert layout["order"] == [0, 2, 1, 3, 4]


def test_vasp_outcar_parsing():
    results = fa.Vasp.parse_outcar(OUTCAR)

    assert results.energy == pytest.approx(-10.84012340)
    assert results.free_energy == pytest.approx(-10.84567890)
    assert results.forces == pytest.approx(
        np.array([[-0.012345, 0.023456, -0.034567], [0.012345, -0.023456, 0.034567]])
    )
    # kB -> eV/A^3, sign flipped, reordered from XX YY ZZ XY YZ ZX to Voigt.
    assert results.stress == pytest.approx(
        np.array([12.34567, 23.45678, 34.56789, -2.34567, -3.45678, -1.23456])
        * KBAR_TO_EV_PER_A3
    )
    assert results.extra["fermi_energy"] == pytest.approx(5.67890)
    assert results.extra["converged"] is True


def test_vasp_end_to_end(tmp_path):
    write_stub_engine(tmp_path, "stub_vasp.sh", "OUTCAR", OUTCAR)

    atoms = silicon()
    atoms.calc = fa.Vasp(
        directory=str(tmp_path),
        command="sh stub_vasp.sh",
        write_potcar=False,
        encut=300,
        kpts=4,
    )

    assert atoms.get_potential_energy() == pytest.approx(-10.84012340)
    assert (tmp_path / "POSCAR").is_file()
    assert (tmp_path / "INCAR").is_file()
    assert (tmp_path / "KPOINTS").is_file()

    assert atoms.get_forces()[0] == pytest.approx([-0.012345, 0.023456, -0.034567])
    assert atoms.get_stress()[0] == pytest.approx(12.34567 * KBAR_TO_EV_PER_A3)
    assert atoms.calc.call_count == 1
    assert atoms.calc.last_process["ok"] is True


def test_vasp_failure_is_reported(tmp_path):
    atoms = silicon()
    atoms.calc = fa.Vasp(
        directory=str(tmp_path),
        command="echo 'ERROR: ZBRENT: fatal error' >&2; exit 1",
        write_potcar=False,
    )
    with pytest.raises(fa.CalculatorError) as caught:
        atoms.get_potential_energy()
    assert "exit code 1" in str(caught.value)
    assert "ZBRENT" in str(caught.value)


# --------------------------------------------------------------------------
# Quantum ESPRESSO
# --------------------------------------------------------------------------
def test_espresso_namelist_routing():
    assert fa.Espresso.namelist_for("calculation") == "control"
    assert fa.Espresso.namelist_for("ecutwfc") == "system"
    assert fa.Espresso.namelist_for("conv_thr") == "electrons"
    assert fa.Espresso.namelist_for("press") == "cell"
    assert fa.QuantumEspresso is fa.Espresso


def test_espresso_input_generation():
    calc = fa.Espresso(
        pseudopotentials={"Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF"},
        pseudo_dir="/opt/pseudos",
        ecutwfc=40.0,
        conv_thr=1e-8,
        kpts=[4, 4, 4],
    )
    deck = calc.input_string(silicon())

    assert "&CONTROL" in deck
    assert "pseudo_dir           = '/opt/pseudos'" in deck
    assert "ecutwfc              = 40" in deck
    assert "conv_thr             = 1e-08" in deck
    assert "ATOMIC_SPECIES" in deck
    assert "CELL_PARAMETERS angstrom" in deck
    assert "K_POINTS automatic" in deck

    with pytest.raises(fa.CalculatorError):
        fa.Espresso().input_string(silicon())


def test_espresso_output_parsing():
    results = fa.Espresso.parse_output_text(PW_OUTPUT)

    assert results.energy == pytest.approx(-22.79981270 * RYDBERG_IN_EV)
    force_scale = RYDBERG_IN_EV / BOHR_IN_ANGSTROM
    assert results.forces[0] == pytest.approx(np.array([0.001, 0.002, 0.003]) * force_scale)

    stress_scale = -RYDBERG_IN_EV / BOHR_IN_ANGSTROM**3
    assert results.stress == pytest.approx(
        np.array([1e-4, 2e-4, 3e-4, 4e-5, 3e-5, 2e-5]) * stress_scale
    )
    assert results.extra["scf_iterations"] == 8
    assert results.extra["job_done"] is True


def test_espresso_end_to_end(tmp_path):
    write_stub_engine(tmp_path, "stub_pw.sh", "pw.out", PW_OUTPUT)

    atoms = silicon()
    atoms.calc = fa.Espresso(
        directory=str(tmp_path),
        command="sh stub_pw.sh",
        pseudopotentials={"Si": "Si.upf"},
        ecutwfc=30.0,
    )
    assert atoms.get_potential_energy() == pytest.approx(-22.79981270 * RYDBERG_IN_EV)
    assert "ecutwfc              = 30" in (tmp_path / "pw.in").read_text()


# --------------------------------------------------------------------------
# SIESTA
# --------------------------------------------------------------------------
def test_siesta_input_generation():
    calc = fa.Siesta(label="run", xc="pbe", mesh_cutoff=250.0, kpts=[2, 2, 3], dm_tolerance=1e-4)
    fdf = calc.fdf_string(silicon())

    assert "SystemLabel     run" in fdf
    assert "NumberOfAtoms   2" in fdf
    assert "%block ChemicalSpeciesLabel" in fdf
    assert "XC.functional  GGA" in fdf
    assert "XC.authors     PBE" in fdf
    assert "MeshCutoff     250.0 Ry" in fdf
    assert "dm.tolerance  0.0001" in fdf
    assert "%block kgrid_Monkhorst_Pack" in fdf
    assert calc.command == "siesta < run.fdf > run.out"


def test_siesta_output_parsing():
    results = fa.Siesta.parse_output_text(SIESTA_OUTPUT)
    assert results.energy == pytest.approx(-215.245972)
    assert results.free_energy == pytest.approx(-215.246500)
    assert results.forces[0] == pytest.approx([0.01, 0.02, 0.03])
    assert results.stress == pytest.approx([-0.001, -0.002, -0.003, 0.0003, 0.0002, 0.0001])


def test_siesta_end_to_end(tmp_path):
    (tmp_path / "Si.psf").write_text("fake pseudopotential\n")
    write_stub_engine(tmp_path, "stub_siesta.sh", "siesta.out", SIESTA_OUTPUT)

    atoms = silicon()
    atoms.calc = fa.Siesta(
        directory=str(tmp_path), command="sh stub_siesta.sh", mesh_cutoff=200.0
    )
    assert atoms.get_potential_energy() == pytest.approx(-215.245972)
    assert (tmp_path / "siesta.fdf").is_file()


def test_siesta_missing_pseudopotential(tmp_path):
    atoms = silicon()
    atoms.calc = fa.Siesta(directory=str(tmp_path), command="true")
    with pytest.raises(fa.CalculatorError) as caught:
        atoms.get_potential_energy()
    assert "no pseudopotential for Si" in str(caught.value)


# --------------------------------------------------------------------------
# Staging, post-processing and the registry
# --------------------------------------------------------------------------
def test_inputs_can_be_staged_without_running(tmp_path):
    calc = fa.Espresso(pseudopotentials={"Si": "Si.upf"}, command="false")
    calc.write_inputs(silicon(), str(tmp_path))
    assert (tmp_path / "pw.in").is_file()
    assert calc.call_count == 0


def test_a_finished_run_can_be_parsed_afterwards(tmp_path):
    (tmp_path / "OUTCAR").write_text(OUTCAR)
    calc = fa.Vasp()
    calc.read_results(str(tmp_path))
    assert calc.results.energy == pytest.approx(-10.84012340)
    assert calc.call_count == 0


def test_registry():
    names = set(fa.registered_calculators())
    # The engines that write input files and run a program, plus the classical
    # force field...
    assert {"vasp", "espresso", "siesta", "lennard_jones", "forcefield"} <= names
    # ...and the machine-learning families, covered in detail by test_mlip.py.
    assert {"mace", "mattersim", "chgnet", "nequip", "allegro", "mlip"} <= names
    assert fa.make_calculator("VASP").name == "vasp"
    assert fa.make_calculator("quantum-espresso").name == "espresso"
    assert fa.make_calculator("lj", epsilon=0.5).parameters["epsilon"] == 0.5
    with pytest.raises(fa.CalculatorError):
        fa.make_calculator("gaussian")


def test_custom_calculator_registered_from_python():
    fa.register_calculator("python_lj", lambda params: fa.LennardJones(**params))
    calc = fa.make_calculator("python_lj", epsilon=0.25)
    assert calc.parameters["epsilon"] == 0.25

    atoms = silicon()
    atoms.calc = calc
    assert atoms.get_potential_energy() != 0.0


# --------------------------------------------------------------------------
# The YAML path
# --------------------------------------------------------------------------
DFT_INPUT = """
structure:
  cell: {a: 5.43, b: 5.43, c: 5.43}
  coordinates: fractional
  symbols: [Si, Si]
  positions: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]

simulation:
  task: single_point
  calculator:
    name: vasp
    parameters:
      xc: pbe
      encut: 400
      kpts: [4, 4, 4]
      write_potcar: false
"""


def test_calculator_comes_from_the_input_file():
    deck = fa.read_input_string(DFT_INPUT)
    assert deck.simulation.calculator.name == "vasp"
    assert deck.simulation.calculator.parameters["encut"] == 400

    calc = deck.make_calculator()
    assert calc.name == "vasp"
    assert "ENCUT = 400" in calc.incar_string()

    # Keyword arguments override the file, which is how one deck drives a series.
    override = deck.make_calculator(encut=500, directory="run-02")
    assert "ENCUT = 500" in override.incar_string()
    assert override.directory == "run-02"

    # The same calculator can also be built through the registry.
    assert fa.make_calculator(deck.simulation.calculator).name == "vasp"


def test_full_yaml_to_energy_round_trip(tmp_path):
    write_stub_engine(tmp_path, "stub_vasp.sh", "OUTCAR", OUTCAR)

    deck = fa.read_input_string(DFT_INPUT)
    atoms = deck.atoms
    atoms.calc = deck.make_calculator(directory=str(tmp_path), command="sh stub_vasp.sh")

    assert atoms.get_potential_energy() == pytest.approx(-10.84012340)
    assert atoms.get_forces().shape == (2, 3)
    assert (tmp_path / "POSCAR").is_file()


def test_command_resolution_and_environment(tmp_path, monkeypatch):
    assert fa.Vasp().command == "vasp_std"
    assert fa.Vasp(command="mpirun -np 4 vasp_gam").command == "mpirun -np 4 vasp_gam"

    monkeypatch.setenv("VASP_COMMAND", "srun vasp_std")
    assert fa.Vasp().command == "srun vasp_std"
    # An explicit parameter still wins over the environment.
    assert fa.Vasp(command="vasp_ncl").command == "vasp_ncl"


def test_child_environment_is_passed_through(tmp_path):
    script = tmp_path / "stub.sh"
    script.write_text('#!/bin/sh\necho "PP=$MY_PSEUDO_DIR" > OUTCAR\n')

    calc = fa.Vasp(
        directory=str(tmp_path),
        command="sh stub.sh",
        write_potcar=False,
        environment={"MY_PSEUDO_DIR": "/opt/pseudos"},
    )
    atoms = silicon()
    atoms.calc = calc
    # The stub writes an OUTCAR with no energy, so parsing fails - but only
    # after the variable has been shown to reach the child.
    with pytest.raises(fa.CalculatorError):
        atoms.get_potential_energy()
    assert "PP=/opt/pseudos" in (tmp_path / "OUTCAR").read_text()
