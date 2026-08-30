// SPDX-License-Identifier: MIT
//
// The VASP, Quantum ESPRESSO and SIESTA interfaces.
//
// Real DFT binaries are not available here (and would take minutes per run
// anyway), so the parsers are checked against representative excerpts of each
// code's output, and the full write/execute/parse pipeline is checked against
// a stand-in engine: a shell script that drops a canned output file in place.
// That exercises everything except the physics of the DFT codes themselves.

#include <cmath>
#include <filesystem>
#include <random>
#include <string>

#include "fastatoms/calculators/quantum_espresso.hpp"
#include "fastatoms/calculators/siesta.hpp"
#include "fastatoms/calculators/vasp.hpp"
#include "fastatoms/util/text.hpp"
#include "testing.hpp"

using namespace fastatoms;

namespace {

class ScratchDirectory {
public:
    explicit ScratchDirectory(const std::string& stem) {
        static std::mt19937_64 rng(20260830);
        path_ = std::filesystem::current_path() /
                (stem + "-" + std::to_string(rng() % 1000000000ull));
        std::filesystem::create_directories(path_);
    }
    ~ScratchDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    ScratchDirectory(const ScratchDirectory&) = delete;
    ScratchDirectory& operator=(const ScratchDirectory&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

Value parameters(std::initializer_list<std::pair<const char*, Value>> entries) {
    Value map{Value::Map{}};
    for (const auto& [key, value] : entries) {
        map.insert(key, value);
    }
    return map;
}

Atoms silicon_dimer() {
    return Atoms(std::vector<std::string>{"Si", "Si"},
                 {0.0, 0.0, 0.0, 1.3575, 1.3575, 1.3575}, Cell::cubic(5.43));
}

/// Conversions written out independently of the library, so the tests check
/// the unit chain rather than restating it.
constexpr double kRydbergInEv = 13.605693122994;
constexpr double kBohrInAngstrom = 0.529177210903;
constexpr double kKbarToEvPerCubicAngstrom = 0.1 / 160.21766208;

// ---------------------------------------------------------------------------
// Fixtures: excerpts of real output, trimmed to the parts the parsers read
// ---------------------------------------------------------------------------

const char* const kOutcar = R"FIXTURE(
 vasp.6.3.2 18Mar2022 (build Apr 01 2022 12:00:00) complex
 POSCAR found :  1 types and       2 ions

--------------------------------------- Iteration      1(   9)  ---------------

 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000      0.00000      0.00000        -0.012345      0.023456     -0.034567
      1.35750      1.35750      1.35750         0.012345     -0.023456      0.034567
 -----------------------------------------------------------------------------------
    total drift:                               -0.000000     -0.000000      0.000000

  FORCE on cell =-STRESS in cart. coord.  units (eV):
  Direction    XX          YY          ZZ          XY          YZ          ZX
  ---------------------------------------------------------------------------
  Total      -1.97382    -3.75026    -5.52670     0.19738     0.37503     0.55267
  in kB     -12.34567   -23.45678   -34.56789     1.23456     2.34567     3.45678
  external pressure =      -23.46 kB  Pullay stress =        0.00 kB

  free  energy   TOTEN  =       -10.84567890 eV

  energy  without entropy=      -10.83456789  energy(sigma->0) =      -10.84012340

 E-fermi :   5.67890     XC(G=0):  -9.12340     alpha+bet : -0.56780
 number of electron       8.0000000 magnetization       0.0000000

 reached required accuracy - stopping structural energy minimisation
)FIXTURE";

const char* const kPwOutput = R"FIXTURE(
     Program PWSCF v.7.1 starts on 29Aug2026 at 23:00:00

     number of k points=    10

     the Fermi energy is     6.2345678 ev

!    total energy              =     -22.79981270 Ry
     estimated scf accuracy    <          1.2E-09 Ry
     smearing contrib. (-TS)   =      -0.00123000 Ry
     internal energy E=F+TS    =     -22.79858270 Ry

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
)FIXTURE";

const char* const kSiestaOutput = R"FIXTURE(
                           ***********************
                           *  WELCOME TO SIESTA  *
                           ***********************

siesta: E_KS(eV) =             -215.2400

siesta: Atomic forces (eV/Ang):
     1    0.010000    0.020000    0.030000
     2   -0.010000   -0.020000   -0.030000
----------------------------------------
   Tot    0.000000    0.000000    0.000000

siesta: Stress tensor (static) (eV/Ang**3):
        -0.001000     0.000100     0.000200
         0.000100    -0.002000     0.000300
         0.000200     0.000300    -0.003000

siesta: Pressure (static):        320.12345678  kBar

siesta: Program's energy decomposition (eV):
siesta: Ebs     =      -137.393352
siesta: Etot    =      -215.245972
siesta: FreeEng =      -215.246500

siesta: Final energy (eV):
siesta:       Band Struct. =    -137.393352
siesta:         Total =      -215.245972
siesta:         Fermi =        -3.123456

siesta: Job completed
)FIXTURE";

const char* const kForceStress = R"FIXTURE(   -215.245972
     -0.00003400      0.00000340      0.00000680
      0.00000340     -0.00006800      0.00001020
      0.00000680      0.00001020     -0.00010200
     2
    1   14    0.00036749    0.00073498    0.00110247
    2   14   -0.00036749   -0.00073498   -0.00110247
)FIXTURE";

/// Write a shell script that stands in for a DFT binary: it drops `content`
/// into `output_name` and exits successfully.
void write_stub_engine(const std::filesystem::path& directory, const std::string& script_name,
                       const std::string& output_name, const std::string& content) {
    util::write_text_file(directory / (output_name + ".reference"), content);
    util::write_text_file(directory / script_name,
                          "#!/bin/sh\n"
                          "echo \"stub engine running in $(pwd)\"\n"
                          "cp \"" + output_name + ".reference\" \"" + output_name + "\"\n");
}

}  // namespace

// ---------------------------------------------------------------------------
// VASP
// ---------------------------------------------------------------------------

FA_TEST(vasp_groups_atoms_by_species) {
    // Species must be contiguous in a POSCAR, so an interleaved structure is
    // reordered and the permutation recorded.
    const Atoms atoms(std::vector<std::string>{"O", "H", "O", "H", "H"},
                      std::vector<double>(15, 0.0), Cell::cubic(10.0));
    const VaspSpeciesLayout layout = vasp_species_layout(atoms);

    FA_CHECK(layout.symbols == (std::vector<std::string>{"O", "H"}));
    FA_CHECK(layout.counts == (std::vector<std::size_t>{2, 3}));
    FA_CHECK(layout.order == (std::vector<std::size_t>{0, 2, 1, 3, 4}));
}

FA_TEST(vasp_writes_a_valid_poscar) {
    const Atoms atoms = silicon_dimer();
    const VaspCalculator calculator;
    const std::string poscar =
        calculator.poscar_string(atoms, vasp_species_layout(atoms));
    const std::vector<std::string> lines = util::split_lines(poscar);

    // comment, scale, 3 lattice vectors, species, counts, "Direct", 2 atoms
    FA_CHECK(lines.size() == 10);
    FA_CHECK_CLOSE(util::scan_numbers(lines[1]).front(), 1.0, 1e-12);
    FA_CHECK_CLOSE(util::scan_numbers(lines[2])[0], 5.43, 1e-12);
    FA_CHECK(lines[5].find("Si") != std::string::npos);
    FA_CHECK_CLOSE(util::scan_numbers(lines[6]).front(), 2.0, 1e-12);
    FA_CHECK(util::trim(lines[7]) == "Direct");
}

FA_TEST(vasp_poscar_holds_fractional_coordinates_in_species_order) {
    const Atoms atoms(std::vector<std::string>{"H", "O", "H"},
                      {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0}, Cell::cubic(10.0));
    const VaspCalculator calculator;
    const VaspSpeciesLayout layout = vasp_species_layout(atoms);
    const std::vector<std::string> lines =
        util::split_lines(calculator.poscar_string(atoms, layout));

    // H first (atoms 0 and 2), then O (atom 1).
    FA_CHECK(lines[5].find("H") != std::string::npos);
    FA_CHECK_CLOSE(util::scan_numbers(lines[8])[0], 0.1, 1e-12);
    FA_CHECK_CLOSE(util::scan_numbers(lines[9])[0], 0.2, 1e-12);
    FA_CHECK_CLOSE(util::scan_numbers(lines[10])[0], 0.0, 1e-12);
}

FA_TEST(vasp_incar_and_kpoints_come_from_the_parameters) {
    const VaspCalculator calculator(parameters({
        {"xc", Value("pbe")},
        {"encut", Value(420.0)},
        {"ismear", Value(0)},
        {"sigma", Value(0.05)},
        {"lwave", Value(false)},
        {"kpts", Value(Value::List{Value(4), Value(4), Value(2)})},
        {"gamma", Value(false)},
    }));

    const std::string incar = calculator.incar_string(default_properties());
    FA_CHECK(incar.find("GGA = PE") != std::string::npos);
    FA_CHECK(incar.find("ENCUT = 420") != std::string::npos);
    FA_CHECK(incar.find("ISMEAR = 0") != std::string::npos);
    FA_CHECK(incar.find("SIGMA = 0.05") != std::string::npos);
    FA_CHECK(incar.find("LWAVE = .FALSE.") != std::string::npos);
    // Stress was requested, so ISIF must allow it.
    FA_CHECK(incar.find("ISIF = 2") != std::string::npos);
    // Interpreted parameters never leak into INCAR.
    FA_CHECK(incar.find("KPTS") == std::string::npos);
    FA_CHECK(incar.find("XC") == std::string::npos);

    const std::string kpoints = calculator.kpoints_string();
    FA_CHECK(kpoints.find("Monkhorst-Pack") != std::string::npos);
    const std::vector<std::string> lines = util::split_lines(kpoints);
    const std::vector<double> grid = util::scan_numbers(lines[3]);
    FA_CHECK(grid.size() == 3);
    FA_CHECK_CLOSE(grid[0], 4.0, 1e-12);
    FA_CHECK_CLOSE(grid[2], 2.0, 1e-12);
}

FA_TEST(vasp_kspacing_replaces_the_kpoints_file) {
    const VaspCalculator with_kspacing(parameters({{"kspacing", Value(0.25)}}));
    FA_CHECK(with_kspacing.kpoints_string().empty());
    FA_CHECK(with_kspacing.incar_string({}).find("KSPACING = 0.25") != std::string::npos);

    const VaspCalculator plain;
    FA_CHECK(plain.kpoints_string().find("Gamma") != std::string::npos);
}

FA_TEST(vasp_rejects_an_unknown_functional) {
    const VaspCalculator calculator(parameters({{"xc", Value("not-a-functional")}}));
    FA_CHECK_THROWS(calculator.incar_string({}), CalculatorError);
}

FA_TEST(vasp_parses_an_outcar) {
    const Results results = VaspCalculator::parse_outcar(kOutcar);

    FA_CHECK(results.energy.has_value());
    FA_CHECK_CLOSE(*results.energy, -10.84012340, 1e-10);
    FA_CHECK_CLOSE(*results.free_energy, -10.84567890, 1e-10);

    FA_CHECK(results.forces.size() == 6);
    FA_CHECK_CLOSE(results.forces[0], -0.012345, 1e-12);
    FA_CHECK_CLOSE(results.forces[2], -0.034567, 1e-12);
    FA_CHECK_CLOSE(results.forces[3], 0.012345, 1e-12);

    // OUTCAR prints XX YY ZZ XY YZ ZX in kB with the opposite sign to ASE;
    // the result must be eV/A^3 in Voigt order (xx, yy, zz, yz, xz, xy).
    FA_CHECK(results.stress.has_value());
    const std::array<double, 6>& stress = *results.stress;
    FA_CHECK_CLOSE(stress[0], 12.34567 * kKbarToEvPerCubicAngstrom, 1e-10);
    FA_CHECK_CLOSE(stress[1], 23.45678 * kKbarToEvPerCubicAngstrom, 1e-10);
    FA_CHECK_CLOSE(stress[2], 34.56789 * kKbarToEvPerCubicAngstrom, 1e-10);
    FA_CHECK_CLOSE(stress[3], -2.34567 * kKbarToEvPerCubicAngstrom, 1e-10);
    FA_CHECK_CLOSE(stress[4], -3.45678 * kKbarToEvPerCubicAngstrom, 1e-10);
    FA_CHECK_CLOSE(stress[5], -1.23456 * kKbarToEvPerCubicAngstrom, 1e-10);

    FA_CHECK_CLOSE(*results.magmom, 0.0, 1e-12);
    FA_CHECK_CLOSE(results.extra["fermi_energy"].as_double(), 5.67890, 1e-10);
    FA_CHECK(results.extra["converged"].as_bool());
    FA_CHECK(results.extra["ionic_steps"].as_int() == 1);
    FA_CHECK(results.extra["electronic_steps"].as_int() == 9);
}

FA_TEST(vasp_runs_end_to_end_against_a_stub_engine) {
    ScratchDirectory scratch("vasp-run");
    write_stub_engine(scratch.path(), "stub_vasp.sh", "OUTCAR", kOutcar);

    Atoms atoms = silicon_dimer();
    auto calculator = std::make_shared<VaspCalculator>(parameters({
        {"directory", Value(scratch.path().string())},
        {"command", Value("sh stub_vasp.sh")},
        {"write_potcar", Value(false)},
        {"encut", Value(300.0)},
        {"kpts", Value(4)},
    }));
    atoms.set_calculator(calculator);

    const double energy = atoms.get_potential_energy();
    FA_CHECK_CLOSE(energy, -10.84012340, 1e-10);

    // The input files really were written.
    FA_CHECK(std::filesystem::is_regular_file(scratch.path() / "POSCAR"));
    FA_CHECK(std::filesystem::is_regular_file(scratch.path() / "INCAR"));
    FA_CHECK(std::filesystem::is_regular_file(scratch.path() / "KPOINTS"));
    FA_CHECK(std::filesystem::is_regular_file(scratch.path() / "vasp.log"));

    // Forces made it back onto the structure.
    const std::vector<double> forces = atoms.get_forces();
    FA_CHECK(forces.size() == 6);
    FA_CHECK_CLOSE(forces[0], -0.012345, 1e-12);
    FA_CHECK(atoms.has_array("forces"));

    const std::array<double, 6> stress = atoms.get_stress();
    FA_CHECK_CLOSE(stress[0], 12.34567 * kKbarToEvPerCubicAngstrom, 1e-10);

    // One engine invocation covered all three properties.
    FA_CHECK(calculator->call_count() == 1);
}

FA_TEST(vasp_maps_forces_back_through_the_species_permutation) {
    ScratchDirectory scratch("vasp-order");
    // Two species interleaved: POSCAR order differs from the structure's.
    const char* const outcar = R"FIXTURE(
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.0  0.0  0.0        1.0   0.0   0.0
      0.0  0.0  0.0        2.0   0.0   0.0
      0.0  0.0  0.0        3.0   0.0   0.0
 -----------------------------------------------------------------------------------
  free  energy   TOTEN  =       -1.00000000 eV
  energy  without entropy=      -1.0  energy(sigma->0) =      -1.00000000
)FIXTURE";
    write_stub_engine(scratch.path(), "stub_vasp.sh", "OUTCAR", outcar);

    // H, O, H -> POSCAR order is H(0), H(2), O(1).
    Atoms atoms(std::vector<std::string>{"H", "O", "H"},
                {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0}, Cell::cubic(10.0));
    atoms.set_calculator(std::make_shared<VaspCalculator>(parameters({
        {"directory", Value(scratch.path().string())},
        {"command", Value("sh stub_vasp.sh")},
        {"write_potcar", Value(false)},
    })));

    const std::vector<double> forces = atoms.get_forces();
    FA_CHECK_CLOSE(forces[0], 1.0, 1e-12);  // atom 0 is POSCAR row 0
    FA_CHECK_CLOSE(forces[3], 3.0, 1e-12);  // atom 1 (O) is POSCAR row 2
    FA_CHECK_CLOSE(forces[6], 2.0, 1e-12);  // atom 2 is POSCAR row 1
}

FA_TEST(a_failing_engine_reports_the_command_and_the_log) {
    ScratchDirectory scratch("vasp-fail");

    Atoms atoms = silicon_dimer();
    atoms.set_calculator(std::make_shared<VaspCalculator>(parameters({
        {"directory", Value(scratch.path().string())},
        {"command", Value("echo 'ERROR: sub-space matrix is not hermitian' >&2; exit 1")},
        {"write_potcar", Value(false)},
    })));

    bool threw = false;
    try {
        (void)atoms.get_potential_energy();
    } catch (const CalculatorError& error) {
        threw = true;
        const std::string message = error.what();
        FA_CHECK(message.find("exit code 1") != std::string::npos);
        FA_CHECK(message.find("sub-space matrix") != std::string::npos);
    }
    FA_CHECK(threw);
}

FA_TEST(a_run_that_produces_no_output_is_reported_clearly) {
    ScratchDirectory scratch("vasp-empty");
    Atoms atoms = silicon_dimer();
    atoms.set_calculator(std::make_shared<VaspCalculator>(parameters({
        {"directory", Value(scratch.path().string())},
        {"command", Value("true")},
        {"write_potcar", Value(false)},
    })));

    bool threw = false;
    try {
        (void)atoms.get_potential_energy();
    } catch (const CalculatorError& error) {
        threw = true;
        FA_CHECK(std::string(error.what()).find("no OUTCAR") != std::string::npos);
    }
    FA_CHECK(threw);
}

FA_TEST(vasp_missing_potcar_library_is_reported_clearly) {
    ScratchDirectory scratch("vasp-potcar");
    Atoms atoms = silicon_dimer();
    atoms.set_calculator(std::make_shared<VaspCalculator>(parameters({
        {"directory", Value(scratch.path().string())},
        {"command", Value("true")},
        {"pp_path", Value((scratch.path() / "nowhere").string())},
    })));

    bool threw = false;
    try {
        (void)atoms.get_potential_energy();
    } catch (const CalculatorError& error) {
        threw = true;
        FA_CHECK(std::string(error.what()).find("no POTCAR found for Si") != std::string::npos);
    }
    FA_CHECK(threw);
}

FA_TEST(vasp_builds_a_potcar_from_a_library) {
    ScratchDirectory scratch("vasp-pp");
    const std::filesystem::path library = scratch.path() / "pp";
    util::write_text_file(library / "potpaw_PBE" / "Si" / "POTCAR", "PAW_PBE Si 05Jan2001\n");
    util::write_text_file(library / "potpaw_PBE" / "O_s" / "POTCAR", "PAW_PBE O_s 07Sep2000\n");

    const Atoms atoms(std::vector<std::string>{"Si", "O"}, std::vector<double>(6, 0.0),
                      Cell::cubic(10.0));
    VaspCalculator calculator(parameters({
        {"directory", Value(scratch.path().string())},
        {"pp_path", Value(library.string())},
        {"setups", Value(Value::Map{{"O", Value("_s")}})},
    }));
    calculator.write_inputs(atoms);

    const std::string potcar = util::read_text_file(scratch.path() / "POTCAR");
    // Concatenated in POSCAR species order.
    FA_CHECK(potcar.find("PAW_PBE Si") != std::string::npos);
    FA_CHECK(potcar.find("PAW_PBE O_s") != std::string::npos);
    FA_CHECK(potcar.find("PAW_PBE Si") < potcar.find("PAW_PBE O_s"));
}

// ---------------------------------------------------------------------------
// Quantum ESPRESSO
// ---------------------------------------------------------------------------

FA_TEST(espresso_routes_keywords_to_the_right_namelist) {
    FA_CHECK(QuantumEspressoCalculator::namelist_for("calculation") == "control");
    FA_CHECK(QuantumEspressoCalculator::namelist_for("ecutwfc") == "system");
    FA_CHECK(QuantumEspressoCalculator::namelist_for("conv_thr") == "electrons");
    FA_CHECK(QuantumEspressoCalculator::namelist_for("ion_dynamics") == "ions");
    FA_CHECK(QuantumEspressoCalculator::namelist_for("press") == "cell");
    FA_CHECK(QuantumEspressoCalculator::namelist_for("ECUTRHO") == "system");
    // Unknown keywords fall through to &SYSTEM, as in ASE.
    FA_CHECK(QuantumEspressoCalculator::namelist_for("some_new_flag") == "system");
}

FA_TEST(espresso_writes_a_complete_input_deck) {
    const Atoms atoms = silicon_dimer();
    const QuantumEspressoCalculator calculator(parameters({
        {"pseudopotentials", Value(Value::Map{{"Si", Value("Si.pbe-n-kjpaw_psl.1.0.0.UPF")}})},
        {"pseudo_dir", Value("/opt/pseudos")},
        {"ecutwfc", Value(40.0)},
        {"conv_thr", Value(1.0e-8)},
        {"occupations", Value("smearing")},
        {"kpts", Value(Value::List{Value(4), Value(4), Value(4)})},
        {"kpts_shift", Value(Value::List{Value(1), Value(1), Value(1)})},
    }));

    const std::string input = calculator.input_string(atoms, default_properties());

    FA_CHECK(input.find("&CONTROL") != std::string::npos);
    FA_CHECK(input.find("calculation          = 'scf'") != std::string::npos);
    FA_CHECK(input.find("pseudo_dir           = '/opt/pseudos'") != std::string::npos);
    FA_CHECK(input.find("tprnfor              = .true.") != std::string::npos);
    FA_CHECK(input.find("tstress              = .true.") != std::string::npos);

    FA_CHECK(input.find("&SYSTEM") != std::string::npos);
    FA_CHECK(input.find("ibrav                = 0") != std::string::npos);
    FA_CHECK(input.find("nat                  = 2") != std::string::npos);
    FA_CHECK(input.find("ntyp                 = 1") != std::string::npos);
    FA_CHECK(input.find("ecutwfc              = 40") != std::string::npos);
    FA_CHECK(input.find("occupations          = 'smearing'") != std::string::npos);

    FA_CHECK(input.find("&ELECTRONS") != std::string::npos);
    FA_CHECK(input.find("conv_thr             = 1e-08") != std::string::npos);

    FA_CHECK(input.find("ATOMIC_SPECIES") != std::string::npos);
    FA_CHECK(input.find("Si.pbe-n-kjpaw_psl.1.0.0.UPF") != std::string::npos);
    FA_CHECK(input.find("CELL_PARAMETERS angstrom") != std::string::npos);
    FA_CHECK(input.find("ATOMIC_POSITIONS crystal") != std::string::npos);
    FA_CHECK(input.find("K_POINTS automatic") != std::string::npos);
    FA_CHECK(input.find("4 4 4 1 1 1") != std::string::npos);
}

FA_TEST(espresso_only_asks_for_what_was_requested) {
    const Atoms atoms = silicon_dimer();
    const QuantumEspressoCalculator calculator(parameters(
        {{"pseudopotentials", Value(Value::Map{{"Si", Value("Si.upf")}})}}));

    const std::string energy_only = calculator.input_string(atoms, {Property::Energy});
    FA_CHECK(energy_only.find("tstress              = .false.") != std::string::npos);
    FA_CHECK(energy_only.find("tprnfor              = .false.") != std::string::npos);
    FA_CHECK(energy_only.find("K_POINTS gamma") != std::string::npos);
}

FA_TEST(espresso_requires_pseudopotentials) {
    const Atoms atoms = silicon_dimer();
    const QuantumEspressoCalculator without;
    FA_CHECK_THROWS(without.input_string(atoms, default_properties()), CalculatorError);

    const QuantumEspressoCalculator wrong_element(parameters(
        {{"pseudopotentials", Value(Value::Map{{"Ge", Value("Ge.upf")}})}}));
    FA_CHECK_THROWS(wrong_element.input_string(atoms, default_properties()), CalculatorError);
}

FA_TEST(espresso_parses_pw_output) {
    const Results results = QuantumEspressoCalculator::parse_output_text(kPwOutput);

    FA_CHECK(results.energy.has_value());
    FA_CHECK_CLOSE(*results.energy, -22.79981270 * kRydbergInEv, 1e-8);

    // Ry/bohr -> eV/angstrom
    const double force_scale = kRydbergInEv / kBohrInAngstrom;
    FA_CHECK(results.forces.size() == 6);
    FA_CHECK_CLOSE(results.forces[0], 0.001 * force_scale, 1e-10);
    FA_CHECK_CLOSE(results.forces[2], 0.003 * force_scale, 1e-10);
    FA_CHECK_CLOSE(results.forces[3], -0.001 * force_scale, 1e-10);

    // Ry/bohr^3 -> eV/angstrom^3, with the sign flipped to the ASE convention.
    const double stress_scale =
        -kRydbergInEv / (kBohrInAngstrom * kBohrInAngstrom * kBohrInAngstrom);
    FA_CHECK(results.stress.has_value());
    const std::array<double, 6>& stress = *results.stress;
    FA_CHECK_CLOSE(stress[0], 0.0001 * stress_scale, 1e-10);
    FA_CHECK_CLOSE(stress[1], 0.0002 * stress_scale, 1e-10);
    FA_CHECK_CLOSE(stress[2], 0.0003 * stress_scale, 1e-10);
    FA_CHECK_CLOSE(stress[3], 0.00004 * stress_scale, 1e-10);   // yz
    FA_CHECK_CLOSE(stress[4], 0.00003 * stress_scale, 1e-10);   // xz
    FA_CHECK_CLOSE(stress[5], 0.00002 * stress_scale, 1e-10);   // xy

    FA_CHECK_CLOSE(results.extra["fermi_energy"].as_double(), 6.2345678, 1e-9);
    FA_CHECK(results.extra["converged"].as_bool());
    FA_CHECK(results.extra["scf_iterations"].as_int() == 8);
    FA_CHECK(results.extra["job_done"].as_bool());
    FA_CHECK_CLOSE(results.extra["smearing_contribution"].as_double(),
                   -0.00123 * kRydbergInEv, 1e-9);
}

FA_TEST(espresso_runs_end_to_end_against_a_stub_engine) {
    ScratchDirectory scratch("espresso-run");
    write_stub_engine(scratch.path(), "stub_pw.sh", "pw.out", kPwOutput);

    Atoms atoms = silicon_dimer();
    auto calculator = std::make_shared<QuantumEspressoCalculator>(parameters({
        {"directory", Value(scratch.path().string())},
        {"command", Value("sh stub_pw.sh")},
        {"pseudopotentials", Value(Value::Map{{"Si", Value("Si.upf")}})},
        {"ecutwfc", Value(30.0)},
    }));
    atoms.set_calculator(calculator);

    FA_CHECK_CLOSE(atoms.get_potential_energy(), -22.79981270 * kRydbergInEv, 1e-8);
    FA_CHECK(std::filesystem::is_regular_file(scratch.path() / "pw.in"));

    const std::string written = util::read_text_file(scratch.path() / "pw.in");
    FA_CHECK(written.find("ecutwfc              = 30") != std::string::npos);
    FA_CHECK(atoms.get_forces().size() == 6);
}

// ---------------------------------------------------------------------------
// SIESTA
// ---------------------------------------------------------------------------

FA_TEST(siesta_writes_a_complete_fdf_deck) {
    const Atoms atoms(std::vector<std::string>{"Si", "O", "Si"},
                      {0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0}, Cell::cubic(10.0));
    const SiestaCalculator calculator(parameters({
        {"label", Value("quartz")},
        {"xc", Value("pbe")},
        {"mesh_cutoff", Value(250.0)},
        {"kpts", Value(Value::List{Value(2), Value(2), Value(3)})},
        {"dm_tolerance", Value(1.0e-4)},
        {"max_scf_iterations", Value(200)},
    }));

    const std::string fdf = calculator.fdf_string(atoms);

    FA_CHECK(fdf.find("SystemLabel     quartz") != std::string::npos);
    FA_CHECK(fdf.find("NumberOfAtoms   3") != std::string::npos);
    FA_CHECK(fdf.find("NumberOfSpecies 2") != std::string::npos);
    FA_CHECK(fdf.find("%block ChemicalSpeciesLabel") != std::string::npos);
    FA_CHECK(fdf.find("1  14  Si") != std::string::npos);
    FA_CHECK(fdf.find("2  8  O") != std::string::npos);
    FA_CHECK(fdf.find("%block LatticeVectors") != std::string::npos);
    FA_CHECK(fdf.find("AtomicCoordinatesFormat Fractional") != std::string::npos);
    FA_CHECK(fdf.find("%block kgrid_Monkhorst_Pack") != std::string::npos);
    FA_CHECK(fdf.find("XC.functional  GGA") != std::string::npos);
    FA_CHECK(fdf.find("XC.authors     PBE") != std::string::npos);
    FA_CHECK(fdf.find("MeshCutoff     250.0 Ry") != std::string::npos);
    // Underscored YAML keys become dotted fdf directives; fdf matching ignores
    // case and separators, so the spelling is passed through unchanged.
    FA_CHECK(fdf.find("dm.tolerance  0.0001") != std::string::npos);
    FA_CHECK(fdf.find("max.scf.iterations  200") != std::string::npos);
    FA_CHECK(fdf.find("WriteForces    .true.") != std::string::npos);

    // Species indices: Si, O, Si -> 1, 2, 1.
    const std::vector<std::string> lines = util::split_lines(fdf);
    const std::ptrdiff_t block =
        util::find_line(lines, "%block AtomicCoordinatesAndAtomicSpecies");
    FA_CHECK(block >= 0);
    FA_CHECK(util::scan_numbers(lines[static_cast<std::size_t>(block) + 1]).back() == 1.0);
    FA_CHECK(util::scan_numbers(lines[static_cast<std::size_t>(block) + 2]).back() == 2.0);
    FA_CHECK(util::scan_numbers(lines[static_cast<std::size_t>(block) + 3]).back() == 1.0);
}

FA_TEST(siesta_rejects_an_unknown_functional_and_a_missing_cell) {
    const Atoms atoms = silicon_dimer();
    const SiestaCalculator bad_xc(parameters({{"xc", Value("made-up")}}));
    FA_CHECK_THROWS(bad_xc.fdf_string(atoms), CalculatorError);

    const Atoms molecule(std::vector<std::string>{"H", "H"},
                         {0.0, 0.0, 0.0, 0.74, 0.0, 0.0});
    const SiestaCalculator calculator;
    FA_CHECK_THROWS(calculator.fdf_string(molecule), CalculatorError);
}

FA_TEST(siesta_parses_its_text_output) {
    const Results results = SiestaCalculator::parse_output_text(kSiestaOutput);

    FA_CHECK(results.energy.has_value());
    FA_CHECK_CLOSE(*results.energy, -215.245972, 1e-9);
    FA_CHECK_CLOSE(*results.free_energy, -215.246500, 1e-9);

    // The text output is already in eV/Ang and eV/Ang^3.
    FA_CHECK(results.forces.size() == 6);
    FA_CHECK_CLOSE(results.forces[0], 0.010000, 1e-12);
    FA_CHECK_CLOSE(results.forces[5], -0.030000, 1e-12);

    FA_CHECK(results.stress.has_value());
    const std::array<double, 6>& stress = *results.stress;
    FA_CHECK_CLOSE(stress[0], -0.001, 1e-12);
    FA_CHECK_CLOSE(stress[1], -0.002, 1e-12);
    FA_CHECK_CLOSE(stress[2], -0.003, 1e-12);
    FA_CHECK_CLOSE(stress[3], 0.0003, 1e-12);  // yz
    FA_CHECK_CLOSE(stress[4], 0.0002, 1e-12);  // xz
    FA_CHECK_CLOSE(stress[5], 0.0001, 1e-12);  // xy

    FA_CHECK_CLOSE(results.extra["fermi_energy"].as_double(), -3.123456, 1e-9);
    FA_CHECK(results.extra["job_completed"].as_bool());
}

FA_TEST(siesta_prefers_the_force_stress_file) {
    Results results = SiestaCalculator::parse_output_text(kSiestaOutput);
    SiestaCalculator::parse_force_stress(kForceStress, results);

    const double force_scale = kRydbergInEv / kBohrInAngstrom;
    const double stress_scale =
        kRydbergInEv / (kBohrInAngstrom * kBohrInAngstrom * kBohrInAngstrom);

    FA_CHECK(results.forces.size() == 6);
    FA_CHECK_CLOSE(results.forces[0], 0.00036749 * force_scale, 1e-9);
    FA_CHECK_CLOSE(results.forces[5], -0.00110247 * force_scale, 1e-9);

    const std::array<double, 6>& stress = *results.stress;
    FA_CHECK_CLOSE(stress[0], -0.000034 * stress_scale, 1e-9);
    FA_CHECK_CLOSE(stress[2], -0.000102 * stress_scale, 1e-9);
    FA_CHECK_CLOSE(stress[5], 0.0000034 * stress_scale, 1e-9);  // xy
}

FA_TEST(siesta_runs_end_to_end_against_a_stub_engine) {
    ScratchDirectory scratch("siesta-run");
    util::write_text_file(scratch.path() / "Si.psf", "fake pseudopotential\n");
    util::write_text_file(scratch.path() / "siesta.out.reference", kSiestaOutput);
    util::write_text_file(scratch.path() / "siesta.FORCE_STRESS.reference", kForceStress);
    util::write_text_file(scratch.path() / "stub_siesta.sh",
                          "#!/bin/sh\n"
                          "cp siesta.out.reference siesta.out\n"
                          "cp siesta.FORCE_STRESS.reference siesta.FORCE_STRESS\n");

    Atoms atoms = silicon_dimer();
    auto calculator = std::make_shared<SiestaCalculator>(parameters({
        {"directory", Value(scratch.path().string())},
        {"command", Value("sh stub_siesta.sh")},
        {"mesh_cutoff", Value(200.0)},
    }));
    atoms.set_calculator(calculator);

    FA_CHECK_CLOSE(atoms.get_potential_energy(), -215.245972, 1e-9);
    FA_CHECK(std::filesystem::is_regular_file(scratch.path() / "siesta.fdf"));

    // The higher-precision FORCE_STRESS values win over the text output.
    const double force_scale = kRydbergInEv / kBohrInAngstrom;
    FA_CHECK_CLOSE(atoms.get_forces()[0], 0.00036749 * force_scale, 1e-9);
}

FA_TEST(siesta_missing_pseudopotential_is_reported_clearly) {
    ScratchDirectory scratch("siesta-pp");
    Atoms atoms = silicon_dimer();
    atoms.set_calculator(std::make_shared<SiestaCalculator>(parameters({
        {"directory", Value(scratch.path().string())},
        {"command", Value("true")},
    })));

    bool threw = false;
    try {
        (void)atoms.get_potential_energy();
    } catch (const CalculatorError& error) {
        threw = true;
        FA_CHECK(std::string(error.what()).find("no pseudopotential for Si") !=
                 std::string::npos);
    }
    FA_CHECK(threw);
}

// ---------------------------------------------------------------------------
// Staging and post-processing without running anything
// ---------------------------------------------------------------------------

FA_TEST(inputs_can_be_staged_without_executing) {
    ScratchDirectory scratch("stage");
    const Atoms atoms = silicon_dimer();

    QuantumEspressoCalculator calculator(parameters({
        {"pseudopotentials", Value(Value::Map{{"Si", Value("Si.upf")}})},
        {"command", Value("false")},  // would fail if it were ever run
    }));
    calculator.write_inputs(atoms, scratch.path());

    FA_CHECK(std::filesystem::is_regular_file(scratch.path() / "pw.in"));
    FA_CHECK(calculator.call_count() == 0);
}

FA_TEST(a_finished_run_can_be_parsed_after_the_fact) {
    ScratchDirectory scratch("post");
    util::write_text_file(scratch.path() / "OUTCAR", kOutcar);

    VaspCalculator calculator;
    calculator.read_results(scratch.path());

    FA_CHECK_CLOSE(*calculator.results().energy, -10.84012340, 1e-10);
    FA_CHECK(calculator.results().forces.size() == 6);
    FA_CHECK(calculator.call_count() == 0);

    FA_CHECK_THROWS(calculator.read_results(scratch.path() / "does-not-exist"), CalculatorError);
}

FA_TEST(commands_resolve_from_parameters_then_the_environment) {
    const VaspCalculator defaulted;
    FA_CHECK(defaulted.command() == "vasp_std");

    const VaspCalculator configured(parameters({{"command", Value("mpirun -np 8 vasp_gam")}}));
    FA_CHECK(configured.command() == "mpirun -np 8 vasp_gam");

    // The default espresso command names the input and output files, so a user
    // who renames them keeps a working command.
    const QuantumEspressoCalculator renamed(parameters(
        {{"input_file", Value("si.pwi")}, {"output_file", Value("si.pwo")}}));
    FA_CHECK(renamed.command() == "pw.x -in si.pwi > si.pwo");

    const SiestaCalculator siesta(parameters({{"label", Value("run")}}));
    FA_CHECK(siesta.command() == "siesta < run.fdf > run.out");
}

FA_TEST_MAIN()
