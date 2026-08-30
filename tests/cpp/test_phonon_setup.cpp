// SPDX-License-Identifier: MIT
//
// The three self-contained pieces a phonon calculation is built from: the
// eigensolver, supercell generation, and symmetry detection. Each is checked
// against answers that are known independently of the code.

#include <cmath>
#include <complex>
#include <vector>

#include "fastatoms/phonon/eigen.hpp"
#include "fastatoms/phonon/supercell.hpp"
#include "fastatoms/phonon/symmetry.hpp"
#include "testing.hpp"

using namespace fastatoms;
using namespace fastatoms::phonon;

namespace {

Atoms simple_cubic(double a = 3.0) {
    return Atoms(std::vector<int>{18}, {0.0, 0.0, 0.0}, Cell::cubic(a));
}

/// The two-atom primitive cell of diamond-structure silicon.
Atoms diamond_silicon(double a = 5.43) {
    const Vec3 a1{0.0, 0.5 * a, 0.5 * a};
    const Vec3 a2{0.5 * a, 0.0, 0.5 * a};
    const Vec3 a3{0.5 * a, 0.5 * a, 0.0};
    Atoms atoms(std::vector<int>{14, 14}, std::vector<double>(6, 0.0),
                Cell(a1, a2, a3, {true, true, true}));
    atoms.set_scaled_positions({0.0, 0.0, 0.0, 0.25, 0.25, 0.25});
    return atoms;
}

/// The face-centred-cubic primitive cell.
Atoms fcc_primitive(double a = 5.26) {
    const Vec3 a1{0.0, 0.5 * a, 0.5 * a};
    const Vec3 a2{0.5 * a, 0.0, 0.5 * a};
    const Vec3 a3{0.5 * a, 0.5 * a, 0.0};
    return Atoms(std::vector<int>{18}, {0.0, 0.0, 0.0}, Cell(a1, a2, a3, {true, true, true}));
}

}  // namespace

// ---------------------------------------------------------------------------
// The eigensolver
// ---------------------------------------------------------------------------

FA_TEST(hermitian_eigen_solves_a_real_symmetric_matrix) {
    // [[2, 1], [1, 2]] has eigenvalues 1 and 3.
    ComplexMatrix m(2);
    m(0, 0) = 2.0;
    m(1, 1) = 2.0;
    m(0, 1) = 1.0;
    m(1, 0) = 1.0;

    const HermitianEigen result = hermitian_eigen(m);
    FA_CHECK(result.converged);
    FA_CHECK_CLOSE(result.values[0], 1.0, 1e-12);
    FA_CHECK_CLOSE(result.values[1], 3.0, 1e-12);
}

FA_TEST(hermitian_eigen_solves_a_genuinely_complex_matrix) {
    // [[1, i], [-i, 1]] has eigenvalues 0 and 2.
    ComplexMatrix m(2);
    m(0, 0) = 1.0;
    m(1, 1) = 1.0;
    m(0, 1) = std::complex<double>(0.0, 1.0);
    m(1, 0) = std::complex<double>(0.0, -1.0);

    const HermitianEigen result = hermitian_eigen(m);
    FA_CHECK_CLOSE(result.values[0], 0.0, 1e-12);
    FA_CHECK_CLOSE(result.values[1], 2.0, 1e-12);
}

FA_TEST(hermitian_eigen_diagonalises_and_stays_unitary) {
    // A larger matrix with no special structure, checked the only way that
    // really matters: U^H A U has to come out diagonal, and U unitary.
    const std::size_t n = 9;
    ComplexMatrix a(n);
    for (std::size_t i = 0; i < n; ++i) {
        a(i, i) = 0.5 * static_cast<double>(i) - 1.0;
        for (std::size_t j = i + 1; j < n; ++j) {
            const double real = std::sin(1.7 * static_cast<double>(i + 1) * (j + 2));
            const double imaginary = std::cos(2.3 * static_cast<double>(i + 3) * (j + 1));
            a(i, j) = std::complex<double>(real, imaginary);
            a(j, i) = std::conj(a(i, j));
        }
    }

    const HermitianEigen result = hermitian_eigen(a);
    FA_CHECK(result.converged);

    for (std::size_t p = 0; p < n; ++p) {
        for (std::size_t q = 0; q < n; ++q) {
            // (U^H A U)_pq
            std::complex<double> value{};
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    value += std::conj(result.vectors(i, p)) * a(i, j) * result.vectors(j, q);
                }
            }
            FA_CHECK_CLOSE(value.real(), p == q ? result.values[p] : 0.0, 1e-9);
            FA_CHECK_CLOSE(value.imag(), 0.0, 1e-9);

            // (U^H U)_pq
            std::complex<double> overlap{};
            for (std::size_t i = 0; i < n; ++i) {
                overlap += std::conj(result.vectors(i, p)) * result.vectors(i, q);
            }
            FA_CHECK_CLOSE(overlap.real(), p == q ? 1.0 : 0.0, 1e-10);
            FA_CHECK_CLOSE(overlap.imag(), 0.0, 1e-10);
        }
    }

    // Ascending order, and the trace is preserved.
    double trace = 0.0;
    for (std::size_t k = 0; k + 1 < n; ++k) {
        FA_CHECK(result.values[k] <= result.values[k + 1]);
    }
    for (std::size_t k = 0; k < n; ++k) {
        trace += result.values[k];
    }
    double original_trace = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        original_trace += a(k, k).real();
    }
    FA_CHECK_CLOSE(trace, original_trace, 1e-10);
}

FA_TEST(hermitian_eigen_handles_degenerate_and_empty_input) {
    const HermitianEigen identity = hermitian_eigen(ComplexMatrix::identity(4));
    for (const double value : identity.values) {
        FA_CHECK_CLOSE(value, 1.0, 1e-12);
    }
    FA_CHECK(hermitian_eigen(ComplexMatrix{}).values.empty());
    FA_CHECK(hermitian_eigen(ComplexMatrix(3)).values.size() == 3);
}

// ---------------------------------------------------------------------------
// Supercells
// ---------------------------------------------------------------------------

FA_TEST(a_diagonal_supercell_is_a_repetition) {
    const Atoms unit = diamond_silicon();
    const SupercellMap map = make_supercell(unit, 2, 2, 2);

    FA_CHECK(map.cell_count == 8);
    FA_CHECK(map.unit_cell_size == 2);
    FA_CHECK(map.size() == 16);
    FA_CHECK_CLOSE(map.atoms.volume(), 8.0 * unit.volume(), 1e-9);

    // The first block is the original cell, untouched, at the zero translation.
    for (std::size_t k = 0; k < unit.size(); ++k) {
        FA_CHECK(map.unit_cell_index[k] == k);
        FA_CHECK(map.lattice_point[k] == (std::array<int, 3>{0, 0, 0}));
        FA_CHECK_CLOSE((map.atoms.position(k) - unit.position(k)).norm(), 0.0, 1e-12);
    }

    // Every atom sits exactly where its unit-cell parent plus its translation
    // says it should.
    for (std::size_t k = 0; k < map.size(); ++k) {
        const Vec3 expected =
            unit.position(map.unit_cell_index[k]) + map.lattice_translation[k];
        FA_CHECK_CLOSE((map.atoms.position(k) - expected).norm(), 0.0, 1e-12);
        FA_CHECK(map.atoms.number(k) == unit.number(map.unit_cell_index[k]));
    }
}

FA_TEST(a_non_diagonal_supercell_builds_the_conventional_cell) {
    // The primitive-to-conventional transformation for fcc: determinant 4, so
    // the cubic cell holds four primitive cells.
    const Atoms primitive = fcc_primitive(5.26);
    const Mat3 transformation(-1, 1, 1, 1, -1, 1, 1, 1, -1);
    const SupercellMap map = make_supercell(primitive, transformation);

    FA_CHECK(map.cell_count == 4);
    FA_CHECK(map.size() == 4);
    FA_CHECK_CLOSE(map.atoms.volume(), 4.0 * primitive.volume(), 1e-9);

    // The result is the cubic conventional cell.
    const Vec3 lengths = map.atoms.cell().lengths();
    FA_CHECK_CLOSE(lengths.x, 5.26, 1e-9);
    FA_CHECK_CLOSE(lengths.y, 5.26, 1e-9);
    FA_CHECK_CLOSE(lengths.z, 5.26, 1e-9);
    FA_CHECK(map.atoms.cell().is_orthorhombic(1e-9));
}

FA_TEST(supercell_errors_are_reported) {
    const Atoms unit = simple_cubic();
    FA_CHECK_THROWS(make_supercell(unit, Mat3::diagonal(1.5, 1.0, 1.0)), PhononError);
    FA_CHECK_THROWS(make_supercell(unit, Mat3::diagonal(0.0, 1.0, 1.0)), PhononError);
    FA_CHECK_THROWS(make_supercell(unit, Mat3::diagonal(-2.0, 1.0, 1.0)), PhononError);
    FA_CHECK_THROWS(make_supercell(unit, 0, 1, 1), PhononError);

    Atoms molecule(std::vector<int>{1}, {0.0, 0.0, 0.0});
    FA_CHECK_THROWS(make_supercell(molecule, 2, 2, 2), PhononError);
}

// ---------------------------------------------------------------------------
// Symmetry
// ---------------------------------------------------------------------------

FA_TEST(lattice_point_groups_have_the_expected_orders) {
    // Cubic: m-3m, 48 operations. This is the standard result, and getting it
    // from an integer search over the metric tensor is the whole test.
    FA_CHECK(lattice_point_group(Cell::cubic(4.0)).size() == 48);

    // The fcc primitive lattice is still cubic, even though its cell is not.
    FA_CHECK(lattice_point_group(fcc_primitive().cell()).size() == 48);

    // Tetragonal: 4/mmm, 16.
    FA_CHECK(lattice_point_group(Cell::orthorhombic(4.0, 4.0, 6.0)).size() == 16);

    // Orthorhombic: mmm, 8.
    FA_CHECK(lattice_point_group(Cell::orthorhombic(3.0, 4.0, 5.0)).size() == 8);

    // Hexagonal: 6/mmm, 24.
    FA_CHECK(lattice_point_group(Cell::from_parameters(3.0, 3.0, 5.0, 90, 90, 120)).size() == 24);

    // Triclinic: only the identity and inversion.
    FA_CHECK(lattice_point_group(Cell::from_parameters(3.0, 4.0, 5.0, 71, 83, 97)).size() == 2);
}

FA_TEST(symmetry_of_simple_cubic_and_diamond) {
    const SymmetryDataset cubic = find_symmetry(simple_cubic());
    FA_CHECK(cubic.order() == 48);
    FA_CHECK(cubic.point_group.size() == 48);
    FA_CHECK(cubic.independent_atoms.size() == 1);

    // Diamond silicon: two atoms, one orbit, and the full 48 operations of the
    // space group act within the primitive cell.
    const SymmetryDataset silicon = find_symmetry(diamond_silicon());
    FA_CHECK(silicon.order() == 48);
    FA_CHECK(silicon.independent_atoms.size() == 1);
    FA_CHECK(silicon.equivalent_atoms[0] == silicon.equivalent_atoms[1]);

    // Half the operations move the two atoms onto each other; the other half
    // leave them alone. Either way the site symmetry has order 24.
    FA_CHECK(site_symmetry(silicon, 0).size() == 24);
    FA_CHECK(mapping_operation(silicon, 0, 1) != nullptr);
}

FA_TEST(symmetry_of_a_supercell_includes_its_translations) {
    // A 2x2x2 supercell has eight pure translations, and each of the 48 point
    // operations combines with all of them.
    const SupercellMap map = make_supercell(simple_cubic(), 2, 2, 2);
    const SymmetryDataset symmetry = find_symmetry(map.atoms);

    FA_CHECK(symmetry.order() == 48 * 8);
    FA_CHECK(symmetry.point_group.size() == 48);
    // Every atom is a translated copy of the one at the origin.
    FA_CHECK(symmetry.independent_atoms.size() == 1);
    FA_CHECK(symmetry.independent_atoms.front() == 0);

    std::size_t translations = 0;
    for (const SymmetryOperation& operation : symmetry.operations) {
        if (operation.is_pure_translation()) {
            ++translations;
        }
    }
    FA_CHECK(translations == 8);
}

FA_TEST(breaking_the_symmetry_is_detected) {
    // Move one atom off its site and the cubic symmetry has to collapse.
    Atoms distorted = diamond_silicon();
    std::vector<double> positions = distorted.positions();
    positions[0] += 0.3;
    distorted.set_positions(positions);

    const SymmetryDataset symmetry = find_symmetry(distorted);
    FA_CHECK(symmetry.order() < 48);
    FA_CHECK(symmetry.order() >= 1);

    // The two atoms are nonetheless still equivalent, and that is not a
    // failure of the search: inversion through the midpoint of any two like
    // atoms is a symmetry of a two-atom cell whatever their positions, because
    // -I is in every lattice point group and the translation is free. Finding
    // it is a sign the search is looking at more than the obvious operations.
    FA_CHECK(symmetry.independent_atoms.size() == 1);
    FA_CHECK(mapping_operation(symmetry, 0, 1) != nullptr);

    // Changing one atom's species does break the equivalence, without moving
    // anything, because no operation may exchange different elements.
    Atoms alloy = diamond_silicon();
    alloy.set_number(1, 32);
    const SymmetryDataset mixed = find_symmetry(alloy);
    FA_CHECK(mixed.independent_atoms.size() == 2);
    FA_CHECK(mapping_operation(mixed, 0, 1) == nullptr);
    // Zincblende keeps the 24 operations of Td, having lost inversion.
    FA_CHECK(mixed.order() == 24);
}

FA_TEST(symmetry_operations_really_map_the_structure_onto_itself) {
    // The check the search itself performs, done again from the outside: apply
    // each operation to every atom and confirm it lands on its image.
    const Atoms atoms = diamond_silicon();
    const SymmetryDataset symmetry = find_symmetry(atoms);
    const Cell& cell = atoms.cell();

    for (const SymmetryOperation& operation : symmetry.operations) {
        // The rotation is an integer matrix in the lattice basis...
        for (const double value : operation.rotation.m) {
            FA_CHECK_CLOSE(value, std::nearbyint(value), 1e-9);
        }
        // ...and orthogonal in Cartesian coordinates.
        const Mat3 product =
            operation.cartesian_rotation * operation.cartesian_rotation.transposed();
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                FA_CHECK_CLOSE(product(i, j), i == j ? 1.0 : 0.0, 1e-9);
            }
        }

        for (std::size_t i = 0; i < atoms.size(); ++i) {
            const Vec3 image = cell.scaled(atoms.position(i)) * operation.rotation +
                               operation.translation;
            const Vec3 target = cell.scaled(atoms.position(operation.atom_map[i]));
            const Vec3 difference = cell.minimum_image(cell.cartesian(image - target));
            FA_CHECK(difference.norm() < 1e-6);
            FA_CHECK(atoms.number(i) == atoms.number(operation.atom_map[i]));
        }
    }
}

FA_TEST_MAIN()
