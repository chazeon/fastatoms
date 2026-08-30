// SPDX-License-Identifier: MIT
#include "fastatoms/phonon/phonons.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>

#include "fastatoms/core/units.hpp"

namespace fastatoms::phonon {
namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

/// sqrt(eV / (angstrom^2 amu)) is an angular frequency in inverse internal
/// time units; this turns it into a frequency in THz.
constexpr double frequency_to_terahertz() { return units::fs * 1000.0 / kTwoPi; }

/// Outer product: `(a (x) b)_{ij} = a_i b_j`.
Mat3 outer(const Vec3& a, const Vec3& b) {
    return {a.x * b.x, a.x * b.y, a.x * b.z, a.y * b.x, a.y * b.y,
            a.y * b.z, a.z * b.x, a.z * b.y, a.z * b.z};
}

/// Rank of a set of 3-vectors, by Gram-Schmidt with a tolerance.
std::size_t span_rank(const std::vector<Vec3>& vectors, double tolerance = 1e-6) {
    std::vector<Vec3> basis;
    for (Vec3 v : vectors) {
        for (const Vec3& b : basis) {
            v -= b * dot(v, b);
        }
        const double length = v.norm();
        if (length > tolerance) {
            basis.push_back(v / length);
        }
        if (basis.size() == 3) {
            break;
        }
    }
    return basis.size();
}

}  // namespace

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

Phonons::Phonons(Atoms unit_cell, Value parameters)
    : unit_cell_(std::move(unit_cell)), parameters_(std::move(parameters)) {
    if (parameters_.is_null()) {
        parameters_ = Value(Value::Map{});
    }
    if (!parameters_.is_map()) {
        throw PhononError("phonon parameters must be a mapping, got " +
                          std::string(parameters_.type_name()));
    }
    if (unit_cell_.empty()) {
        fail("cannot compute phonons for an empty structure");
    }
    if (!unit_cell_.cell().is_invertible() || !unit_cell_.cell().any_pbc()) {
        fail("phonons need a periodic structure with a full 3x3 cell");
    }

    supercell_ = make_supercell(unit_cell_, supercell_matrix());
    symmetry_ = find_symmetry(supercell_.atoms, symmetry_tolerance());
    build_displacements();
    build_shortest_vectors();
}

const Value& Phonons::parameter(std::string_view key) const {
    if (const Value* found = parameters_.find(key)) {
        return *found;
    }
    return Value::null();
}

void Phonons::fail(const std::string& message) const { throw PhononError("phonons: " + message); }

Mat3 Phonons::supercell_matrix() const {
    const Value& value = parameter("supercell");
    if (value.is_null()) {
        return Mat3::diagonal(2.0, 2.0, 2.0);
    }
    if (value.is_number()) {
        const double n = value.as_double();
        return Mat3::diagonal(n, n, n);
    }
    const std::vector<double> entries = value.as_doubles();
    if (entries.size() == 3) {
        return Mat3::diagonal(entries[0], entries[1], entries[2]);
    }
    if (entries.size() == 9) {
        return Mat3(entries[0], entries[1], entries[2], entries[3], entries[4], entries[5],
                    entries[6], entries[7], entries[8]);
    }
    fail("'supercell' must be one integer, three of them, or a full 3x3 matrix");
}

double Phonons::displacement_distance() const {
    const double value = parameter("displacement").value_or(0.01);
    if (!(value > 0.0)) {
        fail("'displacement' must be positive");
    }
    return value;
}

double Phonons::symmetry_tolerance() const {
    const double value = parameter("symmetry_tolerance").value_or(1e-5);
    if (!(value > 0.0)) {
        fail("'symmetry_tolerance' must be positive");
    }
    return value;
}

bool Phonons::use_symmetry() const { return parameter("symmetry").value_or(true); }
bool Phonons::central_differences() const { return parameter("central").value_or(true); }
bool Phonons::acoustic_sum_rule() const {
    return parameter("acoustic_sum_rule").value_or(true);
}

std::size_t Phonons::calculation_count_without_symmetry() const {
    const std::size_t per_direction = central_differences() ? 2 : 1;
    return 3 * supercell_.unit_cell_size * per_direction;
}

// ---------------------------------------------------------------------------
// Choosing the displacements
// ---------------------------------------------------------------------------

void Phonons::build_displacements() {
    displacements_.clear();
    groups_.clear();

    const double distance = displacement_distance();
    const bool central = central_differences();
    const std::array<Vec3, 3> axes{Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}};

    // Only the atoms of the first unit cell need rows in the force constants,
    // and of those only one per symmetry orbit.
    std::vector<std::size_t> atoms_to_displace;
    for (std::size_t atom = 0; atom < supercell_.unit_cell_size; ++atom) {
        if (!use_symmetry() || symmetry_.equivalent_atoms[atom] == atom) {
            atoms_to_displace.push_back(atom);
        }
    }

    for (const std::size_t atom : atoms_to_displace) {
        std::vector<Vec3> directions;

        if (use_symmetry()) {
            // The site symmetry group turns one displacement into several. Add
            // axes only while they buy new directions: once the images of the
            // chosen ones span three dimensions, the rest are determined.
            const std::vector<SymmetryOperation> site = site_symmetry(symmetry_, atom);
            std::vector<Vec3> spanned;
            for (const Vec3& axis : axes) {
                if (span_rank(spanned) == 3) {
                    break;
                }
                std::vector<Vec3> extended = spanned;
                for (const SymmetryOperation& operation : site) {
                    extended.push_back(axis * operation.cartesian_rotation);
                }
                if (span_rank(extended) > span_rank(spanned)) {
                    directions.push_back(axis);
                    spanned = std::move(extended);
                }
            }
            if (span_rank(spanned) < 3) {
                // Should not happen for a well-formed structure, but falling
                // back to all three axes is always correct.
                directions.assign(axes.begin(), axes.end());
            }
        } else {
            directions.assign(axes.begin(), axes.end());
        }

        for (const Vec3& direction : directions) {
            DisplacementGroup group;
            group.atom = atom;
            group.direction = direction;

            group.plus = displacements_.size();
            displacements_.push_back(Displacement{atom, direction * distance});
            if (central) {
                group.minus = static_cast<std::ptrdiff_t>(displacements_.size());
                displacements_.push_back(Displacement{atom, direction * (-distance)});
            }
            groups_.push_back(group);
        }
    }
}

std::vector<Atoms> Phonons::displaced_supercells() const {
    std::vector<Atoms> configurations;
    configurations.reserve(displacements_.size());
    for (const Displacement& displacement : displacements_) {
        Atoms moved = supercell_.atoms;
        moved.set_calculator(nullptr);
        moved.set_position(displacement.atom,
                           moved.position(displacement.atom) + displacement.vector);
        configurations.push_back(std::move(moved));
    }
    return configurations;
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

void Phonons::run(std::shared_ptr<Calculator> calculator) {
    if (!calculator) {
        calculator = unit_cell_.calculator();
    }
    if (!calculator) {
        fail("no calculator: attach one to the structure or pass it to run()");
    }

    Atoms reference = supercell_.atoms;
    reference.set_calculator(calculator);
    equilibrium_forces_ = reference.get_forces();

    std::vector<std::vector<double>> forces;
    forces.reserve(displacements_.size());
    for (const Displacement& displacement : displacements_) {
        Atoms moved = supercell_.atoms;
        moved.set_calculator(calculator);
        moved.set_position(displacement.atom,
                           moved.position(displacement.atom) + displacement.vector);
        forces.push_back(moved.get_forces());
    }

    solve_force_constants(forces);
}

void Phonons::set_forces(const std::vector<std::vector<double>>& forces,
                         const std::vector<double>& equilibrium_forces) {
    if (forces.size() != displacements_.size()) {
        fail("expected " + std::to_string(displacements_.size()) + " force sets, got " +
             std::to_string(forces.size()));
    }
    if (!equilibrium_forces.empty()) {
        if (equilibrium_forces.size() != 3 * supercell_.size()) {
            fail("the equilibrium forces have the wrong length");
        }
        equilibrium_forces_ = equilibrium_forces;
    } else if (equilibrium_forces_.empty()) {
        if (!central_differences()) {
            fail("one-sided differences need the equilibrium forces as well");
        }
        equilibrium_forces_.assign(3 * supercell_.size(), 0.0);
    }
    solve_force_constants(forces);
}

double Phonons::residual_force() const {
    double largest = 0.0;
    for (std::size_t i = 0; i * 3 + 2 < equilibrium_forces_.size(); ++i) {
        largest = std::max(largest, Vec3::load(equilibrium_forces_.data() + 3 * i).norm());
    }
    return largest;
}

void Phonons::solve_force_constants(const std::vector<std::vector<double>>& forces) {
    const std::size_t total = supercell_.size();
    for (const std::vector<double>& set : forces) {
        if (set.size() != 3 * total) {
            fail("a force set has the wrong length: expected " + std::to_string(3 * total));
        }
    }
    if (equilibrium_forces_.size() != 3 * total) {
        equilibrium_forces_.assign(3 * total, 0.0);
    }

    force_constants_ = ForceConstants(supercell_.unit_cell_size, total);
    std::vector<bool> solved(supercell_.unit_cell_size, false);

    // --- rows for the atoms that were actually displaced -------------------
    std::vector<std::size_t> displaced_atoms;
    for (const DisplacementGroup& group : groups_) {
        if (std::find(displaced_atoms.begin(), displaced_atoms.end(), group.atom) ==
            displaced_atoms.end()) {
            displaced_atoms.push_back(group.atom);
        }
    }

    for (const std::size_t atom : displaced_atoms) {
        const std::vector<SymmetryOperation> site =
            use_symmetry() ? site_symmetry(symmetry_, atom) : std::vector<SymmetryOperation>{};

        // Every measurement, real or produced by symmetry, is one linear
        // equation `F(j) = -d Phi(atom, j)`. Accumulate the normal equations:
        // A = sum d^T d and B_j = -sum d^T F(j), then Phi = A^-1 B.
        Mat3 normal{};
        std::vector<Mat3> right(total);

        const auto add_measurement = [&](const Vec3& d, const std::vector<double>& response,
                                         const std::vector<std::size_t>& permutation) {
            const Mat3 contribution = outer(d, d);
            for (std::size_t k = 0; k < 9; ++k) {
                normal.m[k] += contribution.m[k];
            }
            for (std::size_t j = 0; j < total; ++j) {
                // `permutation` maps the original atom index to where the
                // symmetry operation sent it, so the force felt by atom
                // permutation[j] is the rotated force of atom j.
                const Vec3 force = Vec3::load(response.data() + 3 * j);
                const Mat3 term = outer(d, force);
                Mat3& target = right[permutation.empty() ? j : permutation[j]];
                for (std::size_t k = 0; k < 9; ++k) {
                    target.m[k] -= term.m[k];
                }
            }
        };

        for (const DisplacementGroup& group : groups_) {
            if (group.atom != atom) {
                continue;
            }

            // The linear part of the response. Central differences cancel the
            // cubic term; one-sided ones lean on the equilibrium forces.
            std::vector<double> response(3 * total);
            const std::vector<double>& plus = forces[group.plus];
            if (group.minus >= 0) {
                const std::vector<double>& minus = forces[static_cast<std::size_t>(group.minus)];
                for (std::size_t k = 0; k < response.size(); ++k) {
                    response[k] = 0.5 * (plus[k] - minus[k]);
                }
            } else {
                for (std::size_t k = 0; k < response.size(); ++k) {
                    response[k] = plus[k] - equilibrium_forces_[k];
                }
            }

            const Vec3 vector = group.direction * displacement_distance();
            add_measurement(vector, response, {});

            // Every site-symmetry operation turns this one measurement into
            // another: displacing along `d R` gives the forces `F R`, on the
            // atoms the operation permutes them onto.
            for (const SymmetryOperation& operation : site) {
                if (operation.is_pure_translation()) {
                    continue;  // already counted as the measurement itself
                }
                const Mat3& rotation = operation.cartesian_rotation;
                std::vector<double> rotated(3 * total);
                for (std::size_t j = 0; j < total; ++j) {
                    const Vec3 force = Vec3::load(response.data() + 3 * j);
                    (force * rotation).store(rotated.data() + 3 * j);
                }
                add_measurement(vector * rotation, rotated, operation.atom_map);
            }
        }

        if (std::abs(normal.determinant()) < 1e-20) {
            fail("the displacements for atom " + std::to_string(atom) +
                 " do not span three dimensions; try symmetry: false");
        }
        const Mat3 inverse_normal = normal.inverse();
        for (std::size_t j = 0; j < total; ++j) {
            force_constants_.block(atom, j) = inverse_normal * right[j];
        }
        solved[atom] = true;
    }

    // --- rows for the atoms symmetry stood in for --------------------------
    for (std::size_t atom = 0; atom < supercell_.unit_cell_size; ++atom) {
        if (solved[atom]) {
            continue;
        }
        const std::size_t representative = symmetry_.equivalent_atoms[atom];
        if (!solved[representative]) {
            fail("atom " + std::to_string(atom) + " has no solved representative");
        }
        const SymmetryOperation* operation =
            mapping_operation(symmetry_, representative, atom);
        if (operation == nullptr) {
            fail("no symmetry operation maps atom " + std::to_string(representative) +
                 " onto atom " + std::to_string(atom));
        }

        // Phi(S(a), S(j)) = R^T Phi(a, j) R, so walk j back through the map.
        std::vector<std::size_t> inverse_map(total);
        for (std::size_t j = 0; j < total; ++j) {
            inverse_map[operation->atom_map[j]] = j;
        }
        const Mat3& rotation = operation->cartesian_rotation;
        const Mat3 transposed = rotation.transposed();
        for (std::size_t j = 0; j < total; ++j) {
            force_constants_.block(atom, j) =
                transposed * force_constants_.block(representative, inverse_map[j]) * rotation;
        }
        solved[atom] = true;
    }

    if (acoustic_sum_rule()) {
        force_constants_.apply_acoustic_sum_rule();
    }
}

// ---------------------------------------------------------------------------
// Dynamical matrix
// ---------------------------------------------------------------------------

void Phonons::build_shortest_vectors() {
    const std::size_t rows = supercell_.unit_cell_size;
    const std::size_t total = supercell_.size();
    shortest_vectors_.assign(rows * total, {});

    const Mat3 super_matrix = supercell_.atoms.cell().matrix();
    const Mat3& transformation = supercell_.transformation;

    for (std::size_t a = 0; a < rows; ++a) {
        const Vec3 origin = supercell_.atoms.position(a);
        for (std::size_t j = 0; j < total; ++j) {
            const Vec3 offset = supercell_.atoms.position(j) - origin;

            // The pair vector is only defined modulo the supercell lattice.
            // Using the shortest images, with equal weight when several tie,
            // is what makes the Fourier interpolation sensible away from the
            // q-points the supercell commensurately contains.
            double shortest = std::numeric_limits<double>::max();
            std::vector<std::array<int, 3>> best;
            for (int m0 = -1; m0 <= 1; ++m0) {
                for (int m1 = -1; m1 <= 1; ++m1) {
                    for (int m2 = -1; m2 <= 1; ++m2) {
                        const Vec3 shift = super_matrix.row(0) * m0 +
                                           super_matrix.row(1) * m1 + super_matrix.row(2) * m2;
                        const double length = (offset + shift).norm();
                        if (length < shortest - 1e-6) {
                            shortest = length;
                            best.clear();
                        }
                        if (length < shortest + 1e-6) {
                            // The supercell translation in unit-cell integers
                            // is m * P, which is integer because P is.
                            const Vec3 m{static_cast<double>(m0), static_cast<double>(m1),
                                         static_cast<double>(m2)};
                            const Vec3 in_unit_cell = m * transformation;
                            const std::array<int, 3>& base = supercell_.lattice_point[j];
                            best.push_back({base[0] + static_cast<int>(std::nearbyint(in_unit_cell.x)),
                                            base[1] + static_cast<int>(std::nearbyint(in_unit_cell.y)),
                                            base[2] + static_cast<int>(std::nearbyint(in_unit_cell.z))});
                        }
                    }
                }
            }

            const double weight = 1.0 / static_cast<double>(best.size());
            auto& entry = shortest_vectors_[a * total + j];
            entry.reserve(best.size());
            for (const std::array<int, 3>& lattice : best) {
                entry.emplace_back(lattice, weight);
            }
        }
    }
}

const ForceConstants& Phonons::force_constants() const {
    if (force_constants_.empty()) {
        fail("no force constants yet; call run() or set_forces() first");
    }
    return force_constants_;
}

ComplexMatrix Phonons::dynamical_matrix(const Vec3& q) const {
    const ForceConstants& constants = force_constants();
    const std::size_t natoms = supercell_.unit_cell_size;
    const std::size_t total = supercell_.size();
    const std::vector<double> masses = unit_cell_.masses();

    ComplexMatrix matrix(3 * natoms);
    for (std::size_t a = 0; a < natoms; ++a) {
        for (std::size_t j = 0; j < total; ++j) {
            const std::size_t b = supercell_.unit_cell_index[j];

            // The phase is exp(2 pi i q . n) with n the integer lattice
            // translation, because the reciprocal basis is dual to the direct
            // one: q . R = 2 pi q_fractional . n exactly.
            std::complex<double> phase{0.0, 0.0};
            for (const auto& [lattice, weight] : shortest_vectors_[a * total + j]) {
                const double angle = kTwoPi * (q.x * lattice[0] + q.y * lattice[1] +
                                               q.z * lattice[2]);
                phase += weight * std::complex<double>(std::cos(angle), std::sin(angle));
            }

            const double mass_factor = 1.0 / std::sqrt(masses[a] * masses[b]);
            const Mat3& block = constants.block(a, j);
            for (std::size_t alpha = 0; alpha < 3; ++alpha) {
                for (std::size_t beta = 0; beta < 3; ++beta) {
                    matrix(3 * a + alpha, 3 * b + beta) +=
                        block(alpha, beta) * mass_factor * phase;
                }
            }
        }
    }

    // Hermiticity is the permutation symmetry of the force constants, which
    // finite displacements only satisfy to within their numerical error.
    matrix.hermitize();
    return matrix;
}

PhononModes Phonons::modes(const Vec3& q) const {
    const HermitianEigen decomposition = hermitian_eigen(dynamical_matrix(q));

    PhononModes result;
    result.squared_frequencies = decomposition.values;
    result.eigenvectors = decomposition.vectors;
    result.frequencies.reserve(decomposition.values.size());
    for (const double squared : decomposition.values) {
        // An unstable mode has omega^2 < 0; reporting -sqrt(|omega^2|) is the
        // usual way to show an imaginary frequency on a real axis.
        const double magnitude = std::sqrt(std::abs(squared)) * frequency_to_terahertz();
        result.frequencies.push_back(squared < 0.0 ? -magnitude : magnitude);
    }
    return result;
}

std::vector<double> Phonons::frequencies(const Vec3& q) const { return modes(q).frequencies; }

std::vector<std::vector<double>> Phonons::band_structure(const std::vector<Vec3>& path) const {
    std::vector<std::vector<double>> bands;
    bands.reserve(path.size());
    for (const Vec3& q : path) {
        bands.push_back(frequencies(q));
    }
    return bands;
}

std::vector<double> Phonons::mesh_frequencies(int n1, int n2, int n3) const {
    if (n1 < 1 || n2 < 1 || n3 < 1) {
        fail("the mesh must have at least one point along each direction");
    }
    std::vector<double> all;
    all.reserve(static_cast<std::size_t>(n1) * n2 * n3 * 3 * supercell_.unit_cell_size);
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            for (int k = 0; k < n3; ++k) {
                const Vec3 q{static_cast<double>(i) / n1, static_cast<double>(j) / n2,
                             static_cast<double>(k) / n3};
                for (const double value : frequencies(q)) {
                    all.push_back(value);
                }
            }
        }
    }
    return all;
}

std::string Phonons::to_string() const {
    std::ostringstream out;
    out.precision(4);
    const Vec3 diagonal{supercell_.transformation(0, 0), supercell_.transformation(1, 1),
                        supercell_.transformation(2, 2)};
    out << "Phonons(" << unit_cell_.formula() << ", supercell " << supercell_.cell_count
        << " cells / " << supercell_.size() << " atoms, " << displacements_.size()
        << " displacements";
    if (use_symmetry()) {
        out << " of " << calculation_count_without_symmetry();
    }
    out << ", " << symmetry_.summary() << ")";
    (void)diagonal;
    return out.str();
}

}  // namespace fastatoms::phonon
