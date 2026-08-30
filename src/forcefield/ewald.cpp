// SPDX-License-Identifier: MIT
#include "fastatoms/forcefield/ewald.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

#include "fastatoms/core/units.hpp"

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace fastatoms::ff {
namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kSqrtPi = 1.772453850905516027298167483341;

/// Rows of the reciprocal cell, `b_i`, with `a_i . b_j = 2 pi delta_ij`.
[[nodiscard]] Mat3 reciprocal_cell(const Mat3& cell) {
    const Mat3 inverse = cell.inverse();
    // Rows of the inverse transpose, scaled by 2 pi.
    return Mat3(kTwoPi * inverse.m[0], kTwoPi * inverse.m[3], kTwoPi * inverse.m[6],
                kTwoPi * inverse.m[1], kTwoPi * inverse.m[4], kTwoPi * inverse.m[7],
                kTwoPi * inverse.m[2], kTwoPi * inverse.m[5], kTwoPi * inverse.m[8]);
}

[[nodiscard]] Vec3 row(const Mat3& matrix, int index) {
    const auto k = static_cast<std::size_t>(3 * index);
    return Vec3{matrix.m[k], matrix.m[k + 1], matrix.m[k + 2]};
}

/// Perpendicular widths of the cell: the distance between the pairs of faces.
[[nodiscard]] Vec3 cell_widths(const Mat3& cell) {
    const Mat3 reciprocal = reciprocal_cell(cell);
    Vec3 widths{};
    for (int axis = 0; axis < 3; ++axis) {
        const Vec3 b = row(reciprocal, axis);
        const double norm = b.norm();
        (&widths.x)[axis] = norm > 0.0 ? kTwoPi / norm : 0.0;
    }
    return widths;
}

}  // namespace

std::string EwaldEnergy::to_string() const {
    std::ostringstream out;
    out << "EwaldEnergy(real=" << real << ", reciprocal=" << reciprocal << ", self=" << self
        << ", excluded=" << excluded << ", total=" << total() << " eV)";
    return out.str();
}

Ewald::Ewald(EwaldParameters parameters) : parameters_(parameters) {
    if (!(parameters_.accuracy > 0.0) || parameters_.accuracy >= 1.0) {
        throw ForcefieldError("ewald: accuracy must be between 0 and 1");
    }
    if (parameters_.real_cutoff < 0.0 || parameters_.alpha < 0.0 || parameters_.kcutoff < 0.0) {
        throw ForcefieldError("ewald: cutoffs and alpha must not be negative");
    }
}

void Ewald::prepare(const Atoms& atoms) {
    const Mat3 cell = atoms.cell().matrix();
    const std::array<bool, 3> pbc = atoms.pbc();
    if (!(pbc[0] && pbc[1] && pbc[2])) {
        throw ForcefieldError(
            "ewald: the sum is defined for a fully periodic cell; this structure is periodic "
            "in fewer than three directions");
    }
    if (!atoms.cell().is_invertible()) {
        throw ForcefieldError("ewald: the cell is singular, so there is no reciprocal lattice");
    }
    if (prepared_ && cell == prepared_cell_) {
        return;  // nothing about the splitting depends on where the atoms are
    }

    const Vec3 widths = cell_widths(cell);
    const double smallest = std::min({widths.x, widths.y, widths.z});

    // The splitting is chosen once and then kept. It is a property of the
    // model, not of the cell the model is being used in: letting the cutoff
    // follow the cell would make the energy depend on the volume through the
    // truncation as well as through the physics, and the stress - which is a
    // derivative with respect to exactly that volume - would no longer be the
    // derivative of the energy.
    if (!prepared_) {
        // The minimum-image convention needs the cutoff inside half the
        // smallest width; going past 10 A buys nothing for a water box.
        real_cutoff_ = parameters_.real_cutoff > 0.0 ? parameters_.real_cutoff
                                                     : std::min(0.5 * smallest, 10.0);
        if (!(real_cutoff_ > 0.0)) {
            throw ForcefieldError("ewald: could not choose a real-space cutoff for this cell");
        }

        // erfc(alpha r_c) ~ exp(-(alpha r_c)^2), so demanding
        // exp(-(alpha r_c)^2) < 2 delta fixes alpha once the cutoff is chosen.
        const double delta = parameters_.accuracy;
        alpha_ = parameters_.alpha > 0.0 ? parameters_.alpha
                                         : std::sqrt(-std::log(2.0 * delta)) / real_cutoff_;

        // The reciprocal terms fall off as exp(-k^2/4 alpha^2), so the same
        // tolerance fixes the k-space cutoff.
        kcutoff_ = parameters_.kcutoff > 0.0
                       ? parameters_.kcutoff
                       : 2.0 * alpha_ * std::sqrt(-std::log(delta));
    }

    // The reciprocal vectors follow the cell even when the splitting does not.
    build_kvectors(cell);
    prepared_cell_ = cell;
    prepared_ = true;
}

void Ewald::build_kvectors(const Mat3& cell) {
    const Mat3 reciprocal = reciprocal_cell(cell);
    const double volume = std::abs(cell.determinant());
    if (!(volume > 0.0)) {
        throw ForcefieldError("ewald: the cell has no volume");
    }

    // How far to go along each reciprocal axis to cover a sphere of radius
    // kcutoff: the perpendicular spacing of the reciprocal planes.
    const Vec3 widths = cell_widths(cell);
    for (int axis = 0; axis < 3; ++axis) {
        const double spacing = kTwoPi / (&widths.x)[axis];
        kmax_[static_cast<std::size_t>(axis)] =
            std::max(1, static_cast<int>(std::ceil(kcutoff_ / spacing)));
    }

    kvectors_.clear();
    kfactors_.clear();
    kindices_.clear();

    const double kcutoff_squared = kcutoff_ * kcutoff_;
    const double quarter_inverse_alpha_squared = 1.0 / (4.0 * alpha_ * alpha_);

    // Only half of reciprocal space is stored: A(k) and |S(k)|^2 are both even
    // in k, so the other half is accounted for by a factor of two. The
    // half-space is "n1 > 0, or n1 == 0 and (n2 > 0, or n2 == 0 and n3 > 0)".
    for (int n1 = 0; n1 <= kmax_[0]; ++n1) {
        const int n2_start = (n1 == 0) ? 0 : -kmax_[1];
        for (int n2 = n2_start; n2 <= kmax_[1]; ++n2) {
            const int n3_start = (n1 == 0 && n2 == 0) ? 1 : -kmax_[2];
            for (int n3 = n3_start; n3 <= kmax_[2]; ++n3) {
                const Vec3 k = row(reciprocal, 0) * static_cast<double>(n1) +
                               row(reciprocal, 1) * static_cast<double>(n2) +
                               row(reciprocal, 2) * static_cast<double>(n3);
                const double k_squared = k.norm2();
                if (k_squared <= 0.0 || k_squared > kcutoff_squared) {
                    continue;
                }
                kvectors_.push_back(k);
                kfactors_.push_back(std::exp(-k_squared * quarter_inverse_alpha_squared) /
                                    k_squared);
                kindices_.push_back({n1, n2, n3});
            }
        }
    }
}

double Ewald::estimated_error(const Topology& topology, double volume) const {
    // Kolafa and Perram's estimates for the two truncations.
    const std::vector<double>& charges = topology.charges();
    const double charge_squared =
        std::accumulate(charges.begin(), charges.end(), 0.0,
                        [](double sum, double q) { return sum + q * q; });
    const auto n = static_cast<double>(charges.size());
    if (n <= 0.0 || !(volume > 0.0) || !prepared_) {
        return 0.0;
    }
    const double real = charge_squared * std::sqrt(real_cutoff_ / (2.0 * volume * n)) *
                        std::exp(-alpha_ * alpha_ * real_cutoff_ * real_cutoff_) /
                        (alpha_ * real_cutoff_ * real_cutoff_);
    const double reciprocal =
        charge_squared * alpha_ / (std::pow(kcutoff_, 1.5) * std::sqrt(kSqrtPi)) *
        std::exp(-kcutoff_ * kcutoff_ / (4.0 * alpha_ * alpha_));
    return units::coulomb * (real + reciprocal);
}

EwaldResult Ewald::compute(const Atoms& atoms, const Topology& topology,
                           const NeighborList& pairs, bool need_forces, bool need_stress) {
    prepare(atoms);
    topology.validate(atoms);

    const std::size_t n = atoms.size();
    const std::vector<double>& charges = topology.charges();
    const Mat3 cell = atoms.cell().matrix();
    const double volume = std::abs(cell.determinant());
    const double* FASTATOMS_RESTRICT positions = atoms.positions().data();

    EwaldResult result;
    if (need_forces) {
        result.forces.assign(3 * n, 0.0);
    }
    double virial[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    // -----------------------------------------------------------------
    // Real space: screened point charges, over the neighbour list.
    // -----------------------------------------------------------------
    // An excluded pair is skipped only at zero shift. A molecule does interact
    // with its own periodic images, and dropping those would be wrong.
    {
        const double cutoff_squared = real_cutoff_ * real_cutoff_;
        const double two_alpha_over_sqrt_pi = 2.0 * alpha_ / kSqrtPi;
        double energy = 0.0;

        const std::size_t npairs = pairs.size();
        for (std::size_t p = 0; p < npairs; ++p) {
            const auto i = static_cast<std::size_t>(pairs.i[p]);
            const auto j = static_cast<std::size_t>(pairs.j[p]);
            if (i > j) {
                continue;  // a full list holds each pair twice
            }
            const int s0 = pairs.shifts[3 * p];
            const int s1 = pairs.shifts[3 * p + 1];
            const int s2 = pairs.shifts[3 * p + 2];
            const bool home_cell = (s0 == 0 && s1 == 0 && s2 == 0);
            if (i == j && home_cell) {
                continue;
            }
            if (home_cell && topology.excluded(i, j)) {
                continue;
            }
            const double qq = charges[i] * charges[j];
            if (qq == 0.0) {
                continue;
            }

            const double sx = s0 * cell.m[0] + s1 * cell.m[3] + s2 * cell.m[6];
            const double sy = s0 * cell.m[1] + s1 * cell.m[4] + s2 * cell.m[7];
            const double sz = s0 * cell.m[2] + s1 * cell.m[5] + s2 * cell.m[8];
            const double dx = positions[3 * j] + sx - positions[3 * i];
            const double dy = positions[3 * j + 1] + sy - positions[3 * i + 1];
            const double dz = positions[3 * j + 2] + sz - positions[3 * i + 2];
            const double r_squared = dx * dx + dy * dy + dz * dz;
            if (r_squared > cutoff_squared || r_squared <= 0.0) {
                continue;
            }

            const double r = std::sqrt(r_squared);
            const double alpha_r = alpha_ * r;
            const double erfc_term = std::erfc(alpha_r);
            const double prefactor = units::coulomb * qq;
            // A self-image pair (i == j, shift != 0) is counted once but is
            // shared with its own mirror, so it carries half the weight.
            const double weight = (i == j) ? 0.5 : 1.0;
            energy += weight * prefactor * erfc_term / r;

            if (need_forces || need_stress) {
                // -dE/dr / r, so that F_i = -magnitude * d and F_j = +magnitude * d.
                const double magnitude =
                    weight * prefactor *
                    (erfc_term / r + two_alpha_over_sqrt_pi * std::exp(-alpha_r * alpha_r)) /
                    r_squared;
                if (need_forces) {
                    result.forces[3 * i] -= magnitude * dx;
                    result.forces[3 * i + 1] -= magnitude * dy;
                    result.forces[3 * i + 2] -= magnitude * dz;
                    result.forces[3 * j] += magnitude * dx;
                    result.forces[3 * j + 1] += magnitude * dy;
                    result.forces[3 * j + 2] += magnitude * dz;
                }
                if (need_stress) {
                    const double d[3] = {dx, dy, dz};
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            virial[3 * a + b] -= magnitude * d[a] * d[b];
                        }
                    }
                }
            }
        }
        result.energy.real = energy;
    }

    // -----------------------------------------------------------------
    // Self energy: each Gaussian cloud sitting on its own point charge.
    // -----------------------------------------------------------------
    {
        double charge_squared = 0.0;
        for (const double q : charges) {
            charge_squared += q * q;
        }
        result.energy.self = -units::coulomb * alpha_ / kSqrtPi * charge_squared;
    }

    // -----------------------------------------------------------------
    // Excluded pairs: the reciprocal sum included every pair in the cell,
    // including the ones inside a molecule, so their smooth part comes back
    // out here. Only the home-cell image is removed.
    // -----------------------------------------------------------------
    {
        const double two_alpha_over_sqrt_pi = 2.0 * alpha_ / kSqrtPi;
        double energy = 0.0;
        const Mat3 inverse = cell.inverse();
        for (const auto& [i, j] : topology.excluded_pairs()) {
            const double qq = charges[i] * charges[j];
            if (qq == 0.0) {
                continue;
            }
            // Minimum image, so a molecule split across the boundary is still
            // corrected with the distance it actually has.
            Vec3 d{positions[3 * j] - positions[3 * i], positions[3 * j + 1] - positions[3 * i + 1],
                   positions[3 * j + 2] - positions[3 * i + 2]};
            Vec3 fractional = d * inverse;
            fractional.x -= std::round(fractional.x);
            fractional.y -= std::round(fractional.y);
            fractional.z -= std::round(fractional.z);
            d = fractional * cell;

            const double r_squared = d.norm2();
            if (!(r_squared > 0.0)) {
                continue;
            }
            const double r = std::sqrt(r_squared);
            const double alpha_r = alpha_ * r;
            const double prefactor = units::coulomb * qq;
            energy -= prefactor * std::erf(alpha_r) / r;

            if (need_forces || need_stress) {
                const double magnitude =
                    -prefactor *
                    (std::erf(alpha_r) / r -
                     two_alpha_over_sqrt_pi * std::exp(-alpha_r * alpha_r)) /
                    r_squared;
                if (need_forces) {
                    result.forces[3 * i] -= magnitude * d.x;
                    result.forces[3 * i + 1] -= magnitude * d.y;
                    result.forces[3 * i + 2] -= magnitude * d.z;
                    result.forces[3 * j] += magnitude * d.x;
                    result.forces[3 * j + 1] += magnitude * d.y;
                    result.forces[3 * j + 2] += magnitude * d.z;
                }
                if (need_stress) {
                    const double components[3] = {d.x, d.y, d.z};
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            virial[3 * a + b] -= magnitude * components[a] * components[b];
                        }
                    }
                }
            }
        }
        result.energy.excluded = energy;
    }

    // -----------------------------------------------------------------
    // Reciprocal space.
    // -----------------------------------------------------------------
    if (!kvectors_.empty()) {
        const std::size_t nk = kvectors_.size();

        // Phase factors by recursion: exp(i n b.r) = exp(i b.r)^n, so the only
        // trigonometry is O(N) per axis instead of O(N * K).
        const Mat3 reciprocal = reciprocal_cell(cell);
        const std::array<int, 3> extent = kmax_;
        const std::size_t stride0 = static_cast<std::size_t>(extent[0]) + 1;
        const std::size_t stride1 = 2 * static_cast<std::size_t>(extent[1]) + 1;
        const std::size_t stride2 = 2 * static_cast<std::size_t>(extent[2]) + 1;

        std::vector<double> ex_re(n * stride0), ex_im(n * stride0);
        std::vector<double> ey_re(n * stride1), ey_im(n * stride1);
        std::vector<double> ez_re(n * stride2), ez_im(n * stride2);

        const Vec3 b0 = row(reciprocal, 0);
        const Vec3 b1 = row(reciprocal, 1);
        const Vec3 b2 = row(reciprocal, 2);

        for (std::size_t i = 0; i < n; ++i) {
            const Vec3 r{positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]};
            const double t0 = dot(b0, r);
            const double t1 = dot(b1, r);
            const double t2 = dot(b2, r);

            ex_re[i * stride0] = 1.0;
            ex_im[i * stride0] = 0.0;
            if (extent[0] >= 1) {
                ex_re[i * stride0 + 1] = std::cos(t0);
                ex_im[i * stride0 + 1] = std::sin(t0);
            }
            for (int m = 2; m <= extent[0]; ++m) {
                const std::size_t k = i * stride0 + static_cast<std::size_t>(m);
                ex_re[k] = ex_re[k - 1] * ex_re[i * stride0 + 1] -
                           ex_im[k - 1] * ex_im[i * stride0 + 1];
                ex_im[k] = ex_im[k - 1] * ex_re[i * stride0 + 1] +
                           ex_re[k - 1] * ex_im[i * stride0 + 1];
            }

            // y and z run over negative indices too, stored with an offset.
            const auto fill = [&](std::vector<double>& re, std::vector<double>& im,
                                  std::size_t stride, int limit, double theta) {
                const std::size_t centre = i * stride + static_cast<std::size_t>(limit);
                re[centre] = 1.0;
                im[centre] = 0.0;
                if (limit >= 1) {
                    re[centre + 1] = std::cos(theta);
                    im[centre + 1] = std::sin(theta);
                    re[centre - 1] = re[centre + 1];
                    im[centre - 1] = -im[centre + 1];
                }
                for (int m = 2; m <= limit; ++m) {
                    const std::size_t up = centre + static_cast<std::size_t>(m);
                    re[up] = re[up - 1] * re[centre + 1] - im[up - 1] * im[centre + 1];
                    im[up] = im[up - 1] * re[centre + 1] + re[up - 1] * im[centre + 1];
                    const std::size_t down = centre - static_cast<std::size_t>(m);
                    re[down] = re[up];
                    im[down] = -im[up];
                }
            };
            fill(ey_re, ey_im, stride1, extent[1], t1);
            fill(ez_re, ez_im, stride2, extent[2], t2);
        }

        int nthreads = 1;
#ifdef _OPENMP
        nthreads = std::max(1, omp_get_max_threads());
#endif
        std::vector<double> partial_energy(static_cast<std::size_t>(nthreads), 0.0);
        std::vector<double> partial_virial(static_cast<std::size_t>(nthreads) * 9, 0.0);
        std::vector<double> partial_forces(
            need_forces ? static_cast<std::size_t>(nthreads) * 3 * n : 0, 0.0);

        const double energy_prefactor = kTwoPi * units::coulomb / volume;
        const double quarter_inverse_alpha_squared = 1.0 / (4.0 * alpha_ * alpha_);

#ifdef _OPENMP
#  pragma omp parallel num_threads(nthreads)
#endif
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            std::vector<double> cos_kr(n);
            std::vector<double> sin_kr(n);
            double* forces = need_forces
                                 ? partial_forces.data() + static_cast<std::size_t>(tid) * 3 * n
                                 : nullptr;
            double* virial_local = partial_virial.data() + static_cast<std::size_t>(tid) * 9;
            double energy_local = 0.0;

#ifdef _OPENMP
#  pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t kk = 0; kk < static_cast<std::ptrdiff_t>(nk); ++kk) {
                const auto k_index = static_cast<std::size_t>(kk);
                const std::array<int, 3>& indices = kindices_[k_index];
                const std::size_t ox = static_cast<std::size_t>(indices[0]);
                const std::size_t oy = static_cast<std::size_t>(indices[1] + extent[1]);
                const std::size_t oz = static_cast<std::size_t>(indices[2] + extent[2]);

                double structure_re = 0.0;
                double structure_im = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    const double xr = ex_re[i * stride0 + ox];
                    const double xi = ex_im[i * stride0 + ox];
                    const double yr = ey_re[i * stride1 + oy];
                    const double yi = ey_im[i * stride1 + oy];
                    const double zr = ez_re[i * stride2 + oz];
                    const double zi = ez_im[i * stride2 + oz];

                    const double xy_re = xr * yr - xi * yi;
                    const double xy_im = xr * yi + xi * yr;
                    const double c = xy_re * zr - xy_im * zi;
                    const double s = xy_re * zi + xy_im * zr;

                    cos_kr[i] = c;
                    sin_kr[i] = s;
                    structure_re += charges[i] * c;
                    structure_im += charges[i] * s;
                }

                const double factor = kfactors_[k_index];
                const double magnitude_squared =
                    structure_re * structure_re + structure_im * structure_im;
                // The factor of two restores the half of reciprocal space that
                // was not enumerated.
                energy_local += 2.0 * energy_prefactor * factor * magnitude_squared;

                if (need_forces) {
                    const Vec3 k = kvectors_[k_index];
                    const double scale = 4.0 * energy_prefactor * factor;
                    for (std::size_t i = 0; i < n; ++i) {
                        const double weight =
                            scale * charges[i] *
                            (structure_re * sin_kr[i] - structure_im * cos_kr[i]);
                        forces[3 * i] += weight * k.x;
                        forces[3 * i + 1] += weight * k.y;
                        forces[3 * i + 2] += weight * k.z;
                    }
                }

                if (need_stress) {
                    const Vec3 k = kvectors_[k_index];
                    const double common = 2.0 * energy_prefactor * factor * magnitude_squared;
                    const double coefficient =
                        2.0 * (1.0 / k.norm2() + quarter_inverse_alpha_squared);
                    const double components[3] = {k.x, k.y, k.z};
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            virial_local[3 * a + b] +=
                                common * (coefficient * components[a] * components[b] -
                                          (a == b ? 1.0 : 0.0));
                        }
                    }
                }
            }
            partial_energy[static_cast<std::size_t>(tid)] = energy_local;
        }

        double reciprocal_energy = 0.0;
        for (const double part : partial_energy) {
            reciprocal_energy += part;
        }
        result.energy.reciprocal = reciprocal_energy;

        if (need_forces) {
            for (int t = 0; t < nthreads; ++t) {
                const double* part = partial_forces.data() + static_cast<std::size_t>(t) * 3 * n;
                for (std::size_t k = 0; k < 3 * n; ++k) {
                    result.forces[k] += part[k];
                }
            }
        }
        if (need_stress) {
            // The reciprocal contribution is already dE/d(strain), not a pair
            // virial, so it goes in directly.
            for (int t = 0; t < nthreads; ++t) {
                const double* part = partial_virial.data() + static_cast<std::size_t>(t) * 9;
                for (int k = 0; k < 9; ++k) {
                    virial[k] += part[k];
                }
            }
        }
    }

    if (need_stress) {
        // Pair terms entered `virial` as -sum(magnitude * d_a * d_b), which is
        // dE/d(strain); dividing by the volume gives the stress.
        const double inverse_volume = 1.0 / volume;
        result.stress = {inverse_volume * virial[0],
                         inverse_volume * virial[4],
                         inverse_volume * virial[8],
                         inverse_volume * 0.5 * (virial[5] + virial[7]),
                         inverse_volume * 0.5 * (virial[2] + virial[6]),
                         inverse_volume * 0.5 * (virial[1] + virial[3])};
    }
    return result;
}

std::string Ewald::to_string() const {
    std::ostringstream out;
    out << "Ewald(alpha=" << alpha_ << " 1/A, real_cutoff=" << real_cutoff_
        << " A, kcutoff=" << kcutoff_ << " 1/A, " << kvectors_.size() << " k-vectors)";
    return out.str();
}

}  // namespace fastatoms::ff
