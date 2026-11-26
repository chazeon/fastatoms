// SPDX-License-Identifier: MIT
#include "fastatoms/core/atoms.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <cstdio>
#include <stdexcept>

#include "fastatoms/core/elements.hpp"
#include "fastatoms/core/units.hpp"

namespace fastatoms {
namespace {

/// Names that are owned by dedicated members and may not be shadowed by a
/// user-defined array.
bool is_reserved_array_name(const std::string& name) {
    static const std::vector<std::string> reserved{"positions", "numbers",  "masses", "momenta",
                                                   "charges",   "tags",     "symbols", "cell",
                                                   "pbc",       "velocities"};
    return std::find(reserved.begin(), reserved.end(), name) != reserved.end();
}

void require_length(const char* what, std::size_t got, std::size_t expected) {
    if (got != expected) {
        throw std::invalid_argument(std::string(what) + ": expected " + std::to_string(expected) +
                                    " values, got " + std::to_string(got));
    }
}

/// Copy `indices` rows out of a per-atom array with `ncols` components.
template <typename T>
std::vector<T> gather(const std::vector<T>& src, std::size_t ncols,
                      const std::vector<std::size_t>& indices) {
    if (src.empty()) {
        return {};
    }
    std::vector<T> out;
    out.resize(indices.size() * ncols);
    for (std::size_t n = 0; n < indices.size(); ++n) {
        const T* from = src.data() + indices[n] * ncols;
        T* to = out.data() + n * ncols;
        for (std::size_t c = 0; c < ncols; ++c) {
            to[c] = from[c];
        }
    }
    return out;
}

/// Append `src` to `dst`, materialising `dst` with `fill` first when only one
/// of the two structures carries the array.
template <typename T>
void merge_optional(std::vector<T>& dst, const std::vector<T>& src, std::size_t dst_atoms,
                    std::size_t src_atoms, std::size_t ncols, T fill) {
    if (dst.empty() && src.empty()) {
        return;
    }
    if (dst.empty()) {
        dst.assign(dst_atoms * ncols, fill);
    }
    if (src.empty()) {
        dst.insert(dst.end(), src_atoms * ncols, fill);
    } else {
        dst.insert(dst.end(), src.begin(), src.end());
    }
}

}  // namespace

Atoms::Atoms(std::vector<int> numbers, std::vector<double> positions, Cell cell)
    : numbers_(std::move(numbers)), positions_(std::move(positions)), cell_(std::move(cell)) {
    require_length("Atoms: positions", positions_.size(), 3 * numbers_.size());
    for (const int z : numbers_) {
        if (z < 0 || z > kMaxAtomicNumber) {
            throw std::invalid_argument("Atoms: atomic number " + std::to_string(z) +
                                        " is out of range");
        }
    }
}

Atoms::Atoms(const std::vector<std::string>& symbols, std::vector<double> positions, Cell cell)
    : Atoms([&symbols] {
                std::vector<int> numbers;
                numbers.reserve(symbols.size());
                for (const auto& s : symbols) {
                    numbers.push_back(number_from_symbol(s));
                }
                return numbers;
            }(),
            std::move(positions), std::move(cell)) {}

void Atoms::check_index(std::size_t i) const {
    if (i >= size()) {
        throw std::out_of_range("atom index " + std::to_string(i) + " is out of range for " +
                                std::to_string(size()) + " atoms");
    }
}

void Atoms::reserve(std::size_t n) {
    numbers_.reserve(n);
    positions_.reserve(3 * n);
}

void Atoms::clear() {
    numbers_.clear();
    positions_.clear();
    masses_.clear();
    velocities_.clear();
    charges_.clear();
    tags_.clear();
    extra_.clear();
    extra_int_.clear();
    extra_str_.clear();
}

void Atoms::resize_optional(std::size_t n) {
    if (!masses_.empty()) {
        masses_.resize(n, 0.0);
    }
    if (!velocities_.empty()) {
        velocities_.resize(3 * n, 0.0);
    }
    if (!charges_.empty()) {
        charges_.resize(n, 0.0);
    }
    if (!tags_.empty()) {
        tags_.resize(n, 0);
    }
    for (auto& [name, array] : extra_) {
        array.data.resize(n * array.ncols, 0.0);
    }
    for (auto& [name, array] : extra_int_) {
        array.data.resize(n * array.ncols, 0);
    }
    for (auto& [name, array] : extra_str_) {
        // A new atom's label is its symbol, never empty (as in extend).
        const std::size_t old_size = array.size();
        array.resize(n);
        if (name == "labels") {
            for (std::size_t i = old_size; i < n && i < numbers_.size(); ++i) {
                array[i] = std::string(symbol_from_number(numbers_[i]));
            }
        }
    }
}

void Atoms::set_positions(const std::vector<double>& positions) {
    require_length("set_positions", positions.size(), 3 * size());
    positions_ = positions;
}

void Atoms::set_numbers(const std::vector<int>& numbers) {
    require_length("set_numbers", numbers.size(), size());
    for (const int z : numbers) {
        if (z < 0 || z > kMaxAtomicNumber) {
            throw std::invalid_argument("set_numbers: atomic number " + std::to_string(z) +
                                        " is out of range");
        }
    }
    numbers_ = numbers;
}

void Atoms::set_number(std::size_t i, int z) {
    check_index(i);
    if (z < 0 || z > kMaxAtomicNumber) {
        throw std::invalid_argument("set_number: atomic number " + std::to_string(z) +
                                    " is out of range");
    }
    numbers_[i] = z;
}

std::vector<std::string> Atoms::symbols() const {
    std::vector<std::string> out;
    out.reserve(size());
    for (const int z : numbers_) {
        out.emplace_back(symbol_from_number(z));
    }
    return out;
}

std::string Atoms::symbol(std::size_t i) const {
    check_index(i);
    return std::string(symbol_from_number(numbers_[i]));
}

void Atoms::set_symbols(const std::vector<std::string>& symbols) {
    require_length("set_symbols", symbols.size(), size());
    std::vector<int> numbers;
    numbers.reserve(symbols.size());
    for (const auto& s : symbols) {
        numbers.push_back(number_from_symbol(s));
    }
    numbers_ = std::move(numbers);
}

std::string Atoms::formula() const {
    std::map<std::string, std::size_t> counts;
    for (const int z : numbers_) {
        ++counts[std::string(symbol_from_number(z))];
    }

    // Hill notation: carbon first, then hydrogen, then everything else in
    // alphabetical order (and purely alphabetical when there is no carbon).
    std::vector<std::string> order;
    order.reserve(counts.size());
    const bool organic = counts.count("C") != 0;
    if (organic) {
        order.emplace_back("C");
        if (counts.count("H") != 0) {
            order.emplace_back("H");
        }
    }
    for (const auto& [sym, count] : counts) {
        if (organic && (sym == "C" || sym == "H")) {
            continue;
        }
        order.push_back(sym);
    }

    std::string out;
    for (const auto& sym : order) {
        out += sym;
        const std::size_t n = counts[sym];
        if (n > 1) {
            out += std::to_string(n);
        }
    }
    return out;
}

void Atoms::set_cell(const Mat3& matrix, bool scale_atoms) {
    Cell updated = cell_;
    updated.set_matrix(matrix);
    set_cell(updated, scale_atoms);
}

void Atoms::set_cell(const Cell& cell, bool scale_atoms) {
    if (scale_atoms) {
        if (!cell_.is_invertible()) {
            throw std::invalid_argument(
                "set_cell(scale_atoms=true): the current cell is not invertible");
        }
        const std::vector<double> scaled = scaled_positions();
        cell_ = cell;
        set_scaled_positions(scaled);
    } else {
        cell_ = cell;
    }
}

std::vector<double> Atoms::scaled_positions(bool wrapped) const {
    const std::size_t n = size();
    std::vector<double> out(3 * n);
    if (!cell_.is_invertible()) {
        throw std::invalid_argument(
            "scaled_positions: the cell is not invertible (it spans fewer than three dimensions)");
    }
    for (std::size_t i = 0; i < n; ++i) {
        Vec3 s = cell_.scaled(position(i));
        if (wrapped) {
            s = cell_.wrap_scaled(s);
        }
        s.store(out.data() + 3 * i);
    }
    return out;
}

void Atoms::set_scaled_positions(const std::vector<double>& scaled) {
    require_length("set_scaled_positions", scaled.size(), 3 * size());
    for (std::size_t i = 0; i < size(); ++i) {
        set_position(i, cell_.cartesian(Vec3::load(scaled.data() + 3 * i)));
    }
}

void Atoms::wrap(double eps) {
    if (!cell_.any_pbc() || !cell_.is_invertible()) {
        return;
    }
    const std::size_t n = size();
    double* FASTATOMS_RESTRICT p = positions_.data();
    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 s = cell_.wrap_scaled(cell_.scaled(Vec3::load(p + 3 * i)), eps);
        cell_.cartesian(s).store(p + 3 * i);
    }
}

std::vector<double> Atoms::masses() const {
    if (!masses_.empty()) {
        return masses_;
    }
    std::vector<double> out;
    out.reserve(size());
    for (const int z : numbers_) {
        out.push_back(atomic_mass(z));
    }
    return out;
}

double Atoms::mass(std::size_t i) const {
    check_index(i);
    return masses_.empty() ? atomic_mass(numbers_[i]) : masses_[i];
}

void Atoms::set_masses(const std::vector<double>& masses) {
    if (masses.empty()) {
        masses_.clear();
        return;
    }
    require_length("set_masses", masses.size(), size());
    masses_ = masses;
}

std::vector<double> Atoms::momenta() const {
    std::vector<double> out(3 * size(), 0.0);
    if (velocities_.empty()) {
        return out;
    }
    for (std::size_t i = 0; i < size(); ++i) {
        const double m = mass(i);
        for (std::size_t k = 0; k < 3; ++k) {
            out[3 * i + k] = velocities_[3 * i + k] * m;
        }
    }
    return out;
}

void Atoms::set_momenta(const std::vector<double>& momenta) {
    if (momenta.empty()) {
        velocities_.clear();
        return;
    }
    require_length("set_momenta", momenta.size(), 3 * size());
    velocities_.resize(3 * size());
    for (std::size_t i = 0; i < size(); ++i) {
        const double m = mass(i);
        const double inv = m > 0.0 ? 1.0 / m : 0.0;
        for (std::size_t k = 0; k < 3; ++k) {
            velocities_[3 * i + k] = momenta[3 * i + k] * inv;
        }
    }
}

std::vector<double> Atoms::velocities() const {
    return velocities_.empty() ? std::vector<double>(3 * size(), 0.0) : velocities_;
}

void Atoms::set_velocities(const std::vector<double>& velocities) {
    if (velocities.empty()) {
        velocities_.clear();
        return;
    }
    require_length("set_velocities", velocities.size(), 3 * size());
    velocities_ = velocities;
}

std::vector<int> Atoms::tags() const {
    return tags_.empty() ? std::vector<int>(size(), 0) : tags_;
}

void Atoms::set_tags(const std::vector<int>& tags) {
    if (tags.empty()) {
        tags_.clear();
        return;
    }
    require_length("set_tags", tags.size(), size());
    tags_ = tags;
}

std::vector<double> Atoms::charges() const {
    return charges_.empty() ? std::vector<double>(size(), 0.0) : charges_;
}

void Atoms::set_info(Value info) {
    if (!info.is_null() && !info.is_map()) {
        throw std::invalid_argument("Atoms::set_info: info must be a mapping");
    }
    info_ = info.is_null() ? Value(Value::Map{}) : std::move(info);
}

void Atoms::set_charges(const std::vector<double>& charges) {
    if (charges.empty()) {
        charges_.clear();
        return;
    }
    require_length("set_charges", charges.size(), size());
    charges_ = charges;
}

void Atoms::set_array(const std::string& name, const std::vector<double>& data, std::size_t ncols) {
    if (is_reserved_array_name(name)) {
        throw std::invalid_argument("set_array: '" + name +
                                    "' is a built-in array; use its dedicated setter");
    }
    if (ncols == 0) {
        throw std::invalid_argument("set_array: ncols must be at least 1");
    }
    require_length(("set_array('" + name + "')").c_str(), data.size(), size() * ncols);
    extra_[name] = Array{data, ncols};
}

bool Atoms::has_array(const std::string& name) const { return extra_.count(name) != 0; }

const Atoms::Array& Atoms::array(const std::string& name) const {
    const auto it = extra_.find(name);
    if (it == extra_.end()) {
        throw std::out_of_range("no per-atom array named '" + name + "'");
    }
    return it->second;
}

void Atoms::remove_array(const std::string& name) { extra_.erase(name); }

std::vector<std::string> Atoms::array_names() const {
    std::vector<std::string> names;
    names.reserve(extra_.size());
    for (const auto& [name, array] : extra_) {
        names.push_back(name);
    }
    return names;
}

void Atoms::set_int_array(const std::string& name, const std::vector<std::int64_t>& data,
                          std::size_t ncols) {
    if (ncols == 0) {
        throw std::invalid_argument("set_int_array: ncols must be at least 1");
    }
    require_length(("set_int_array('" + name + "')").c_str(), data.size(), size() * ncols);
    extra_int_[name] = IntArray{data, ncols};
}

bool Atoms::has_int_array(const std::string& name) const { return extra_int_.count(name) != 0; }

const Atoms::IntArray& Atoms::int_array(const std::string& name) const {
    const auto it = extra_int_.find(name);
    if (it == extra_int_.end()) {
        throw std::out_of_range("no per-atom integer array named '" + name + "'");
    }
    return it->second;
}

void Atoms::remove_int_array(const std::string& name) { extra_int_.erase(name); }

std::vector<std::string> Atoms::int_array_names() const {
    std::vector<std::string> names;
    names.reserve(extra_int_.size());
    for (const auto& [name, array] : extra_int_) {
        names.push_back(name);
    }
    return names;
}

void Atoms::set_string_array(const std::string& name, const std::vector<std::string>& data) {
    require_length(("set_string_array('" + name + "')").c_str(), data.size(), size());
    extra_str_[name] = data;
}

bool Atoms::has_string_array(const std::string& name) const {
    return extra_str_.count(name) != 0;
}

const std::vector<std::string>& Atoms::string_array(const std::string& name) const {
    const auto it = extra_str_.find(name);
    if (it == extra_str_.end()) {
        throw std::out_of_range("no per-atom string array named '" + name + "'");
    }
    return it->second;
}

void Atoms::remove_string_array(const std::string& name) { extra_str_.erase(name); }

std::vector<std::string> Atoms::string_array_names() const {
    std::vector<std::string> names;
    names.reserve(extra_str_.size());
    for (const auto& [name, array] : extra_str_) {
        names.push_back(name);
    }
    return names;
}

std::vector<std::string> Atoms::labels() const {
    const auto it = extra_str_.find("labels");
    return it != extra_str_.end() ? it->second : symbols();
}

Vec3 Atoms::distance_vector(std::size_t i, std::size_t j, bool mic) const {
    check_index(i);
    check_index(j);
    const Vec3 dr = position(j) - position(i);
    return mic ? cell_.minimum_image(dr) : dr;
}

double Atoms::distance(std::size_t i, std::size_t j, bool mic) const {
    return distance_vector(i, j, mic).norm();
}

std::vector<double> Atoms::distances_from(std::size_t i, bool mic) const {
    check_index(i);
    const std::size_t n = size();
    std::vector<double> out(n);
    const Vec3 ri = position(i);
    for (std::size_t j = 0; j < n; ++j) {
        const Vec3 dr = position(j) - ri;
        out[j] = (mic ? cell_.minimum_image(dr) : dr).norm();
    }
    return out;
}

std::vector<double> Atoms::all_distances(bool mic) const {
    const std::size_t n = size();
    std::vector<double> out(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 ri = position(i);
        for (std::size_t j = i + 1; j < n; ++j) {
            const Vec3 dr = position(j) - ri;
            const double d = (mic ? cell_.minimum_image(dr) : dr).norm();
            out[i * n + j] = d;
            out[j * n + i] = d;
        }
    }
    return out;
}

Vec3 Atoms::center_of_mass() const {
    Vec3 sum{};
    double total = 0.0;
    for (std::size_t i = 0; i < size(); ++i) {
        const double m = mass(i);
        sum += position(i) * m;
        total += m;
    }
    if (total <= 0.0) {
        return center_of_geometry();
    }
    return sum / total;
}

Vec3 Atoms::center_of_geometry() const {
    if (empty()) {
        return {};
    }
    Vec3 sum{};
    for (std::size_t i = 0; i < size(); ++i) {
        sum += position(i);
    }
    return sum / static_cast<double>(size());
}

Mat3 Atoms::moments_of_inertia_tensor() const {
    const Vec3 com = center_of_mass();
    Mat3 inertia{};
    for (std::size_t i = 0; i < size(); ++i) {
        const Vec3 r = position(i) - com;
        const double m = mass(i);
        inertia(0, 0) += m * (r.y * r.y + r.z * r.z);
        inertia(1, 1) += m * (r.x * r.x + r.z * r.z);
        inertia(2, 2) += m * (r.x * r.x + r.y * r.y);
        inertia(0, 1) -= m * r.x * r.y;
        inertia(0, 2) -= m * r.x * r.z;
        inertia(1, 2) -= m * r.y * r.z;
    }
    inertia(1, 0) = inertia(0, 1);
    inertia(2, 0) = inertia(0, 2);
    inertia(2, 1) = inertia(1, 2);
    return inertia;
}

void Atoms::translate(const Vec3& shift) {
    const std::size_t n = size();
    double* FASTATOMS_RESTRICT p = positions_.data();
    for (std::size_t i = 0; i < n; ++i) {
        p[3 * i] += shift.x;
        p[3 * i + 1] += shift.y;
        p[3 * i + 2] += shift.z;
    }
}

void Atoms::translate(const std::vector<double>& shifts) {
    require_length("translate", shifts.size(), 3 * size());
    const std::size_t n = 3 * size();
    double* FASTATOMS_RESTRICT p = positions_.data();
    const double* FASTATOMS_RESTRICT s = shifts.data();
    for (std::size_t k = 0; k < n; ++k) {
        p[k] += s[k];
    }
}

void Atoms::rotate(double angle, const Vec3& axis, const Vec3& center) {
    if (axis.norm2() == 0.0) {
        throw std::invalid_argument("rotate: the rotation axis must be non-zero");
    }
    const Mat3 R = rotation_matrix(axis, angle);
    for (std::size_t i = 0; i < size(); ++i) {
        set_position(i, (position(i) - center) * R + center);
    }
    if (!velocities_.empty()) {
        for (std::size_t i = 0; i < size(); ++i) {
            const Vec3 v = Vec3::load(velocities_.data() + 3 * i) * R;
            v.store(velocities_.data() + 3 * i);
        }
    }
}

void Atoms::rotate(double angle, const Vec3& axis) { rotate(angle, axis, center_of_mass()); }

void Atoms::center(double vacuum, const std::array<bool, 3>& axes) {
    if (empty()) {
        return;
    }

    if (cell_.is_invertible()) {
        // Work in lattice space so that triclinic cells behave sensibly.
        Mat3 matrix = cell_.matrix();
        if (vacuum >= 0.0) {
            const std::vector<double> scaled = scaled_positions();
            const Vec3 heights = cell_.heights();
            for (std::size_t k = 0; k < 3; ++k) {
                if (!axes[k]) {
                    continue;
                }
                double lo = std::numeric_limits<double>::max();
                double hi = std::numeric_limits<double>::lowest();
                for (std::size_t i = 0; i < size(); ++i) {
                    lo = std::min(lo, scaled[3 * i + k]);
                    hi = std::max(hi, scaled[3 * i + k]);
                }
                // Scaling a lattice vector scales the cell volume, and hence
                // this perpendicular width, by exactly the same factor.
                const double extent = (hi - lo) * heights[k];
                const double target = extent + 2.0 * vacuum;
                if (heights[k] > 0.0 && target > 0.0) {
                    matrix.set_row(k, matrix.row(k) * (target / heights[k]));
                }
            }
            cell_.set_matrix(matrix);
        }

        const std::vector<double> scaled = scaled_positions();
        Vec3 shift{};
        for (std::size_t k = 0; k < 3; ++k) {
            if (!axes[k]) {
                continue;
            }
            double lo = std::numeric_limits<double>::max();
            double hi = std::numeric_limits<double>::lowest();
            for (std::size_t i = 0; i < size(); ++i) {
                lo = std::min(lo, scaled[3 * i + k]);
                hi = std::max(hi, scaled[3 * i + k]);
            }
            shift += cell_.matrix().row(k) * (0.5 - 0.5 * (lo + hi));
        }
        translate(shift);
        return;
    }

    // Non-invertible cell: fall back to centring along the Cartesian axes.
    Mat3 matrix = cell_.matrix();
    Vec3 lo{std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()};
    Vec3 hi{std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(),
            std::numeric_limits<double>::lowest()};
    for (std::size_t i = 0; i < size(); ++i) {
        const Vec3 r = position(i);
        for (std::size_t k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], r[k]);
            hi[k] = std::max(hi[k], r[k]);
        }
    }

    Vec3 shift{};
    for (std::size_t k = 0; k < 3; ++k) {
        if (!axes[k]) {
            continue;
        }
        double length = matrix.row(k).norm();
        if (vacuum >= 0.0) {
            length = (hi[k] - lo[k]) + 2.0 * vacuum;
            Vec3 row{};
            row[k] = length;
            matrix.set_row(k, row);
        }
        shift[k] = 0.5 * length - 0.5 * (lo[k] + hi[k]);
    }
    cell_.set_matrix(matrix);
    translate(shift);
}

double Atoms::kinetic_energy() const {
    if (velocities_.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (std::size_t i = 0; i < size(); ++i) {
        const double m = mass(i);
        total += m * Vec3::load(velocities_.data() + 3 * i).norm2();
    }
    return 0.5 * total;
}

double Atoms::temperature(int dof) const {
    const int ndof = dof >= 0 ? dof : 3 * static_cast<int>(size());
    if (ndof <= 0) {
        return 0.0;
    }
    return 2.0 * kinetic_energy() / (static_cast<double>(ndof) * units::kB);
}

Vec3 Atoms::total_momentum() const {
    Vec3 total{};
    if (velocities_.empty()) {
        return total;
    }
    for (std::size_t i = 0; i < size(); ++i) {
        total += Vec3::load(velocities_.data() + 3 * i) * mass(i);
    }
    return total;
}

void Atoms::zero_center_of_mass_momentum() {
    if (velocities_.empty() || empty()) {
        return;
    }
    double total_mass = 0.0;
    for (std::size_t i = 0; i < size(); ++i) {
        total_mass += mass(i);
    }
    if (total_mass <= 0.0) {
        return;
    }
    const Vec3 v_cm = total_momentum() / total_mass;
    for (std::size_t i = 0; i < size(); ++i) {
        const Vec3 v = Vec3::load(velocities_.data() + 3 * i) - v_cm;
        v.store(velocities_.data() + 3 * i);
    }
}

void Atoms::append(int number, const Vec3& position) {
    if (number < 0 || number > kMaxAtomicNumber) {
        throw std::invalid_argument("append: atomic number " + std::to_string(number) +
                                    " is out of range");
    }
    numbers_.push_back(number);
    positions_.push_back(position.x);
    positions_.push_back(position.y);
    positions_.push_back(position.z);
    resize_optional(numbers_.size());
}

void Atoms::append(const std::string& symbol, const Vec3& position) {
    append(number_from_symbol(symbol), position);
}

void Atoms::extend(const Atoms& other) {
    if (&other == this) {
        const Atoms copy = other;  // inserting a vector into itself is UB
        extend(copy);
        return;
    }
    const std::size_t n_this = size();
    const std::size_t n_other = other.size();
    if (n_other == 0) {
        return;
    }

    numbers_.insert(numbers_.end(), other.numbers_.begin(), other.numbers_.end());
    positions_.insert(positions_.end(), other.positions_.begin(), other.positions_.end());

    // A quantity that is explicit in one structure and implicit in the other
    // is materialised from the implicit default so that lengths stay in sync.
    if (!masses_.empty() || !other.masses_.empty()) {
        std::vector<double> mine = masses();
        const std::vector<double> theirs = other.masses();
        mine.insert(mine.end(), theirs.begin(), theirs.end());
        masses_ = std::move(mine);
    }
    merge_optional(velocities_, other.velocities_, n_this, n_other, 3, 0.0);
    merge_optional(charges_, other.charges_, n_this, n_other, 1, 0.0);
    merge_optional(tags_, other.tags_, n_this, n_other, 1, 0);

    for (auto& [name, array] : extra_) {
        const auto it = other.extra_.find(name);
        if (it != other.extra_.end() && it->second.ncols == array.ncols) {
            array.data.insert(array.data.end(), it->second.data.begin(), it->second.data.end());
        } else {
            array.data.resize(array.data.size() + n_other * array.ncols, 0.0);
        }
    }
    for (const auto& [name, array] : other.extra_) {
        if (extra_.count(name) != 0) {
            continue;
        }
        Array merged;
        merged.ncols = array.ncols;
        merged.data.assign(n_this * array.ncols, 0.0);
        merged.data.insert(merged.data.end(), array.data.begin(), array.data.end());
        extra_.emplace(name, std::move(merged));
    }

    for (auto& [name, array] : extra_int_) {
        const auto it = other.extra_int_.find(name);
        if (it != other.extra_int_.end() && it->second.ncols == array.ncols) {
            array.data.insert(array.data.end(), it->second.data.begin(), it->second.data.end());
        } else {
            array.data.resize(array.data.size() + n_other * array.ncols, 0);
        }
    }
    for (const auto& [name, array] : other.extra_int_) {
        if (extra_int_.count(name) != 0) {
            continue;
        }
        IntArray merged;
        merged.ncols = array.ncols;
        merged.data.assign(n_this * array.ncols, 0);
        merged.data.insert(merged.data.end(), array.data.begin(), array.data.end());
        extra_int_.emplace(name, std::move(merged));
    }

    // A string array missing on one side is filled with that side's symbols
    // for "labels" (so a label is never empty) and with "" otherwise.
    std::vector<std::string> names = string_array_names();
    for (const auto& name : other.string_array_names()) {
        if (extra_str_.count(name) == 0) {
            names.push_back(name);
        }
    }
    for (const auto& name : names) {
        auto fill = [&name](const Atoms& atoms, std::size_t begin, std::size_t count) {
            std::vector<std::string> out;
            out.reserve(count);
            for (std::size_t i = begin; i < begin + count; ++i) {
                out.push_back(name == "labels" ? atoms.symbol(i) : std::string{});
            }
            return out;
        };
        auto& mine = extra_str_[name];
        if (mine.empty()) {
            mine = fill(*this, 0, n_this);
        }
        const auto it = other.extra_str_.find(name);
        if (it != other.extra_str_.end()) {
            mine.insert(mine.end(), it->second.begin(), it->second.end());
        } else {
            const auto extra = fill(other, 0, n_other);
            mine.insert(mine.end(), extra.begin(), extra.end());
        }
    }
}

Atoms Atoms::subset(const std::vector<std::size_t>& indices) const {
    for (const std::size_t i : indices) {
        check_index(i);
    }

    Atoms out;
    out.numbers_ = gather(numbers_, 1, indices);
    out.positions_ = gather(positions_, 3, indices);
    out.masses_ = gather(masses_, 1, indices);
    out.velocities_ = gather(velocities_, 3, indices);
    out.charges_ = gather(charges_, 1, indices);
    out.tags_ = gather(tags_, 1, indices);
    for (const auto& [name, array] : extra_) {
        out.extra_.emplace(name, Array{gather(array.data, array.ncols, indices), array.ncols});
    }
    for (const auto& [name, array] : extra_int_) {
        out.extra_int_.emplace(name,
                               IntArray{gather(array.data, array.ncols, indices), array.ncols});
    }
    for (const auto& [name, array] : extra_str_) {
        out.extra_str_.emplace(name, gather(array, 1, indices));
    }
    out.cell_ = cell_;
    out.info_ = info_;
    return out;
}

void Atoms::remove(std::vector<std::size_t> indices) {
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    if (!indices.empty()) {
        check_index(indices.back());
    }

    std::vector<std::size_t> keep;
    keep.reserve(size() - indices.size());
    auto next = indices.begin();
    for (std::size_t i = 0; i < size(); ++i) {
        if (next != indices.end() && *next == i) {
            ++next;
            continue;
        }
        keep.push_back(i);
    }
    *this = subset(keep);
}

Atoms Atoms::repeat(int nx, int ny, int nz) const {
    if (nx < 1 || ny < 1 || nz < 1) {
        throw std::invalid_argument("repeat: all repetition counts must be at least 1");
    }
    const std::size_t n = size();
    const std::size_t images = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                               static_cast<std::size_t>(nz);

    Atoms out;
    out.numbers_.reserve(n * images);
    out.positions_.reserve(3 * n * images);

    const Mat3& m = cell_.matrix();
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int iz = 0; iz < nz; ++iz) {
                const Vec3 shift = m.row(0) * ix + m.row(1) * iy + m.row(2) * iz;
                for (std::size_t i = 0; i < n; ++i) {
                    out.numbers_.push_back(numbers_[i]);
                    const Vec3 r = position(i) + shift;
                    out.positions_.push_back(r.x);
                    out.positions_.push_back(r.y);
                    out.positions_.push_back(r.z);
                }
            }
        }
    }

    auto tile = [images](const auto& src) {
        using T = typename std::decay_t<decltype(src)>::value_type;
        std::vector<T> out_data;
        if (src.empty()) {
            return out_data;
        }
        out_data.reserve(src.size() * images);
        for (std::size_t k = 0; k < images; ++k) {
            out_data.insert(out_data.end(), src.begin(), src.end());
        }
        return out_data;
    };

    out.masses_ = tile(masses_);
    out.velocities_ = tile(velocities_);
    out.charges_ = tile(charges_);
    out.tags_ = tile(tags_);
    for (const auto& [name, array] : extra_) {
        out.extra_.emplace(name, Array{tile(array.data), array.ncols});
    }
    for (const auto& [name, array] : extra_int_) {
        out.extra_int_.emplace(name, IntArray{tile(array.data), array.ncols});
    }
    for (const auto& [name, array] : extra_str_) {
        out.extra_str_.emplace(name, tile(array));
    }
    out.info_ = info_;

    out.cell_ = cell_;
    out.cell_.set_matrix(Mat3::from_rows(m.row(0) * nx, m.row(1) * ny, m.row(2) * nz));
    return out;
}

std::string Atoms::to_string() const {
    std::string out = "Atoms(" + formula() + ", n=" + std::to_string(size());
    if (!cell_.is_zero()) {
        const Vec3 l = cell_.lengths();
        char buffer[96];
        const int n = std::snprintf(buffer, sizeof buffer, ", cell=[%.4f, %.4f, %.4f]", l.x, l.y, l.z);
        out.append(buffer, static_cast<std::size_t>(n));
    }
    const auto& p = cell_.pbc();
    out += std::string(", pbc=[") + (p[0] ? "True" : "False") + ", " + (p[1] ? "True" : "False") +
           ", " + (p[2] ? "True" : "False") + "])";
    return out;
}

}  // namespace fastatoms
