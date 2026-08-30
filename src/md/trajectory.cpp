// SPDX-License-Identifier: MIT
#include "fastatoms/md/trajectory.hpp"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "fastatoms/util/text.hpp"

namespace fastatoms::md {
namespace {

std::string format(double value, int precision = 8) {
    std::ostringstream out;
    out << std::setprecision(precision) << value;
    return out.str();
}

/// The forces from the last evaluation, if the calculator left any behind.
const std::vector<double>* stored_forces(const Atoms& atoms) {
    if (!atoms.has_array("forces")) {
        return nullptr;
    }
    const Atoms::Array& array = atoms.array("forces");
    if (array.ncols != 3 || array.data.size() != 3 * atoms.size()) {
        return nullptr;
    }
    return &array.data;
}

}  // namespace

// ---------------------------------------------------------------------------
// Trajectory
// ---------------------------------------------------------------------------

TrajectoryWriter::TrajectoryWriter(std::filesystem::path path, TrajectoryOptions options)
    : path_(std::move(path)), options_(options) {
    if (path_.empty()) {
        throw DynamicsError("the trajectory needs a file name");
    }
    if (options_.interval < 1) {
        throw DynamicsError("the trajectory interval must be at least 1");
    }
    if (path_.has_parent_path() && !path_.parent_path().empty()) {
        std::error_code error;
        std::filesystem::create_directories(path_.parent_path(), error);
    }
    if (!options_.append) {
        std::ofstream truncate(path_, std::ios::trunc);
        if (!truncate) {
            throw DynamicsError("cannot open '" + path_.string() + "' for writing");
        }
    }
}

bool TrajectoryWriter::wants(int step) const { return step % options_.interval == 0; }

std::string TrajectoryWriter::frame(const Atoms& atoms, const DynamicsStep& step,
                                    const TrajectoryOptions& options) {
    std::ostringstream out;
    out << atoms.size() << "\n";

    // The comment line carries the cell and the per-frame scalars, in the
    // key=value form extended XYZ readers expect.
    const Mat3& cell = atoms.cell().matrix();
    out << "Lattice=\"";
    for (std::size_t k = 0; k < 9; ++k) {
        out << (k ? " " : "") << format(cell.m[k]);
    }
    out << "\" Properties=species:S:1";
    if (options.positions) {
        out << ":pos:R:3";
    }
    if (options.velocities) {
        out << ":velo:R:3";
    }
    if (options.forces) {
        out << ":forces:R:3";
    }

    const auto& pbc = atoms.pbc();
    out << " pbc=\"" << (pbc[0] ? 'T' : 'F') << " " << (pbc[1] ? 'T' : 'F') << " "
        << (pbc[2] ? 'T' : 'F') << "\"";
    out << " step=" << step.step;
    out << " time=" << format(step.time, 10);
    out << " energy=" << format(step.potential_energy, 12);
    out << " kinetic_energy=" << format(step.kinetic_energy, 12);
    out << " total_energy=" << format(step.total_energy, 12);
    out << " conserved_energy=" << format(step.conserved_energy, 12);
    out << " temperature=" << format(step.temperature, 8);
    out << " pressure=" << format(step.pressure, 8);
    out << " volume=" << format(step.volume, 10);
    out << "\n";

    const std::vector<double> velocities =
        options.velocities ? atoms.velocities() : std::vector<double>{};
    const std::vector<double>* forces = options.forces ? stored_forces(atoms) : nullptr;

    for (std::size_t i = 0; i < atoms.size(); ++i) {
        out << atoms.symbol(i);
        if (options.positions) {
            const Vec3 r = atoms.position(i);
            out << " " << format(r.x) << " " << format(r.y) << " " << format(r.z);
        }
        if (options.velocities) {
            out << " " << format(velocities[3 * i]) << " " << format(velocities[3 * i + 1])
                << " " << format(velocities[3 * i + 2]);
        }
        if (options.forces) {
            // A run that never evaluated forces still produces a valid frame.
            const double fx = forces != nullptr ? (*forces)[3 * i] : 0.0;
            const double fy = forces != nullptr ? (*forces)[3 * i + 1] : 0.0;
            const double fz = forces != nullptr ? (*forces)[3 * i + 2] : 0.0;
            out << " " << format(fx) << " " << format(fy) << " " << format(fz);
        }
        out << "\n";
    }
    return out.str();
}

void TrajectoryWriter::write(const Atoms& atoms, const DynamicsStep& step) {
    std::ofstream stream(path_, std::ios::app);
    if (!stream) {
        throw DynamicsError("cannot append to '" + path_.string() + "'");
    }
    stream << frame(atoms, step, options_);
    if (!stream) {
        throw DynamicsError("failed while writing '" + path_.string() + "'");
    }
    ++frames_;
}

// ---------------------------------------------------------------------------
// Log
// ---------------------------------------------------------------------------

LogWriter::LogWriter(std::filesystem::path path, int interval)
    : path_(std::move(path)), interval_(interval) {
    if (interval_ < 1) {
        throw DynamicsError("the log interval must be at least 1");
    }
    if (!path_.empty()) {
        if (path_.has_parent_path() && !path_.parent_path().empty()) {
            std::error_code error;
            std::filesystem::create_directories(path_.parent_path(), error);
        }
        std::ofstream truncate(path_, std::ios::trunc);
        if (!truncate) {
            throw DynamicsError("cannot open '" + path_.string() + "' for writing");
        }
    }
}

bool LogWriter::wants(int step) const { return step % interval_ == 0; }

std::string LogWriter::header() {
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), "%8s %14s %16s %16s %16s %16s %10s %14s %14s\n",
                  "step", "time[fs]", "E_pot[eV]", "E_kin[eV]", "E_tot[eV]", "E_cons[eV]",
                  "T[K]", "P[eV/A^3]", "V[A^3]");
    return buffer;
}

std::string LogWriter::row(const DynamicsStep& step) {
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer),
                  "%8d %14.4f %16.8f %16.8f %16.8f %16.8f %10.3f %14.6e %14.4f\n", step.step,
                  step.time, step.potential_energy, step.kinetic_energy, step.total_energy,
                  step.conserved_energy, step.temperature, step.pressure, step.volume);
    return buffer;
}

void LogWriter::write(const DynamicsStep& step) {
    if (path_.empty()) {
        if (!started_) {
            std::fputs(header().c_str(), stdout);
            started_ = true;
        }
        std::fputs(row(step).c_str(), stdout);
        std::fflush(stdout);
        return;
    }

    std::ofstream stream(path_, std::ios::app);
    if (!stream) {
        throw DynamicsError("cannot append to '" + path_.string() + "'");
    }
    if (!started_) {
        stream << header();
        started_ = true;
    }
    stream << row(step);
}

}  // namespace fastatoms::md
