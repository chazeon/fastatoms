// SPDX-License-Identifier: MIT
#include "fastatoms/md/trajectory.hpp"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "fastatoms/util/text.hpp"

namespace fastatoms::md {
namespace {


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

Value TrajectoryWriter::frame_info(const DynamicsStep& step) {
    Value info(Value::Map{});
    info.insert("step", Value(static_cast<std::int64_t>(step.step)));
    info.insert("time", Value(step.time));
    info.insert("energy", Value(step.potential_energy));
    info.insert("kinetic_energy", Value(step.kinetic_energy));
    info.insert("total_energy", Value(step.total_energy));
    info.insert("conserved_energy", Value(step.conserved_energy));
    info.insert("temperature", Value(step.temperature));
    info.insert("pressure", Value(step.pressure));
    info.insert("volume", Value(step.volume));
    return info;
}

std::string TrajectoryWriter::frame(const Atoms& atoms, const DynamicsStep& step,
                                    const TrajectoryOptions& options) {
    io::ExtxyzOptions extxyz;
    extxyz.velocities = options.velocities;
    extxyz.forces = options.forces;
    extxyz.info = frame_info(step);
    return io::extxyz_string(atoms, extxyz);
}

void TrajectoryWriter::write(const Atoms& atoms, const DynamicsStep& step) {
    if (!writer_) {
        io::ExtxyzOptions extxyz;
        extxyz.velocities = options_.velocities;
        extxyz.forces = options_.forces;
        // The constructor has already truncated the file when asked to, so the
        // writer always appends from here.
        extxyz.append = true;
        try {
            writer_ = std::make_unique<io::ExtxyzWriter>(path_.string(), extxyz);
        } catch (const io::ExtxyzError& error) {
            throw DynamicsError(error.what());
        }
    }
    try {
        writer_->write(atoms, frame_info(step));
        // Flushed every frame, not every megabyte. A trajectory that only
        // reaches the disk when the run ends cannot be watched while it runs,
        // and a run that dies takes its output with it. The file handle stays
        // open, which is where the cost of the old reopen-per-frame was.
        writer_->flush();
    } catch (const io::ExtxyzError& error) {
        throw DynamicsError(error.what());
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
