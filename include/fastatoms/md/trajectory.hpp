// SPDX-License-Identifier: MIT
#pragma once

/// \file trajectory.hpp
/// \brief Writing a run out to disk.

#include <filesystem>
#include <string>

#include "fastatoms/md/types.hpp"

namespace fastatoms::md {

/// What goes into a trajectory frame.
struct TrajectoryOptions {
    /// Write a frame every this many steps. The first and last are always
    /// written.
    int interval{1};
    bool positions{true};
    bool velocities{true};
    bool forces{true};
    /// Append to an existing file rather than truncating it.
    bool append{false};
};

/// Writes frames in extended XYZ.
///
/// The format is the de-facto standard for atomistic trajectories - ASE,
/// OVITO and VMD all read it - and it carries the cell, per-atom vectors and
/// per-frame scalars in a plain text file:
///
/// \code
/// 4
/// Lattice="5.3 0 0 0 5.3 0 0 0 5.3" Properties=species:S:1:pos:R:3:velo:R:3:forces:R:3 ...
/// Ar  0.0 0.0 0.0   0.001 0.0 0.0   -0.02 0.0 0.0
/// \endcode
///
/// Forces are the ones from the most recent evaluation, taken from the
/// structure's `forces` array rather than triggering a new calculation. Under
/// a barostat the cell moves after that evaluation, so they belong to the
/// geometry half a deformation earlier.
class TrajectoryWriter {
public:
    /// \throws DynamicsError if the file cannot be opened.
    TrajectoryWriter(std::filesystem::path path, TrajectoryOptions options = {});

    /// Append one frame.
    void write(const Atoms& atoms, const DynamicsStep& step);

    /// True when `step` falls on the sampling interval.
    [[nodiscard]] bool wants(int step) const;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] int frames_written() const noexcept { return frames_; }

    /// One frame as text, without writing anything.
    [[nodiscard]] static std::string frame(const Atoms& atoms, const DynamicsStep& step,
                                           const TrajectoryOptions& options);

private:
    std::filesystem::path path_;
    TrajectoryOptions options_;
    int frames_{0};
};

/// Writes a plain text table of the per-step scalars.
class LogWriter {
public:
    /// \throws DynamicsError if the file cannot be opened. An empty path
    /// writes to standard output.
    explicit LogWriter(std::filesystem::path path, int interval = 1);

    void write(const DynamicsStep& step);
    [[nodiscard]] bool wants(int step) const;

    /// The header line and one formatted row, for reuse elsewhere.
    [[nodiscard]] static std::string header();
    [[nodiscard]] static std::string row(const DynamicsStep& step);

private:
    std::filesystem::path path_;
    int interval_{1};
    bool started_{false};
};

}  // namespace fastatoms::md
