// SPDX-License-Identifier: MIT
#pragma once

/// \file extxyz.hpp
/// \brief Extended XYZ: FastAtoms' default structure and trajectory format.
///
/// A frame is three parts:
///
///     3
///     Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 pbc="T T T" energy=-1.5
///     O   0.000  0.000  0.000
///     H   0.958  0.000  0.000
///     H  -0.240  0.927  0.000
///
/// an atom count, a comment line of `key=value` pairs, and one line per atom
/// whose columns are described by `Properties`. Frames follow one another with
/// no separator, which is what makes the format usable for trajectories: a
/// writer appends, and a reader that wants frame 900 of 1000 does not have to
/// hold the first 899 in memory.
///
/// `Properties` is a colon-separated list of triples, `name:type:count`, where
/// the type is `S` (string), `R` (real), `I` (integer) or `L` (logical). The
/// columns are matched to the structure like this:
///
///   - `species` or `Z`      -- the elements
///   - `pos`                 -- positions, in angstrom
///   - `forces` or `force`   -- stored as the per-atom array "forces"
///   - `velo`/`velocities`   -- velocities, converted to momenta on the way in
///   - `momenta`             -- momenta directly
///   - `masses`, `charges`, `tags`
///   - anything else, real or integer, becomes a per-atom array of its own name
///
/// String columns other than the species are skipped: FastAtoms has no per-atom
/// string array to put them in, and dropping them loudly in the documentation
/// beats dropping them silently in the code.
///
/// The comment line's other keys land in `Atoms::info()`, so `energy=-1.5`
/// survives a read and comes back out on a write.
///
/// Performance
/// -----------
/// The reader maps the file rather than copying it, scans with pointers rather
/// than streams, and turns text into numbers with `std::from_chars`, which
/// neither allocates nor consults the locale. A `std::istringstream` per line -
/// the obvious implementation - is roughly thirty times slower, which is the
/// difference between reading a million-atom trajectory and not bothering.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "fastatoms/core/atoms.hpp"
#include "fastatoms/io/parameters.hpp"

namespace fastatoms::io {

/// Raised when a file is not extended XYZ, or is but is malformed.
class ExtxyzError : public std::runtime_error {
public:
    explicit ExtxyzError(const std::string& message) : std::runtime_error(message) {}
};

/// One entry of a `Properties` specification.
struct PropertyColumn {
    std::string name{};
    char type{'R'};      ///< S, R, I or L
    int count{1};        ///< how many columns this entry occupies

    [[nodiscard]] std::string to_string() const;
};

/// Parse a `Properties=` specification into its columns.
/// \throws ExtxyzError if the spec is not a series of `name:type:count` triples.
[[nodiscard]] std::vector<PropertyColumn> parse_properties(std::string_view spec);

/// Render a column list back into a `Properties=` value.
[[nodiscard]] std::string format_properties(const std::vector<PropertyColumn>& columns);

/// Parse a comment line into its `key=value` pairs.
///
/// Values may be quoted, and a quoted value may contain spaces - which
/// `Lattice` always does. A bare key with no `=` becomes `true`.
[[nodiscard]] Value parse_comment_line(std::string_view line);

/// What to put in a frame when writing.
struct ExtxyzOptions {
    /// Append to the file rather than replacing it. This is what a trajectory
    /// needs: one open, many frames, no rewriting of what came before.
    bool append{false};

    /// Per-atom arrays to write, beyond species and positions. Empty means
    /// "whatever the structure has": forces, velocities, masses and charges
    /// are written when present.
    std::vector<std::string> arrays{};

    /// Write velocities as `velo:R:3` when the structure has momenta.
    bool velocities{true};
    /// Write the "forces" array as `forces:R:3` when it is there.
    bool forces{true};
    /// Record the attached calculator's energy, free energy and stress in the
    /// comment line.
    ///
    /// Forces already travel as a column because they are per-atom, but the
    /// scalars live on the calculator rather than on the structure, so without
    /// this a written frame silently loses the energy it was computed with -
    /// and a trajectory of results becomes a trajectory of geometries.
    bool results{true};

    /// Write masses and charges when they are set.
    bool masses{false};
    bool charges{false};

    /// Significant digits for real columns and for real info values.
    ///
    /// Zero, the default, writes the shortest text that reads back as exactly
    /// the same double - which is both exact and usually shorter than a fixed
    /// number of digits would be. Set a positive value to trade accuracy for
    /// file size on a long trajectory.
    int precision{0};

    /// Extra `key=value` pairs for the comment line, merged over
    /// `Atoms::info()`.
    Value info{};
};

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// A frame reader that holds the file open and hands out one frame at a time.
///
/// The file is memory mapped where the platform allows it, so a trajectory
/// larger than RAM is read a page at a time and a frame costs only the
/// structure it produces.
///
///     ExtxyzReader reader("md.extxyz");
///     Atoms frame;
///     while (reader.next(frame)) {
///         ...
///     }
class ExtxyzReader {
public:
    /// \throws ExtxyzError if the file cannot be opened.
    explicit ExtxyzReader(const std::string& path);

    /// Read from text already in memory.
    static ExtxyzReader from_string(std::string text);

    ~ExtxyzReader();

    ExtxyzReader(ExtxyzReader&&) noexcept;
    ExtxyzReader& operator=(ExtxyzReader&&) noexcept;
    ExtxyzReader(const ExtxyzReader&) = delete;
    ExtxyzReader& operator=(const ExtxyzReader&) = delete;

    /// Read the next frame into `atoms`. Returns false at end of file.
    /// \throws ExtxyzError on a malformed frame, naming the line.
    bool next(Atoms& atoms);

    /// Frames read so far.
    [[nodiscard]] std::size_t frames_read() const noexcept { return frames_read_; }

    /// Rewind to the first frame.
    void rewind();

    /// Byte offset of the start of every frame, scanning the whole file once.
    ///
    /// This is what makes random access cheap: the scan reads only the atom
    /// count of each frame and skips the rest, so an index over a million-frame
    /// file costs a fraction of parsing it.
    [[nodiscard]] const std::vector<std::size_t>& index();

    /// Read one frame by position, negative counting from the end.
    /// \throws ExtxyzError if the index is out of range.
    [[nodiscard]] Atoms at(std::ptrdiff_t frame);

    /// How many frames the file holds. Builds the index.
    [[nodiscard]] std::size_t size();

    /// The mapped file or the held text. Opaque: named here only so the
    /// implementation can hand one around.
    struct Source;

private:
    ExtxyzReader() = default;

    std::unique_ptr<Source> source_{};
    std::size_t cursor_{0};
    std::size_t frames_read_{0};
    std::vector<std::size_t> index_{};
    bool indexed_{false};
};

/// Read one frame from a file. `frame` counts from the end when negative, so
/// the default reads the last frame of a trajectory - which is what a
/// relaxation or a dynamics run leaves behind.
[[nodiscard]] Atoms read_extxyz(const std::string& path, std::ptrdiff_t frame = -1);

/// Read every frame.
[[nodiscard]] std::vector<Atoms> read_extxyz_frames(const std::string& path);

/// Read from text already in memory.
[[nodiscard]] Atoms parse_extxyz(const std::string& text, std::ptrdiff_t frame = -1);
[[nodiscard]] std::vector<Atoms> parse_extxyz_frames(const std::string& text);

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// A writer that keeps the file open across frames.
///
/// Appending through this class costs one buffered write per frame. Reopening
/// the file per frame - which a naive trajectory writer does - costs a system
/// call and a seek to the end each time, and on a network filesystem rather
/// more than that.
class ExtxyzWriter {
public:
    /// \throws ExtxyzError if the file cannot be opened.
    explicit ExtxyzWriter(const std::string& path, ExtxyzOptions options = {});
    ~ExtxyzWriter();

    ExtxyzWriter(ExtxyzWriter&&) noexcept;
    ExtxyzWriter& operator=(ExtxyzWriter&&) noexcept;
    ExtxyzWriter(const ExtxyzWriter&) = delete;
    ExtxyzWriter& operator=(const ExtxyzWriter&) = delete;

    /// Append one frame. `info` is merged over the structure's own.
    void write(const Atoms& atoms, const Value& info = Value{});

    /// Flush the buffer to the file. The destructor does this too.
    void flush();

    [[nodiscard]] std::size_t frames_written() const noexcept { return frames_written_; }

private:
    struct Sink;
    std::unique_ptr<Sink> sink_{};
    ExtxyzOptions options_{};
    std::size_t frames_written_{0};
};

/// Write one frame, replacing the file unless `options.append` is set.
void write_extxyz(const std::string& path, const Atoms& atoms, ExtxyzOptions options = {});

/// Write a whole trajectory in one call.
void write_extxyz(const std::string& path, const std::vector<Atoms>& frames,
                  ExtxyzOptions options = {});

/// One frame as a string.
[[nodiscard]] std::string extxyz_string(const Atoms& atoms, ExtxyzOptions options = {});

/// Append one frame's text to `out`. The building block the writer uses, and
/// the one to reach for when the destination is not a file.
void append_extxyz_frame(std::string& out, const Atoms& atoms,
                         const ExtxyzOptions& options = {});

}  // namespace fastatoms::io
