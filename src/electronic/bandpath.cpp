// SPDX-License-Identifier: MIT

#include "fastatoms/electronic/bandpath.hpp"

#include <algorithm>
#include <cctype>
#include <optional>
#include <cmath>
#include <sstream>

#include "fastatoms/symmetry/standardize.hpp"

namespace fastatoms::electronic {
namespace {

constexpr double kPi = 3.14159265358979323846;

/// The conventional path for each tabulated lattice. A comma means the path
/// jumps rather than interpolating.
struct Tabulated {
    std::map<std::string, Vec3> points;
    std::string path;
};

/// Simple cubic.
Tabulated cubic() {
    return {{{"G", {0.0, 0.0, 0.0}},
             {"X", {0.0, 0.5, 0.0}},
             {"M", {0.5, 0.5, 0.0}},
             {"R", {0.5, 0.5, 0.5}}},
            "GXMGRX,MR"};
}

/// Face-centred cubic.
Tabulated face_centred_cubic() {
    return {{{"G", {0.0, 0.0, 0.0}},
             {"X", {0.5, 0.0, 0.5}},
             {"L", {0.5, 0.5, 0.5}},
             {"W", {0.5, 0.25, 0.75}},
             {"K", {0.375, 0.375, 0.75}},
             {"U", {0.625, 0.25, 0.625}}},
            "GXWKGLUWLK,UX"};
}

/// Body-centred cubic.
Tabulated body_centred_cubic() {
    return {{{"G", {0.0, 0.0, 0.0}},
             {"H", {0.5, -0.5, 0.5}},
             {"P", {0.25, 0.25, 0.25}},
             {"N", {0.0, 0.0, 0.5}}},
            "GHNGPH,PN"};
}

/// Simple tetragonal.
Tabulated tetragonal() {
    return {{{"G", {0.0, 0.0, 0.0}},
             {"X", {0.0, 0.5, 0.0}},
             {"M", {0.5, 0.5, 0.0}},
             {"Z", {0.0, 0.0, 0.5}},
             {"R", {0.0, 0.5, 0.5}},
             {"A", {0.5, 0.5, 0.5}}},
            "GXMGZRAZ,XR,MA"};
}

/// Simple orthorhombic.
Tabulated orthorhombic() {
    return {{{"G", {0.0, 0.0, 0.0}},
             {"X", {0.5, 0.0, 0.0}},
             {"Y", {0.0, 0.5, 0.0}},
             {"Z", {0.0, 0.0, 0.5}},
             {"S", {0.5, 0.5, 0.0}},
             {"T", {0.0, 0.5, 0.5}},
             {"U", {0.5, 0.0, 0.5}},
             {"R", {0.5, 0.5, 0.5}}},
            "GXSYGZURTZ,YT,UX,SR"};
}

/// Hexagonal.
Tabulated hexagonal() {
    return {{{"G", {0.0, 0.0, 0.0}},
             {"A", {0.0, 0.0, 0.5}},
             {"K", {1.0 / 3.0, 1.0 / 3.0, 0.0}},
             {"H", {1.0 / 3.0, 1.0 / 3.0, 0.5}},
             {"M", {0.5, 0.0, 0.0}},
             {"L", {0.5, 0.0, 0.5}}},
            "GMKGALHA,LM,KH"};
}

/// Rhombohedral, which has two variants set apart by whether the rhombohedral
/// angle is under or over ninety degrees.
Tabulated rhombohedral(double alpha_degrees) {
    const double alpha = alpha_degrees * kPi / 180.0;
    if (alpha_degrees < 90.0) {
        const double eta = (1.0 + 4.0 * std::cos(alpha)) / (2.0 + 4.0 * std::cos(alpha));
        const double nu = 0.75 - eta / 2.0;
        return {{{"G", {0.0, 0.0, 0.0}},
                 {"B", {eta, 0.5, 1.0 - eta}},
                 {"B1", {0.5, 1.0 - eta, eta - 1.0}},
                 {"F", {0.5, 0.5, 0.0}},
                 {"L", {0.5, 0.0, 0.0}},
                 {"L1", {0.0, 0.0, -0.5}},
                 {"P", {eta, nu, nu}},
                 {"P1", {1.0 - nu, 1.0 - nu, 1.0 - eta}},
                 {"P2", {nu, nu, eta - 1.0}},
                 {"Q", {1.0 - nu, nu, 0.0}},
                 {"X", {nu, 0.0, -nu}},
                 {"Z", {0.5, 0.5, 0.5}}},
                "GLB1,BZGX,QFP1Z,LP"};
    }
    const double half = std::tan(alpha / 2.0);
    const double eta = 1.0 / (2.0 * half * half);
    const double nu = 0.75 - eta / 2.0;
    return {{{"G", {0.0, 0.0, 0.0}},
             {"F", {0.5, -0.5, 0.0}},
             {"L", {0.5, 0.0, 0.0}},
             {"P", {1.0 - nu, -nu, 1.0 - nu}},
             {"P1", {nu, nu - 1.0, nu - 1.0}},
             {"Q", {eta, eta, eta}},
             {"Q1", {1.0 - eta, -eta, -eta}},
             {"Z", {0.5, -0.5, 0.5}}},
            "GPZQGFP1Q1LZ"};
}

/// The tabulated lattices and their data, or nothing when there is none.
std::optional<Tabulated> table_for(BravaisLattice lattice, const Cell& cell) {
    switch (lattice) {
        case BravaisLattice::CUB: return cubic();
        case BravaisLattice::FCC: return face_centred_cubic();
        case BravaisLattice::BCC: return body_centred_cubic();
        case BravaisLattice::TET: return tetragonal();
        case BravaisLattice::ORC: return orthorhombic();
        case BravaisLattice::HEX: return hexagonal();
        case BravaisLattice::RHL: {
            // The rhombohedral angle of the primitive cell decides the variant.
            const Vec3 angles = cell.angles();
            return rhombohedral(angles.x);
        }
        default:
            return std::nullopt;
    }
}

}  // namespace

std::string_view bravais_name(BravaisLattice lattice) {
    switch (lattice) {
        case BravaisLattice::CUB:  return "CUB";
        case BravaisLattice::FCC:  return "FCC";
        case BravaisLattice::BCC:  return "BCC";
        case BravaisLattice::TET:  return "TET";
        case BravaisLattice::BCT:  return "BCT";
        case BravaisLattice::ORC:  return "ORC";
        case BravaisLattice::ORCF: return "ORCF";
        case BravaisLattice::ORCI: return "ORCI";
        case BravaisLattice::ORCC: return "ORCC";
        case BravaisLattice::HEX:  return "HEX";
        case BravaisLattice::RHL:  return "RHL";
        case BravaisLattice::MCL:  return "MCL";
        case BravaisLattice::MCLC: return "MCLC";
        case BravaisLattice::TRI:  return "TRI";
    }
    return "TRI";
}

std::vector<BravaisLattice> tabulated_lattices() {
    return {BravaisLattice::CUB, BravaisLattice::FCC, BravaisLattice::BCC,
            BravaisLattice::TET, BravaisLattice::ORC, BravaisLattice::HEX,
            BravaisLattice::RHL};
}

BravaisLattice identify_bravais(const Cell& cell, double tolerance) {
    if (!cell.is_invertible()) {
        throw ElectronicError("identifying a lattice needs a cell that spans three dimensions");
    }
    // A lattice with one atom in it has the lattice's own holohedry as its
    // point group, so the symmetry module answers this question directly.
    Atoms lattice_only(std::vector<std::string>{"H"}, {0.0, 0.0, 0.0}, cell);
    lattice_only.set_pbc({true, true, true});

    symmetry::CellAnalysis analysis;
    try {
        analysis = symmetry::analyse_cell(lattice_only, tolerance);
    } catch (const symmetry::SymmetryError& error) {
        throw ElectronicError(std::string("could not identify the lattice: ") + error.what());
    }

    using symmetry::Centring;
    using symmetry::CrystalSystem;
    switch (analysis.system) {
        case CrystalSystem::Cubic:
            if (analysis.centring == Centring::F) return BravaisLattice::FCC;
            if (analysis.centring == Centring::I) return BravaisLattice::BCC;
            return BravaisLattice::CUB;
        case CrystalSystem::Tetragonal:
            return analysis.centring == Centring::I ? BravaisLattice::BCT : BravaisLattice::TET;
        case CrystalSystem::Orthorhombic:
            if (analysis.centring == Centring::F) return BravaisLattice::ORCF;
            if (analysis.centring == Centring::I) return BravaisLattice::ORCI;
            if (analysis.centring == Centring::A || analysis.centring == Centring::B ||
                analysis.centring == Centring::C) {
                return BravaisLattice::ORCC;
            }
            return BravaisLattice::ORC;
        case CrystalSystem::Hexagonal:
            return BravaisLattice::HEX;
        case CrystalSystem::Trigonal:
            // A trigonal point group sits on a rhombohedral lattice when the
            // cell is centred and on a hexagonal one otherwise.
            return analysis.centring == Centring::R ? BravaisLattice::RHL
                                                    : BravaisLattice::HEX;
        case CrystalSystem::Monoclinic:
            return analysis.centring == Centring::P ? BravaisLattice::MCL
                                                    : BravaisLattice::MCLC;
        case CrystalSystem::Triclinic:
            return BravaisLattice::TRI;
    }
    return BravaisLattice::TRI;
}

std::map<std::string, Vec3> special_points(const Cell& cell, double tolerance) {
    const BravaisLattice lattice = identify_bravais(cell, tolerance);
    const std::optional<Tabulated> table = table_for(lattice, cell);
    if (!table) {
        throw ElectronicError(
            "the special points of the " + std::string(bravais_name(lattice)) +
            " lattice are not tabulated in this build; give the path explicitly with "
            "bandpath_from_points");
    }
    return table->points;
}

std::string default_path(const Cell& cell, double tolerance) {
    const BravaisLattice lattice = identify_bravais(cell, tolerance);
    const std::optional<Tabulated> table = table_for(lattice, cell);
    if (!table) {
        throw ElectronicError("there is no conventional path for the " +
                              std::string(bravais_name(lattice)) + " lattice in this build");
    }
    return table->path;
}

std::string BandPath::to_string() const {
    std::ostringstream out;
    out << "BandPath(" << path << ", " << size() << " points, "
        << std::string(bravais_name(lattice)) << ")";
    return out.str();
}

namespace {

/// One uninterrupted run of the path. A path with a comma in it is several of
/// these, and the gap between them is crossed without being sampled.
struct Run {
    std::vector<Vec3> corners;
    std::vector<std::string> labels;
};

/// Sample a sequence of runs, sharing `points` between every segment in
/// proportion to its length in reciprocal space.
BandPath sample(const Cell& cell, const std::vector<Run>& runs, int points) {
    if (!cell.is_invertible()) {
        throw ElectronicError("a band path needs a cell that spans three dimensions");
    }
    std::size_t corner_count = 0;
    for (const Run& run : runs) {
        if (run.corners.size() < 2) {
            throw ElectronicError("each run of a path needs at least two points");
        }
        corner_count += run.corners.size();
    }
    if (corner_count < 2) {
        throw ElectronicError("a path needs at least two points");
    }
    if (points < static_cast<int>(corner_count)) {
        throw ElectronicError("asked for " + std::to_string(points) +
                              " k-points on a path with " + std::to_string(corner_count) +
                              " corners");
    }

    // Segment lengths in Cartesian reciprocal space, so the sampling is even
    // there rather than even in fractional coordinates - a segment twice as
    // long in the zone gets twice the points.
    const Mat3 reciprocal = cell.reciprocal();
    const auto length_of = [&](const Vec3& a, const Vec3& b) {
        const Vec3 difference{b.x - a.x, b.y - a.y, b.z - a.z};
        return (2.0 * kPi * (difference * reciprocal)).norm();
    };

    double total = 0.0;
    for (const Run& run : runs) {
        for (std::size_t k = 0; k + 1 < run.corners.size(); ++k) {
            total += length_of(run.corners[k], run.corners[k + 1]);
        }
    }

    BandPath out;
    double travelled = 0.0;
    for (const Run& run : runs) {
        for (std::size_t k = 0; k + 1 < run.corners.size(); ++k) {
            const double length = length_of(run.corners[k], run.corners[k + 1]);
            const int steps =
                total > 0.0 && length > 0.0
                    ? std::max(1, static_cast<int>(std::lround(points * length / total)))
                    : 1;
            for (int step = 0; step < steps; ++step) {
                const double t = static_cast<double>(step) / steps;
                out.kpoints.push_back(
                    Vec3{run.corners[k].x + t * (run.corners[k + 1].x - run.corners[k].x),
                         run.corners[k].y + t * (run.corners[k + 1].y - run.corners[k].y),
                         run.corners[k].z + t * (run.corners[k + 1].z - run.corners[k].z)});
                out.distances.push_back(travelled + t * length);
                out.labels.push_back(step == 0 && !run.labels.empty() ? run.labels[k]
                                                                     : std::string{});
            }
            travelled += length;
        }
        // The run's last corner, which no segment loop reaches. The next run
        // starts from the same distance, so the jump costs no axis length.
        out.kpoints.push_back(run.corners.back());
        out.distances.push_back(travelled);
        out.labels.push_back(run.labels.empty() ? std::string{} : run.labels.back());
    }

    for (std::size_t k = 0; k < out.labels.size(); ++k) {
        if (!out.labels[k].empty()) {
            out.special_distances.push_back(out.distances[k]);
            out.special_labels.push_back(out.labels[k]);
        }
    }
    return out;
}

}  // namespace

BandPath bandpath_from_points(const Cell& cell, const std::vector<Vec3>& corners,
                              const std::vector<std::string>& labels, int points) {
    if (!labels.empty() && labels.size() != corners.size()) {
        throw ElectronicError("there are " + std::to_string(labels.size()) + " labels for " +
                              std::to_string(corners.size()) + " points");
    }
    return sample(cell, {Run{corners, labels}}, points);
}

BandPath bandpath(const Cell& cell, const std::string& path, int points, double tolerance) {
    const BravaisLattice lattice = identify_bravais(cell, tolerance);
    const std::optional<Tabulated> table = table_for(lattice, cell);
    if (!table) {
        throw ElectronicError(
            "the " + std::string(bravais_name(lattice)) +
            " lattice has no tabulated special points in this build; give the corners "
            "explicitly with bandpath_from_points");
    }
    const std::string wanted = path.empty() ? table->path : path;

    // Split on commas: each piece is a run the path interpolates along, and the
    // gap between pieces is jumped rather than crossed - which is the whole
    // point of the comma, and why the plotted axis has no length there.
    std::vector<Run> runs;
    std::size_t start = 0;
    while (start <= wanted.size()) {
        const std::size_t comma = wanted.find(',', start);
        const std::string piece =
            wanted.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        Run run;
        for (std::size_t k = 0; k < piece.size(); ++k) {
            // A label is a letter, optionally followed by digits: "G", "B1".
            std::string label(1, piece[k]);
            while (k + 1 < piece.size() && std::isdigit(static_cast<unsigned char>(piece[k + 1]))) {
                label += piece[++k];
            }
            const auto it = table->points.find(label);
            if (it == table->points.end()) {
                throw ElectronicError("'" + label + "' is not a special point of the " +
                                      std::string(bravais_name(lattice)) + " lattice");
            }
            run.corners.push_back(it->second);
            run.labels.push_back(label);
        }
        if (!run.corners.empty()) {
            runs.push_back(std::move(run));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }

    BandPath out = sample(cell, runs, points);
    out.path = wanted;
    out.lattice = lattice;
    return out;
}

}  // namespace fastatoms::electronic
