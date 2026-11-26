// SPDX-License-Identifier: MIT

#include "fastatoms/symmetry/spacegroup.hpp"

#include <spglib.h>

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "fastatoms/symmetry/standardize.hpp"

namespace fastatoms::symmetry {
namespace {

using DatasetPtr = std::unique_ptr<SpglibDataset, decltype(&spg_free_dataset)>;

[[noreturn]] void fail(const char* what) {
    throw SymmetryError(std::string(what) + ": " + spg_get_error_message(spg_get_error_code()));
}

/// spglib's lattices are column vectors: lattice[i][j] is component i of
/// vector j. Ours are rows, so converting either way is a transpose.
Mat3 from_spglib(const double (&lattice)[3][3]) {
    Mat3 out;
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            out(r, c) = lattice[c][r];
        }
    }
    return out;
}

void to_spglib(const Mat3& matrix, double (&lattice)[3][3]) {
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            lattice[c][r] = matrix(r, c);
        }
    }
}

/// spglib takes N x 3 arrays; a flat vector of 3N doubles has the same layout.
const double (*rows(const std::vector<double>& flat))[3] {
    return reinterpret_cast<const double (*)[3]>(flat.data());
}
double (*rows(std::vector<double>& flat))[3] {
    return reinterpret_cast<double (*)[3]>(flat.data());
}

/// A structure in spglib's terms: column lattice, fractional positions, types.
struct Crystal {
    double lattice[3][3]{};
    std::vector<double> positions;  ///< 3N, fractional
    std::vector<int> types;         ///< atomic numbers, or label ids with use_labels
};

void require_periodic(const Atoms& atoms) {
    if (atoms.empty()) {
        throw SymmetryError("symmetry: the structure has no atoms");
    }
    if (!atoms.cell().all_pbc() || !atoms.cell().is_invertible()) {
        throw SymmetryError(
            "symmetry: the cell must be periodic in all directions with non-zero volume");
    }
}

Crystal to_crystal(const Atoms& atoms, bool use_labels) {
    require_periodic(atoms);
    Crystal c;
    to_spglib(atoms.cell().matrix(), c.lattice);
    c.positions = atoms.scaled_positions();
    if (use_labels) {
        std::map<std::string, int> ids;
        for (const std::string& label : atoms.labels()) {
            c.types.push_back(ids.try_emplace(label, static_cast<int>(ids.size()) + 1)
                                  .first->second);
        }
    } else {
        c.types = atoms.numbers();
    }
    return c;
}

DatasetPtr search(const Crystal& c, double symprec, double angle_tolerance) {
    DatasetPtr ds(spgat_get_dataset(c.lattice, rows(c.positions), c.types.data(),
                                    static_cast<int>(c.types.size()), symprec, angle_tolerance),
                  spg_free_dataset);
    if (!ds) {
        fail("symmetry search failed");
    }
    return ds;
}

SymmetryDataset convert(const SpglibDataset& d) {
    const SpglibSpacegroupType type = spg_get_spacegroup_type(d.hall_number);
    const auto n = static_cast<std::size_t>(d.n_atoms);
    const auto n_std = static_cast<std::size_t>(d.n_std_atoms);

    SymmetryDataset out;
    out.number = d.spacegroup_number;
    out.international = type.international_short;
    out.hall = type.hall_symbol;
    out.hall_number = d.hall_number;
    out.choice = d.choice;
    out.crystal_system = crystal_system_of(out.number);

    std::set<IntMatrix3> distinct;
    for (int i = 0; i < d.n_operations; ++i) {
        IntMatrix3 rotation{};
        for (std::size_t a = 0; a < 3; ++a) {
            for (std::size_t b = 0; b < 3; ++b) {
                rotation[a][b] = d.rotations[i][a][b];
            }
        }
        distinct.insert(rotation);
        out.rotations.push_back(rotation);
        out.translations.emplace_back(d.translations[i][0], d.translations[i][1],
                                      d.translations[i][2]);
    }
    out.pointgroup = {d.pointgroup_symbol, type.pointgroup_schoenflies,
                      static_cast<int>(distinct.size()), out.crystal_system};

    for (std::size_t i = 0; i < n; ++i) {
        out.wyckoffs.push_back(static_cast<char>('a' + d.wyckoffs[i]));
        out.site_symmetry_symbols.emplace_back(d.site_symmetry_symbols[i]);
    }
    out.equivalent_atoms.assign(d.equivalent_atoms, d.equivalent_atoms + n);
    out.crystallographic_orbits.assign(d.crystallographic_orbits, d.crystallographic_orbits + n);

    out.std_lattice = from_spglib(d.std_lattice);
    out.std_types.assign(d.std_types, d.std_types + n_std);
    for (std::size_t i = 0; i < n_std; ++i) {
        out.std_positions.emplace_back(d.std_positions[i][0], d.std_positions[i][1],
                                       d.std_positions[i][2]);
    }
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            out.transformation_matrix(r, c) = d.transformation_matrix[r][c];
        }
    }
    out.origin_shift = Vec3{d.origin_shift[0], d.origin_shift[1], d.origin_shift[2]};

    out.primitive_lattice = from_spglib(d.primitive_lattice);
    out.mapping_to_primitive.assign(d.mapping_to_primitive, d.mapping_to_primitive + n);
    out.std_mapping_to_primitive.assign(d.std_mapping_to_primitive,
                                        d.std_mapping_to_primitive + n_std);
    return out;
}

/// The standardised structure. Labels come from the input atom that shares
/// both the primitive-cell id and the spglib type with the output atom.
Atoms standardized(const Atoms& atoms, bool to_primitive, bool no_idealize, double symprec,
                   double angle_tolerance, bool use_labels) {
    const Crystal input = to_crystal(atoms, use_labels);
    const DatasetPtr ds = search(input, symprec, angle_tolerance);
    const std::size_t n = input.types.size();
    const std::size_t capacity = std::max(n, static_cast<std::size_t>(ds->n_std_atoms));

    // spglib overwrites the arrays in place and needs room for the largest cell.
    Crystal work = input;
    work.positions.resize(3 * capacity, 0.0);
    work.types.resize(capacity, 0);
    const int n_out = spgat_standardize_cell(work.lattice, rows(work.positions),
                                             work.types.data(), static_cast<int>(n),
                                             to_primitive ? 1 : 0, no_idealize ? 1 : 0, symprec,
                                             angle_tolerance);
    if (n_out <= 0) {
        fail("cell standardisation failed");
    }
    const auto count = static_cast<std::size_t>(n_out);

    std::map<int, int> number_of_type;
    for (std::size_t i = 0; i < n; ++i) {
        number_of_type.try_emplace(input.types[i], atoms.number(i));
    }
    std::vector<int> numbers(count);
    for (std::size_t o = 0; o < count; ++o) {
        numbers[o] = number_of_type.at(work.types[o]);
    }
    Atoms out(numbers, std::vector<double>(3 * count, 0.0),
              Cell(from_spglib(work.lattice), {true, true, true}));
    out.set_scaled_positions(
        std::vector<double>(work.positions.begin(),
                            work.positions.begin() + static_cast<std::ptrdiff_t>(3 * count)));

    if (atoms.has_string_array("labels")) {
        const std::vector<std::string> labels = atoms.labels();
        std::map<std::pair<int, int>, std::size_t> source;
        for (std::size_t k = 0; k < n; ++k) {
            source.try_emplace({ds->mapping_to_primitive[k], input.types[k]}, k);
        }
        // Unidealised primitive atoms are numbered by their primitive id; all
        // other outputs follow the standard cell.
        const bool by_primitive_id = to_primitive && no_idealize;
        std::vector<std::string> out_labels;
        for (std::size_t o = 0; o < count; ++o) {
            int id = -1;
            if (by_primitive_id) {
                id = static_cast<int>(o);
            } else if (o < static_cast<std::size_t>(ds->n_std_atoms)) {
                id = ds->std_mapping_to_primitive[o];
            }
            const auto found = source.find({id, work.types[o]});
            out_labels.push_back(found == source.end() ? out.symbol(o) : labels[found->second]);
        }
        out.set_string_array("labels", out_labels);
    }
    return out;
}

}  // namespace

SymmetryDataset get_symmetry_dataset(const Atoms& atoms, double symprec, double angle_tolerance,
                                     bool use_labels) {
    return convert(*search(to_crystal(atoms, use_labels), symprec, angle_tolerance));
}

SpaceGroup analyse(const Atoms& atoms, double tolerance, double angle_tolerance,
                   bool use_labels) {
    const SymmetryDataset ds =
        get_symmetry_dataset(atoms, tolerance, angle_tolerance, use_labels);
    SpaceGroup out;
    out.number = ds.number;
    out.international = ds.international;
    out.pointgroup = ds.pointgroup;
    out.crystal_system = ds.crystal_system;

    std::map<char, WyckoffPosition> orbits;
    for (std::size_t i = 0; i < ds.wyckoffs.size(); ++i) {
        WyckoffPosition& orbit =
            orbits.try_emplace(ds.wyckoffs[i], WyckoffPosition{ds.wyckoffs[i],
                                                               ds.site_symmetry_symbols[i], {}})
                .first->second;
        orbit.atoms.push_back(i);
    }
    for (auto& entry : orbits) {
        out.wyckoff.push_back(std::move(entry.second));
    }
    return out;
}

const std::vector<std::pair<int, std::string>>& space_group_table() {
    static const std::vector<std::pair<int, std::string>> table = [] {
        // Hall numbers are listed in order, so the first setting of each
        // space group is the one kept here.
        std::map<int, std::string> symbols;
        for (int hall = 1; hall <= 530; ++hall) {
            const SpglibSpacegroupType type = spg_get_spacegroup_type(hall);
            symbols.try_emplace(type.number, type.international_short);
        }
        symbols.erase(0);
        return std::vector<std::pair<int, std::string>>(symbols.begin(), symbols.end());
    }();
    return table;
}

int space_group_number(const std::string& hermann_mauguin) {
    for (const auto& [number, symbol] : space_group_table()) {
        if (symbol == hermann_mauguin) {
            return number;
        }
    }
    return 0;
}

std::string space_group_symbol(int number) {
    for (const auto& [n, symbol] : space_group_table()) {
        if (n == number) {
            return symbol;
        }
    }
    throw SymmetryError("space group number out of range: " + std::to_string(number));
}

CellAnalysis analyse_cell(const Atoms& atoms, double symprec, double angle_tolerance,
                          bool use_labels) {
    const SymmetryDataset ds = get_symmetry_dataset(atoms, symprec, angle_tolerance, use_labels);
    CellAnalysis out;
    out.number = ds.number;
    out.international = ds.international;
    out.pointgroup = ds.pointgroup;
    out.crystal_system = ds.crystal_system;
    out.primitive_atoms = find_primitive(atoms, symprec, angle_tolerance, use_labels).size();
    out.to_primitive = ds.primitive_lattice * atoms.cell().inverse_matrix();
    return out;
}

Atoms standardize_cell(const Atoms& atoms, bool to_primitive, bool no_idealize, double symprec,
                       double angle_tolerance, bool use_labels) {
    return standardized(atoms, to_primitive, no_idealize, symprec, angle_tolerance, use_labels);
}

Atoms find_primitive(const Atoms& atoms, double symprec, double angle_tolerance,
                     bool use_labels) {
    return standardized(atoms, true, false, symprec, angle_tolerance, use_labels);
}

Atoms niggli_reduce(const Atoms& atoms, double symprec) {
    require_periodic(atoms);
    double lattice[3][3];
    to_spglib(atoms.cell().matrix(), lattice);
    if (spg_niggli_reduce(lattice, symprec) == 0) {
        fail("Niggli reduction failed");
    }
    Atoms out = atoms;
    out.set_cell(Cell(from_spglib(lattice), atoms.pbc()));
    return out;
}

}  // namespace fastatoms::symmetry
