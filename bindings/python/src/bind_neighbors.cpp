// SPDX-License-Identifier: MIT
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "common.hpp"

#ifdef FASTATOMS_WITH_NEIGHBORS
#include "fastatoms/neighbor/neighbor_list.hpp"
#endif

namespace fastatoms::bindings {

#ifdef FASTATOMS_WITH_NEIGHBORS

void bind_neighbors(py::module_& m) {
    // Same signature and tuple order as ase.neighborlist.neighbor_list, with
    // the full (both directions) pair list that ASE returns.
    m.def(
        "neighbor_list",
        [](const std::string& quantities, const Atoms& atoms, double cutoff) {
            if (quantities.empty()) {
                throw py::value_error("neighbor_list: quantities must not be empty");
            }
            for (char q : quantities) {
                if (std::string("ijSdD").find(q) == std::string::npos) {
                    throw py::value_error(std::string("neighbor_list: unknown quantity '") + q +
                                          "' (use characters from 'ijSdD')");
                }
            }
            NeighborOptions options;
            options.cutoff = cutoff;
            NeighborPairs pairs = neighbor_pairs(atoms, options);
            const auto n = static_cast<py::ssize_t>(pairs.size());

            // Each quantity is moved out of `pairs` on first use, so the arrays
            // take ownership of the buffers without a copy. A repeated letter
            // ("ii") reuses the same array object.
            py::list out;
            py::dict built;
            for (char q : quantities) {
                const std::string key(1, q);
                if (built.contains(key.c_str())) {
                    out.append(built[key.c_str()]);
                    continue;
                }
                py::object item;
                switch (q) {
                    case 'i': {
                        auto copy = std::move(pairs.i);
                        item = move_to_array(std::move(copy), {n});
                        break;
                    }
                    case 'j': {
                        auto copy = std::move(pairs.j);
                        item = move_to_array(std::move(copy), {n});
                        break;
                    }
                    case 'S': {
                        auto copy = std::move(pairs.S);
                        item = move_to_array(std::move(copy), {n, 3});
                        break;
                    }
                    case 'd': {
                        auto copy = std::move(pairs.d);
                        item = move_to_array(std::move(copy), {n});
                        break;
                    }
                    default: {  // 'D'
                        auto copy = std::move(pairs.D);
                        item = move_to_array(std::move(copy), {n, 3});
                        break;
                    }
                }
                built[key.c_str()] = item;
                out.append(item);
            }
            return py::tuple(out);
        },
        py::arg("quantities"), py::arg("atoms"), py::arg("cutoff"),
        "Neighbour pairs as a tuple in the order of `quantities` (ASE semantics).\n\n"
        "Characters: i, j (atom indices), S (cell shifts, (n, 3)), d (distances),\n"
        "D (vectors, (n, 3)). Pairs are sorted by (i, j, S); i and j are int32.");
}

#else

void bind_neighbors(py::module_&) {}

#endif

}  // namespace fastatoms::bindings
