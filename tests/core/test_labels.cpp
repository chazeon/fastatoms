// SPDX-License-Identifier: MIT
#include "fastatoms/core/elements.hpp"
#include "testing.hpp"

using namespace fastatoms;

FA_TEST(labels_resolve_to_elements) {
    const std::pair<const char*, int> cases[] = {
        {"Fe", 26},     {"Fe1", 26},  {"Fe_pv", 26}, {"Fe_pv1", 26}, {"Fe1_pv", 26},
        {"Fe/abc123", 26}, {"Fe_sv_GW", 26}, {"O2-", 8}, {"Cu1+", 29}, {"H1", 1},
        {"Ow", 8},      {"Os2", 76},  {"CA", 20},    {"FE", 26},     {"X", 0},
        {" Si ", 14},   {"D", 1},     {"D1", 1},     {"T2", 1},   {"Li_sv", 3}, {"Ga_d", 31},  {"W_pv", 74},   {"Nb_sv", 41},
    };
    for (const auto& [label, z] : cases) {
        FA_CHECK(try_number_from_label(label) == z);
    }
}

FA_TEST(bad_labels_are_rejected) {
    for (const char* label : {"", "1Fe", "Zz", "_pv", "123", "Carbon", "Sodium", "Nitrogen",
                              "Chlorine", "Cubic", "species", "Lanthanide"}) {
        FA_CHECK(try_number_from_label(label) == -1);
    }
    FA_CHECK_THROWS(number_from_label("Qq1"), std::invalid_argument);
}

FA_TEST_MAIN()
