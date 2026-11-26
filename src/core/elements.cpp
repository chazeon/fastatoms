// SPDX-License-Identifier: MIT
#include "fastatoms/core/elements.hpp"

#include <array>
#include <cctype>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace fastatoms {
namespace {

/// One row of the periodic table.
///
/// `mass` is the IUPAC standard atomic weight in amu (the conventional value
/// for elements with an interval, the longest-lived isotope for the
/// radioactive ones). `radius` is the Cordero (2008) covalent radius in
/// angstrom; 0.0 marks elements with no tabulated value.
struct ElementRecord {
    const char* symbol;
    double mass;
    double radius;
};

// clang-format off
constexpr std::array<ElementRecord, kMaxAtomicNumber + 1> kElements{{
    {"X",   0.0,          0.00},
    {"H",   1.008,        0.31}, {"He",  4.002602,     0.28},
    {"Li",  6.94,         1.28}, {"Be",  9.0121831,    0.96},
    {"B",   10.81,        0.84}, {"C",   12.011,       0.76},
    {"N",   14.007,       0.71}, {"O",   15.999,       0.66},
    {"F",   18.998403163, 0.57}, {"Ne",  20.1797,      0.58},
    {"Na",  22.98976928,  1.66}, {"Mg",  24.305,       1.41},
    {"Al",  26.9815385,   1.21}, {"Si",  28.085,       1.11},
    {"P",   30.973761998, 1.07}, {"S",   32.06,        1.05},
    {"Cl",  35.45,        1.02}, {"Ar",  39.948,       1.06},
    {"K",   39.0983,      2.03}, {"Ca",  40.078,       1.76},
    {"Sc",  44.955908,    1.70}, {"Ti",  47.867,       1.60},
    {"V",   50.9415,      1.53}, {"Cr",  51.9961,      1.39},
    {"Mn",  54.938044,    1.61}, {"Fe",  55.845,       1.52},
    {"Co",  58.933194,    1.50}, {"Ni",  58.6934,      1.24},
    {"Cu",  63.546,       1.32}, {"Zn",  65.38,        1.22},
    {"Ga",  69.723,       1.22}, {"Ge",  72.630,       1.20},
    {"As",  74.921595,    1.19}, {"Se",  78.971,       1.20},
    {"Br",  79.904,       1.20}, {"Kr",  83.798,       1.16},
    {"Rb",  85.4678,      2.20}, {"Sr",  87.62,        1.95},
    {"Y",   88.90584,     1.90}, {"Zr",  91.224,       1.75},
    {"Nb",  92.90637,     1.64}, {"Mo",  95.95,        1.54},
    {"Tc",  97.90721,     1.47}, {"Ru",  101.07,       1.46},
    {"Rh",  102.90550,    1.42}, {"Pd",  106.42,       1.39},
    {"Ag",  107.8682,     1.45}, {"Cd",  112.414,      1.44},
    {"In",  114.818,      1.42}, {"Sn",  118.710,      1.39},
    {"Sb",  121.760,      1.39}, {"Te",  127.60,       1.38},
    {"I",   126.90447,    1.39}, {"Xe",  131.293,      1.40},
    {"Cs",  132.90545196, 2.44}, {"Ba",  137.327,      2.15},
    {"La",  138.90547,    2.07}, {"Ce",  140.116,      2.04},
    {"Pr",  140.90766,    2.03}, {"Nd",  144.242,      2.01},
    {"Pm",  144.91276,    1.99}, {"Sm",  150.36,       1.98},
    {"Eu",  151.964,      1.98}, {"Gd",  157.25,       1.96},
    {"Tb",  158.92535,    1.94}, {"Dy",  162.500,      1.92},
    {"Ho",  164.93033,    1.92}, {"Er",  167.259,      1.89},
    {"Tm",  168.93422,    1.90}, {"Yb",  173.045,      1.87},
    {"Lu",  174.9668,     1.87}, {"Hf",  178.49,       1.75},
    {"Ta",  180.94788,    1.70}, {"W",   183.84,       1.62},
    {"Re",  186.207,      1.51}, {"Os",  190.23,       1.44},
    {"Ir",  192.217,      1.41}, {"Pt",  195.084,      1.36},
    {"Au",  196.966569,   1.36}, {"Hg",  200.592,      1.32},
    {"Tl",  204.38,       1.45}, {"Pb",  207.2,        1.46},
    {"Bi",  208.98040,    1.48}, {"Po",  208.98243,    1.40},
    {"At",  209.98715,    1.50}, {"Rn",  222.01758,    1.50},
    {"Fr",  223.01974,    2.60}, {"Ra",  226.02541,    2.21},
    {"Ac",  227.02775,    2.15}, {"Th",  232.0377,     2.06},
    {"Pa",  231.03588,    2.00}, {"U",   238.02891,    1.96},
    {"Np",  237.04817,    1.90}, {"Pu",  244.06421,    1.87},
    {"Am",  243.06138,    1.80}, {"Cm",  247.07035,    1.69},
    {"Bk",  247.07031,    0.00}, {"Cf",  251.07959,    0.00},
    {"Es",  252.0830,     0.00}, {"Fm",  257.09511,    0.00},
    {"Md",  258.09843,    0.00}, {"No",  259.1010,     0.00},
    {"Lr",  262.110,      0.00}, {"Rf",  267.122,      0.00},
    {"Db",  268.126,      0.00}, {"Sg",  271.134,      0.00},
    {"Bh",  270.133,      0.00}, {"Hs",  269.1338,     0.00},
    {"Mt",  278.156,      0.00}, {"Ds",  281.165,      0.00},
    {"Rg",  281.166,      0.00}, {"Cn",  285.177,      0.00},
    {"Nh",  286.182,      0.00}, {"Fl",  289.190,      0.00},
    {"Mc",  289.194,      0.00}, {"Lv",  293.204,      0.00},
    {"Ts",  293.208,      0.00}, {"Og",  294.214,      0.00},
}};
// clang-format on

const std::unordered_map<std::string, int>& symbol_table() {
    static const std::unordered_map<std::string, int> table = [] {
        std::unordered_map<std::string, int> m;
        m.reserve(kElements.size() * 2);
        for (int z = 0; z <= kMaxAtomicNumber; ++z) {
            m.emplace(kElements[static_cast<std::size_t>(z)].symbol, z);
        }
        return m;
    }();
    return table;
}

/// "si" -> "Si", "HE" -> "He". Anything not one or two letters is left alone
/// so that the caller reports it as an unknown symbol.
std::string canonicalize(std::string_view symbol) {
    if (symbol.empty() || symbol.size() > 2) {
        return std::string(symbol);
    }
    std::string out(symbol);
    out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
    if (out.size() == 2) {
        out[1] = static_cast<char>(std::tolower(static_cast<unsigned char>(out[1])));
    }
    return out;
}

void check_range(int z) {
    if (z < 0 || z > kMaxAtomicNumber) {
        throw std::out_of_range("atomic number " + std::to_string(z) + " is outside [0, " +
                                std::to_string(kMaxAtomicNumber) + "]");
    }
}

}  // namespace

std::string_view symbol_from_number(int z) {
    check_range(z);
    return kElements[static_cast<std::size_t>(z)].symbol;
}

int try_number_from_symbol(std::string_view symbol) noexcept {
    const auto& table = symbol_table();
    if (const auto it = table.find(std::string(symbol)); it != table.end()) {
        return it->second;
    }
    if (const auto it = table.find(canonicalize(symbol)); it != table.end()) {
        return it->second;
    }
    return -1;
}

int number_from_symbol(std::string_view symbol) {
    const int z = try_number_from_symbol(symbol);
    if (z < 0) {
        throw std::invalid_argument("unknown chemical symbol '" + std::string(symbol) + "'");
    }
    return z;
}

int try_number_from_label(std::string_view label) noexcept {
    // Trim surrounding blanks.
    while (!label.empty() && std::isspace(static_cast<unsigned char>(label.front()))) {
        label.remove_prefix(1);
    }
    while (!label.empty() && std::isspace(static_cast<unsigned char>(label.back()))) {
        label.remove_suffix(1);
    }
    if (const int z = try_number_from_symbol(label); z >= 0) {
        return z;
    }
    if (const auto cut = label.find_first_of("/_"); cut != std::string_view::npos) {
        label = label.substr(0, cut);
    }
    std::size_t letters = 0;
    while (letters < label.size() && std::isalpha(static_cast<unsigned char>(label[letters]))) {
        ++letters;
    }
    // A run of three or more letters is a word ("Carbon", "Cubic",
    // "species"), not a decorated symbol; matching its prefix would turn
    // "Sodium" into S.
    if (letters == 0 || letters > 2) {
        return -1;
    }
    const auto is_lower = [](char c) { return std::islower(static_cast<unsigned char>(c)) != 0; };
    const auto is_upper = [](char c) { return std::isupper(static_cast<unsigned char>(c)) != 0; };
    if (letters >= 2 && is_lower(label[1])) {
        if (const int z = try_number_from_symbol(label.substr(0, 2)); z >= 0) {
            return z;
        }
    }
    const int one = try_number_from_symbol(label.substr(0, 1));
    if (one >= 0) {
        return one;
    }
    // Deuterium and tritium are hydrogen; the label keeps the isotope.
    if (label[0] == 'D' || label[0] == 'T' || label[0] == 'd' || label[0] == 't') {
        return 1;
    }
    if (letters >= 2 && is_upper(label[0]) && is_upper(label[1])) {
        return try_number_from_symbol(label.substr(0, 2));
    }
    return -1;
}

int number_from_label(std::string_view label) {
    const int z = try_number_from_label(label);
    if (z < 0) {
        throw std::invalid_argument("cannot tell which element the species label '" +
                                    std::string(label) + "' names");
    }
    return z;
}

double atomic_mass(int z) {
    check_range(z);
    return kElements[static_cast<std::size_t>(z)].mass;
}


/// Ambient-condition crystal structures, indexed by atomic number.
///
/// Experimental room-temperature values. They are where a calculation
/// starts, not where it ends: relaxing under any given functional moves
/// them by a per cent or two.
const std::array<ReferenceState, kMaxAtomicNumber + 1> kReferenceStates{{
    {"", 0.0, 0.0, 0.0, 0.0},
    {"diatomic", 0.74, 0, 0, 0},
    {"hcp", 3.57, 1.633, 0, 0},
    {"bcc", 3.49, 0, 0, 0},
    {"hcp", 2.29, 1.567, 0, 0},
    {"rhombohedral", 5.05, 0, 58.06, 0},
    {"diamond", 3.57, 0, 0, 0},
    {"diatomic", 1.1, 0, 0, 0},
    {"diatomic", 1.21, 0, 0, 0},
    {"diatomic", 1.42, 0, 0, 0},
    {"fcc", 4.43, 0, 0, 0},
    {"bcc", 4.23, 0, 0, 0},
    {"hcp", 3.21, 1.624, 0, 0},
    {"fcc", 4.05, 0, 0, 0},
    {"diamond", 5.43, 0, 0, 0},
    {"cubic", 7.17, 0, 0, 0},
    {"orthorhombic", 10.47, 0, 0, 1.229},
    {"diatomic", 1.99, 0, 0, 0},
    {"fcc", 5.26, 0, 0, 0},
    {"bcc", 5.23, 0, 0, 0},
    {"fcc", 5.58, 0, 0, 0},
    {"hcp", 3.31, 1.594, 0, 0},
    {"hcp", 2.95, 1.588, 0, 0},
    {"bcc", 3.02, 0, 0, 0},
    {"bcc", 2.88, 0, 0, 0},
    {"cubic", 8.89, 0, 0, 0},
    {"bcc", 2.87, 0, 0, 0},
    {"hcp", 2.51, 1.622, 0, 0},
    {"fcc", 3.52, 0, 0, 0},
    {"fcc", 3.61, 0, 0, 0},
    {"hcp", 2.66, 1.856, 0, 0},
    {"orthorhombic", 4.51, 0, 0, 1.001},
    {"diamond", 5.66, 0, 0, 0},
    {"rhombohedral", 4.13, 0, 54.16, 0},
    {"hexagonal", 4.36, 1.136, 0, 0},
    {"diatomic", 2.28, 0, 0, 0},
    {"fcc", 5.72, 0, 0, 0},
    {"bcc", 5.59, 0, 0, 0},
    {"fcc", 6.08, 0, 0, 0},
    {"hcp", 3.65, 1.571, 0, 0},
    {"hcp", 3.23, 1.593, 0, 0},
    {"bcc", 3.3, 0, 0, 0},
    {"bcc", 3.15, 0, 0, 0},
    {"hcp", 2.74, 1.604, 0, 0},
    {"hcp", 2.7, 1.584, 0, 0},
    {"fcc", 3.8, 0, 0, 0},
    {"fcc", 3.89, 0, 0, 0},
    {"fcc", 4.09, 0, 0, 0},
    {"hcp", 2.98, 1.886, 0, 0},
    {"tetragonal", 3.25, 1.518, 0, 0},
    {"tetragonal", 5.83, 0.546, 0, 0},
    {"rhombohedral", 4.51, 0, 57.6, 0},
    {"hexagonal", 4.45, 1.33, 0, 0},
    {"orthorhombic", 7.27, 0, 0, 1.347},
    {"fcc", 6.2, 0, 0, 0},
    {"bcc", 6.05, 0, 0, 0},
    {"bcc", 5.02, 0, 0, 0},
    {"hexagonal", 3.75, 3.22, 0, 0},
    {"fcc", 5.16, 0, 0, 0},
    {"hexagonal", 3.67, 3.22, 0, 0},
    {"hexagonal", 3.66, 3.22, 0, 0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"rhombohedral", 9.0, 0, 23.13, 0},
    {"bcc", 4.61, 0, 0, 0},
    {"hcp", 3.64, 1.588, 0, 0},
    {"hcp", 3.6, 1.581, 0, 0},
    {"hcp", 3.59, 1.573, 0, 0},
    {"hcp", 3.58, 1.57, 0, 0},
    {"hcp", 3.56, 1.57, 0, 0},
    {"hcp", 3.54, 1.57, 0, 0},
    {"fcc", 5.49, 0, 0, 0},
    {"hcp", 3.51, 1.585, 0, 0},
    {"hcp", 3.19, 1.582, 0, 0},
    {"bcc", 3.31, 0, 0, 0},
    {"bcc", 3.16, 0, 0, 0},
    {"hcp", 2.76, 1.615, 0, 0},
    {"hcp", 2.74, 1.579, 0, 0},
    {"fcc", 3.84, 0, 0, 0},
    {"fcc", 3.92, 0, 0, 0},
    {"fcc", 4.08, 0, 0, 0},
    {"rhombohedral", 2.99, 0, 70.75, 0},
    {"hcp", 3.46, 1.599, 0, 0},
    {"fcc", 4.95, 0, 0, 0},
    {"rhombohedral", 4.75, 0, 57.23, 0},
    {"sc", 3.35, 0, 0, 0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"fcc", 5.9, 0, 0, 0},
    {"bcc", 6.15, 0, 0, 0},
    {"bcc", 5.15, 0, 0, 0},
    {"fcc", 5.31, 0, 0, 0},
    {"fcc", 5.08, 0, 0, 0},
    {"tetragonal", 3.93, 0.825, 0, 0},
    {"orthorhombic", 2.85, 0, 0, 2.056},
    {"orthorhombic", 4.72, 0, 0, 1.001},
    {"monoclinic", 6.18, 0, 0, 0},
    {"hexagonal", 3.47, 3.208, 0, 0},
    {"hexagonal", 3.5, 3.199, 0, 0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
    {"", 0.0, 0.0, 0.0, 0.0},
}};

const ReferenceState& reference_state(int z) {
    check_range(z);
    return kReferenceStates[static_cast<std::size_t>(z)];
}

double covalent_radius(int z) {
    check_range(z);
    return kElements[static_cast<std::size_t>(z)].radius;
}

std::vector<double> natural_cutoffs(const std::vector<int>& numbers, double mult, double fallback) {
    std::vector<double> cutoffs;
    cutoffs.reserve(numbers.size());
    for (const int z : numbers) {
        const double r = covalent_radius(z);
        cutoffs.push_back(mult * (r > 0.0 ? r : fallback));
    }
    return cutoffs;
}

}  // namespace fastatoms
