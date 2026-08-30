// SPDX-License-Identifier: MIT

#include "fastatoms/adsorption/sites.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>

#include "fastatoms/core/elements.hpp"
#include "fastatoms/core/neighbor_list.hpp"

namespace fastatoms::adsorption {
namespace {

Vec3 normalised(const Vec3& v) {
    const double length = v.norm();
    return length > 1e-12 ? Vec3{v.x / length, v.y / length, v.z / length}
                          : Vec3{0.0, 0.0, 1.0};
}

/// The displacement from `a` to `b`, over the minimum image when the structure
/// is periodic - so a slab that wraps in x and y behaves.
Vec3 displacement(const Atoms& atoms, const Vec3& a, const Vec3& b) {
    Vec3 d{b.x - a.x, b.y - a.y, b.z - a.z};
    const std::array<bool, 3> pbc = atoms.pbc();
    if ((pbc[0] || pbc[1] || pbc[2]) && atoms.cell().is_invertible()) {
        d = atoms.cell().minimum_image(d);
    }
    return d;
}

double separation(const Atoms& atoms, const Vec3& a, const Vec3& b) {
    return displacement(atoms, a, b).norm();
}

}  // namespace

std::string_view site_type_name(SiteType type) {
    switch (type) {
        case SiteType::Atop:     return "atop";
        case SiteType::Bridge:   return "bridge";
        case SiteType::Hollow:   return "hollow";
        case SiteType::FourFold: return "4-fold";
    }
    return "atop";
}

SiteType site_type_from_string(std::string_view name) {
    std::string key(name);
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    key.erase(std::remove_if(key.begin(), key.end(),
                             [](unsigned char c) { return c == '-' || c == '_' || c == ' '; }),
              key.end());
    if (key == "atop" || key == "top" || key == "ontop") return SiteType::Atop;
    if (key == "bridge") return SiteType::Bridge;
    if (key == "hollow" || key == "threefold" || key == "3fold") return SiteType::Hollow;
    if (key == "fourfold" || key == "4fold") return SiteType::FourFold;
    throw AdsorptionError("'" + std::string(name) +
                          "' is not a site type; use atop, bridge, hollow or 4-fold");
}

std::string AdsorptionSite::to_string() const {
    std::ostringstream out;
    out << site_type_name(type) << "(";
    for (std::size_t k = 0; k < atoms.size(); ++k) {
        out << (k ? "," : "") << atoms[k];
    }
    out << ")";
    return out.str();
}

Atoms molecule(const std::string& name) {
    // The binding atom comes first, and the rest sits along +z, so the default
    // placement points the molecule away from the surface.
    if (name == "H")   return Atoms(std::vector<std::string>{"H"}, {0.0, 0.0, 0.0});
    if (name == "O")   return Atoms(std::vector<std::string>{"O"}, {0.0, 0.0, 0.0});
    if (name == "N")   return Atoms(std::vector<std::string>{"N"}, {0.0, 0.0, 0.0});
    if (name == "C")   return Atoms(std::vector<std::string>{"C"}, {0.0, 0.0, 0.0});
    if (name == "OH")
        return Atoms(std::vector<std::string>{"O", "H"}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.97});
    if (name == "CO")
        return Atoms(std::vector<std::string>{"C", "O"}, {0.0, 0.0, 0.0, 0.0, 0.0, 1.13});
    if (name == "NO")
        return Atoms(std::vector<std::string>{"N", "O"}, {0.0, 0.0, 0.0, 0.0, 0.0, 1.15});
    if (name == "O2")
        return Atoms(std::vector<std::string>{"O", "O"}, {0.0, 0.0, 0.0, 0.0, 0.0, 1.21});
    if (name == "H2")
        return Atoms(std::vector<std::string>{"H", "H"}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.74});
    if (name == "N2")
        return Atoms(std::vector<std::string>{"N", "N"}, {0.0, 0.0, 0.0, 0.0, 0.0, 1.10});
    if (name == "OOH")
        return Atoms(std::vector<std::string>{"O", "O", "H"},
                     {0.0, 0.0, 0.0, 0.0, 1.10, 0.79, 0.0, 0.83, 1.62});
    if (name == "H2O")
        return Atoms(std::vector<std::string>{"O", "H", "H"},
                     {0.0, 0.0, 0.0, 0.76, 0.0, 0.59, -0.76, 0.0, 0.59});
    if (name == "CH3")
        return Atoms(std::vector<std::string>{"C", "H", "H", "H"},
                     {0.0, 0.0, 0.0, 1.03, 0.0, 0.36, -0.51, 0.89, 0.36, -0.51, -0.89, 0.36});
    throw AdsorptionError("'" + name +
                          "' is not one of the built-in adsorbates; build the Atoms yourself "
                          "and pass it in");
}

std::vector<std::string> known_molecules() {
    return {"H", "O", "N", "C", "OH", "CO", "NO", "O2", "H2", "N2", "OOH", "H2O", "CH3"};
}

AdsorptionFinder::AdsorptionFinder(Atoms surface, SiteOptions options)
    : surface_(std::move(surface)), options_(std::move(options)) {
    if (surface_.empty()) {
        throw AdsorptionError("there is no surface to find sites on");
    }
    if (!(options_.height > 0.0)) {
        throw AdsorptionError("the adsorption height must be positive");
    }
    if (!(options_.bond_scale > 0.0)) {
        throw AdsorptionError("the bond scale must be positive");
    }

    // Bonds from covalent radii, which is what decides both the coordination
    // numbers and the topology the sites are built from.
    std::vector<double> radii;
    radii.reserve(surface_.size());
    double largest = 0.0;
    for (std::size_t k = 0; k < surface_.size(); ++k) {
        const double radius = covalent_radius(surface_.number(k)) * options_.bond_scale;
        radii.push_back(radius);
        largest = std::max(largest, radius);
    }
    NeighborListOptions neighbours;
    neighbours.per_atom_cutoffs = radii;
    neighbours.cutoff = 2.0 * largest;
    neighbours.full = true;
    neighbours.store_vectors = true;
    neighbours.store_distances = true;
    const NeighborList list = build_neighbor_list(surface_, neighbours);
    if (list.empty()) {
        throw AdsorptionError(
            "no bonds were found at this bond scale, so there is no surface topology to work "
            "from; try a larger bond_scale");
    }

    // Bonded neighbours of each atom, and the outward direction, which is the
    // opposite of where its neighbours are. That definition needs no axis and
    // no centre, so it works on a facet of a nanoparticle and on a slab alike.
    const std::size_t n = surface_.size();
    std::vector<std::vector<std::size_t>> bonded(n);
    std::vector<Vec3> outward(n, Vec3{0.0, 0.0, 0.0});
    for (std::size_t p = 0; p < list.size(); ++p) {
        const auto a = static_cast<std::size_t>(list.i[p]);
        const auto b = static_cast<std::size_t>(list.j[p]);
        bonded[a].push_back(b);
        const Vec3 to_neighbour{list.vectors[3 * p], list.vectors[3 * p + 1],
                                list.vectors[3 * p + 2]};
        const Vec3 unit = normalised(to_neighbour);
        outward[a] = Vec3{outward[a].x - unit.x, outward[a].y - unit.y, outward[a].z - unit.z};
    }
    for (Vec3& direction : outward) {
        direction = normalised(direction);
    }

    // Which atoms are exposed. The default threshold is one below the highest
    // coordination present, which separates the skin of a particle from its
    // interior without needing to be told the bulk number.
    std::size_t highest = 0;
    for (const std::vector<std::size_t>& list_of : bonded) {
        highest = std::max(highest, list_of.size());
    }
    const std::size_t threshold =
        options_.surface_coordination > 0
            ? static_cast<std::size_t>(options_.surface_coordination)
            : (highest > 0 ? highest - 1 : 0);
    for (std::size_t k = 0; k < n; ++k) {
        if (bonded[k].size() <= threshold) {
            surface_atoms_.push_back(k);
        }
    }
    if (surface_atoms_.empty()) {
        throw AdsorptionError("no atom was judged to be on the surface; every one has the "
                              "highest coordination, so this looks like bulk");
    }
    std::vector<char> exposed(n, 0);
    for (const std::size_t k : surface_atoms_) {
        exposed[k] = 1;
    }

    // Build a candidate site from a group of atoms: its centre is their
    // average, its normal the average of their outward directions.
    const auto make_site = [&](SiteType type, std::vector<std::size_t> group) {
        AdsorptionSite site;
        site.type = type;
        site.atoms = std::move(group);
        std::sort(site.atoms.begin(), site.atoms.end());

        // Averaged around the first member using minimum-image offsets, so a
        // site straddling a periodic boundary lands in the right place rather
        // than in the middle of the cell.
        const Vec3 anchor = surface_.position(site.atoms.front());
        Vec3 offset{0.0, 0.0, 0.0};
        Vec3 normal{0.0, 0.0, 0.0};
        for (const std::size_t member : site.atoms) {
            const Vec3 d = displacement(surface_, anchor, surface_.position(member));
            offset = Vec3{offset.x + d.x, offset.y + d.y, offset.z + d.z};
            normal = Vec3{normal.x + outward[member].x, normal.y + outward[member].y,
                          normal.z + outward[member].z};
        }
        const double count = static_cast<double>(site.atoms.size());
        site.centre = Vec3{anchor.x + offset.x / count, anchor.y + offset.y / count,
                           anchor.z + offset.z / count};
        site.normal = normalised(normal);
        site.position = Vec3{site.centre.x + options_.height * site.normal.x,
                             site.centre.y + options_.height * site.normal.y,
                             site.centre.z + options_.height * site.normal.z};
        return site;
    };

    // A site is kept only if the place the adsorbate would go is actually
    // clear. This is what removes the underside of a bridge and the hollows
    // that are really inside the particle.
    const double minimum_clearance = options_.clearance * options_.height;
    const auto is_clear = [&](const AdsorptionSite& site) {
        if (site.normal.norm() < 0.5) {
            return false;
        }
        for (std::size_t k = 0; k < n; ++k) {
            if (separation(surface_, site.position, surface_.position(k)) < minimum_clearance) {
                return false;
            }
        }
        return true;
    };

    const auto are_bonded = [&](std::size_t a, std::size_t b) {
        return std::find(bonded[a].begin(), bonded[a].end(), b) != bonded[a].end();
    };

    if (options_.atop) {
        for (const std::size_t a : surface_atoms_) {
            AdsorptionSite site = make_site(SiteType::Atop, {a});
            if (is_clear(site)) {
                sites_.push_back(std::move(site));
            }
        }
    }
    if (options_.bridge) {
        for (const std::size_t a : surface_atoms_) {
            for (const std::size_t b : bonded[a]) {
                if (b <= a || !exposed[b]) {
                    continue;
                }
                AdsorptionSite site = make_site(SiteType::Bridge, {a, b});
                if (is_clear(site)) {
                    sites_.push_back(std::move(site));
                }
            }
        }
    }
    if (options_.hollow || options_.four_fold) {
        for (const std::size_t a : surface_atoms_) {
            for (const std::size_t b : bonded[a]) {
                if (b <= a || !exposed[b]) {
                    continue;
                }
                for (const std::size_t c : bonded[b]) {
                    if (c <= b || !exposed[c] || !are_bonded(a, c)) {
                        continue;
                    }
                    if (options_.hollow) {
                        AdsorptionSite site = make_site(SiteType::Hollow, {a, b, c});
                        if (is_clear(site)) {
                            sites_.push_back(std::move(site));
                        }
                    }
                }
            }
        }
    }
    if (options_.four_fold) {
        // A four-fold site is a genuine square: a four-cycle whose diagonals
        // are *not* bonded. Building it from a triangle plus a fourth atom
        // instead finds every rhombus of two adjacent triangles, and a
        // close-packed surface is made of nothing else - which puts dozens of
        // four-fold sites on an icosahedron, a shape with no square facet
        // anywhere on it. The centre of such a rhombus is over the shared
        // edge, which is to say it is the bridge site already found.
        for (const std::size_t a : surface_atoms_) {
            for (const std::size_t b : bonded[a]) {
                if (b <= a || !exposed[b]) {
                    continue;
                }
                for (const std::size_t c : bonded[b]) {
                    if (c == a || !exposed[c] || are_bonded(a, c)) {
                        continue;   // a and c must be the non-bonded diagonal
                    }
                    for (const std::size_t d : bonded[c]) {
                        if (d <= a || d == b || !exposed[d]) {
                            continue;   // d <= a fixes a as the smallest, once each
                        }
                        if (d < b) {
                            continue;   // and b < d picks one of the two ways round
                        }
                        if (!are_bonded(d, a) || are_bonded(d, b)) {
                            continue;   // the other diagonal must be open too
                        }
                        AdsorptionSite site = make_site(SiteType::FourFold, {a, b, c, d});
                        if (is_clear(site)) {
                            sites_.push_back(std::move(site));
                        }
                    }
                }
            }
        }
    }

    // Which sites are the same. Two are equivalent when an adsorbate at one
    // would see exactly what it sees at the other, so the invariant is the
    // sorted list of distances from the site to every atom, together with what
    // kind of site it is. That catches the real symmetry of a particle without
    // needing its point group, and it works for a slab too.
    std::vector<std::vector<long long>> fingerprints;
    fingerprints.reserve(sites_.size());
    for (const AdsorptionSite& site : sites_) {
        std::vector<long long> print;
        print.push_back(static_cast<long long>(site.type));
        std::vector<std::pair<double, int>> neighbours_of;
        for (std::size_t k = 0; k < n; ++k) {
            neighbours_of.emplace_back(separation(surface_, site.position, surface_.position(k)),
                                       surface_.number(k));
        }
        std::sort(neighbours_of.begin(), neighbours_of.end());
        for (const auto& [distance, number] : neighbours_of) {
            print.push_back(std::llround(distance / options_.tolerance));
            print.push_back(number);
        }
        fingerprints.push_back(std::move(print));
    }

    std::map<std::vector<long long>, std::size_t> seen;
    for (std::size_t k = 0; k < sites_.size(); ++k) {
        const auto [it, inserted] = seen.emplace(fingerprints[k], unique_.size());
        if (inserted) {
            unique_.push_back(k);
            equivalent_.emplace_back();
        }
        sites_[k].unique = it->second;
        equivalent_[it->second].push_back(k);
    }
}

const std::vector<std::size_t>& AdsorptionFinder::equivalent_sites(std::size_t unique) const {
    if (unique >= equivalent_.size()) {
        throw AdsorptionError("there are " + std::to_string(equivalent_.size()) +
                              " unique sites, so " + std::to_string(unique) + " is not one");
    }
    return equivalent_[unique];
}

std::vector<std::size_t> AdsorptionFinder::sites_of_type(SiteType type, bool unique_only) const {
    std::vector<std::size_t> out;
    if (unique_only) {
        for (const std::size_t index : unique_) {
            if (sites_[index].type == type) {
                out.push_back(index);
            }
        }
        return out;
    }
    for (std::size_t k = 0; k < sites_.size(); ++k) {
        if (sites_[k].type == type) {
            out.push_back(k);
        }
    }
    return out;
}

std::string AdsorptionFinder::summary() const {
    std::map<SiteType, std::pair<int, int>> counts;
    for (const AdsorptionSite& site : sites_) {
        ++counts[site.type].first;
    }
    for (const std::size_t index : unique_) {
        ++counts[sites_[index].type].second;
    }

    std::ostringstream out;
    out << surface_.size() << " atoms, " << surface_atoms_.size() << " on the surface; "
        << sites_.size() << " sites, " << unique_.size() << " unique";
    for (const auto& [type, pair] : counts) {
        out << "\n  " << site_type_name(type) << ": " << pair.first << " sites, " << pair.second
            << " unique";
    }
    return out.str();
}

Atoms AdsorptionFinder::place_atom(const std::string& symbol, std::size_t site) const {
    if (site >= sites_.size()) {
        throw AdsorptionError("there are " + std::to_string(sites_.size()) + " sites, so " +
                              std::to_string(site) + " is not one");
    }
    Atoms out = surface_;
    out.append(symbol, sites_[site].position);
    return out;
}

Atoms AdsorptionFinder::place(const Atoms& adsorbate, std::size_t site,
                              std::size_t binding_atom, std::ptrdiff_t anchor) const {
    return place_many(adsorbate, {site}, binding_atom, anchor);
}

Atoms AdsorptionFinder::place_many(const Atoms& adsorbate, const std::vector<std::size_t>& sites,
                                   std::size_t binding_atom, std::ptrdiff_t anchor) const {
    if (adsorbate.empty()) {
        throw AdsorptionError("the adsorbate is empty");
    }
    if (binding_atom >= adsorbate.size()) {
        throw AdsorptionError("the adsorbate has " + std::to_string(adsorbate.size()) +
                              " atoms, so " + std::to_string(binding_atom) +
                              " cannot be the binding one");
    }
    if (anchor >= 0 && static_cast<std::size_t>(anchor) >= adsorbate.size()) {
        throw AdsorptionError("the anchor atom is outside the adsorbate");
    }
    if (anchor >= 0 && static_cast<std::size_t>(anchor) == binding_atom) {
        throw AdsorptionError("the anchor and the binding atom cannot be the same; there would "
                              "be no direction to orient along");
    }

    // The direction inside the molecule that should end up pointing away from
    // the surface: from the binding atom towards the anchor, or towards
    // everything else when no anchor was named.
    const Vec3 binding = adsorbate.position(binding_atom);
    Vec3 axis{0.0, 0.0, 0.0};
    if (adsorbate.size() > 1) {
        if (anchor >= 0) {
            const Vec3 target = adsorbate.position(static_cast<std::size_t>(anchor));
            axis = Vec3{target.x - binding.x, target.y - binding.y, target.z - binding.z};
        } else {
            for (std::size_t k = 0; k < adsorbate.size(); ++k) {
                if (k == binding_atom) {
                    continue;
                }
                const Vec3 p = adsorbate.position(k);
                axis = Vec3{axis.x + p.x - binding.x, axis.y + p.y - binding.y,
                            axis.z + p.z - binding.z};
            }
        }
    }
    const bool orientable = axis.norm() > 1e-9;
    const Vec3 from = orientable ? normalised(axis) : Vec3{0.0, 0.0, 1.0};

    Atoms out = surface_;
    for (const std::size_t index : sites) {
        if (index >= sites_.size()) {
            throw AdsorptionError("there are " + std::to_string(sites_.size()) + " sites, so " +
                                  std::to_string(index) + " is not one");
        }
        const AdsorptionSite& site = sites_[index];
        const Vec3 to = site.normal;

        // The rotation taking `from` onto `to`, by Rodrigues' formula. The
        // antiparallel case has no unique axis, so any perpendicular one will
        // do and the result is a half turn.
        Mat3 rotation = Mat3::identity();
        if (orientable) {
            const Vec3 axis_of = cross(from, to);
            const double sine = axis_of.norm();
            const double cosine = dot(from, to);
            if (sine > 1e-9) {
                const Vec3 k = normalised(axis_of);
                Mat3 cross_matrix{};
                cross_matrix(0, 0) = 0.0;   cross_matrix(0, 1) = -k.z; cross_matrix(0, 2) = k.y;
                cross_matrix(1, 0) = k.z;   cross_matrix(1, 1) = 0.0;  cross_matrix(1, 2) = -k.x;
                cross_matrix(2, 0) = -k.y;  cross_matrix(2, 1) = k.x;  cross_matrix(2, 2) = 0.0;
                const Mat3 squared = cross_matrix * cross_matrix;
                for (std::size_t c = 0; c < 9; ++c) {
                    rotation.m[c] += cross_matrix.m[c] * sine + squared.m[c] * (1.0 - cosine);
                }
                // Rodrigues' formula gives the matrix that acts on column
                // vectors. Positions here are rows - `r -> r R` - so what is
                // wanted is its transpose; using it as it stands applies the
                // rotation backwards, which lays the molecule down instead of
                // standing it up and is not obviously wrong to look at.
                rotation = rotation.transposed();
            } else if (cosine < 0.0) {
                // Antiparallel: turn half a circle about anything perpendicular.
                const Vec3 helper = std::abs(from.z) < 0.9 ? Vec3{0.0, 0.0, 1.0}
                                                           : Vec3{1.0, 0.0, 0.0};
                const Vec3 k = normalised(cross(from, helper));
                rotation = Mat3::diagonal(-1.0, -1.0, -1.0);
                for (std::size_t r = 0; r < 3; ++r) {
                    for (std::size_t c = 0; c < 3; ++c) {
                        const double kk = (r == 0 ? k.x : r == 1 ? k.y : k.z) *
                                          (c == 0 ? k.x : c == 1 ? k.y : k.z);
                        rotation(r, c) += 2.0 * kk;
                    }
                }
            }
        }

        for (std::size_t k = 0; k < adsorbate.size(); ++k) {
            const Vec3 relative{adsorbate.position(k).x - binding.x,
                                adsorbate.position(k).y - binding.y,
                                adsorbate.position(k).z - binding.z};
            // Row-vector convention: r -> r R.
            const Vec3 turned = relative * rotation;
            out.append(adsorbate.symbol(k), Vec3{site.position.x + turned.x,
                                                 site.position.y + turned.y,
                                                 site.position.z + turned.z});
        }
    }
    return out;
}

std::vector<Atoms> AdsorptionFinder::trajectory(const Atoms& adsorbate, bool unique_only,
                                                std::size_t binding_atom,
                                                std::ptrdiff_t anchor) const {
    std::vector<Atoms> frames;
    if (unique_only) {
        frames.reserve(unique_.size());
        for (const std::size_t index : unique_) {
            frames.push_back(place(adsorbate, index, binding_atom, anchor));
        }
        return frames;
    }
    frames.reserve(sites_.size());
    for (std::size_t k = 0; k < sites_.size(); ++k) {
        frames.push_back(place(adsorbate, k, binding_atom, anchor));
    }
    return frames;
}

std::vector<Atoms> AdsorptionFinder::trajectory_atom(const std::string& symbol,
                                                     bool unique_only) const {
    return trajectory(Atoms(std::vector<std::string>{symbol}, {0.0, 0.0, 0.0}), unique_only);
}

std::vector<std::size_t> AdsorptionFinder::spread_order(
    const std::vector<std::size_t>& candidates) const {
    std::vector<std::size_t> pool = candidates;
    if (pool.empty()) {
        pool.resize(sites_.size());
        for (std::size_t k = 0; k < sites_.size(); ++k) {
            pool[k] = k;
        }
    }
    for (const std::size_t index : pool) {
        if (index >= sites_.size()) {
            throw AdsorptionError("site " + std::to_string(index) + " does not exist");
        }
    }
    if (pool.empty()) {
        return pool;
    }

    // Greedy farthest-point ordering: start somewhere, then repeatedly take
    // whichever candidate is furthest from everything chosen so far. A
    // coverage series built this way spreads over the particle instead of
    // filling one facet and leaving the rest bare.
    std::vector<std::size_t> order;
    std::vector<char> taken(pool.size(), 0);
    order.push_back(pool.front());
    taken[0] = 1;

    std::vector<double> closest(pool.size(), std::numeric_limits<double>::infinity());
    while (order.size() < pool.size()) {
        const Vec3 last = sites_[order.back()].position;
        std::size_t best = pool.size();
        double best_distance = -1.0;
        for (std::size_t k = 0; k < pool.size(); ++k) {
            if (taken[k]) {
                continue;
            }
            closest[k] = std::min(closest[k],
                                  separation(surface_, sites_[pool[k]].position, last));
            if (closest[k] > best_distance) {
                best_distance = closest[k];
                best = k;
            }
        }
        if (best == pool.size()) {
            break;
        }
        taken[best] = 1;
        order.push_back(pool[best]);
    }
    return order;
}

std::vector<Atoms> AdsorptionFinder::coverage_series(const Atoms& adsorbate,
                                                     const std::vector<std::size_t>& candidates,
                                                     int steps, std::size_t binding_atom,
                                                     std::ptrdiff_t anchor) const {
    const std::vector<std::size_t> order = spread_order(candidates);
    if (order.empty()) {
        throw AdsorptionError("there are no candidate sites to cover");
    }
    const std::size_t wanted =
        steps > 0 ? std::min(order.size(), static_cast<std::size_t>(steps)) : order.size();

    std::vector<Atoms> frames;
    frames.reserve(wanted);
    for (std::size_t k = 1; k <= wanted; ++k) {
        const std::vector<std::size_t> chosen(order.begin(),
                                              order.begin() + static_cast<std::ptrdiff_t>(k));
        frames.push_back(place_many(adsorbate, chosen, binding_atom, anchor));
    }
    return frames;
}

std::vector<double> AdsorptionFinder::coverage_fractions(
    const std::vector<std::size_t>& candidates, int steps) const {
    const std::vector<std::size_t> order = spread_order(candidates);
    const std::size_t wanted =
        steps > 0 ? std::min(order.size(), static_cast<std::size_t>(steps)) : order.size();
    std::vector<double> out;
    out.reserve(wanted);
    for (std::size_t k = 1; k <= wanted; ++k) {
        out.push_back(static_cast<double>(k) / static_cast<double>(surface_atoms_.size()));
    }
    return out;
}

}  // namespace fastatoms::adsorption
