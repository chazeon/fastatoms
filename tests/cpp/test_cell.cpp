// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include "fastatoms/core/cell.hpp"
#include "testing.hpp"

using namespace fastatoms;

FA_TEST(mat3_inverse_round_trips) {
    const Mat3 m(2.0, 0.5, 0.0, -1.0, 3.0, 0.25, 0.0, 1.5, 4.0);
    const Mat3 identity = m * m.inverse();
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            FA_CHECK_CLOSE(identity(r, c), r == c ? 1.0 : 0.0, 1e-12);
        }
    }
    FA_CHECK(Mat3{}.inverse().is_zero());
}

FA_TEST(cell_from_parameters_reproduces_them) {
    const Cell cell = Cell::from_parameters(4.0, 5.0, 6.0, 70.0, 80.0, 95.0);
    const Vec3 lengths = cell.lengths();
    const Vec3 angles = cell.angles();
    FA_CHECK_CLOSE(lengths.x, 4.0, 1e-12);
    FA_CHECK_CLOSE(lengths.y, 5.0, 1e-12);
    FA_CHECK_CLOSE(lengths.z, 6.0, 1e-12);
    FA_CHECK_CLOSE(angles.x, 70.0, 1e-12);
    FA_CHECK_CLOSE(angles.y, 80.0, 1e-12);
    FA_CHECK_CLOSE(angles.z, 95.0, 1e-12);

    FA_CHECK_THROWS(Cell::from_parameters(-1.0, 5.0, 6.0, 90.0, 90.0, 90.0),
                    std::invalid_argument);
    FA_CHECK_THROWS(Cell::from_parameters(4.0, 5.0, 6.0, 10.0, 10.0, 170.0),
                    std::invalid_argument);
}

FA_TEST(cubic_cell_geometry) {
    const Cell cell = Cell::cubic(3.0);
    FA_CHECK_CLOSE(cell.volume(), 27.0, 1e-12);
    FA_CHECK(cell.is_orthorhombic());
    FA_CHECK(cell.is_invertible());
    const Vec3 heights = cell.heights();
    FA_CHECK_CLOSE(heights.x, 3.0, 1e-12);
    FA_CHECK_CLOSE(heights.z, 3.0, 1e-12);
    FA_CHECK_CLOSE(cell.maximum_unique_cutoff(), 1.5, 1e-12);
}

FA_TEST(heights_are_perpendicular_widths) {
    // A sheared cell keeps a height of 2 along z even though |a3| is larger.
    const Cell cell(Vec3{4.0, 0.0, 0.0}, Vec3{0.0, 4.0, 0.0}, Vec3{2.0, 2.0, 2.0});
    FA_CHECK_CLOSE(cell.volume(), 32.0, 1e-12);
    FA_CHECK_CLOSE(cell.heights().z, 2.0, 1e-12);
    FA_CHECK(!cell.is_orthorhombic());
}

FA_TEST(scaled_and_cartesian_are_inverses) {
    const Cell cell = Cell::from_parameters(3.0, 4.0, 5.0, 80.0, 100.0, 110.0);
    const Vec3 r{1.3, -0.7, 2.2};
    const Vec3 back = cell.cartesian(cell.scaled(r));
    FA_CHECK_CLOSE(back.x, r.x, 1e-12);
    FA_CHECK_CLOSE(back.y, r.y, 1e-12);
    FA_CHECK_CLOSE(back.z, r.z, 1e-12);
}

FA_TEST(wrap_folds_into_the_cell) {
    Cell cell = Cell::cubic(10.0);
    const Vec3 wrapped = cell.wrap(Vec3{12.5, -3.0, 25.0});
    FA_CHECK_CLOSE(wrapped.x, 2.5, 1e-9);
    FA_CHECK_CLOSE(wrapped.y, 7.0, 1e-9);
    FA_CHECK_CLOSE(wrapped.z, 5.0, 1e-9);

    // Free directions are left alone.
    cell.set_pbc({true, false, false});
    const Vec3 partial = cell.wrap(Vec3{12.5, -3.0, 25.0});
    FA_CHECK_CLOSE(partial.x, 2.5, 1e-9);
    FA_CHECK_CLOSE(partial.y, -3.0, 1e-9);
}

FA_TEST(minimum_image_picks_the_nearest_copy) {
    const Cell cell = Cell::cubic(10.0);
    std::array<int, 3> offset{};
    const Vec3 dr = cell.minimum_image(Vec3{9.0, 0.0, 0.0}, offset);
    FA_CHECK_CLOSE(dr.x, -1.0, 1e-12);
    FA_CHECK(offset[0] == -1);

    // Exactly half a cell is a tie; either image is acceptable, but the length
    // must be right.
    FA_CHECK_CLOSE(cell.minimum_image(Vec3{5.0, 0.0, 0.0}).norm(), 5.0, 1e-12);

    // A lattice vector is its own zero image.
    const Cell sheared(Vec3{10.0, 0.0, 0.0}, Vec3{9.0, 4.0, 0.0}, Vec3{0.0, 0.0, 10.0});
    FA_CHECK_CLOSE(sheared.minimum_image(Vec3{9.0, 4.0, 0.0}).norm(), 0.0, 1e-12);
}

FA_TEST(minimum_image_agrees_with_a_brute_force_search) {
    // Strongly sheared cells are where rounding the fractional coordinates is
    // not enough on its own, so check against an exhaustive image search.
    const Cell cells[] = {
        Cell(Vec3{10.0, 0.0, 0.0}, Vec3{9.0, 4.0, 0.0}, Vec3{0.0, 0.0, 10.0}),
        Cell::from_parameters(6.0, 7.0, 8.0, 65.0, 72.0, 115.0),
        Cell::from_parameters(5.0, 5.0, 5.0, 60.0, 60.0, 60.0),
    };

    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> spread(-12.0, 12.0);

    for (const Cell& cell : cells) {
        for (int trial = 0; trial < 400; ++trial) {
            const Vec3 dr{spread(rng), spread(rng), spread(rng)};

            // Centre the exhaustive search on the fractional coordinates of dr,
            // since a strongly sheared cell can need large integer offsets.
            const Vec3 s = cell.scaled(dr);
            const std::array<int, 3> centre{-static_cast<int>(std::lround(s.x)),
                                            -static_cast<int>(std::lround(s.y)),
                                            -static_cast<int>(std::lround(s.z))};

            double best = std::numeric_limits<double>::max();
            for (int a = centre[0] - 3; a <= centre[0] + 3; ++a) {
                for (int b = centre[1] - 3; b <= centre[1] + 3; ++b) {
                    for (int c = centre[2] - 3; c <= centre[2] + 3; ++c) {
                        const Vec3 offset = cell.vector(0) * a + cell.vector(1) * b +
                                            cell.vector(2) * c;
                        best = std::min(best, (dr + offset).norm2());
                    }
                }
            }

            std::array<int, 3> offset{};
            const Vec3 image = cell.minimum_image(dr, offset);
            FA_CHECK_CLOSE(image.norm2(), best, 1e-9);

            // The reported offset must reproduce the returned vector.
            const Vec3 rebuilt = dr + cell.vector(0) * offset[0] + cell.vector(1) * offset[1] +
                                 cell.vector(2) * offset[2];
            FA_CHECK_CLOSE((rebuilt - image).norm(), 0.0, 1e-9);
        }
    }
}

FA_TEST(non_periodic_cells_do_not_wrap_or_image) {
    Cell cell = Cell::cubic(10.0);
    cell.set_pbc(false);
    const Vec3 dr = cell.minimum_image(Vec3{9.0, 0.0, 0.0});
    FA_CHECK_CLOSE(dr.x, 9.0, 1e-12);
    FA_CHECK(cell.maximum_unique_cutoff() > 1e300);
}

FA_TEST_MAIN()
