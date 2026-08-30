// SPDX-License-Identifier: MIT
//
// Build the starting band for a NEB calculation.
//
//     fastatoms-interpolate initial.extxyz final.extxyz -o band.extxyz -n 9
//
// The output is a multi-frame extended XYZ file, which is what the `neb:`
// section of an input deck reads through `band:`.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "fastatoms/fastatoms.hpp"

using namespace fastatoms;

namespace {

void usage(const char* program) {
    std::fprintf(stderr,
                 "usage: %s <initial> <final> [options]\n"
                 "\n"
                 "Write the starting band for a NEB run, as multi-frame extended XYZ.\n"
                 "The endpoints should already be relaxed.\n"
                 "\n"
                 "  -o, --output PATH   where to write the band (default: band.extxyz)\n"
                 "  -n, --images N      images including both endpoints (default: 7)\n"
                 "      --idpp          place the images by image-dependent pair potential\n"
                 "                      instead of in a straight line\n"
                 "      --idpp-fmax F   IDPP force tolerance (default: 0.1)\n"
                 "      --idpp-steps N  IDPP step ceiling (default: 200)\n"
                 "      --no-mic        interpolate raw coordinate differences, without\n"
                 "                      the minimum-image convention\n"
                 "  -h, --help          this message\n"
                 "\n"
                 "A straight line is fine when little moves and the atoms stay out of\n"
                 "each other's way. When they do not - an exchange, a diffusion hop, a\n"
                 "molecule turning over - it puts atoms on top of one another, and the\n"
                 "first force call on such a band is unrecoverable. The closest approach\n"
                 "of each image is reported below so you can see which case you are in;\n"
                 "if it drops far under the endpoints', use --idpp.\n",
                 program);
}

/// Parse a positive count, or exit with a message naming the flag.
long positive(const char* text, const char* flag) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0) {
        std::fprintf(stderr, "%s: expected a positive integer, got '%s'\n", flag, text);
        std::exit(2);
    }
    return value;
}

double positive_real(const char* text, const char* flag) {
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value > 0.0)) {
        std::fprintf(stderr, "%s: expected a positive number, got '%s'\n", flag, text);
        std::exit(2);
    }
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    std::string initial_path;
    std::string final_path;
    std::string output = "band.extxyz";
    std::size_t images = 7;
    bool idpp = false;
    bool mic = true;
    neb::IDPPOptions idpp_options;

    for (int k = 1; k < argc; ++k) {
        const std::string arg = argv[k];
        const auto next = [&](const char* flag) -> const char* {
            if (k + 1 >= argc) {
                std::fprintf(stderr, "%s: expected a value\n", flag);
                std::exit(2);
            }
            return argv[++k];
        };

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        } else if (arg == "-o" || arg == "--output") {
            output = next("--output");
        } else if (arg == "-n" || arg == "--images") {
            images = static_cast<std::size_t>(positive(next("--images"), "--images"));
        } else if (arg == "--idpp") {
            idpp = true;
        } else if (arg == "--idpp-fmax") {
            idpp_options.fmax = positive_real(next("--idpp-fmax"), "--idpp-fmax");
        } else if (arg == "--idpp-steps") {
            idpp_options.max_steps = static_cast<int>(positive(next("--idpp-steps"), "--idpp-steps"));
        } else if (arg == "--no-mic") {
            mic = false;
        } else if (!arg.empty() && arg.front() == '-') {
            std::fprintf(stderr, "unknown option '%s'\n", arg.c_str());
            usage(argv[0]);
            return 2;
        } else if (initial_path.empty()) {
            initial_path = arg;
        } else if (final_path.empty()) {
            final_path = arg;
        } else {
            std::fprintf(stderr, "unexpected argument '%s'\n", arg.c_str());
            return 2;
        }
    }

    if (initial_path.empty() || final_path.empty()) {
        usage(argv[0]);
        return 2;
    }
    if (images < 3) {
        std::fprintf(stderr, "--images: a band needs at least 3, including both endpoints\n");
        return 2;
    }

    try {
        const Atoms initial = io::read(initial_path);
        const Atoms final = io::read(final_path);
        idpp_options.minimum_image = mic;

        const std::vector<Atoms> band =
            idpp ? neb::idpp_interpolate(initial, final, images, idpp_options)
                 : neb::NEB::interpolate(initial, final, images - 2, mic);

        io::write(output, band);

        std::printf("FastAtoms %s\n\n", version());
        std::printf("initial     : %s\n", initial.to_string().c_str());
        std::printf("final       : %s\n", final.to_string().c_str());
        std::printf("method      : %s\n", idpp ? "image-dependent pair potential" : "straight line");
        std::printf("images      : %zu, endpoints included\n", band.size());

        // The distance each atom has to travel, which is the band's length.
        double travelled = 0.0;
        for (std::size_t k = 0; k + 1 < band.size(); ++k) {
            double sum = 0.0;
            for (std::size_t a = 0; a < band[k].size(); ++a) {
                Vec3 dr = band[k + 1].position(a) - band[k].position(a);
                if (mic) {
                    dr = band[k].cell().minimum_image(dr);
                }
                sum += dr.norm() * dr.norm();
            }
            travelled += std::sqrt(sum);
        }
        std::printf("band length : %.4f A\n", travelled);

        std::printf("\n%5s %18s\n", "image", "closest pair [A]");
        const double ends = std::min(neb::closest_approach(initial, mic),
                                     neb::closest_approach(final, mic));
        double worst = ends;
        for (std::size_t k = 0; k < band.size(); ++k) {
            const double closest = neb::closest_approach(band[k], mic);
            worst = std::min(worst, closest);
            const bool endpoint = k == 0 || k + 1 == band.size();
            std::printf("%5zu %18.4f%s\n", k, closest, endpoint ? "   (endpoint)" : "");
        }

        std::printf("\nwritten to %s\n", output.c_str());
        if (worst < 0.5 * ends) {
            std::printf(
                "\nWARNING: the closest pair in the band is %.3f A, against %.3f A at the\n"
                "endpoints. Atoms are being pushed through each other and no calculator\n"
                "will give a sensible force on this band.%s\n",
                worst, ends, idpp ? "" : " Try --idpp.");
            return 1;
        }
    } catch (const InputError& error) {
        std::fprintf(stderr, "input error: %s\n", error.what());
        return 1;
    } catch (const neb::NEBError& error) {
        std::fprintf(stderr, "neb error: %s\n", error.what());
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
