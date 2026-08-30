// SPDX-License-Identifier: MIT
#pragma once

/// \file testing.hpp
/// \brief A ~60-line test harness, so the test suite needs no extra dependency.

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace fastatoms::testing {

struct Case {
    std::string name;
    std::function<void()> body;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

inline int& failures() {
    static int count = 0;
    return count;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> body) {
        registry().push_back({name, std::move(body)});
    }
};

inline void report(const char* file, int line, const std::string& message) {
    std::printf("    FAIL %s:%d: %s\n", file, line, message.c_str());
    ++failures();
}

inline bool close(double a, double b, double tol) {
    return std::abs(a - b) <= tol * (1.0 + std::abs(a) + std::abs(b));
}

inline int run_all() {
    int failed_cases = 0;
    for (const Case& test : registry()) {
        const int before = failures();
        std::printf("  %s\n", test.name.c_str());
        try {
            test.body();
        } catch (const std::exception& error) {
            report(__FILE__, 0, std::string("unexpected exception: ") + error.what());
        }
        if (failures() > before) {
            ++failed_cases;
        }
    }
    std::printf("%s: %d/%zu cases passed\n", failed_cases ? "FAILED" : "PASSED",
                static_cast<int>(registry().size()) - failed_cases, registry().size());
    return failed_cases == 0 ? 0 : 1;
}

}  // namespace fastatoms::testing

#define FA_TEST(name)                                                             \
    static void name();                                                           \
    static const ::fastatoms::testing::Registrar registrar_##name(#name, name);   \
    static void name()

#define FA_CHECK(expr)                                                            \
    do {                                                                          \
        if (!(expr)) {                                                            \
            ::fastatoms::testing::report(__FILE__, __LINE__, "expected " #expr);  \
        }                                                                         \
    } while (false)

#define FA_CHECK_CLOSE(a, b, tol)                                                        \
    do {                                                                                 \
        const double fa_lhs = (a);                                                       \
        const double fa_rhs = (b);                                                       \
        if (!::fastatoms::testing::close(fa_lhs, fa_rhs, (tol))) {                       \
            ::fastatoms::testing::report(__FILE__, __LINE__,                             \
                                         #a " = " + std::to_string(fa_lhs) +             \
                                             " but expected " + std::to_string(fa_rhs)); \
        }                                                                                \
    } while (false)

#define FA_CHECK_THROWS(expr, exception_type)                                             \
    do {                                                                                  \
        bool fa_threw = false;                                                            \
        try {                                                                             \
            (void)(expr);                                                                 \
        } catch (const exception_type&) {                                                 \
            fa_threw = true;                                                              \
        } catch (...) {                                                                   \
        }                                                                                 \
        if (!fa_threw) {                                                                  \
            ::fastatoms::testing::report(__FILE__, __LINE__,                              \
                                         #expr " did not throw " #exception_type);        \
        }                                                                                 \
    } while (false)

#define FA_TEST_MAIN() \
    int main() { return ::fastatoms::testing::run_all(); }
