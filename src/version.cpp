// SPDX-License-Identifier: MIT
#include "fastatoms/fastatoms.hpp"

#ifndef FASTATOMS_VERSION
#  define FASTATOMS_VERSION "0.0.0-unknown"
#endif

namespace fastatoms {

const char* version() noexcept { return FASTATOMS_VERSION; }

}  // namespace fastatoms
