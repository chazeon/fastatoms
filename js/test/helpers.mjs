// SPDX-License-Identifier: MIT
// Shared fixture paths for the node:test suites. Nothing here is a test.
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

/** Repository fixtures (tests/data), shared with the C++ and Python suites. */
export const REPO_DATA = join(here, "..", "..", "tests", "data");

/**
 * The gemmy example structures. Defaults to a gemmy checkout next to this
 * repository (../gemmy); override with FASTATOMS_GEMMY_EXAMPLES. Tests that
 * need a file skip when it is absent.
 */
export const GEMMY_EXAMPLES =
  process.env.FASTATOMS_GEMMY_EXAMPLES ?? join(here, "..", "..", "..", "gemmy", "examples");

export function gemmyPath(relative) {
  return join(GEMMY_EXAMPLES, relative);
}

export function haveFile(path) {
  return existsSync(path);
}

export function readText(path) {
  return readFileSync(path, "utf8");
}

/** Skip reason for a test that needs a file, or false when it exists. */
export function skipUnlessFile(path) {
  return haveFile(path) ? false : `missing fixture ${path}`;
}
