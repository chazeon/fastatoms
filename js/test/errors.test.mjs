// SPDX-License-Identifier: MIT
// Errors cross the wasm boundary as messages, not as JS throws from C++. A
// module that raised many errors used to trap afterwards; these tests keep
// that from coming back.
import { test } from "node:test";
import assert from "node:assert/strict";
import { init, io, Atoms } from "../dist/index.js";
import { requireModule } from "../dist/module.js";

await init();

const VALID = "1\n\nH 0 0 0\n";
const ERRORS = 10_000;

/** Bytes of wasm linear memory, from the heap view the module exports. */
const heapBytes = () => requireModule().HEAP8.length;

test("10,000 rejected calls throw Error each time, a valid read still works, memory stays bounded", () => {
  // Warm up so the first growth of the heap is not counted.
  for (let i = 0; i < 100; i++) {
    assert.throws(() => io.read(VALID, { format: "no-such-format" }), Error);
  }
  const before = heapBytes();

  let rejected = 0;
  for (let i = 0; i < ERRORS; i++) {
    // Alternate unknown formats and malformed input, with a valid read every 500 calls.
    const bad =
      i % 2 === 0
        ? () => io.read(VALID, { format: "no-such-format" })
        : () => io.read("garbage ???\n", { format: "xyz" });
    assert.throws(bad, (err) => {
      assert.ok(err instanceof Error, "a JS Error");
      assert.ok(err.message.length > 0, "with the C++ message");
      return true;
    });
    rejected++;
    if (i % 500 === 0) {
      const atoms = io.read(VALID, { format: "xyz" });
      assert.ok(atoms instanceof Atoms);
      assert.equal(atoms.length, 1);
      atoms.dispose();
    }
  }
  assert.equal(rejected, ERRORS);

  const after = heapBytes();
  assert.ok(after - before <= 1024 * 1024, `heap grew by ${after - before} bytes over ${ERRORS} errors`);

  const last = io.read(VALID, { format: "xyz" });
  assert.deepEqual(last.getChemicalSymbols(), ["H"]);
  last.dispose();
});

test("the message of a rejected call is the C++ one, not a stale one", () => {
  assert.throws(
    () => io.read(VALID, { format: "no-such-format" }),
    (err) => {
      assert.match(err.message, /no-such-format|unknown|format/i);
      return true;
    },
  );
  assert.throws(
    () => io.read("garbage ???\n", { format: "xyz" }),
    (err) => {
      assert.ok(err.message.length > 0);
      return true;
    },
  );
  // A later failure with a different cause reports its own message.
  assert.throws(
    () => io.read("garbage ???\n", { format: "xyz" }),
    (err) => {
      assert.doesNotMatch(err.message, /no-such-format/);
      return true;
    },
  );
});
