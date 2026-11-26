// SPDX-License-Identifier: MIT
// init() given the glue factory, as a bundler user passes it: the glue is
// imported statically, not loaded from dist/wasm/ at run time.
import { test } from "node:test";
import assert from "node:assert/strict";
import createFastAtoms from "../dist/wasm/fastatoms.mjs";
import { init, read, formats } from "../dist/index.js";

await init(createFastAtoms);

const POSCAR = `NaCl
1.0
5.64 0 0
0 5.64 0
0 0 5.64
Na Cl
1 1
Direct
0 0 0
0.5 0.5 0.5
`;

test("init(glue) loads the module and reads a POSCAR", () => {
  assert.ok(formats().includes("vasp"));
  const atoms = read(POSCAR, { filename: "POSCAR" });
  assert.equal(atoms.length, 2);
  assert.deepEqual(atoms.getChemicalSymbols(), ["Na", "Cl"]);
  assert.ok(Math.abs(atoms.getVolume() - 5.64 ** 3) < 1e-9);
});

test("a second init() resolves with or without glue and keeps the module", async () => {
  await init();
  await init(createFastAtoms);
  assert.deepEqual(read(POSCAR, { filename: "POSCAR" }).getChemicalSymbols(), ["Na", "Cl"]);
});
