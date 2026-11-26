// SPDX-License-Identifier: MIT
import { test } from "node:test";
import assert from "node:assert/strict";
import { init, formats, read, readAll, canWrite } from "../dist/index.js";
import { gemmyPath, readText, REPO_DATA, skipUnlessFile } from "./helpers.mjs";

// Top-level await: `skip` options below are evaluated while the file loads.
await init();

const has = (name) => formats().includes(name);

test("formats() reports the compiled-in formats", () => {
  const list = formats();
  assert.ok(Array.isArray(list));
  assert.ok(list.length > 0, "the core registry always has at least one format");
  for (const name of list) {
    assert.equal(typeof name, "string");
  }
});

test("canWrite agrees with the formats that have a writer", () => {
  if (has("vasp")) assert.equal(canWrite("vasp"), true);
  if (has("extxyz")) assert.equal(canWrite("extxyz"), true);
  assert.equal(canWrite("no-such-format"), false);
});

test("ice-ih POSCAR: species line 'O H1 H2' keeps labels, elements H", {
  skip: !has("vasp") || skipUnlessFile(gemmyPath("ice-ih/POSCAR")),
}, () => {
  const atoms = read(readText(gemmyPath("ice-ih/POSCAR")), { format: "vasp" });
  assert.equal(atoms.length, 96);
  const labels = atoms.getLabels();
  const symbols = atoms.getChemicalSymbols();
  assert.equal(labels.filter((l) => l === "O").length, 32);
  assert.equal(labels.filter((l) => l === "H1").length, 28);
  assert.equal(labels.filter((l) => l === "H2").length, 36);
  labels.forEach((label, i) => {
    if (label.startsWith("H")) assert.equal(symbols[i], "H", `atom ${i} (${label})`);
    if (label === "O") assert.equal(symbols[i], "O", `atom ${i}`);
  });
  assert.deepEqual(atoms.pbc, [true, true, true]);
  assert.ok(Math.abs(atoms.getVolume() - 9.04 * 15.6577393004 * 7.37) < 1e-6);
});

test("NaCl, stishovite and C60 POSCARs parse with the right species counts", { skip: !has("vasp") }, async (t) => {
  const cases = [
    ["nacl/POSCAR", 8, { Na: 4, Cl: 4 }],
    ["stishovite/POSCAR", 6, { Si: 2, O: 4 }],
    ["c60/POSCAR", 240, { C: 240 }],
  ];
  for (const [file, n, counts] of cases) {
    const path = gemmyPath(file);
    if (skipUnlessFile(path)) {
      t.diagnostic(`skipped ${file}: ${skipUnlessFile(path)}`);
      continue;
    }
    const atoms = read(readText(path), { format: "vasp" });
    assert.equal(atoms.length, n, file);
    const tally = {};
    for (const s of atoms.getChemicalSymbols()) tally[s] = (tally[s] ?? 0) + 1;
    assert.deepEqual(tally, counts, file);
  }
});

test("dna-1bna extxyz: Lattice comment becomes the cell, frames read", {
  skip: !has("extxyz") && !has("xyz") ? "no extxyz reader" : skipUnlessFile(gemmyPath("dna-1bna/1bna.extxyz")),
}, () => {
  const text = readText(gemmyPath("dna-1bna/1bna.extxyz"));
  const expected = Number.parseInt(text.split("\n", 1)[0], 10);
  const atoms = read(text, { format: "extxyz" });
  assert.equal(atoms.length, expected);
  assert.ok(atoms.getVolume() > 0, "Lattice= gives a non-zero cell");
  assert.equal(atoms.getLabels().length, expected);
});

test("read() returns the last frame by default, index selects others", {
  skip: !has("xyz") && !has("extxyz") ? "no xyz reader" : skipUnlessFile(`${REPO_DATA}/xyz/multi.xyz`),
}, () => {
  const text = readText(`${REPO_DATA}/xyz/multi.xyz`);
  const frames = readAll(text, { format: "extxyz" });
  assert.ok(frames.length > 1, "fixture has several frames");
  const last = read(text, { format: "extxyz" });
  assert.equal(last.length, frames[frames.length - 1].length);
  const first = read(text, { format: "extxyz", index: 0 });
  assert.deepEqual(first.getPositions(), frames[0].getPositions());
});

test("the format is guessed from the file name when none is given", {
  skip: !has("vasp") || skipUnlessFile(gemmyPath("nacl/POSCAR")),
}, () => {
  const text = readText(gemmyPath("nacl/POSCAR"));
  const atoms = read(text, { filename: "POSCAR" });
  assert.equal(atoms.length, 8);
});

test("a format that is not compiled in is an error with a message", () => {
  assert.throws(
    () => read("1\n\nH 0 0 0\n", { format: "no-such-format" }),
    (err) => {
      assert.ok(err instanceof Error);
      assert.match(err.message, /no-such-format/);
      return true;
    },
  );
});
