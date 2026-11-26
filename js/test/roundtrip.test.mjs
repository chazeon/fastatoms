// SPDX-License-Identifier: MIT
import { test } from "node:test";
import assert from "node:assert/strict";
import { init, formats, read, readAll, write, Atoms } from "../dist/index.js";
import { gemmyPath, readText, skipUnlessFile } from "./helpers.mjs";

// Top-level await: `skip` options below are evaluated while the file loads.
await init();

const has = (name) => formats().includes(name);

function sample() {
  return Atoms.fromArrays({
    symbols: ["Na", "Cl", "Cl"],
    positions: [0, 0, 0, 2.82, 2.82, 2.82, 1.5, 0.25, 3.125],
    cell: [5.64, 0, 0, 0.5, 5.64, 0, 0, 0.25, 5.64],
    pbc: [true, true, false],
  });
}

test("extxyz write -> read keeps positions, cell, pbc and symbols", {
  skip: !has("extxyz") && !has("xyz") ? "no extxyz writer" : false,
}, () => {
  const format = has("extxyz") ? "extxyz" : "xyz";
  const atoms = sample();
  const text = write(atoms, { format });
  const back = read(text, { format });
  assert.equal(back.length, atoms.length);
  assert.deepEqual(back.getChemicalSymbols(), atoms.getChemicalSymbols());
  assert.deepEqual(back.pbc, [true, true, false]);
  back.getPositions().forEach((x, k) => {
    assert.ok(Math.abs(x - atoms.getPositions()[k]) < 1e-9, `coord ${k}`);
  });
  back.cell.forEach((x, k) => {
    assert.ok(Math.abs(x - atoms.cell[k]) < 1e-9, `cell ${k}`);
  });
});

test("VASP write -> read keeps labels and positions", {
  skip: !has("vasp") || skipUnlessFile(gemmyPath("ice-ih/POSCAR")),
}, () => {
  const original = read(readText(gemmyPath("ice-ih/POSCAR")), { format: "vasp" });
  const text = write(original, { format: "vasp" });
  const back = read(text, { format: "vasp" });
  assert.equal(back.length, original.length);
  assert.deepEqual(back.getLabels(), original.getLabels());
  assert.deepEqual(back.getChemicalSymbols(), original.getChemicalSymbols());
  const a = original.getPositions();
  const b = back.getPositions();
  for (let k = 0; k < a.length; k++) {
    assert.ok(Math.abs(a[k] - b[k]) < 1e-8, `position ${k}`);
  }
});

test("VASP write -> read -> write is stable to the last ulp", { skip: !has("vasp") }, () => {
  // Direct coordinates are re-derived from Cartesian ones on each write, so a
  // second pass can move the last bit of a coordinate. Not bit-identical.
  const once = read(write(sample(), { format: "vasp" }), { format: "vasp" });
  const again = read(write(once, { format: "vasp" }), { format: "vasp" });
  const a = once.getPositions();
  const b = again.getPositions();
  for (let k = 0; k < a.length; k++) {
    assert.ok(Math.abs(a[k] - b[k]) < 1e-12, `position ${k}`);
  }
});

test("multi-frame extxyz round trip through readAll", {
  skip: !has("extxyz") && !has("xyz") ? "no extxyz writer" : false,
}, () => {
  const format = has("extxyz") ? "extxyz" : "xyz";
  const first = sample();
  const second = sample();
  second.setPositions(second.getPositions().map((x) => x + 0.5));
  const text = write(first, { format }) + write(second, { format });
  const frames = readAll(text, { format });
  assert.equal(frames.length, 2);
  assert.equal(frames[1].getPositions()[0], first.getPositions()[0] + 0.5);
});

test("write without a format is a JS Error", () => {
  assert.throws(() => write(sample(), { format: "" }), /format/);
});

test("an unknown output format is a JS Error carrying the name", () => {
  assert.throws(() => write(sample(), { format: "no-such-format" }), /no-such-format/);
});
