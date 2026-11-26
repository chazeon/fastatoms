// SPDX-License-Identifier: MIT
import { test } from "node:test";
import assert from "node:assert/strict";
import { init, hasNeighbors, neighborList, Atoms } from "../dist/index.js";

// Top-level await: `skip` options below are evaluated while the file loads.
await init();

/** Rock-salt NaCl, conventional 8-atom cell, a = 5.64 A (gemmy nacl example). */
function rockSalt() {
  const a = 5.64;
  const basis = [
    ["Na", 0, 0, 0],
    ["Na", 0, 0.5, 0.5],
    ["Na", 0.5, 0, 0.5],
    ["Na", 0.5, 0.5, 0],
    ["Cl", 0.5, 0, 0],
    ["Cl", 0, 0.5, 0],
    ["Cl", 0, 0, 0.5],
    ["Cl", 0.5, 0.5, 0.5],
  ];
  const symbols = basis.map((b) => b[0]);
  const positions = basis.flatMap((b) => [b[1] * a, b[2] * a, b[3] * a]);
  return Atoms.fromArrays({
    symbols,
    positions,
    cell: [a, 0, 0, 0, a, 0, 0, 0, a],
    pbc: true,
  });
}

const skip = hasNeighbors() ? false : "neighbour list not compiled in";

test("NaCl: every ion has six nearest neighbours inside 3.0 A", { skip }, () => {
  const atoms = rockSalt();
  const { i, j, S, d, D } = neighborList("ijSdD", atoms, 3.0);
  const n = atoms.length;
  assert.equal(i.length, 6 * n, "six pairs per atom, full list");
  const counts = new Array(n).fill(0);
  for (const k of i) counts[k] += 1;
  assert.ok(
    counts.every((c) => c === 6),
    `coordination ${counts}`,
  );

  const nearest = 5.64 / 2; // Na-Cl distance in rock salt
  for (let k = 0; k < i.length; k++) {
    assert.ok(d[k] < 3.0, "distance below cutoff");
    assert.ok(Math.abs(d[k] - nearest) < 1e-9, `nearest-neighbour distance ${d[k]}`);
    const dx = D[3 * k],
      dy = D[3 * k + 1],
      dz = D[3 * k + 2];
    assert.ok(Math.abs(Math.hypot(dx, dy, dz) - d[k]) < 1e-9, "D is the vector behind d");
    const selfAtZero = i[k] === j[k] && S[3 * k] === 0 && S[3 * k + 1] === 0 && S[3 * k + 2] === 0;
    assert.ok(!selfAtZero, "no self pair at S=0");
  }
});

test("NaCl: pairs are sorted by (i, j, S) and the full list is symmetric", { skip }, () => {
  const { i, j, S } = neighborList("ijS", rockSalt(), 3.0);
  const key = (p) => [i[p], j[p], S[3 * p], S[3 * p + 1], S[3 * p + 2]];
  for (let k = 1; k < i.length; k++) {
    const a = key(k - 1);
    const b = key(k);
    let c = 0;
    while (c < 5 && a[c] === b[c]) c++;
    assert.ok(c === 5 || a[c] < b[c], `pairs out of order at ${k}`);
  }
  const forward = new Set();
  for (let k = 0; k < i.length; k++) {
    forward.add(`${i[k]}|${j[k]}|${S[3 * k]},${S[3 * k + 1]},${S[3 * k + 2]}`);
  }
  for (let k = 0; k < i.length; k++) {
    const mirror = `${j[k]}|${i[k]}|${-S[3 * k]},${-S[3 * k + 1]},${-S[3 * k + 2]}`;
    assert.ok(forward.has(mirror), "every pair has its mirror");
  }
});

test("full=false keeps one of each pair", { skip }, () => {
  const { i } = neighborList("i", rockSalt(), 3.0, { full: false });
  assert.equal(i.length, 24);
});

test("only the requested keys are returned", { skip }, () => {
  const result = neighborList("ij", rockSalt(), 3.0);
  assert.deepEqual(Object.keys(result).sort(), ["i", "j"]);
  assert.ok(result.i instanceof Int32Array);
});

test("a cutoff larger than the cell gives periodic images", { skip }, () => {
  const atoms = Atoms.fromArrays({
    symbols: ["Ar"],
    positions: [0, 0, 0],
    cell: [2, 0, 0, 0, 2, 0, 0, 0, 2],
    pbc: true,
  });
  const { i, d } = neighborList("id", atoms, 2.1);
  assert.equal(i.length, 6, "six images at distance 2 (no self pair at S=0)");
  for (const dist of d) assert.ok(Math.abs(dist - 2) < 1e-9);
});

test("unknown quantity letters and bad arguments are JS Errors", { skip }, () => {
  assert.throws(() => neighborList("ix", rockSalt(), 3.0), /unknown quantity 'x'/);
  assert.throws(() => neighborList("", rockSalt(), 3.0), /at least one/);
  assert.throws(
    () => neighborList("i", rockSalt(), -1),
    (err) => err instanceof Error,
  );
});
