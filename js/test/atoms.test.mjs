// SPDX-License-Identifier: MIT
import { test } from "node:test";
import assert from "node:assert/strict";
import { init, formats, read, Atoms } from "../dist/index.js";
import { gemmyPath, readText, skipUnlessFile } from "./helpers.mjs";

// Top-level await: `skip` options below are evaluated while the file loads.
await init();

const has = (name) => formats().includes(name);

function nacl() {
  return Atoms.fromArrays({
    symbols: ["Na", "Cl"],
    positions: [0, 0, 0, 2.82, 2.82, 2.82],
    cell: [5.64, 0, 0, 0, 5.64, 0, 0, 0, 5.64],
    pbc: true,
  });
}

test("fromArrays builds atoms with symbols, labels and a row-major cell", () => {
  const atoms = nacl();
  assert.equal(atoms.length, 2);
  assert.ok(atoms.getAtomicNumbers() instanceof Int32Array);
  assert.deepEqual([...atoms.getAtomicNumbers()], [11, 17]);
  assert.deepEqual(atoms.getChemicalSymbols(), ["Na", "Cl"]);
  assert.deepEqual(atoms.getLabels(), ["Na", "Cl"], "labels default to symbols");
  assert.ok(atoms.cell instanceof Float64Array);
  assert.deepEqual([...atoms.cell], [5.64, 0, 0, 0, 5.64, 0, 0, 0, 5.64]);
  assert.deepEqual(atoms.pbc, [true, true, true]);
  assert.ok(Math.abs(atoms.getVolume() - 5.64 ** 3) < 1e-9);
});

test("fromArrays accepts atomic numbers and nested cell rows", () => {
  const atoms = Atoms.fromArrays({
    numbers: [8],
    positions: [1, 2, 3],
    cell: [
      [10, 0, 0],
      [0, 10, 0],
      [0, 0, 10],
    ],
  });
  assert.deepEqual(atoms.getChemicalSymbols(), ["O"]);
  assert.deepEqual(atoms.pbc, [false, false, false]);
  assert.deepEqual([...atoms.getPositions()], [1, 2, 3]);
});

test("fromArrays validates its arguments", () => {
  assert.throws(() => Atoms.fromArrays({ positions: [0, 0, 0] }), /exactly one of/);
  assert.throws(() => Atoms.fromArrays({ symbols: ["H"], numbers: [1], positions: [0, 0, 0] }), /exactly one of/);
  assert.throws(
    () => Atoms.fromArrays({ symbols: ["H", "H"], positions: [0, 0, 0] }),
    (err) => {
      assert.ok(err instanceof Error);
      assert.match(err.message, /3 \* N|values|expected/i);
      return true;
    },
  );
  assert.throws(() => Atoms.fromArrays({ symbols: ["Xx"], positions: [0, 0, 0] }), /Xx|element|symbol/i);
});

test("setPositions and setCell (with and without scaleAtoms) and setPbc", () => {
  const atoms = nacl();
  atoms.setPositions([0, 0, 0, 1, 1, 1]);
  assert.deepEqual([...atoms.getPositions()], [0, 0, 0, 1, 1, 1]);

  const fractional = [...atoms.getScaledPositions()];
  atoms.setCell([10, 0, 0, 0, 10, 0, 0, 0, 10], true);
  assert.deepEqual([...atoms.getScaledPositions()], fractional, "fractional kept with scaleAtoms");
  assert.deepEqual(
    [...atoms.getPositions()].slice(3),
    fractional.slice(3).map((f) => 10 * f),
  );

  const cartesian = [...atoms.getPositions()];
  atoms.setCell([20, 0, 0, 0, 20, 0, 0, 0, 20]);
  assert.deepEqual([...atoms.getPositions()], cartesian, "Cartesian positions kept without scaling");

  atoms.setPbc([true, false, false]);
  assert.deepEqual(atoms.pbc, [true, false, false]);
  atoms.setPbc(false);
  assert.deepEqual(atoms.pbc, [false, false, false]);
});

test("setPositions with the wrong length is a JS Error with a message", () => {
  const atoms = nacl();
  assert.throws(
    () => atoms.setPositions([1, 2]),
    (err) => {
      assert.ok(err instanceof Error);
      assert.ok(err.message.length > 0);
      return true;
    },
  );
});

test("wrap folds atoms into the cell; getScaledPositions can wrap", () => {
  const atoms = Atoms.fromArrays({
    symbols: ["H"],
    positions: [6.5, -1, 0],
    cell: [5, 0, 0, 0, 5, 0, 0, 0, 5],
    pbc: true,
  });
  assert.deepEqual([...atoms.getScaledPositions({ wrap: false })], [1.3, -0.2, 0]);
  const wrapped = [...atoms.getScaledPositions({ wrap: true })];
  assert.ok(Math.abs(wrapped[0] - 0.3) < 1e-9 && Math.abs(wrapped[1] - 0.8) < 1e-9);
  atoms.wrap();
  const p = atoms.getPositions();
  assert.ok(Math.abs(p[0] - 1.5) < 1e-9 && Math.abs(p[1] - 4) < 1e-9);
});

test("repeat keeps labels in ASE order (atom index varies fastest)", {
  skip: !has("vasp") || skipUnlessFile(gemmyPath("ice-ih/POSCAR")),
}, () => {
  const base = read(readText(gemmyPath("ice-ih/POSCAR")), { format: "vasp" });
  const n = base.length;
  const labels = base.getLabels();
  const big = base.repeat([1, 1, 2]);
  assert.equal(big.length, 2 * n);
  const repeated = big.getLabels();
  for (let k = 0; k < repeated.length; k++) {
    assert.equal(repeated[k], labels[k % n], `label of atom ${k}`);
  }
  assert.deepEqual(big.getChemicalSymbols().slice(0, n), base.getChemicalSymbols());
  assert.equal(big.cell[8], 2 * base.cell[8]);
  assert.equal(base.repeat(2).length, 8 * n);
});

test("repeat with a scalar repeats along all three axes", () => {
  const atoms = nacl().repeat(2);
  assert.equal(atoms.length, 16);
  assert.deepEqual([...atoms.cell], [11.28, 0, 0, 0, 11.28, 0, 0, 0, 11.28]);
  assert.throws(() => nacl().repeat([0, 1, 1]));
});

test("copy is independent, subset keeps order and labels", () => {
  const atoms = nacl();
  const copy = atoms.copy();
  copy.setPositions([9, 9, 9, 9, 9, 9]);
  assert.deepEqual([...atoms.getPositions()], [0, 0, 0, 2.82, 2.82, 2.82]);

  const sub = atoms.subset([1, 0]);
  assert.deepEqual(sub.getChemicalSymbols(), ["Cl", "Na"]);
  assert.deepEqual(sub.getLabels(), ["Cl", "Na"]);
});

test("getArray and arrayInfo expose the named per-atom arrays", () => {
  const atoms = read(
    '3\nLattice="4 0 0 0 4 0 0 0 4" Properties=species:S:1:pos:R:3:charge:R:1 pbc="T T T"\n' +
      "Na 0 0 0 1\nCl 1 1 1 -1\nCl 2 2 2 -1\n",
    { format: "extxyz" },
  );
  const info = atoms.arrayInfo();
  const names = atoms.arrays;
  assert.ok(names.includes("labels"));
  assert.ok(names.includes("charges"), "the charge column lands in the built-in charges");
  const charge = info.find((item) => item.name === "charges");
  assert.equal(charge.kind, "float");
  assert.equal(charge.ncols, 1);
  assert.deepEqual([...atoms.getArray("charges")], [1, -1, -1]);
  assert.deepEqual(atoms.getArray("labels"), ["Na", "Cl", "Cl"]);
  assert.throws(() => atoms.getArray("no-such-array"), /no-such-array/);
});

test("info is parsed from the JSON form and is a copy", {
  skip: !has("vasp") || skipUnlessFile(gemmyPath("ice-ih/POSCAR")),
}, () => {
  const atoms = read(readText(gemmyPath("ice-ih/POSCAR")), { format: "vasp" });
  assert.equal(typeof atoms.info, "object");
  assert.match(String(atoms.info.comment), /Ice Ih/);
  atoms.info.comment = "changed";
  assert.match(String(atoms.info.comment), /Ice Ih/, "mutating the copy does not change the atoms");
});

test("dispose frees the handle; later use throws, a second dispose is harmless", () => {
  const atoms = nacl();
  assert.equal(atoms.disposed, false);
  atoms.dispose();
  assert.equal(atoms.disposed, true);
  assert.throws(() => atoms.length, /disposed/);
  assert.throws(() => atoms.getPositions(), /disposed/);
  assert.doesNotThrow(() => atoms.dispose());
});

test("dispose does not affect atoms derived from it", () => {
  const atoms = nacl();
  const copy = atoms.copy();
  atoms.dispose();
  assert.equal(copy.length, 2);
  copy.dispose();
});

test("info numbers survive the JSON hop with their shortest exact form", () => {
  const atoms = read(
    '1\nenergy=-1.1 label="a b" Lattice="3 0 0 0 3 0 0 0 3" Properties=species:S:1:pos:R:3\nH 0 0 0\n',
    {
      format: "extxyz",
    },
  );
  assert.equal(atoms.info.energy, -1.1);
  assert.equal(atoms.info.label, "a b");
});
