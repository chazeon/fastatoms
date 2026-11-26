// SPDX-License-Identifier: MIT
import { test } from "node:test";
import assert from "node:assert/strict";
import { init, io, Atoms, read, symmetry } from "../dist/index.js";
import { gemmyPath, readText, REPO_DATA, skipUnlessFile } from "./helpers.mjs";

// Top-level await: `skip` options below are evaluated while the file loads.
await init();

const WITH = symmetry.hasSymmetry();
const skipAll = WITH ? false : "built without symmetry (FASTATOMS_WITH_SYMMETRY=OFF)";

/** Conventional fcc Cu, 4 atoms in a 3.61 A cube. */
function fccCu() {
  const a = 3.61;
  const basis = [
    [0, 0, 0],
    [0.5, 0.5, 0],
    [0.5, 0, 0.5],
    [0, 0.5, 0.5],
  ];
  return Atoms.fromArrays({
    symbols: ["Cu", "Cu", "Cu", "Cu"],
    positions: basis.flatMap((f) => f.map((x) => x * a)),
    cell: [a, 0, 0, 0, a, 0, 0, 0, a],
    pbc: true,
  });
}

/** Primitive fcc Cu, one atom, with the basis vectors as rows. */
function primitiveFcc() {
  const h = 3.61 / 2;
  return Atoms.fromArrays({
    symbols: ["Cu"],
    positions: [0, 0, 0],
    cell: [0, h, h, h, 0, h, h, h, 0],
    pbc: true,
  });
}

test("hasSymmetry reports the build", () => {
  assert.equal(typeof symmetry.hasSymmetry(), "boolean");
});

test("without symmetry in the build, every symmetry call is a clear Error", {
  skip: WITH ? "built with symmetry" : false,
}, () => {
  assert.throws(() => symmetry.getSymmetryDataset(fccCu()), /FASTATOMS_WITH_SYMMETRY/);
  assert.throws(() => symmetry.standardizeCell(fccCu()), /FASTATOMS_WITH_SYMMETRY/);
});

test("fcc Cu is space group 225", { skip: skipAll }, () => {
  const ds = symmetry.getSymmetryDataset(fccCu());
  assert.equal(ds.number, 225);
  assert.equal(ds.international, "Fm-3m");
  assert.equal(ds.pointgroup, "m-3m");
  assert.equal(ds.hallNumber >= 1, true);
  assert.equal(ds.wyckoffs.length, 4);
  assert.ok(
    ds.wyckoffs.every((w) => w === "a"),
    "all four Cu on 4a",
  );
});

test("dataset arrays have the documented shapes", { skip: skipAll }, () => {
  const ds = symmetry.getSymmetryDataset(fccCu());
  const ops = ds.translations.length / 3;
  assert.equal(ds.rotations.length, 9 * ops);
  assert.ok(ds.rotations instanceof Int32Array);
  assert.ok(ds.translations instanceof Float64Array);
  assert.equal(ds.siteSymmetrySymbols.length, 4);
  assert.equal(ds.equivalentAtoms.length, 4);
  assert.equal(ds.crystallographicOrbits.length, 4);
  assert.equal(ds.stdLattice.length, 9);
  assert.equal(ds.stdPositions.length, 3 * ds.stdTypes.length);
  assert.equal(ds.transformationMatrix.length, 9);
  assert.equal(ds.originShift.length, 3);
  assert.equal(ds.primitiveLattice.length, 9);
  assert.equal(ds.mappingToPrimitive.length, 4);
  assert.equal(ds.stdMappingToPrimitive.length, ds.stdTypes.length);
  assert.equal(typeof ds.hall, "string");
  assert.equal(typeof ds.choice, "string");
});

test("rutile is space group 136 with Ti on 2a and O on 4f", {
  skip: skipAll || skipUnlessFile(`${REPO_DATA}/cif/rutile.cif`),
}, () => {
  const atoms = io.read(readText(`${REPO_DATA}/cif/rutile.cif`), { filename: "rutile.cif" });
  const ds = symmetry.getSymmetryDataset(atoms);
  assert.equal(ds.number, 136);
  const labels = atoms.getLabels();
  labels.forEach((label, k) => {
    if (label.startsWith("Ti")) assert.equal(ds.wyckoffs[k], "a", `${label} ${k}`);
    if (label.startsWith("O")) assert.equal(ds.wyckoffs[k], "f", `${label} ${k}`);
  });
});

test("the gemmy stishovite POSCAR is space group 136", {
  skip: skipAll || skipUnlessFile(gemmyPath("stishovite/POSCAR")),
}, () => {
  const atoms = read(readText(gemmyPath("stishovite/POSCAR")), { format: "vasp" });
  assert.equal(symmetry.getSymmetryDataset(atoms).number, 136);
});

test("standardizeCell of primitive fcc gives the 4-atom conventional cell", { skip: skipAll }, () => {
  const conventional = symmetry.standardizeCell(primitiveFcc());
  assert.ok(conventional instanceof Atoms);
  assert.equal(conventional.length, 4);
  assert.equal(conventional.pbc.every(Boolean), true);
});

test("findPrimitive gives back the one-atom fcc cell; niggliReduce keeps the atom count", { skip: skipAll }, () => {
  assert.equal(symmetry.findPrimitive(fccCu()).length, 1);
  const reduced = symmetry.niggliReduce(primitiveFcc());
  assert.equal(reduced.length, 1);
  assert.ok(Math.abs(reduced.getVolume() - primitiveFcc().getVolume()) < 1e-6);
});

test("useLabels types atoms by label and lowers the symmetry", { skip: skipAll }, () => {
  // Two labels of one element: Cu1 on the (0,0,0) sublattice, Cu2 on the other.
  const split = read(
    [
      "fcc Cu, two labels",
      "1.0",
      "3.61 0 0",
      "0 3.61 0",
      "0 0 3.61",
      "Cu1 Cu2",
      "2 2",
      "Direct",
      "0 0 0",
      "0.5 0.5 0",
      "0.5 0 0.5",
      "0 0.5 0.5",
      "",
    ].join("\n"),
    { format: "vasp" },
  );
  assert.deepEqual(split.getLabels(), ["Cu1", "Cu1", "Cu2", "Cu2"]);
  assert.equal(symmetry.getSymmetryDataset(fccCu()).number, 225);
  assert.equal(symmetry.getSymmetryDataset(split, { useLabels: false }).number, 225, "labels ignored");
  const labelled = symmetry.getSymmetryDataset(split, { useLabels: true }).number;
  assert.ok(labelled < 225, `expected a lower group, got ${labelled}`);
});

test("a non-periodic structure is a JS Error", { skip: skipAll }, () => {
  const cluster = Atoms.fromArrays({ symbols: ["H", "H"], positions: [0, 0, 0, 0.7, 0, 0] });
  assert.throws(
    () => symmetry.getSymmetryDataset(cluster),
    (err) => err instanceof Error,
  );
});
