// SPDX-License-Identifier: MIT
import { test } from "node:test";
import assert from "node:assert/strict";
import { init, formats, io, Atoms } from "../dist/index.js";
import { readText, REPO_DATA, skipUnlessFile } from "./helpers.mjs";

// Top-level await: `skip` options below are evaluated while the file loads.
await init();

const has = (name) => formats().includes(name);

/** A small VASP 5 file: NaCl in a 5.64 A cubic cell. */
const POSCAR = [
  "NaCl",
  "1.0",
  "5.64 0 0",
  "0 5.64 0",
  "0 0 5.64",
  "Na Cl",
  "1 1",
  "Direct",
  "0 0 0",
  "0.5 0.5 0.5",
  "",
].join("\n");

/** Five single-atom frames whose x coordinate is the frame number. */
function trajectory() {
  const frames = [0, 1, 2, 3, 4].map((k) =>
    Atoms.fromArrays({ symbols: ["H"], positions: [k, 0, 0], cell: [10, 0, 0, 0, 10, 0, 0, 0, 10], pbc: true }),
  );
  const text = io.write(frames, { format: "extxyz" });
  frames.forEach((frame) => {
    frame.dispose();
  });
  return text;
}

/** The x coordinate of each frame, in order. */
function xs(frames) {
  return frames.map((atoms) => atoms.getPositions()[0]);
}

test("io.read takes a string, Uint8Array and ArrayBuffer (UTF-8, BOM stripped)", { skip: !has("vasp") }, () => {
  const bytes = new TextEncoder().encode(POSCAR);
  const withBom = new Uint8Array([0xef, 0xbb, 0xbf, ...bytes]);
  for (const input of [POSCAR, bytes, withBom, withBom.buffer]) {
    const atoms = io.read(input, { format: "vasp" });
    assert.equal(atoms.length, 2);
    assert.deepEqual(atoms.getChemicalSymbols(), ["Na", "Cl"]);
  }
  assert.equal(io.read(`\uFEFF${POSCAR}`, { format: "vasp" }).length, 2, "BOM in a string");
});

test("io.read takes a Blob and a File; a File names itself for the format guess", { skip: !has("vasp") }, async () => {
  const blob = new Blob([POSCAR]);
  const fromBlob = await io.read(blob, { format: "vasp" });
  assert.equal(fromBlob.length, 2);
  const file = new File([POSCAR], "POSCAR");
  const fromFile = await io.read(file);
  assert.deepEqual(fromFile.getChemicalSymbols(), ["Na", "Cl"]);
  assert.ok(fromFile instanceof Atoms, "a single frame by default");
});

test("a Blob or File gives a Promise, the other inputs do not", { skip: !has("vasp") }, () => {
  assert.ok(io.read(POSCAR, { format: "vasp" }) instanceof Atoms);
  assert.ok(io.read(new TextEncoder().encode(POSCAR), { format: "vasp" }) instanceof Atoms);
  const pending = io.read(new Blob([POSCAR]), { format: "vasp" });
  assert.ok(pending instanceof Promise);
  return pending;
});

test("index: undefined is the last frame, negatives count from the end", { skip: !has("extxyz") }, () => {
  const text = trajectory();
  assert.deepEqual(xs([io.read(text, { format: "extxyz" })]), [4]);
  assert.deepEqual(xs([io.read(text, { format: "extxyz", index: -1 })]), [4]);
  assert.deepEqual(xs([io.read(text, { format: "extxyz", index: -2 })]), [3]);
  assert.deepEqual(xs([io.read(text, { format: "extxyz", index: 0 })]), [0]);
  assert.deepEqual(xs([io.read(text, { format: "extxyz", index: 2 })]), [2]);
});

test("index: a plain integer string gives one Atoms", { skip: !has("extxyz") }, () => {
  const text = trajectory();
  const one = io.read(text, { format: "extxyz", index: "2" });
  assert.ok(one instanceof Atoms);
  assert.deepEqual(xs([one]), [2]);
  assert.deepEqual(xs([io.read(text, { format: "extxyz", index: "-1" })]), [4]);
  assert.throws(() => io.read(text, { format: "extxyz", index: "two" }), /invalid index/);
});

test("index: slice strings give Atoms[] in ASE/Python order", { skip: !has("extxyz") }, () => {
  const text = trajectory();
  const every = (index) => io.read(text, { format: "extxyz", index });
  const all = every(":");
  assert.ok(Array.isArray(all));
  assert.deepEqual(xs(all), [0, 1, 2, 3, 4]);
  assert.deepEqual(xs(every("::2")), [0, 2, 4]);
  assert.deepEqual(xs(every("-3:")), [2, 3, 4]);
  assert.deepEqual(xs(every("1:5")), [1, 2, 3, 4]);
  assert.deepEqual(xs(every("1:-1")), [1, 2, 3]);
  assert.deepEqual(xs(every("::-1")), [4, 3, 2, 1, 0]);
  assert.deepEqual(xs(every("10:")), []);
  assert.throws(() => every("1:2:3:4"), /invalid index slice/);
  assert.throws(() => every("::0"), /step/);
});

test("index: the Blob path honours the same rules", { skip: !has("extxyz") }, async () => {
  const text = trajectory();
  const frames = await io.read(new Blob([text]), { format: "extxyz", index: "-3:" });
  assert.deepEqual(xs(frames), [2, 3, 4]);
  const last = await io.read(new File([text], "traj.xyz"));
  assert.deepEqual(xs([last]), [4]);
});

test("a gzipped Blob is inflated (CompressionStream made it) and read", { skip: !has("vasp") }, async () => {
  const stream = new Blob([POSCAR]).stream().pipeThrough(new CompressionStream("gzip"));
  const gz = new Uint8Array(await new Response(stream).arrayBuffer());
  assert.deepEqual([gz[0], gz[1]], [0x1f, 0x8b], "fixture is gzip");

  const file = new File([gz], "POSCAR.gz");
  const atoms = await io.read(file);
  assert.deepEqual(atoms.getChemicalSymbols(), ["Na", "Cl"], "format from POSCAR after .gz is stripped");

  const named = await io.read(new Blob([gz]), { format: "vasp" });
  assert.equal(named.length, 2);
});

test("gzipped bytes in a synchronous input are an error that says to use a Blob", { skip: !has("vasp") }, () => {
  const gz = new Uint8Array([0x1f, 0x8b, 0x08, 0x00]);
  assert.throws(() => io.read(gz, { format: "vasp" }), /Blob or File/);
});

test("format is guessed from the content when there is no filename", {
  skip: !has("cif") || skipUnlessFile(`${REPO_DATA}/cif/nacl.cif`),
}, () => {
  const atoms = io.read(readText(`${REPO_DATA}/cif/nacl.cif`));
  assert.ok(atoms.length > 0);
});

test("cif: info.occupancy and spacegroup_kinds follow ASE", {
  skip: !has("cif") || skipUnlessFile(`${REPO_DATA}/cif/disorder.cif`),
}, () => {
  const atoms = io.read(readText(`${REPO_DATA}/cif/disorder.cif`), {
    format: "cif",
    disorder: "max_occupancy",
  });
  assert.deepEqual(atoms.info.occupancy, {
    0: { Fe: 0.6, Co: 0.4 },
    1: { Co: 0.4, Fe: 0.6 },
    2: { Ni: 0.5, Cu: 0.5 },
    3: { Cu: 0.5, Ni: 0.5 },
  });
  assert.deepEqual(Object.keys(atoms.info.occupancy[1]), ["Co", "Fe"]);
  assert.deepEqual([...atoms.getArray("spacegroup_kinds")], [0, 2]);
});

test("format is guessed from the filename, else an error names the file", { skip: !has("vasp") }, () => {
  assert.equal(io.read(POSCAR, { filename: "CONTCAR" }).length, 2);
  assert.throws(() => io.read("just some text\n", { filename: "notes.txt" }), /notes\.txt/);
  assert.throws(() => io.read("just some text\n"), /cannot determine/);
});

test("io.read forwards format options and rejects unknown ones", { skip: !has("vasp") }, () => {
  assert.throws(() => io.read(POSCAR, { format: "vasp", no_such_option: 1 }));
});

test("io.write returns text; a list needs a multi-frame format", { skip: !has("extxyz") }, () => {
  const one = io.read(POSCAR, { format: "vasp" });
  const text = io.write(one, { format: "extxyz" });
  assert.equal(typeof text, "string");
  assert.equal(io.read(text, { format: "extxyz" }).length, 2);

  const list = io.read(trajectory(), { format: "extxyz", index: ":" });
  const written = io.write(list, { format: "extxyz" });
  assert.deepEqual(xs(io.read(written, { format: "extxyz", index: ":" })), [0, 1, 2, 3, 4]);

  assert.throws(() => io.write(list, { format: "vasp" }), /one structure/);
  assert.throws(() => io.write([], { format: "extxyz" }), /empty/);
});

test("io.write guesses the format from the filename, and needs one of the two", {
  skip: !has("vasp") || !has("extxyz"),
}, () => {
  const atoms = io.read(POSCAR, { format: "vasp" });
  assert.match(io.write(atoms, { filename: "CONTCAR" }), /Direct|Cartesian/);
  assert.match(io.write(atoms, { filename: "out.xyz" }), /Na/);
  assert.throws(() => io.write(atoms, {}), /format/);
  assert.throws(() => io.write(atoms, { filename: "structure.unknown" }), /structure\.unknown|cannot determine/);
  assert.throws(() => io.write(atoms, { filename: "out.extxyz.gz" }), /gzip/);
});

/** A two-frame extxyz written by io.write, with frame k at x = k. */
const LAMMPS_DUMP = [
  "ITEM: TIMESTEP",
  "42",
  "ITEM: NUMBER OF ATOMS",
  "1",
  "ITEM: BOX BOUNDS pp pp pp",
  "0 10",
  "0 10",
  "0 10",
  "ITEM: ATOMS id type x y z",
  "1 1 0 0 0",
  "",
].join("\n");

test("format guess with no filename: plain XYZ text is read as xyz", { skip: !has("xyz") && !has("extxyz") }, () => {
  const atoms = io.read("1\n\nH 0 0 0\n");
  assert.equal(atoms.length, 1);
  assert.deepEqual(atoms.getChemicalSymbols(), ["H"]);
});

test("format guess: a content signature beats the extension (LAMMPS dump named traj.xyz)", {
  skip: !has("lammps-dump"),
}, () => {
  const atoms = io.read(LAMMPS_DUMP, { filename: "traj.xyz" });
  assert.equal(atoms.length, 1);
  assert.ok(
    atoms.arrayInfo().some((info) => info.name === "id"),
    "a LAMMPS dump has an id array",
  );
  assert.equal(atoms.info.timestep, 42);
});

test("index: null is the last frame, like undefined", { skip: !has("extxyz") }, () => {
  const text = trajectory();
  const last = io.read(text, { format: "extxyz", index: null });
  assert.ok(last instanceof Atoms);
  assert.deepEqual(xs([last]), [4]);
});

test("index: a malformed slice is an error before any frame is read", { skip: !has("extxyz") }, () => {
  const text = trajectory();
  for (const bad of ["::+", "+:", "1x:", "::-", ":-:", "1::2:3", "::0", "1:2:3:4", "--1:"]) {
    assert.throws(() => io.read(text, { format: "extxyz", index: bad }), /index|step/, `index "${bad}"`);
  }
  assert.deepEqual(xs(io.read(text, { format: "extxyz", index: "+1:" })), [1, 2, 3, 4], "a signed start is fine");
  assert.deepEqual(xs(io.read(text, { format: "extxyz", index: " 1 : 3 " })), [1, 2]);
});

test("io.write takes a list for cif: a 2-frame CIF reads back as 2 frames", {
  skip: !has("cif"),
}, () => {
  const make = (a) =>
    Atoms.fromArrays({
      symbols: ["Na", "Cl"],
      positions: [0, 0, 0, a / 2, a / 2, a / 2],
      cell: [a, 0, 0, 0, a, 0, 0, 0, a],
      pbc: true,
    });
  const frames = [make(5.64), make(5.9)];
  const text = io.write(frames, { format: "cif" });
  assert.equal((text.match(/^data_/gm) ?? []).length, 2, "one data block per frame");
  frames.forEach((frame) => {
    frame.dispose();
  });

  const back = io.read(text, { format: "cif", index: ":" });
  assert.equal(back.length, 2);
  const cells = back.map((atoms) => atoms.cell[0]);
  assert.ok(Math.abs(cells[0] - 5.64) < 1e-6 && Math.abs(cells[1] - 5.9) < 1e-6, `cell lengths ${cells}`);
});
