// SPDX-License-Identifier: MIT
// Type-level checks for the io.read overloads. Nothing here runs: `npm run
// check` type-checks it with tsconfig.test.json (tsc --noEmit). Each expect
// directive below marks a call that must fail to compile, so the overloads
// cannot drift.
import { type Atoms, io, type IndexResult } from "../src/index.js";

declare const text: string;
declare const bytes: Uint8Array;
declare const blob: Blob;
declare const dynamicIndex: string;

// No options: one frame.
const one: Atoms = io.read(text);
const oneBytes: Atoms = io.read(bytes);
const oneFormat: Atoms = io.read(text, { format: "extxyz" });
const oneNull: Atoms = io.read(text, { format: "extxyz", index: null });
const oneInt: Atoms = io.read(text, { format: "extxyz", index: -1 });

// Slice strings: a list.
const all = io.read(text, { format: "extxyz", index: ":" });
const sliced: Atoms[] = io.read(text, { format: "extxyz", index: "::2" });
const dyn: Atoms | Atoms[] = io.read(text, { format: "extxyz", index: dynamicIndex });

// Blob and File: a Promise of the same.
const asyncOne: Promise<Atoms> = io.read(blob);
const asyncList: Promise<Atoms[]> = io.read(blob, { index: ":" });

// The type-level result for a literal index.
const literalList: IndexResult<":"> = [] as Atoms[];
declare const literalOne: IndexResult<-1>;

// Explicit type arguments without options are rejected: they must say what
// they mean, and the options argument is required for the generic overloads.
// @ts-expect-error a slice index with no options argument is not one frame
io.read<{ index: ":" }>(bytes);
// @ts-expect-error the generic overloads need the options argument
io.read<{ format: "extxyz"; index: ":" }>(text);

// A slice string never gives one Atoms.
// @ts-expect-error `index: ":"` gives Atoms[], not Atoms
const wrong: Atoms = io.read(text, { format: "extxyz", index: ":" });
// @ts-expect-error a Blob with a slice gives Promise<Atoms[]>, not Promise<Atoms>
const wrongAsync: Promise<Atoms> = io.read(blob, { index: ":" });

// Silence unused locals.
void [
  one,
  oneBytes,
  oneFormat,
  oneNull,
  oneInt,
  all,
  sliced,
  dyn,
  asyncOne,
  asyncList,
  literalList,
  literalOne,
  wrong,
  wrongAsync,
];
