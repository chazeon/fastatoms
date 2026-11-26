// SPDX-License-Identifier: MIT
/**
 * fastatoms: structure IO, symmetry and neighbour lists, compiled to WebAssembly.
 *
 *   import { init, read, neighborList, io, symmetry } from "fastatoms";
 *   await init();
 *   const atoms = read(poscarText, { filename: "POSCAR" });
 *   const nl = neighborList("ijd", atoms, 3.0);
 *   const frames = io.read(trajectory, { filename: "traj.xyz", index: ":" });
 *
 * `init()` must resolve before any other call. Everything else is synchronous,
 * except `io.read` on a Blob or File, which returns a Promise.
 */

import { Atoms } from "./atoms.js";
import * as io from "./io.js";
import { init, native, requireModule } from "./module.js";
import type { EmscriptenFactory, NeighborKey } from "./raw.js";

export { init, Atoms, io };
export type { EmscriptenFactory };
export * as symmetry from "./symmetry.js";
export type { ArrayInfo, ArrayKind, FromArraysOptions, Vec3 } from "./atoms.js";
export type { IndexResult, IOInput, IOOptions, TextInput, WriteOptions as IOWriteOptions } from "./io.js";

/** Options that are not forwarded to the format reader. */
export interface ReadOptions {
  /** Format name ("vasp", "xyz", "extxyz", "cif", "lammps-data", "lammps-dump"). */
  format?: string;
  /** File name, used to guess the format when `format` is omitted. */
  filename?: string;
  /** Frame to return (`read` only). Default -1: the last frame, like ASE. */
  index?: number;
  /** Format-specific options, e.g. `{ species: ["Fe", "O"] }` for VASP 4. */
  [option: string]: unknown;
}

export interface WriteOptions {
  /** Output format. Required. */
  format: string;
  /** Format-specific options, e.g. `{ labels: false }` for POSCAR. */
  [option: string]: unknown;
}

export interface NeighborOptions {
  /** Both (i, j, S) and (j, i, -S). Default true, as in ASE's full lists. */
  full?: boolean;
}

export interface NeighborArrays {
  /** Central atom index. */
  i: Int32Array;
  /** Neighbour atom index. */
  j: Int32Array;
  /** Cell offsets, 3 per pair. */
  S: Int32Array;
  /** Distances. */
  d: Float64Array;
  /** Vectors r[j] - r[i] + S @ cell, 3 per pair. */
  D: Float64Array;
}

type CharsOf<Q extends string> = Q extends `${infer C}${infer R}` ? C | CharsOf<R> : never;

/** The arrays named in a quantities string: `NeighborResult<"ijd">` has i, j and d. */
export type NeighborResult<Q extends string> = {
  [K in NeighborKey as K extends CharsOf<Q> ? K : never]: NeighborArrays[K];
};

const KNOWN_QUANTITIES = new Set<string>(["i", "j", "S", "d", "D"]);

/** Formats compiled into the loaded module, e.g. ["vasp", "xyz", "extxyz"]. */
export function formats(): string[] {
  return requireModule().availableFormats();
}

/** Whether `format` can be written. */
export function canWrite(format: string): boolean {
  return requireModule().canWrite(format);
}

/** Whether the neighbour-list function is compiled into the module. */
export function hasNeighbors(): boolean {
  return typeof requireModule().neighborList === "function";
}

/** Options of `readAll`: `read` without the frame index. */
export interface ReadAllOptions {
  format?: string;
  filename?: string;
  [option: string]: unknown;
}

/**
 * The frame at `index` (default -1, the last frame, as in ASE). The format is
 * taken from `format`, else guessed from `filename`, else from the text.
 * Same as `io.read` for text input.
 */
export function read(text: string, options: ReadOptions = {}): Atoms {
  return io.read(text, options) as Atoms;
}

/** Every frame in `text`, in file order. Same format rules as `read`. */
export function readAll(text: string, options: ReadAllOptions = {}): Atoms[] {
  return io.read(text, { ...options, index: ":" }) as Atoms[];
}

/** Serialise `atoms` in `options.format`. Returns the file contents. Same as `io.write`. */
export function write(atoms: Atoms, options: WriteOptions): string {
  return io.write(atoms, options);
}

/**
 * Neighbour pairs with ASE semantics (see the C++ header): sorted by (i, j, S),
 * no self pair at S = 0, periodic images included.
 *
 * `quantities` picks the arrays to return, letters from "ijSdD"; the result
 * holds only those keys.
 *
 * @example
 *   const { i, j, d } = neighborList("ijd", atoms, 3.0);
 */
export function neighborList<Q extends string>(
  quantities: Q,
  atoms: Atoms,
  cutoff: number,
  options: NeighborOptions = {},
): NeighborResult<Q> {
  const module = requireModule();
  if (typeof module.neighborList !== "function") {
    throw new Error("neighborList is not compiled into this build (FASTATOMS_WITH_NEIGHBORS=OFF)");
  }
  if (quantities.length === 0) {
    throw new Error("neighborList: give at least one quantity from i, j, S, d, D");
  }
  for (const letter of quantities) {
    if (!KNOWN_QUANTITIES.has(letter)) {
      throw new Error(`neighborList: unknown quantity '${letter}' (use letters from i, j, S, d, D)`);
    }
  }
  const handle = atoms.rawHandle();
  const full = options.full ?? true;
  return native(() => module.neighborList?.(handle, cutoff, full, quantities)) as NeighborResult<Q>;
}
