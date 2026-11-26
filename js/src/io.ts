// SPDX-License-Identifier: MIT
/**
 * ASE-style structure IO: `io.read` and `io.write`.
 *
 *   import { io } from "fastatoms";
 *   const atoms = io.read(text, { format: "vasp" });        // Atoms
 *   const frames = io.read(text, { filename: "traj.xyz", index: ":" });
 *   const file = await io.read(new File([bytes], "POSCAR.gz")); // Promise<Atoms>
 *   const out = io.write(atoms, { filename: "CONTCAR" });       // string
 *
 * Input types for `read`: a string, an ArrayBuffer or Uint8Array (decoded as
 * UTF-8, BOM stripped), or a Blob/File. A Blob is read asynchronously, so for
 * a Blob or File `read` returns a Promise; every other input gives the result
 * at once. Gzipped bytes (magic 1f 8b) are inflated with DecompressionStream,
 * which needs the async path, so they are an error for the synchronous inputs.
 */

import { Atoms } from "./atoms.js";
import { requireModule } from "./module.js";

/** What `read` accepts as text-bearing input. */
export type TextInput = string | ArrayBuffer | Uint8Array;

/** Every input `read` accepts. A Blob (or File) makes the call async. */
export type IOInput = TextInput | Blob;

/**
 * Options of `read` and `write`. `format` and `filename` steer the format
 * guess; `index` selects frames (read only); anything else goes to the
 * format-specific reader or writer, as in `read(text, { format: "vasp",
 * species: ["Fe", "O"] })`.
 */
export interface IOOptions {
  /** Format name ("vasp", "xyz", "extxyz", "cif", "lammps-data", "lammps-dump"). */
  format?: string;
  /**
   * File name. Used to guess the format and named in error messages. A File
   * supplies its own `name` when this is omitted.
   */
  filename?: string;
  /**
   * Frames to read, as in ASE. `undefined` (default): the last frame. An
   * integer or an integer string: that one frame, negatives count from the
   * end. A slice string (":", "::2", "-3:", "1:5"): an array of frames.
   */
  index?: number | string;
  [option: string]: unknown;
}

/** The type `read` returns for the `index` option `I`. */
export type IndexResult<I> = [I] extends [undefined]
  ? Atoms
  : [I] extends [number]
    ? Atoms
    : string extends I
      ? Atoms | Atoms[]
      : I extends `${string}:${string}`
        ? Atoms[]
        : Atoms;

/** The `index` type of an options object `O`; `undefined` when not given. */
type IndexOf<O> = "index" extends keyof O ? O["index"] : undefined;

const MULTI_FRAME_WRITERS: ReadonlySet<string> = new Set(["xyz", "extxyz"]);
const SLICE_PATTERN = /^[+-]?\d*$/;

// ---------------------------------------------------------------------------
// Input decoding
// ---------------------------------------------------------------------------

function isBlob(input: IOInput): input is Blob {
  return typeof Blob !== "undefined" && input instanceof Blob;
}

function isGzip(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
}

function stripBom(text: string): string {
  return text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
}

/** UTF-8 decode. TextDecoder drops a leading BOM by default. */
function decodeUtf8(bytes: Uint8Array): string {
  return new TextDecoder("utf-8").decode(bytes);
}

function toBytes(input: ArrayBuffer | Uint8Array): Uint8Array {
  return input instanceof Uint8Array ? input : new Uint8Array(input);
}

async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream === "undefined") {
    throw new Error("gzip input needs DecompressionStream (Node 18+ or a current browser)");
  }
  const compressed = new Blob([bytes as BlobPart]).stream();
  const inflated = compressed.pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(inflated).arrayBuffer());
}

// ---------------------------------------------------------------------------
// Format guessing
// ---------------------------------------------------------------------------

/** The filename without a trailing ".gz", for the format guess. */
function baseName(filename: string | undefined): string | undefined {
  return filename?.replace(/\.gz$/i, "");
}

function guessOrFail(filename: string | undefined, head: string, label: string): string {
  const guessed = requireModule().guessFormat(baseName(filename) ?? "", head);
  if (guessed === "") {
    throw new Error(
      `cannot determine the structure format of ${label}: pass { format }, or a { filename } with a known extension`,
    );
  }
  return guessed;
}

function resolveFormat(format: string | undefined, filename: string | undefined, text: string, label: string): string {
  if (format !== undefined && format !== "") {
    return format;
  }
  return guessOrFail(filename, text.slice(0, 4096), label);
}

// ---------------------------------------------------------------------------
// Index handling
// ---------------------------------------------------------------------------

/** Python slice semantics over `n` frames: the frame positions it selects. */
function sliceIndices(n: number, start?: number, stop?: number, step = 1): number[] {
  if (step === 0) {
    throw new Error("index slice: step cannot be zero");
  }
  let from: number;
  let to: number;
  if (step > 0) {
    from = start === undefined ? 0 : start < 0 ? Math.max(start + n, 0) : Math.min(start, n);
    to = stop === undefined ? n : stop < 0 ? Math.max(stop + n, 0) : Math.min(stop, n);
  } else {
    from = start === undefined ? n - 1 : start < 0 ? Math.max(start + n, -1) : Math.min(start, n - 1);
    to = stop === undefined ? -1 : stop < 0 ? Math.max(stop + n, -1) : Math.min(stop, n - 1);
  }
  const out: number[] = [];
  for (let i = from; step > 0 ? i < to : i > to; i += step) {
    out.push(i);
  }
  return out;
}

function parseSliceField(field: string | undefined, spec: string): number | undefined {
  if (field === undefined || field.trim() === "") {
    return undefined;
  }
  if (!SLICE_PATTERN.test(field.trim())) {
    throw new Error(`invalid index slice "${spec}"`);
  }
  return Number.parseInt(field, 10);
}

function parseIndex(index: number | string | undefined): number | "slice" {
  if (index === undefined) {
    return -1;
  }
  if (typeof index === "number") {
    if (!Number.isInteger(index)) {
      throw new Error(`index must be an integer or a slice string, got ${index}`);
    }
    return index;
  }
  if (index.includes(":")) {
    return "slice";
  }
  if (!/^\s*[+-]?\d+\s*$/.test(index)) {
    throw new Error(`invalid index "${index}": use an integer or a slice such as ":", "::2", "-3:"`);
  }
  return Number.parseInt(index, 10);
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

function readText(text: string, filename: string | undefined, options: IOOptions): Atoms | Atoms[] {
  const formatOptions: Record<string, unknown> = { ...options };
  delete formatOptions.format;
  delete formatOptions.filename;
  delete formatOptions.index;
  const label = filename === undefined ? "the input" : `"${filename}"`;
  const resolved = resolveFormat(options.format, baseName(filename), text, label);
  const mod = requireModule();
  const frameIndex = parseIndex(options.index);
  const json = (extra: Record<string, unknown>): string => JSON.stringify({ ...formatOptions, ...extra });

  if (frameIndex !== "slice") {
    return new Atoms(mod.read(text, resolved, json({ index: frameIndex })));
  }

  const spec = options.index as string;
  const parts = spec.split(":");
  if (parts.length > 3) {
    throw new Error(`invalid index slice "${spec}"`);
  }
  const start = parseSliceField(parts[0], spec);
  const stop = parseSliceField(parts[1], spec);
  const step = parseSliceField(parts[2], spec) ?? 1;
  const frames = mod.readAll(text, resolved, json({})).map((raw) => new Atoms(raw));
  const order = sliceIndices(frames.length, start, stop, step);
  const keep = new Set(order);
  const chosen = new Map<number, Atoms>();
  frames.forEach((frame, k) => {
    if (keep.has(k)) {
      chosen.set(k, frame);
    } else {
      frame.dispose();
    }
  });
  // Slice order, which may be reversed or strided; not file order.
  return order.map((k) => chosen.get(k) as Atoms);
}

async function readBlob(blob: Blob, options: IOOptions): Promise<Atoms | Atoms[]> {
  const filename = options.filename ?? (blob as Partial<File>).name;
  let bytes: Uint8Array = new Uint8Array(await blob.arrayBuffer());
  if (isGzip(bytes)) {
    bytes = await gunzip(bytes);
  }
  return readText(decodeUtf8(bytes), filename, options);
}

/**
 * Read structures from text or bytes. Returns one `Atoms` for `index` undefined,
 * an integer or an integer string; an `Atoms[]` for a slice string. A Blob or
 * File gives a Promise of the same.
 *
 * @example
 *   io.read(text, { format: "extxyz" });               // last frame
 *   io.read(text, { format: "extxyz", index: ":" });   // all frames
 *   await io.read(file, { index: -2 });                // from a File, gzip ok
 */
export function read<const O extends IOOptions = Record<string, never>>(
  input: TextInput,
  options?: O,
): IndexResult<IndexOf<O>>;
export function read<const O extends IOOptions = Record<string, never>>(
  input: Blob,
  options?: O,
): Promise<IndexResult<IndexOf<O>>>;
export function read(input: IOInput, options: IOOptions = {}): Atoms | Atoms[] | Promise<Atoms | Atoms[]> {
  if (isBlob(input)) {
    return readBlob(input, options);
  }
  if (typeof input === "string") {
    return readText(stripBom(input), options.filename, options);
  }
  const bytes = toBytes(input);
  if (isGzip(bytes)) {
    throw new Error("gzip data needs the async path: pass a Blob or File to io.read");
  }
  return readText(decodeUtf8(bytes), options.filename, options);
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

export interface WriteOptions {
  /** Output format. Guessed from `filename` when omitted. */
  format?: string;
  /** Used only to guess the format. A ".gz" name is refused: write returns text. */
  filename?: string;
  /** Format-specific options, e.g. `{ labels: false }` for POSCAR. */
  [option: string]: unknown;
}

/**
 * Serialise one structure, or a list of them for multi-frame formats (xyz,
 * extxyz), and return the text. The format comes from `format`, else from
 * `filename`.
 *
 * @throws if no format can be found, if a list is given for a single-frame
 *   format (as ASE does), or if the list is empty.
 */
export function write(atoms: Atoms | readonly Atoms[], options: WriteOptions = {}): string {
  const { format: given, filename, ...formatOptions } = options;
  if (filename !== undefined && /\.gz$/i.test(filename)) {
    throw new Error(`write: "${filename}" is gzipped; write returns text, so compress the output yourself`);
  }
  let format = given ?? "";
  if (format === "") {
    if (filename === undefined || filename === "") {
      throw new Error("write: { format } is required (or a { filename } with a known extension)");
    }
    format = guessOrFail(filename, "", `"${filename}"`);
  }

  const mod = requireModule();
  const json = JSON.stringify(formatOptions);
  if (atoms instanceof Atoms) {
    return mod.write(atoms.rawHandle(), format, json);
  }
  if (!MULTI_FRAME_WRITERS.has(format)) {
    throw new Error(
      `write: format "${format}" holds one structure; pass a single Atoms (multi-frame formats: xyz, extxyz)`,
    );
  }
  if (atoms.length === 0) {
    throw new Error("write: the list of structures is empty");
  }
  return atoms.map((frame) => mod.write(frame.rawHandle(), format, json)).join("");
}
