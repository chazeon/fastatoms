// SPDX-License-Identifier: MIT
import type { EmscriptenFactory, RawModule } from "./raw.js";

let loading: Promise<RawModule> | undefined;
let loaded: RawModule | undefined;

async function load(): Promise<RawModule> {
  // The glue sits next to this file in dist/wasm/ (see scripts/build-wasm.sh).
  // A computed specifier keeps tsc from looking for it at build time.
  const glueUrl = new URL("./wasm/fastatoms.mjs", import.meta.url).href;
  const glue = (await import(glueUrl)) as { default: EmscriptenFactory };
  return glue.default();
}

/**
 * Load the WebAssembly module. Idempotent: the module is fetched and
 * instantiated once, and later calls reuse it. Every other function in this
 * package needs `init()` to have resolved first.
 */
export function init(): Promise<void> {
  if (loaded !== undefined) {
    return Promise.resolve();
  }
  loading ??= load().then(
    (mod) => {
      loaded = mod;
      return mod;
    },
    (error: unknown) => {
      loading = undefined; // allow a retry after a failed load
      throw error;
    },
  );
  return loading.then(() => undefined);
}

/** The loaded module. Throws if `init()` has not resolved yet. */
export function requireModule(): RawModule {
  if (loaded === undefined) {
    throw new Error("fastatoms is not initialised: await init() before using it");
  }
  return loaded;
}
