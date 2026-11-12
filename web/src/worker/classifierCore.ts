/**
 * All TensorFlow.js work: backend selection (WebGL -> WASM -> CPU), download of both models in parallel with
 * progress (the classifier and the input check's image encoder), IndexedDB caching keyed by each model's
 * content hash, warm-up, preprocessing and inference. Normally runs inside a Web Worker (classifier.worker.ts);
 * browsers without WebGL in workers run it on the main thread instead.
 *
 * Per photo: preprocess once, run the input check, then the classifier only if the photo passed (or the user
 * chose "Analyse anyway").
 */
import * as Comlink from 'comlink';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import { loadGraphModel, type GraphModel } from '@tensorflow/tfjs-converter';
import wasmPlain from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm?url';
import wasmSimd from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm?url';
import wasmThreadedSimd from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm?url';
import { checkEmbedding } from '../lib/inputCheck';
import { bitmapToRGBA } from '../lib/pixels';
import { rgbToRGBA, toModelInput, toSeenImage } from '../lib/preprocess';
import { MODEL_SIZE } from '../lib/resize';
import { GATE_INPUT, GATE_OUTPUT, runGate } from './gate';
import { INPUT_NAME, OUTPUT_NAMES, runModel } from './inference';
import type { BackendName, ClassifierApi, ClassifyOptions, ModelInfo, RunsIn, WorkerResult } from './types';

setWasmPaths({
  'tfjs-backend-wasm.wasm': wasmPlain,
  'tfjs-backend-wasm-simd.wasm': wasmSimd,
  'tfjs-backend-wasm-threaded-simd.wasm': wasmThreadedSimd,
});

interface CacheSlot {
  prefix: string;
  key: string;
}
const MODEL_CACHE: CacheSlot = {
  prefix: 'indexeddb://poultry-disease-',
  key: `indexeddb://poultry-disease-${__MODEL_VERSION__}`,
};
const GATE_CACHE: CacheSlot = {
  prefix: 'indexeddb://poultry-gate-',
  key: `indexeddb://poultry-gate-${__GATE_VERSION__}`,
};
const TOTAL_BYTES = __MODEL_BYTES__ + __GATE_BYTES__;
const BACKENDS: BackendName[] = ['webgl', 'wasm', 'cpu'];

async function selectBackend(preferred?: BackendName): Promise<BackendName> {
  const order = preferred ? [preferred, ...BACKENDS.filter((b) => b !== preferred)] : BACKENDS;
  for (const name of order) {
    try {
      if (await tf.setBackend(name)) {
        await tf.ready();
        return name;
      }
    } catch {
      /* try the next backend */
    }
  }
  throw new Error('No TensorFlow.js backend is available in this browser.');
}

async function loadFromCache(slot: CacheSlot): Promise<GraphModel | null> {
  try {
    const cached = await tf.io.listModels();
    if (cached[slot.key]) return await loadGraphModel(slot.key);
  } catch {
    /* IndexedDB unavailable (private browsing, blocked storage): fall back to the network */
  }
  return null;
}

async function saveToCache(m: GraphModel, slot: CacheSlot): Promise<void> {
  try {
    const cached = await tf.io.listModels();
    await Promise.all(
      Object.keys(cached)
        .filter((key) => key.startsWith(slot.prefix) && key !== slot.key)
        .map((key) => tf.io.removeModel(key)),
    );
    await m.save(slot.key);
  } catch (e) {
    console.warn('Could not cache the model in IndexedDB', e);
  }
}

/** Cache first, then the network; reports this model's download fraction. */
async function loadModel(
  url: string,
  slot: CacheSlot,
  onFraction: (f: number) => void,
): Promise<{ model: GraphModel; fromCache: boolean }> {
  const cached = await loadFromCache(slot);
  if (cached) {
    onFraction(1);
    return { model: cached, fromCache: true };
  }
  const model = await loadGraphModel(url, { onProgress: onFraction });
  return { model, fromCache: false };
}

/**
 * The first inference compiles every WebGL shader, which is slow one by one. A compile-only pass first lets
 * the driver compile them in parallel (KHR_parallel_shader_compile); one real run then primes the rest.
 */
async function compileShaders(m: GraphModel, input: string, outputs: string[]): Promise<void> {
  const webgl = tf.backend() as unknown as {
    checkCompileCompletionAsync(): Promise<boolean[]>;
    getUniformLocations(): void;
  };
  const x = tf.zeros([1, MODEL_SIZE, MODEL_SIZE, 3]);
  try {
    tf.env().set('ENGINE_COMPILE_ONLY', true);
    const out = m.execute({ [input]: x }, outputs) as tf.Tensor[];
    tf.env().set('ENGINE_COMPILE_ONLY', false);
    await webgl.checkCompileCompletionAsync();
    webgl.getUniformLocations();
    tf.dispose(out);
  } catch (e) {
    console.warn('Parallel shader compilation unavailable; compiling during warm-up instead', e);
  } finally {
    tf.env().set('ENGINE_COMPILE_ONLY', false);
    x.dispose();
  }
}

async function warmUp(m: GraphModel, gate: GraphModel, backend: BackendName): Promise<void> {
  if (backend === 'webgl') {
    await compileShaders(m, INPUT_NAME, [...OUTPUT_NAMES]);
    await compileShaders(gate, GATE_INPUT, [GATE_OUTPUT]);
  }
  await runModel(m, new Float32Array(MODEL_SIZE * MODEL_SIZE * 3));
  await runGate(gate, new Uint8Array(MODEL_SIZE * MODEL_SIZE * 3));
}

export function createClassifierApi(runsIn: RunsIn): ClassifierApi {
  let model: GraphModel | null = null;
  let gate: GraphModel | null = null;
  let info: ModelInfo | null = null;

  async function predict(
    data: Uint8ClampedArray,
    width: number,
    height: number,
    exact: boolean,
    options: ClassifyOptions,
  ): Promise<WorkerResult> {
    if (!model || !gate || !info) throw new Error('The model is not loaded yet.');
    const t0 = performance.now();
    const seenRgb = toSeenImage(data, width, height, 4);
    const t1 = performance.now();
    const inputCheck = checkEmbedding(await runGate(gate, seenRgb));
    const t2 = performance.now();
    let prediction: WorkerResult['prediction'] = null;
    if (inputCheck.passed || options.force) {
      const { probabilities, cam } = await runModel(model, toModelInput(seenRgb));
      prediction = { probabilities: Array.from(probabilities), cam, inferenceMs: performance.now() - t2 };
    }
    const seen = rgbToRGBA(seenRgb);
    const result: WorkerResult = {
      inputCheck,
      inputCheckMs: t2 - t1,
      prediction,
      seen,
      exact,
      preprocessMs: t1 - t0,
      backend: info.backend,
    };
    const transfers = prediction ? [prediction.cam.buffer, seen.buffer] : [seen.buffer];
    return runsIn === 'worker' ? Comlink.transfer(result, transfers) : result;
  }

  return {
    async init(urls, onProgress, preferredBackend) {
      if (model && gate && info) return info;
      const t0 = performance.now();
      const backend = await selectBackend(preferredBackend);

      // both downloads at once; progress is the byte-weighted share of the two
      const fractions = { model: 0, gate: 0 };
      const report = () =>
        onProgress({
          phase: 'download',
          fraction: (fractions.model * __MODEL_BYTES__ + fractions.gate * __GATE_BYTES__) / TOTAL_BYTES,
        });
      const [main, check] = await Promise.all([
        loadModel(urls.model, MODEL_CACHE, (f) => {
          fractions.model = f;
          report();
        }),
        loadModel(urls.gate, GATE_CACHE, (f) => {
          fractions.gate = f;
          report();
        }),
      ]);
      model = main.model;
      gate = check.model;
      const loadMs = performance.now() - t0;

      onProgress({ phase: 'warmup', fraction: 0 });
      const t1 = performance.now();
      await warmUp(model, gate, backend);
      const warmupMs = performance.now() - t1;
      onProgress({ phase: 'warmup', fraction: 1 });

      info = { backend, fromCache: main.fromCache && check.fromCache, loadMs, warmupMs, runsIn };
      // after warm-up, so caching never delays the first prediction
      if (!main.fromCache) void saveToCache(model, MODEL_CACHE);
      if (!check.fromCache) void saveToCache(gate, GATE_CACHE);
      return info;
    },

    async classify(bitmap, options) {
      const { data, width, height, exact } = bitmapToRGBA(bitmap);
      bitmap.close();
      return predict(data, width, height, exact, options);
    },

    async classifyPixels(data, width, height, exact, options) {
      return predict(data, width, height, exact, options);
    },

    async memory() {
      return { numTensors: tf.memory().numTensors, numBytes: tf.memory().numBytes };
    },
  };
}
