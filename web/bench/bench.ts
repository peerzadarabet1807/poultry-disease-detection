/**
 * Browser latency of one input-check image encoder candidate. Query parameters:
 *   runtime  tfjs | ort
 *   model    URL of model.json (tfjs) or of the .onnx file (ort), relative to the bench server
 *   backend  webgl | wasm | cpu (tfjs) or wasm (ort)
 *   runs     timed runs after warm-up (default 20)
 * The result (load, warm-up and median inference times, the WebGL renderer) is written to
 * window.__benchResult for scripts/bench-gate.ts to collect.
 */
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import { loadGraphModel } from '@tensorflow/tfjs-converter';
import wasmPlain from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm?url';
import wasmSimd from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm?url';
import wasmThreadedSimd from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm?url';

setWasmPaths({
  'tfjs-backend-wasm.wasm': wasmPlain,
  'tfjs-backend-wasm-simd.wasm': wasmSimd,
  'tfjs-backend-wasm-threaded-simd.wasm': wasmThreadedSimd,
});

declare global {
  interface Window {
    __benchResult?: unknown;
  }
}

const SIZE = 224;
const params = new URLSearchParams(location.search);
const runtime = params.get('runtime') ?? 'tfjs';
const modelUrl = params.get('model') ?? '';
const backend = params.get('backend') ?? 'webgl';
const runs = Number(params.get('runs') ?? 20);

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
}

function renderer(): string {
  const gl = document.createElement('canvas').getContext('webgl2');
  if (!gl) return 'no webgl2';
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  return ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : String(gl.getParameter(gl.RENDERER));
}

function input(): Float32Array {
  const x = new Float32Array(SIZE * SIZE * 3);
  for (let i = 0; i < x.length; i++) x[i] = (i * 37) % 256;
  return x;
}

async function benchTfjs() {
  await tf.setBackend(backend);
  await tf.ready();
  const t0 = performance.now();
  const model = await loadGraphModel(modelUrl);
  const loadMs = performance.now() - t0;
  const x = tf.tensor4d(input(), [1, SIZE, SIZE, 3]);
  const run = async () => {
    const y = model.execute(x) as tf.Tensor;
    await y.data();
    y.dispose();
  };
  const t1 = performance.now();
  await run();
  const warmupMs = performance.now() - t1;
  const times: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t = performance.now();
    await run();
    times.push(performance.now() - t);
  }
  x.dispose();
  return {
    loadMs,
    warmupMs,
    medianMs: median(times),
    minMs: Math.min(...times),
    activeBackend: tf.getBackend(),
  };
}

async function benchOrt() {
  const ort = await import('onnxruntime-web');
  ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency) : 1;
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
  const loadMs = performance.now() - t0;
  const feeds = { image: new ort.Tensor('float32', input(), [1, SIZE, SIZE, 3]) };
  const t1 = performance.now();
  await session.run(feeds);
  const warmupMs = performance.now() - t1;
  const times: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t = performance.now();
    await session.run(feeds);
    times.push(performance.now() - t);
  }
  return {
    loadMs,
    warmupMs,
    medianMs: median(times),
    minMs: Math.min(...times),
    activeBackend: `ort-wasm (${ort.env.wasm.numThreads} thread${ort.env.wasm.numThreads === 1 ? '' : 's'})`,
  };
}

(async () => {
  const status = document.getElementById('status')!;
  try {
    const r = runtime === 'ort' ? await benchOrt() : await benchTfjs();
    window.__benchResult = { ok: true, runtime, model: modelUrl, renderer: renderer(), runs, ...r };
  } catch (e) {
    window.__benchResult = { ok: false, runtime, model: modelUrl, error: String(e) };
  }
  status.textContent = 'done';
})();
