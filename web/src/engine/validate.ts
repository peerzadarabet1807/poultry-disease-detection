/** Boundary check: whatever the worker returns is checked here before the UI sees it. */
import { CLASS_IDS } from '../lib/classes';
import { CAM_SIZE } from '../lib/cam';
import { MODEL_SIZE } from '../lib/resize';
import { EngineError, type Prediction } from './types';

const N = CLASS_IDS.length;
const CAM_CELLS = CAM_SIZE * CAM_SIZE;
const SEEN_BYTES = MODEL_SIZE * MODEL_SIZE * 4;

function fail(what: string): never {
  throw new EngineError(`The model returned an invalid result (${what}).`);
}

export function validatePrediction(p: Prediction): Prediction {
  const probs = p.probabilities;
  if (!Array.isArray(probs) || probs.length !== N) fail(`expected ${N} probabilities`);
  if (probs.some((x) => typeof x !== 'number' || !Number.isFinite(x) || x < 0 || x > 1)) {
    fail('probabilities outside [0, 1]');
  }
  const sum = probs.reduce((a, b) => a + b, 0);
  if (Math.abs(sum - 1) > 1e-3) fail(`probabilities sum to ${sum.toFixed(4)}`);
  if (!Array.isArray(p.camMaps) || p.camMaps.length !== N) fail(`expected ${N} class maps`);
  for (const m of p.camMaps) {
    if (!(m instanceof Float32Array) || m.length !== CAM_CELLS) fail(`class map size ${m?.length}`);
    for (const v of m) if (!(v >= 0 && v <= 1)) fail('class map values outside [0, 1]');
  }
  if (!(p.seen instanceof Uint8ClampedArray) || p.seen.length !== SEEN_BYTES) fail('model input image size');
  if (typeof p.exactPreprocessing !== 'boolean') fail('preprocessing flag');
  if (!(p.inferenceMs >= 0)) fail('timing');
  return p;
}
