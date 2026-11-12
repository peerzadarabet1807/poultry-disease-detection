/**
 * What the in-browser engine (TensorFlow.js in a Web Worker, browserEngine.ts) hands to the UI: one validated
 * EngineOutput per photo (the input check, and the classifier's Prediction unless the check failed). The photo
 * never leaves the device; there is no server.
 */
import type { InputCheckResult } from '../lib/inputCheck';

export type { InputCheckResult };
export interface LoadProgress {
  phase: 'download' | 'warmup';
  /** 0..1 */
  fraction: number;
}

export type ProgressCallback = (p: LoadProgress) => void;

export interface EngineInfo {
  /** e.g. "WebGL (GPU)", "WebAssembly (CPU)" */
  backend: string;
  /** one short line for the status bar */
  detail: string;
  loadMs: number;
}

export interface Prediction {
  /** softmax, model order (models/class_names.json) */
  probabilities: number[];
  /** per class (model order): 7 x 7 class activation map after ReLU, divided by its maximum */
  camMaps: Float32Array[];
  /** 224 x 224 RGBA: exactly the image the model saw */
  seen: Uint8ClampedArray;
  /** false only when a browser had to pre-shrink a very large photo before the resize */
  exactPreprocessing: boolean;
  inferenceMs: number;
}

export interface EngineOutput {
  inputCheck: InputCheckResult;
  inputCheckMs: number;
  /** null when the input check failed and the classifier was not asked */
  prediction: Prediction | null;
  /** 224 x 224 RGBA: exactly the image both models saw */
  seen: Uint8ClampedArray;
}

export class EngineError extends Error {}
