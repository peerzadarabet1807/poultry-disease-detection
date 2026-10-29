/**
 * One interface for both ways of running the model: in the browser (BrowserEngine, TensorFlow.js in a Web
 * Worker, the default on GitHub Pages) or on the FastAPI backend (ApiEngine). Both return the same validated
 * Prediction, so the UI cannot tell them apart. Selected by VITE_INFERENCE_MODE ("browser" or "api").
 */
import type { DecodedPhoto } from '../lib/decode';
import type { ImageKind } from '../lib/fileValidation';

export type EngineKind = 'browser' | 'api';

export interface LoadProgress {
  phase: 'download' | 'warmup' | 'connect';
  /** 0..1 */
  fraction: number;
}

export type ProgressCallback = (p: LoadProgress) => void;

export interface EngineInfo {
  kind: EngineKind;
  /** e.g. "WebGL (GPU)", "WebAssembly (CPU)", "API" */
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

export interface ClassifyInput {
  file: Blob;
  kind: ImageKind;
  /** decoded pixels; the engine takes ownership and closes the bitmap */
  photo: DecodedPhoto;
}

export interface InferenceEngine {
  readonly kind: EngineKind;
  init(onProgress: ProgressCallback): Promise<EngineInfo>;
  classify(input: ClassifyInput): Promise<Prediction>;
  /** live tensor count (browser engine only), for leak checks */
  memory?(): Promise<{ numTensors: number; numBytes: number }>;
  dispose(): void;
}

export class EngineError extends Error {}
