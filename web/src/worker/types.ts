import type { InputCheckResult } from '../lib/inputCheck';

export type BackendName = 'webgl' | 'wasm' | 'cpu';

/** Where TensorFlow.js runs: a Web Worker, or the page itself when workers cannot use WebGL. */
export type RunsIn = 'worker' | 'main-thread';

export interface WorkerLoadProgress {
  phase: 'download' | 'warmup';
  /** 0..1 */
  fraction: number;
}

export interface ModelInfo {
  backend: BackendName;
  /** both models (classifier and input check) loaded from IndexedDB instead of the network */
  fromCache: boolean;
  /** backend setup + download (or cache read) + graph construction of both models, ms */
  loadMs: number;
  /** first inference of both models (shader compilation), ms */
  warmupMs: number;
  runsIn: RunsIn;
}

export interface ClassifyOptions {
  /** run the classifier even when the input check fails ("Analyse anyway") */
  force: boolean;
}

export interface WorkerPrediction {
  /** softmax, model output order */
  probabilities: number[];
  /** flat [7, 7, 4] class activation maps (pre-ReLU) */
  cam: Float32Array;
  inferenceMs: number;
}

export interface WorkerResult {
  /** always computed, before the classifier */
  inputCheck: InputCheckResult;
  inputCheckMs: number;
  /** null when the input check failed and the classifier was not run */
  prediction: WorkerPrediction | null;
  /** the 224 x 224 RGBA image both models actually saw */
  seen: Uint8ClampedArray;
  /** false when the photo was above the canvas limit and pre-shrunk before the resize */
  exact: boolean;
  preprocessMs: number;
  backend: BackendName;
}

export interface ClassifierApi {
  init(
    urls: { model: string; gate: string },
    onProgress: (p: WorkerLoadProgress) => void,
    preferredBackend?: BackendName,
  ): Promise<ModelInfo>;
  /** takes ownership of (and closes) the bitmap */
  classify(bitmap: ImageBitmap, options: ClassifyOptions): Promise<WorkerResult>;
  /** fallback when the worker has no OffscreenCanvas: RGBA pixels extracted on the main thread */
  classifyPixels(
    data: Uint8ClampedArray,
    width: number,
    height: number,
    exact: boolean,
    options: ClassifyOptions,
  ): Promise<WorkerResult>;
  memory(): Promise<{ numTensors: number; numBytes: number }>;
}
