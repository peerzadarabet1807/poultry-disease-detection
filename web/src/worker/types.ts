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
  /** loaded from IndexedDB instead of the network */
  fromCache: boolean;
  /** backend setup + download (or cache read) + graph construction, ms */
  loadMs: number;
  /** first inference (shader compilation), ms */
  warmupMs: number;
  runsIn: RunsIn;
}

export interface WorkerPrediction {
  /** softmax, model output order */
  probabilities: number[];
  /** flat [7, 7, 4] class activation maps (pre-ReLU) */
  cam: Float32Array;
  /** the 224 x 224 RGBA image the model actually saw */
  seen: Uint8ClampedArray;
  /** false when the photo was above the canvas limit and pre-shrunk before the resize */
  exact: boolean;
  preprocessMs: number;
  inferenceMs: number;
  backend: BackendName;
}

export interface ClassifierApi {
  init(
    modelUrl: string,
    onProgress: (p: WorkerLoadProgress) => void,
    preferredBackend?: BackendName,
  ): Promise<ModelInfo>;
  /** takes ownership of (and closes) the bitmap */
  classify(bitmap: ImageBitmap): Promise<WorkerPrediction>;
  /** fallback when the worker has no OffscreenCanvas: RGBA pixels extracted on the main thread */
  classifyPixels(
    data: Uint8ClampedArray,
    width: number,
    height: number,
    exact: boolean,
  ): Promise<WorkerPrediction>;
  memory(): Promise<{ numTensors: number; numBytes: number }>;
}
