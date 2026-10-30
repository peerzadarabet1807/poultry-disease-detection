/** TensorFlow.js in the browser (Web Worker when possible): the photo never leaves the device. */
import { CLASS_IDS } from '../lib/classes';
import { extractClassMap, reluMaxNormalize } from '../lib/cam';
import { ClassifierClient } from '../worker/client';
import type { BackendName } from '../worker/types';
import { validatePrediction } from './validate';
import type { EngineInfo, Prediction, ProgressCallback } from './types';

const BACKEND_LABEL: Record<BackendName, string> = {
  webgl: 'WebGL (GPU)',
  wasm: 'WebAssembly (CPU)',
  cpu: 'plain JavaScript (slow)',
};

const seconds = (ms: number) => `${(ms / 1000).toFixed(1)} s`;

export class BrowserEngine {
  private client = new ClassifierClient();

  async init(onProgress: ProgressCallback): Promise<EngineInfo> {
    const info = await this.client.init(onProgress);
    const where = info.runsIn === 'main-thread' ? ' on the page thread' : '';
    return {
      backend: BACKEND_LABEL[info.backend],
      detail: `${BACKEND_LABEL[info.backend]}${where} · ${info.fromCache ? 'loaded from browser cache' : 'downloaded'} in ${seconds(info.loadMs)}`,
      loadMs: info.loadMs,
    };
  }

  /** Classify a decoded photo; the engine takes ownership of the bitmap and closes it. */
  async classify(bitmap: ImageBitmap): Promise<Prediction> {
    const raw = await this.client.classify(bitmap);
    return validatePrediction({
      probabilities: raw.probabilities,
      camMaps: CLASS_IDS.map((_, i) => reluMaxNormalize(extractClassMap(raw.cam, i, CLASS_IDS.length))),
      seen: raw.seen,
      exactPreprocessing: raw.exact,
      inferenceMs: raw.inferenceMs,
    });
  }

  memory() {
    return this.client.memory();
  }

  dispose() {
    this.client.terminate();
  }
}
