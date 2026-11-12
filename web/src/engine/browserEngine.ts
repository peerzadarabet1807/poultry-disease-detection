/** TensorFlow.js in the browser (Web Worker when possible): the photo never leaves the device. */
import { CLASS_IDS } from '../lib/classes';
import { extractClassMap, reluMaxNormalize } from '../lib/cam';
import { ClassifierClient } from '../worker/client';
import type { BackendName } from '../worker/types';
import { validateOutput } from './validate';
import type { EngineInfo, EngineOutput, ProgressCallback } from './types';

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

  /**
   * Check, then classify, a decoded photo; the engine takes ownership of the bitmap and closes it. The
   * classifier runs only when the input check passes, or when `force` is set ("Analyse anyway").
   */
  async classify(bitmap: ImageBitmap, force = false): Promise<EngineOutput> {
    const raw = await this.client.classify(bitmap, { force });
    const p = raw.prediction;
    return validateOutput({
      inputCheck: raw.inputCheck,
      inputCheckMs: raw.inputCheckMs,
      seen: raw.seen,
      prediction: p && {
        probabilities: p.probabilities,
        camMaps: CLASS_IDS.map((_, i) => reluMaxNormalize(extractClassMap(p.cam, i, CLASS_IDS.length))),
        seen: raw.seen,
        exactPreprocessing: raw.exact,
        inferenceMs: p.inferenceMs,
      },
    });
  }

  memory() {
    return this.client.memory();
  }

  dispose() {
    this.client.terminate();
  }
}
