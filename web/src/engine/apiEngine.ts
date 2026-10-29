/**
 * The FastAPI backend (backend/poultry_api). The photo is uploaded; the server runs the same preprocessing
 * and model as the Python package and returns probabilities, class maps and the image the model saw.
 */
import { CLASS_IDS } from '../lib/classes';
import { MODEL_SIZE } from '../lib/resize';
import { parseApiResponse, validatePrediction } from './validate';
import {
  EngineError,
  type ClassifyInput,
  type EngineInfo,
  type InferenceEngine,
  type Prediction,
  type ProgressCallback,
} from './types';

const TIMEOUT_MS = 60_000;

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { error?: { message?: string } };
    if (body.error?.message) return body.error.message;
  } catch {
    /* not the API's error shape */
  }
  return `The server answered with HTTP ${res.status}.`;
}

async function pngToRGBA(base64: string): Promise<Uint8ClampedArray> {
  const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
  const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/png' }), {
    colorSpaceConversion: 'none',
    premultiplyAlpha: 'none',
  });
  const canvas = document.createElement('canvas');
  canvas.width = MODEL_SIZE;
  canvas.height = MODEL_SIZE;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('2D canvas is not available');
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
}

export class ApiEngine implements InferenceEngine {
  readonly kind = 'api' as const;

  constructor(private readonly baseUrl: string) {
    if (!baseUrl) throw new EngineError('VITE_API_URL is not set, so the app cannot reach the API.');
  }

  private url(path: string) {
    return `${this.baseUrl.replace(/\/$/, '')}/api/v1${path}`;
  }

  async init(onProgress: ProgressCallback): Promise<EngineInfo> {
    const t0 = performance.now();
    onProgress({ phase: 'connect', fraction: 0 });
    let res: Response;
    try {
      res = await fetch(this.url('/health'), { signal: AbortSignal.timeout(TIMEOUT_MS) });
    } catch {
      throw new EngineError(`The API at ${this.baseUrl} is not reachable.`);
    }
    if (!res.ok) throw new EngineError(await errorMessage(res));
    onProgress({ phase: 'connect', fraction: 1 });
    const loadMs = performance.now() - t0;
    return {
      kind: 'api',
      backend: 'API',
      detail: `Server inference at ${new URL(this.baseUrl).host}`,
      loadMs,
    };
  }

  async classify({ file, photo }: ClassifyInput): Promise<Prediction> {
    photo.bitmap.close(); // the server decodes the original file itself
    const form = new FormData();
    form.append('file', file, 'photo');
    const t0 = performance.now();
    let res: Response;
    try {
      res = await fetch(this.url('/predict?cam=true'), {
        method: 'POST',
        body: form,
        signal: AbortSignal.timeout(TIMEOUT_MS),
      });
    } catch {
      throw new EngineError('The upload to the API failed. Check your connection and try again.');
    }
    if (!res.ok) throw new EngineError(await errorMessage(res));
    const body = parseApiResponse(await res.json());
    const inferenceMs = performance.now() - t0;
    return validatePrediction({
      probabilities: CLASS_IDS.map((c) => body.probabilities[c]),
      camMaps: CLASS_IDS.map((c) => Float32Array.from(body.cam.maps[c].flat())),
      seen: await pngToRGBA(body.cam.input_png_base64),
      exactPreprocessing: true,
      inferenceMs,
    });
  }

  dispose() {}
}
