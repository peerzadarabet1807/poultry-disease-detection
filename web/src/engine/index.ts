import type { InferenceEngine } from './types';

export type InferenceMode = 'browser' | 'api';

export function inferenceMode(value = import.meta.env.VITE_INFERENCE_MODE): InferenceMode {
  return value === 'api' ? 'api' : 'browser';
}

/** The API engine is code-split, so the default browser build never downloads it (and vice versa). */
export async function createEngine(mode = inferenceMode()): Promise<InferenceEngine> {
  if (mode === 'api') {
    const { ApiEngine } = await import('./apiEngine');
    return new ApiEngine(import.meta.env.VITE_API_URL ?? '');
  }
  const { BrowserEngine } = await import('./browserEngine');
  return new BrowserEngine();
}

export type { ClassifyInput, EngineInfo, InferenceEngine, LoadProgress, Prediction } from './types';
export { EngineError } from './types';
