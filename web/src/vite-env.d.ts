/// <reference types="vite/client" />

declare const __MODEL_VERSION__: string;
declare const __MODEL_BYTES__: number;

interface ImportMetaEnv {
  /** "browser" (default, TensorFlow.js) or "api" (the FastAPI backend at VITE_API_URL) */
  readonly VITE_INFERENCE_MODE?: string;
  readonly VITE_API_URL?: string;
}
