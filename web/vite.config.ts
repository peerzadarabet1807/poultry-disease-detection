/// <reference types="vitest/config" />
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const MODEL_DIR = join(__dirname, 'public', 'model');
const modelFiles = () => (existsSync(MODEL_DIR) ? readdirSync(MODEL_DIR).sort() : []);

/** Content hash of the shipped TF.js model: the IndexedDB cache key, so a new model invalidates old caches. */
function modelVersion(): string {
  const hash = createHash('sha256');
  for (const file of modelFiles()) hash.update(file).update(readFileSync(join(MODEL_DIR, file)));
  return hash.digest('hex').slice(0, 12);
}

function modelBytes(): number {
  return modelFiles().reduce((n, f) => n + statSync(join(MODEL_DIR, f)).size, 0);
}

export default defineConfig({
  base: '/poultry-disease-detection/',
  plugins: [react(), tailwindcss()],
  define: {
    __MODEL_VERSION__: JSON.stringify(modelVersion()),
    __MODEL_BYTES__: JSON.stringify(modelBytes()),
  },
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    // tfjs lives in the worker chunk; the main bundle stays small.
    chunkSizeWarningLimit: 1600,
  },
  preview: { port: 4173, strictPort: true },
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['tests/unit/**/*.test.{ts,tsx}'],
    setupFiles: ['tests/setup.ts'],
    css: false,
  },
});
