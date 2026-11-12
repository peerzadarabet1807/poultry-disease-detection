// Dev server for the input-check bench page only (not part of the app build). Models are served from the
// repository's build/ folder: build/gate/<name>/tfjs/<variant>/model.json and build/gate/<name>/onnx/*.onnx.
import { join } from 'node:path';
import { defineConfig } from 'vite';

export default defineConfig({
  root: __dirname,
  publicDir: join(__dirname, '..', '..', 'build'),
  server: {
    port: 4180,
    strictPort: true,
    fs: { allow: [join(__dirname, '..')] },
    // No cross-origin isolation headers: GitHub Pages cannot send them, so the app gets no WASM threads
    // either. The bench measures what users get.
  },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
