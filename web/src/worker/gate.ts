/**
 * The input check's image encoder, shared by the Web Worker and the Node parity test.
 *
 * Contract of the exported graph (scripts/build_gate_model.py, TinyCLIP ViT-8M/16 vision tower):
 *   input  "image"      float32 [N, 224, 224, 3], the RGB image the classifier sees, 0..255 (normalised inside)
 *   output "embedding"  float32 [N, 512], unit length
 */
import * as tf from '@tensorflow/tfjs-core';
import type { GraphModel } from '@tensorflow/tfjs-converter';
import { MODEL_SIZE } from '../lib/resize';

export const GATE_INPUT = 'image';
export const GATE_OUTPUT = 'embedding';

/** Embed one 224 x 224 RGB image (bytes). Frees every tensor it creates. */
export async function runGate(model: GraphModel, seenRgb: Uint8Array): Promise<Float32Array> {
  if (seenRgb.length !== MODEL_SIZE * MODEL_SIZE * 3) {
    throw new Error(`expected ${MODEL_SIZE}x${MODEL_SIZE}x3 bytes, got ${seenRgb.length}`);
  }
  const out = tf.tidy(() => {
    const x = tf.tensor4d(Float32Array.from(seenRgb), [1, MODEL_SIZE, MODEL_SIZE, 3], 'float32');
    return model.execute({ [GATE_INPUT]: x }, GATE_OUTPUT) as tf.Tensor;
  });
  try {
    return Float32Array.from(await out.data());
  } finally {
    out.dispose();
  }
}
