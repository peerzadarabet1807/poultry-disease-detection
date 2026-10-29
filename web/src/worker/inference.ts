/**
 * Backend-agnostic model execution, shared by the Web Worker and the Node parity test.
 *
 * Contract of the exported graph (scripts/build_web_model.py):
 *   input  "image"  float32 [N, 224, 224, 3], ResNet50 "caffe" input (see lib/preprocess.ts)
 *   output "probs"  float32 [N, 4] softmax, order = models/class_names.json
 *   output "cam"    float32 [N, 7, 7, 4] class activation maps (pre-ReLU)
 */
import * as tf from '@tensorflow/tfjs-core';
import type { GraphModel } from '@tensorflow/tfjs-converter';
import { MODEL_SIZE } from '../lib/resize';

export const INPUT_NAME = 'image';
export const OUTPUT_NAMES = ['probs', 'cam'] as const;

export interface RawPrediction {
  probabilities: Float32Array;
  /** flat [7, 7, numClasses] (NHWC without the batch axis) */
  cam: Float32Array;
}

/** Run one preprocessed 224 x 224 x 3 input through the model. Frees every tensor it creates. */
export async function runModel(model: GraphModel, input: Float32Array): Promise<RawPrediction> {
  if (input.length !== MODEL_SIZE * MODEL_SIZE * 3) {
    throw new Error(`expected ${MODEL_SIZE}x${MODEL_SIZE}x3 values, got ${input.length}`);
  }
  const [probT, camT] = tf.tidy(() => {
    const x = tf.tensor4d(input, [1, MODEL_SIZE, MODEL_SIZE, 3], 'float32');
    return model.execute({ [INPUT_NAME]: x }, [...OUTPUT_NAMES]) as tf.Tensor[];
  });
  try {
    const [probabilities, cam] = await Promise.all([probT.data(), camT.data()]);
    // copies: some backends hand out their internal buffers, and the worker transfers these to the UI
    return { probabilities: Float32Array.from(probabilities), cam: Float32Array.from(cam) };
  } finally {
    tf.dispose([probT, camT]);
  }
}
