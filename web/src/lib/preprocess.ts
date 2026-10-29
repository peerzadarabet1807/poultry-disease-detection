/**
 * The preprocessing contract, identical to training and to src/poultry_disease/preprocess.py:
 *
 * 1. Decode to RGB without EXIF rotation and without colour management (done by the caller, see decode.ts).
 * 2. Antialiased bilinear resize of the full frame to 224 x 224 (aspect ratio not kept), round half to
 *    even, clip, uint8 (resize.ts).
 * 3. ResNet50 "caffe" input: RGB -> BGR, subtract the ImageNet channel means, float32 (below).
 */
import { MODEL_SIZE, resizeToRGB } from './resize';

/** keras.applications.resnet50.preprocess_input means, in BGR order. */
export const CAFFE_MEANS_BGR = [103.939, 116.779, 123.68] as const;

const MEANS = CAFFE_MEANS_BGR.map(Math.fround);

/** Decoded pixels (RGBA or RGB) -> the 224 x 224 RGB bytes the model sees. */
export function toSeenImage(
  pixels: ArrayLike<number>,
  width: number,
  height: number,
  channels = 4,
): Uint8Array {
  return resizeToRGB(pixels, width, height, channels, MODEL_SIZE, MODEL_SIZE);
}

/** 224 x 224 RGB bytes -> float32 model input (BGR, means subtracted), as numpy does it in float32. */
export function toModelInput(seen: Uint8Array): Float32Array {
  if (seen.length !== MODEL_SIZE * MODEL_SIZE * 3) {
    throw new Error(`expected ${MODEL_SIZE}x${MODEL_SIZE}x3 bytes, got ${seen.length}`);
  }
  const out = new Float32Array(seen.length);
  for (let i = 0; i < seen.length; i += 3) {
    out[i] = Math.fround(seen[i + 2] - MEANS[0]);
    out[i + 1] = Math.fround(seen[i + 1] - MEANS[1]);
    out[i + 2] = Math.fround(seen[i] - MEANS[2]);
  }
  return out;
}

/** 224 x 224 RGB bytes -> RGBA for display ("what the model sees"). */
export function rgbToRGBA(rgb: Uint8Array): Uint8ClampedArray {
  const out = new Uint8ClampedArray((rgb.length / 3) * 4);
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    out[j] = rgb[i];
    out[j + 1] = rgb[i + 1];
    out[j + 2] = rgb[i + 2];
    out[j + 3] = 255;
  }
  return out;
}
