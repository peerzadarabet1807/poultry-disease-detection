/**
 * Class activation map (CAM) post-processing.
 *
 * The model's `cam` output is [1, 7, 7, 4]: for every class c, sum_k w_k^c A^k over the final ResNet50
 * feature maps. For the GAP -> Dense head this is the notebook's Grad-CAM up to a positive constant
 * (1 / 49), so after ReLU and division by the maximum it is exactly the notebook's heatmap
 * (src/poultry_disease/model.py has the derivation).
 */

export const CAM_SIZE = 7;

/** Extract one class channel from a flat NHWC [1, H, W, C] tensor (row-major H x W result). */
export function extractClassMap(
  cam: ArrayLike<number>,
  classIndex: number,
  numClasses: number,
  size = CAM_SIZE,
): Float32Array {
  const out = new Float32Array(size * size);
  for (let i = 0; i < size * size; i++) out[i] = cam[i * numClasses + classIndex];
  return out;
}

/** ReLU, then divide by the maximum (the notebook's normalisation). No positive evidence: all zero. */
export function reluMaxNormalize(values: ArrayLike<number>): Float32Array {
  const out = new Float32Array(values.length);
  let peak = 0;
  for (let i = 0; i < values.length; i++) {
    const v = Math.max(0, values[i]);
    out[i] = v;
    if (v > peak) peak = v;
  }
  if (!(peak > 0)) return out.fill(0);
  for (let i = 0; i < out.length; i++) out[i] /= peak;
  return out;
}

/**
 * Bilinear upsampling with half-pixel centres and clamped edges, the convention of the notebook's
 * `tf.image.resize(cam, (224, 224))`. Interpolating the scalar map before colour-mapping avoids banding.
 */
export function bilinearUpsample(
  src: ArrayLike<number>,
  sw: number,
  sh: number,
  dw: number,
  dh: number,
): Float32Array {
  const out = new Float32Array(dw * dh);
  const sx = sw / dw;
  const sy = sh / dh;
  for (let y = 0; y < dh; y++) {
    const fy = Math.min(Math.max((y + 0.5) * sy - 0.5, 0), sh - 1);
    const y0 = Math.floor(fy);
    const y1 = Math.min(y0 + 1, sh - 1);
    const wy = fy - y0;
    for (let x = 0; x < dw; x++) {
      const fx = Math.min(Math.max((x + 0.5) * sx - 0.5, 0), sw - 1);
      const x0 = Math.floor(fx);
      const x1 = Math.min(x0 + 1, sw - 1);
      const wx = fx - x0;
      const top = src[y0 * sw + x0] * (1 - wx) + src[y0 * sw + x1] * wx;
      const bottom = src[y1 * sw + x0] * (1 - wx) + src[y1 * sw + x1] * wx;
      out[y * dw + x] = top * (1 - wy) + bottom * wy;
    }
  }
  return out;
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Classic "jet" colormap, v in [0, 1] -> [r, g, b]. Same formula as src/poultry_disease/cam.py. */
export function jet(v: number): [number, number, number] {
  const t = clamp01(v);
  return [
    Math.round(255 * clamp01(1.5 - Math.abs(4 * t - 3))),
    Math.round(255 * clamp01(1.5 - Math.abs(4 * t - 2))),
    Math.round(255 * clamp01(1.5 - Math.abs(4 * t - 1))),
  ];
}

/** Opacity of a heatmap pixel: proportional to activation, so the photo stays readable where the map is weak. */
export const heatmapAlpha = (v: number) => clamp01(0.04 + 0.96 * clamp01(v));

/** Colour-map a [0, 1] field into RGBA pixels (alpha weighted by activation, see heatmapAlpha). */
export function heatmapRGBA(field: ArrayLike<number>): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(field.length * 4);
  for (let i = 0; i < field.length; i++) {
    const [r, g, b] = jet(field[i]);
    out[i * 4] = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = Math.round(255 * heatmapAlpha(field[i]));
  }
  return out;
}

export function pearson(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = a.length;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma;
    const db = b[i] - mb;
    cov += da * db;
    va += da * da;
    vb += db * db;
  }
  if (va === 0 || vb === 0) {
    let same = true;
    for (let i = 0; i < n && same; i++) same = Math.abs(a[i] - b[i]) < 1e-6;
    return same ? 1 : 0;
  }
  return cov / Math.sqrt(va * vb);
}
