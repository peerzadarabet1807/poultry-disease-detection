/**
 * TensorFlow's antialiased bilinear resize, ported bit for bit:
 * `tf.image.resize(float32(img), (224, 224), method="bilinear", antialias=True)`, then round half to
 * even, clip and cast to uint8. That is the training preprocessing, so the browser feeds the model exactly
 * the pixels the Python package does (tests/resize_reference.py documents the same algorithm and the
 * golden tests compare against TensorFlow's output).
 *
 * The kernel is TF's ScaleAndTranslate with a triangle filter: for each output index a span of source
 * pixels with normalised weights; rows are resampled first into an intermediate of out_h x in_w, then
 * columns. All arithmetic is float32. `Math.fround(a op b)` on float32 operands equals the float32
 * operation (double has more than 2 x 24 + 2 mantissa bits), so wrapping every step in `fround` gives
 * TF's exact results.
 */

export const MODEL_SIZE = 224;

const f = Math.fround;
const EPS32 = 2 ** -23;

export interface Spans {
  starts: Int32Array;
  /** row-major [outSize, spanSize] */
  weights: Float32Array;
  spanSize: number;
}

export function computeSpans(outSize: number, inSize: number): Spans {
  const scale = f(outSize / inSize);
  const invScale = f(1 / scale);
  const kernelScale = Math.max(invScale, 1);
  const radius = kernelScale; // triangle kernel radius 1, scaled
  const spanSize = Math.min(2 * Math.ceil(radius) + 1, inSize);
  const starts = new Int32Array(outSize);
  const weights = new Float32Array(outSize * spanSize);
  const oneOverKernelScale = f(1 / kernelScale);
  const ws = new Float32Array(spanSize);

  for (let x = 0; x < outSize; x++) {
    const sample = f(f(x + 0.5) * invScale);
    if (sample < 0 || sample > inSize) continue;
    let start = Math.ceil(f(f(sample - radius) - 0.5));
    let end = Math.floor(f(f(sample + radius) - 0.5));
    start = Math.min(Math.max(start, 0), inSize - 1);
    end = Math.min(Math.max(end, 0), inSize - 1) + 1;
    let total = 0;
    let n = 0;
    for (let src = start; src < end; src++) {
      const pos = f(f(src + 0.5) - sample);
      const w = Math.max(0, f(1 - Math.abs(f(pos * oneOverKernelScale))));
      ws[n++] = w;
      total = f(total + w);
    }
    if (Math.abs(total) >= 1000 * EPS32) {
      const invTotal = f(1 / total);
      for (let i = 0; i < n; i++) ws[i] = f(ws[i] * invTotal);
    }
    starts[x] = start;
    weights.set(ws.subarray(0, n), x * spanSize);
    ws.fill(0);
  }
  return { starts, weights, spanSize };
}

/** Round half to even (tf.round / np.round), for non-negative values. */
function roundHalfEven(v: number): number {
  const r = Math.round(v);
  return r - v === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/**
 * Resize interleaved pixels to `outW x outH` RGB bytes, exactly as the training preprocessing.
 * @param channels channels per source pixel (4 for canvas ImageData, 3 for RGB); only R, G, B are used
 */
export function resizeToRGB(
  src: ArrayLike<number>,
  width: number,
  height: number,
  channels = 4,
  outW = MODEL_SIZE,
  outH = MODEL_SIZE,
): Uint8Array {
  if (width < 1 || height < 1) throw new Error(`empty image (${width} x ${height})`);
  if (src.length < width * height * channels) throw new Error('pixel buffer is smaller than width x height');

  // Pass 1, rows: tmp[oy, x, c] = sum_i w[oy, i] * src[start + i, x, c]
  const rows = computeSpans(outH, height);
  const tmp = new Float32Array(outH * width * 3);
  for (let oy = 0; oy < outH; oy++) {
    const start = rows.starts[oy];
    const taps = Math.min(start + rows.spanSize, height) - start;
    const out = oy * width * 3;
    for (let i = 0; i < taps; i++) {
      const w = rows.weights[oy * rows.spanSize + i];
      const line = (start + i) * width * channels;
      for (let x = 0; x < width; x++) {
        const s = line + x * channels;
        const d = out + x * 3;
        tmp[d] = f(tmp[d] + f(w * src[s]));
        tmp[d + 1] = f(tmp[d + 1] + f(w * src[s + 1]));
        tmp[d + 2] = f(tmp[d + 2] + f(w * src[s + 2]));
      }
    }
  }

  // Pass 2, columns: out[oy, ox, c] = sum_i w[ox, i] * tmp[oy, start + i, c]
  const cols = computeSpans(outW, width);
  const result = new Uint8Array(outW * outH * 3);
  for (let oy = 0; oy < outH; oy++) {
    const line = oy * width * 3;
    for (let ox = 0; ox < outW; ox++) {
      const start = cols.starts[ox];
      const taps = Math.min(start + cols.spanSize, width) - start;
      let r = 0;
      let g = 0;
      let b = 0;
      for (let i = 0; i < taps; i++) {
        const w = cols.weights[ox * cols.spanSize + i];
        const s = line + (start + i) * 3;
        r = f(r + f(w * tmp[s]));
        g = f(g + f(w * tmp[s + 1]));
        b = f(b + f(w * tmp[s + 2]));
      }
      const d = (oy * outW + ox) * 3;
      result[d] = Math.min(255, Math.max(0, roundHalfEven(r)));
      result[d + 1] = Math.min(255, Math.max(0, roundHalfEven(g)));
      result[d + 2] = Math.min(255, Math.max(0, roundHalfEven(b)));
    }
  }
  return result;
}
