/**
 * Golden tests: the TypeScript resize must be bit-identical to TensorFlow's
 * tf.image.resize(bilinear, antialias=True) -> round -> clip -> uint8 (scripts/make_web_references.py).
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { computeSpans, resizeToRGB } from '../../src/lib/resize';
import { CAFFE_MEANS_BGR, toModelInput } from '../../src/lib/preprocess';
import { readNpy } from '../utils/npy';

const GOLDEN = join(__dirname, '..', 'fixtures', 'golden');
const doc = JSON.parse(readFileSync(join(GOLDEN, 'golden.json'), 'utf8')) as {
  images: { height: number; width: number; file: string }[];
};

/** Same formula as scripts/make_web_references.py, as RGBA (what a canvas returns). */
function synthetic(height: number, width: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(height * width * 4);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      for (let c = 0; c < 4; c++)
        out[(y * width + x) * 4 + c] = c === 3 ? 255 : (x * 7 + y * 13 + c * 101 + ((x * y) % 251)) % 256;
  return out;
}

describe('resizeToRGB matches TensorFlow bit for bit', () => {
  for (const { height, width, file } of doc.images) {
    it(`${width} x ${height}`, () => {
      const expected = readNpy(readFileSync(join(GOLDEN, file))).data as Uint8Array;
      const actual = resizeToRGB(synthetic(height, width), width, height, 4);
      let mismatches = 0;
      for (let i = 0; i < expected.length; i++) if (actual[i] !== expected[i]) mismatches++;
      expect(mismatches).toBe(0);
    });
  }

  it('accepts RGB input as well as RGBA', () => {
    const rgba = synthetic(300, 200);
    const rgb = new Uint8Array(300 * 200 * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) rgb.set(rgba.subarray(i, i + 3), j);
    expect(resizeToRGB(rgb, 200, 300, 3)).toEqual(resizeToRGB(rgba, 200, 300, 4));
  });

  it('rejects empty or short buffers', () => {
    expect(() => resizeToRGB(new Uint8Array(0), 0, 0)).toThrow();
    expect(() => resizeToRGB(new Uint8Array(10), 10, 10)).toThrow();
  });
});

describe('computeSpans', () => {
  it('normalises weights to one when downsampling', () => {
    const { weights, spanSize } = computeSpans(224, 3264);
    for (let x = 0; x < 224; x++) {
      const sum = weights.subarray(x * spanSize, (x + 1) * spanSize).reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1, 5);
    }
  });

  it('is the identity at the same size', () => {
    const { starts, weights, spanSize } = computeSpans(224, 224);
    expect(spanSize).toBe(3);
    for (let x = 0; x < 224; x++) {
      const row = Array.from(weights.subarray(x * spanSize, (x + 1) * spanSize));
      expect(row[x - starts[x]]).toBe(1);
    }
  });
});

describe('toModelInput (caffe)', () => {
  it('swaps to BGR and subtracts the ImageNet means in float32', () => {
    const seen = new Uint8Array(224 * 224 * 3);
    seen.set([10, 20, 30]);
    const x = toModelInput(seen);
    expect(x[0]).toBe(Math.fround(30 - Math.fround(CAFFE_MEANS_BGR[0])));
    expect(x[1]).toBe(Math.fround(20 - Math.fround(CAFFE_MEANS_BGR[1])));
    expect(x[2]).toBe(Math.fround(10 - Math.fround(CAFFE_MEANS_BGR[2])));
    expect(x[3]).toBe(Math.fround(-CAFFE_MEANS_BGR[0]));
  });
});
