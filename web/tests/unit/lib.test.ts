import {
  bilinearUpsample,
  extractClassMap,
  heatmapRGBA,
  jet,
  pearson,
  reluMaxNormalize,
} from '../../src/lib/cam';
import { readJpegInfo, swapsAxes } from '../../src/lib/exif';
import { formatBytes, MAX_FILE_BYTES, sniffImageKind, validateFile } from '../../src/lib/fileValidation';
import { EXTERNAL, INTERNAL, RUNS } from '../../src/lib/metrics';
import { sourceCanvasSize } from '../../src/lib/pixels';

const bytes = (...b: number[]) => new Uint8Array(b);
const ascii = (s: string) => Uint8Array.from(s, (c) => c.charCodeAt(0));

describe('file validation', () => {
  it('sniffs formats from magic bytes', () => {
    expect(sniffImageKind(bytes(0xff, 0xd8, 0xff, 0xe0))).toBe('jpeg');
    expect(sniffImageKind(ascii('\x89PNG\r\n\x1a\n'))).toBe('png');
    expect(sniffImageKind(ascii('\0\0\0\x18ftypheic'))).toBe('heic');
    expect(sniffImageKind(ascii('RIFF\0\0\0\0WEBP'))).toBeNull();
    expect(sniffImageKind(ascii('GIF89a'))).toBeNull();
  });

  it('rejects empty, oversized and non-image files with a reason', async () => {
    expect(await validateFile(new File([], 'a.jpg'))).toMatchObject({ ok: false, reason: 'empty' });
    const big = new File([new Uint8Array(MAX_FILE_BYTES + 1)], 'big.jpg');
    expect(await validateFile(big)).toMatchObject({ ok: false, reason: 'too-large' });
    const gif = new File([ascii('GIF89a......')], 'x.gif', { type: 'image/gif' });
    expect(await validateFile(gif)).toMatchObject({ ok: false, reason: 'unsupported-image' });
    const txt = new File([ascii('hello world!')], 'x.txt', { type: 'text/plain' });
    expect(await validateFile(txt)).toMatchObject({ ok: false, reason: 'not-an-image' });
    const jpg = new File([bytes(0xff, 0xd8, 0xff, 0xdb, 0, 0)], 'x.bin', { type: '' });
    expect(await validateFile(jpg)).toEqual({ ok: true, kind: 'jpeg' });
  });

  it('formats sizes in decimal units', () => {
    expect(formatBytes(MAX_FILE_BYTES)).toBe('25 MB');
    expect(formatBytes(47_189_000)).toBe('47.2 MB');
    expect(formatBytes(312_000)).toBe('312 KB');
  });
});

describe('EXIF orientation', () => {
  it('reads the orientation and frame size of a JPEG', () => {
    // SOI, APP1 Exif (big-endian TIFF, one IFD entry: orientation 6), SOF0 100 x 50, SOS
    const exif = [
      0x45, 0x78, 0x69, 0x66, 0, 0, 0x4d, 0x4d, 0, 0x2a, 0, 0, 0, 8, 0, 1, 0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, 6,
      0, 0, 0, 0, 0, 0,
    ];
    const jpeg = bytes(
      0xff, 0xd8,
      0xff, 0xe1, 0, exif.length + 2, ...exif,
      0xff, 0xc0, 0, 11, 8, 0, 50, 0, 100, 3, 0, 0, 0,
      0xff, 0xda,
    ); // prettier-ignore
    expect(readJpegInfo(jpeg.buffer)).toEqual({ orientation: 6, width: 100, height: 50 });
    expect(swapsAxes(6)).toBe(true);
    expect(swapsAxes(3)).toBe(false);
    expect(readJpegInfo(ascii('not a jpeg').buffer)).toBeNull();
  });
});

describe('CAM helpers', () => {
  it('extracts one class from the NHWC map', () => {
    const flat = Float32Array.from({ length: 49 * 4 }, (_, i) => i);
    const m = extractClassMap(flat, 2, 4);
    expect(m[0]).toBe(2);
    expect(m[1]).toBe(6);
  });

  it('applies ReLU and divides by the max (the notebook normalisation)', () => {
    expect(Array.from(reluMaxNormalize([-1, 0, 2, 4]))).toEqual([0, 0, 0.5, 1]);
    expect(Array.from(reluMaxNormalize([-3, -1]))).toEqual([0, 0]);
  });

  it('upsamples with clamped edges and keeps constants constant', () => {
    const up = bilinearUpsample(new Float32Array(49).fill(0.3), 7, 7, 224, 224);
    expect(up.every((v) => Math.abs(v - 0.3) < 1e-6)).toBe(true);
  });

  it('uses the same jet formula as the Python package', () => {
    expect(jet(0)).toEqual([0, 0, 128]);
    expect(jet(0.5)).toEqual([128, 255, 128]);
    expect(jet(1)).toEqual([128, 0, 0]);
    expect(heatmapRGBA([0, 1])).toHaveLength(8);
  });

  it('computes Pearson correlation', () => {
    expect(pearson([1, 2, 3], [2, 4, 6])).toBeCloseTo(1);
    expect(pearson([1, 2, 3], [3, 2, 1])).toBeCloseTo(-1);
  });
});

describe('canvas size', () => {
  it('keeps photos at native size up to the canvas limit, and flags the rest as inexact', () => {
    expect(sourceCanvasSize(3120, 4160)).toEqual({ width: 3120, height: 4160, exact: true });
    const big = sourceCanvasSize(8000, 6000);
    expect(big.exact).toBe(false);
    expect(big.width * big.height).toBeLessThanOrEqual(16_777_216);
  });
});

describe('metrics come from the committed results', () => {
  it('shows the internal/external gap', () => {
    expect(INTERNAL.n).toBe(974);
    expect(EXTERNAL.n).toBe(1201);
    expect(INTERNAL.accuracy.value).toBeGreaterThan(EXTERNAL.accuracy.value);
    expect(INTERNAL.accuracy.ci95[0]).toBeLessThan(INTERNAL.accuracy.value);
    expect(RUNS.count).toBe(5);
  });
});
