/**
 * ImageBitmap -> decoded RGBA pixels at native resolution. Runs in the worker (OffscreenCanvas) or, as a
 * fallback, on the main thread. The antialiased resize (resize.ts) then needs every source pixel.
 */

/** Mobile Safari refuses canvases above 16.7 M pixels; larger photos are pre-shrunk on the canvas. */
export const MAX_CANVAS_PIXELS = 16_777_216;

type AnyCanvas = OffscreenCanvas | HTMLCanvasElement;
type Any2D = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;

export function createCanvas(width: number, height: number): AnyCanvas {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

/**
 * Size of the canvas the bitmap is drawn onto: native size, unless the photo is above the canvas limit.
 * Only then is the input not exactly the training preprocessing (the canvas pre-shrinks it first).
 */
export function sourceCanvasSize(
  width: number,
  height: number,
): { width: number; height: number; exact: boolean } {
  if (width * height <= MAX_CANVAS_PIXELS) return { width, height, exact: true };
  const scale = Math.sqrt(MAX_CANVAS_PIXELS / (width * height));
  return { width: Math.floor(width * scale), height: Math.floor(height * scale), exact: false };
}

export function bitmapToRGBA(bitmap: ImageBitmap): {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  exact: boolean;
} {
  const { width, height, exact } = sourceCanvasSize(bitmap.width, bitmap.height);
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true }) as Any2D | null;
  if (!ctx) throw new Error('2D canvas is not available');
  if (!exact) {
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
  }
  ctx.drawImage(bitmap, 0, 0, width, height);
  return { data: ctx.getImageData(0, 0, width, height).data, width, height, exact };
}
