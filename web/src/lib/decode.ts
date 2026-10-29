/**
 * Browser image decoding for the model, matching training as closely as a browser can:
 *  - no EXIF rotation (`imageOrientation: "none"`): tf.io.decode_image reads the stored pixels;
 *  - no colour management (`colorSpaceConversion: "none"`): TensorFlow ignores embedded ICC profiles;
 *  - JPEGs are decoded with the browser's accurate IDCT, while training used TensorFlow's fast IDCT. No browser
 *    API can select it; models/PARITY.md measures the effect.
 * HEIC (the iPhone default) is decoded natively where possible (Safari), otherwise via a lazily loaded
 * heic2any. Training never saw HEIC.
 */
import { readJpegInfo } from './exif';
import type { ImageKind } from './fileValidation';

const MODEL_DECODE: ImageBitmapOptions = {
  imageOrientation: 'none',
  colorSpaceConversion: 'none',
  premultiplyAlpha: 'none',
};

export const MIN_SIDE_PX = 32;

export class DecodeError extends Error {}

export interface DecodedPhoto {
  /** pixels as stored in the file, for the model */
  bitmap: ImageBitmap;
  /** EXIF orientation of a JPEG (1 = none) */
  orientation: number;
}

export async function decodeForModel(file: Blob, kind: ImageKind): Promise<DecodedPhoto> {
  let source: Blob = file;
  let bitmap: ImageBitmap | null = null;
  if (kind === 'heic') {
    try {
      bitmap = await createImageBitmap(file, MODEL_DECODE);
    } catch {
      source = await heicToPng(file);
    }
  }
  if (!bitmap) {
    try {
      bitmap = await createImageBitmap(source, MODEL_DECODE);
    } catch {
      throw new DecodeError('This file could not be decoded as an image. It may be damaged or truncated.');
    }
  }
  if (Math.min(bitmap.width, bitmap.height) < MIN_SIDE_PX) {
    const { width, height } = bitmap;
    bitmap.close();
    throw new DecodeError(`This image is only ${width} × ${height} px. Please use a full-size photo.`);
  }
  const orientation =
    kind === 'jpeg' ? (readJpegInfo(await file.slice(0, 256 * 1024).arrayBuffer())?.orientation ?? 1) : 1;
  return { bitmap, orientation };
}

async function heicToPng(file: Blob): Promise<Blob> {
  try {
    const { default: heic2any } = await import('heic2any');
    const out = await heic2any({ blob: file, toType: 'image/png' });
    return Array.isArray(out) ? out[0] : out;
  } catch {
    throw new DecodeError('This HEIC photo could not be decoded. Please export it as JPEG and try again.');
  }
}

/** A downscaled JPEG of the photo for display, upright (EXIF applied) as the user expects to see it. */
export async function makeDisplayUrl(file: Blob, fallback: ImageBitmap, maxSide = 1400): Promise<string> {
  let upright: ImageBitmap | null = null;
  try {
    upright = await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch {
    /* HEIC outside Safari: show the decoded pixels */
  }
  const source = upright ?? fallback;
  const scale = Math.min(1, maxSide / Math.max(source.width, source.height));
  const width = Math.round(source.width * scale);
  const height = Math.round(source.height * scale);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas is not available');
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, width, height);
  upright?.close();
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.9));
  if (!blob) throw new Error('Could not render the image preview');
  return URL.createObjectURL(blob);
}
