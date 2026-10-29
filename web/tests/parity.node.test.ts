/**
 * Parity test 2: TF.js (Node, WASM backend) through the TypeScript preprocessing vs Keras.
 *
 * For every committed fixture photo (tests/fixtures/images, original JPEGs):
 *   1. decode with sharp (libjpeg-turbo's accurate integer IDCT, as browsers do; no rotation, no colour
 *      management) and check the pixels equal TensorFlow's decode with the same IDCT (SHA-256 from
 *      scripts/make_web_references.py);
 *   2. resize with web/src/lib/resize.ts and compare with the image the Python package makes from them;
 *   3. apply the caffe step (lib/preprocess.ts), run the converted model and compare probabilities and
 *      CAMs with Keras on the same input: this isolates the TF.js conversion;
 *   4. also report the end-to-end difference from the training pipeline, whose JPEG decode uses the fast
 *      integer IDCT (TensorFlow's default), on this machine and as recorded in Colab.
 * Required: identical pixels in 1 and 2, identical argmax in 3. Results: results/parity/tfjs_<variant>.json.
 *
 *   npm run test:parity                                   (the shipped model, web/public/model)
 *   PARITY_MODEL_DIR=../build/tfjs/uint8 npm run test:parity
 */
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import * as tf from '@tensorflow/tfjs-core';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import type { GraphModel } from '@tensorflow/tfjs-converter';
import sharp from 'sharp';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { extractClassMap, pearson, reluMaxNormalize } from '../src/lib/cam';
import { toModelInput, toSeenImage } from '../src/lib/preprocess';
import { runModel } from '../src/worker/inference';
import { loadGraphModelFromDir } from './utils/nodeModel';
import { readNpy } from './utils/npy';

const ROOT = resolve(__dirname, '..', '..');
const MODEL_DIR = resolve(process.env.PARITY_MODEL_DIR ?? join(ROOT, 'web', 'public', 'model'));
// A named variant (PARITY_MODEL_DIR) writes the committed report; the shipped model writes a scratch copy.
const VARIANT =
  process.env.PARITY_VARIANT ?? (process.env.PARITY_MODEL_DIR ? basename(MODEL_DIR) : 'shipped');
const REPORT_DIR = join(ROOT, VARIANT === 'shipped' ? 'build' : 'results', 'parity');
const REF_DIR = join(ROOT, 'web', 'tests', 'fixtures', 'reference');
const IMAGES = join(ROOT, 'tests', 'fixtures', 'images');

interface Reference {
  id: string;
  file: string;
  width: number;
  height: number;
  decodedSha256: string;
  seen: string;
  kerasPredicted: string;
  kerasProbabilities: number[];
  trainingPathProbabilities: number[];
  colabProbabilities: number[];
  cams: Record<string, number[]>;
}

const { classNames, images } = JSON.parse(readFileSync(join(REF_DIR, 'reference.json'), 'utf8')) as {
  classNames: string[];
  images: Reference[];
};

const argmax = (p: ArrayLike<number>) => Array.from(p).indexOf(Math.max(...Array.from(p)));
let model: GraphModel;

beforeAll(async () => {
  setWasmPaths(join(ROOT, 'web', 'node_modules', '@tensorflow', 'tfjs-backend-wasm', 'dist') + '/');
  await tf.setBackend('wasm');
  await tf.ready();
  model = await loadGraphModelFromDir(MODEL_DIR);
}, 120_000);

afterAll(() => model?.dispose());

describe(`TF.js (${VARIANT}) through the TypeScript preprocessing matches Keras`, () => {
  it('all fixtures', async () => {
    const before = tf.memory().numTensors;
    const rows = [];
    for (const ref of images) {
      await new Promise((r) => setImmediate(r)); // keep the vitest worker RPC responsive
      const { data, info } = await sharp(readFileSync(join(IMAGES, ref.file)))
        .raw()
        .toBuffer({ resolveWithObject: true });
      const decodedMatchesTf = createHash('sha256').update(data).digest('hex') === ref.decodedSha256;

      const seen = toSeenImage(data, info.width, info.height, info.channels);
      const expectedSeen = readNpy(readFileSync(join(REF_DIR, ref.seen))).data as Uint8Array;
      let seenMismatch = 0;
      let seenMaxDiff = 0;
      for (let i = 0; i < seen.length; i++) {
        const d = Math.abs(seen[i] - expectedSeen[i]);
        if (d) seenMismatch++;
        seenMaxDiff = Math.max(seenMaxDiff, d);
      }

      const { probabilities, cam } = await runModel(model, toModelInput(seen));
      const pred = argmax(probabilities);
      const diffs = ref.kerasProbabilities.map((p, i) => Math.abs(p - probabilities[i]));
      const e2e = ref.trainingPathProbabilities.map((p, i) => Math.abs(p - probabilities[i]));
      const camCorr = pearson(
        reluMaxNormalize(extractClassMap(cam, pred, classNames.length)),
        ref.cams[classNames[pred]],
      );
      rows.push({
        id: ref.id,
        size: `${info.width}x${info.height}`,
        decodedMatchesTf,
        seenPixelsDiffering: seenMismatch,
        seenMaxDiff,
        keras: ref.kerasPredicted,
        tfjs: classNames[pred],
        trainingPath: classNames[argmax(ref.trainingPathProbabilities)],
        colab: classNames[argmax(ref.colabProbabilities)],
        maxProbDiff: Math.max(...diffs),
        meanProbDiff: diffs.reduce((a, b) => a + b, 0) / diffs.length,
        endToEndMaxProbDiff: Math.max(...e2e),
        camCorr,
      });
    }

    const summary = {
      description:
        'TF.js graph model in Node (WASM backend), fed by the TypeScript preprocessing from the original ' +
        'JPEGs (sharp decode), compared with Keras on CPU. Engineering check, not a model evaluation.',
      variant: VARIANT,
      backend: tf.getBackend(),
      images: rows.length,
      decodedIdenticalToTf: rows.filter((r) => r.decodedMatchesTf).length,
      seenIdenticalToPython: rows.filter((r) => r.seenPixelsDiffering === 0).length,
      conversion: {
        argmaxIdenticalToKeras: rows.filter((r) => r.tfjs === r.keras).length,
        maxProbDiff: Math.max(...rows.map((r) => r.maxProbDiff)),
        meanProbDiff: rows.reduce((a, r) => a + r.meanProbDiff, 0) / rows.length,
        minCamCorr: Math.min(...rows.map((r) => r.camCorr)),
        meanCamCorr: rows.reduce((a, r) => a + r.camCorr, 0) / rows.length,
      },
      endToEnd: {
        argmaxIdenticalToTrainingPath: rows.filter((r) => r.tfjs === r.trainingPath).length,
        argmaxIdenticalToColab: rows.filter((r) => r.tfjs === r.colab).length,
        maxProbDiff: Math.max(...rows.map((r) => r.endToEndMaxProbDiff)),
      },
      leakedTensors: tf.memory().numTensors - before,
    };
    console.log(JSON.stringify(summary, null, 1));
    mkdirSync(REPORT_DIR, { recursive: true });
    writeFileSync(
      join(REPORT_DIR, `tfjs_${VARIANT}.json`),
      JSON.stringify({ summary, perImage: rows }, null, 2) + '\n',
    );

    expect(summary.decodedIdenticalToTf).toBe(rows.length);
    expect(summary.seenIdenticalToPython).toBe(rows.length);
    expect(summary.conversion.argmaxIdenticalToKeras).toBe(rows.length);
    expect(summary.leakedTensors).toBe(0);
  }, 600_000);
});
