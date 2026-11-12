/**
 * Parity test 3: the input check's image encoder in TF.js (Node, WASM backend) through the TypeScript
 * preprocessing vs the original PyTorch model (scripts/make_gate_references.py).
 *
 * For every fixture (18 dataset photos, 12 original OOD photos):
 *   1. conversion: feed TF.js exactly the 224 x 224 image Python made (committed .npy), compare the embedding
 *      with PyTorch's and apply the rule in web/src/lib/inputCheck.ts;
 *   2. end to end: decode the JPEG with sharp, resize with web/src/lib/resize.ts, embed and decide. Most
 *      fixtures decode to the same pixels as TensorFlow; a few JPEGs decode slightly differently in sharp's
 *      libjpeg-turbo (browsers ship their own decoders too), so here the image identity and cosine are
 *      reported and the decision is what must match.
 * Required: conversion cosine >= 0.999 and the same pass/fail decision on every fixture in both paths, no
 * leaked tensors. Results: results/parity/gate_<variant>.json.
 *
 *   npm run test:parity                                                    (the shipped gate, web/public/gate)
 *   PARITY_GATE_DIR=../build/gate/tinyclip-vit-8m-16-yfcc15m/tfjs/uint8 npm run test:parity
 */
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import * as tf from '@tensorflow/tfjs-core';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import type { GraphModel } from '@tensorflow/tfjs-converter';
import sharp from 'sharp';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { checkEmbedding } from '../src/lib/inputCheck';
import { toSeenImage } from '../src/lib/preprocess';
import { runGate } from '../src/worker/gate';
import { loadGraphModelFromDir } from './utils/nodeModel';
import { readNpy } from './utils/npy';

const ROOT = resolve(__dirname, '..', '..');
const FIXTURES = join(ROOT, 'web', 'tests', 'fixtures');
const GATE_DIR = resolve(process.env.PARITY_GATE_DIR ?? join(ROOT, 'web', 'public', 'gate'));
const VARIANT = process.env.PARITY_GATE_DIR ? basename(GATE_DIR) : 'shipped';
const REPORT_DIR = join(ROOT, VARIANT === 'shipped' ? 'build' : 'results', 'parity');
const MIN_COSINE = 0.999;

interface Reference {
  id: string;
  file: string;
  kind: 'droppings' | 'ood';
  /** Python's 224 x 224 image, relative to web/tests/fixtures */
  seen: string;
  seenSha256: string;
  score: number;
  passed: boolean;
  embedding: number[];
}

const ref = JSON.parse(readFileSync(join(FIXTURES, 'input_check', 'reference.json'), 'utf8')) as {
  model: string;
  version: string;
  threshold: number;
  images: Reference[];
};

let gate: GraphModel;

beforeAll(async () => {
  setWasmPaths(join(ROOT, 'web', 'node_modules', '@tensorflow', 'tfjs-backend-wasm', 'dist') + '/');
  await tf.setBackend('wasm');
  await tf.ready();
  gate = await loadGraphModelFromDir(GATE_DIR);
}, 120_000);

afterAll(() => gate?.dispose());

function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / Math.sqrt(na * nb);
}

describe(`input check TF.js (${VARIANT}) matches the PyTorch reference`, () => {
  it('all fixtures', async () => {
    const before = tf.memory().numTensors;
    const rows = [];
    for (const r of ref.images) {
      await new Promise((resolveTick) => setImmediate(resolveTick)); // keep the vitest worker RPC responsive
      const { data, info } = await sharp(readFileSync(join(ROOT, r.file)))
        .raw()
        .toBuffer({ resolveWithObject: true });
      const pySeen = readNpy(readFileSync(join(FIXTURES, r.seen))).data as Uint8Array;
      const conv = await runGate(gate, pySeen);
      const convCheck = checkEmbedding(conv);

      const seen = toSeenImage(data, info.width, info.height, info.channels);
      const seenMatches = createHash('sha256').update(seen).digest('hex') === r.seenSha256;
      const e2e = seenMatches ? conv : await runGate(gate, seen);
      const e2eCheck = checkEmbedding(e2e);
      rows.push({
        id: r.id,
        kind: r.kind,
        pythonScore: r.score,
        pythonPassed: r.passed,
        cosine: cosine(conv, r.embedding),
        tfjsScore: convCheck.score,
        scoreDiff: Math.abs(convCheck.score - r.score),
        tfjsPassed: convCheck.passed,
        endToEnd: {
          seenIdenticalToPython: seenMatches,
          cosine: cosine(e2e, r.embedding),
          score: e2eCheck.score,
          passed: e2eCheck.passed,
        },
      });
    }
    const summary = {
      description:
        'Input check image encoder as a TF.js graph model in Node (WASM backend) vs the original PyTorch model. ' +
        'conversion: TF.js fed exactly the 224 x 224 image Python made. endToEnd: from the original JPEG through ' +
        'sharp decode and the TypeScript resize. Engineering check, not a model evaluation.',
      variant: VARIANT,
      model: ref.model,
      version: ref.version,
      backend: tf.getBackend(),
      images: rows.length,
      conversion: {
        minCosine: Math.min(...rows.map((r) => r.cosine)),
        meanCosine: rows.reduce((a, r) => a + r.cosine, 0) / rows.length,
        maxScoreDiff: Math.max(...rows.map((r) => r.scoreDiff)),
        sameDecision: rows.filter((r) => r.pythonPassed === r.tfjsPassed).length,
      },
      endToEnd: {
        seenIdenticalToPython: rows.filter((r) => r.endToEnd.seenIdenticalToPython).length,
        minCosine: Math.min(...rows.map((r) => r.endToEnd.cosine)),
        sameDecision: rows.filter((r) => r.pythonPassed === r.endToEnd.passed).length,
      },
      passedDroppings: rows.filter((r) => r.kind === 'droppings' && r.tfjsPassed).length,
      droppings: rows.filter((r) => r.kind === 'droppings').length,
      rejectedOod: rows.filter((r) => r.kind === 'ood' && !r.tfjsPassed).length,
      ood: rows.filter((r) => r.kind === 'ood').length,
      requiredMinCosine: MIN_COSINE,
      leakedTensors: tf.memory().numTensors - before,
    };
    console.log(JSON.stringify(summary, null, 1));
    mkdirSync(REPORT_DIR, { recursive: true });
    writeFileSync(
      join(REPORT_DIR, `gate_${VARIANT}.json`),
      JSON.stringify({ summary, perImage: rows }, null, 2) + '\n',
    );

    expect(summary.conversion.minCosine).toBeGreaterThanOrEqual(MIN_COSINE);
    expect(summary.conversion.sameDecision).toBe(rows.length);
    expect(summary.endToEnd.sameDecision).toBe(rows.length);
    expect(summary.leakedTensors).toBe(0);
  }, 600_000);
});
