import config from '../../src/data/input_check.json';
import summary from '../../src/data/input_check_summary.json';
import { EngineError, type EngineOutput } from '../../src/engine/types';
import { validateInputCheck, validateOutput } from '../../src/engine/validate';
import {
  checkEmbedding,
  EMBEDDING_DIM,
  INPUT_CHECK_THRESHOLD,
  INPUT_CHECK_VERSION,
  inputCheckScore,
} from '../../src/lib/inputCheck';
import { INPUT_CHECK_COPY, rejectionAnnouncement, safeguardNote } from '../../src/lib/inputCheckCopy';

const protos = config.prototypes.vectors;

/** A vector orthogonal to every prototype (Gram-Schmidt against the four). */
function orthogonalToPrototypes(seed = 1): number[] {
  let v = Array.from({ length: EMBEDDING_DIM }, (_, i) => (Math.sin(i * 12.9898 + seed) * 43758.5453) % 1);
  const basis: number[][] = [];
  for (const p of protos) {
    let b = [...p];
    for (const q of basis) {
      const d = b.reduce((a, x, i) => a + x * q[i], 0);
      b = b.map((x, i) => x - d * q[i]);
    }
    const n = Math.hypot(...b);
    basis.push(b.map((x) => x / n));
  }
  for (const q of basis) {
    const d = v.reduce((a, x, i) => a + x * q[i], 0);
    v = v.map((x, i) => x - d * q[i]);
  }
  return v;
}

describe('input check rule (mirrors src/poultry_disease/input_check.py)', () => {
  it('loads the committed threshold and version', () => {
    expect(INPUT_CHECK_THRESHOLD).toBe(config.threshold);
    expect(INPUT_CHECK_THRESHOLD).toBe(summary.threshold);
    expect(INPUT_CHECK_VERSION).toBe(summary.version);
    expect(INPUT_CHECK_THRESHOLD).toBeGreaterThan(0);
    expect(INPUT_CHECK_THRESHOLD).toBeLessThan(1);
    expect(protos).toHaveLength(4);
    expect(config.prototypes.classes).toEqual(['Coccidiosis', 'Healthy', 'Salmonella', 'NCD']);
  });

  it('scores a prototype at 1 whatever its length, and passes it', () => {
    for (const p of protos) {
      expect(inputCheckScore(p)).toBeCloseTo(1, 6);
      expect(checkEmbedding(p.map((x) => x * 0.2)).passed).toBe(true);
    }
  });

  it('fails an embedding unlike every prototype', () => {
    const r = checkEmbedding(orthogonalToPrototypes());
    expect(r.score).toBeCloseTo(0, 9);
    expect(r.passed).toBe(false);
    expect(r.threshold).toBe(INPUT_CHECK_THRESHOLD);
  });

  it('passes exactly at the threshold and fails just below it', () => {
    const o = orthogonalToPrototypes(2);
    const on = Math.hypot(...o);
    const at = (c: number) => protos[0].map((x, i) => c * x + Math.sqrt(1 - c * c) * (o[i] / on));
    expect(inputCheckScore(at(INPUT_CHECK_THRESHOLD))).toBeCloseTo(INPUT_CHECK_THRESHOLD, 6);
    expect(checkEmbedding(at(INPUT_CHECK_THRESHOLD + 1e-4)).passed).toBe(true);
    expect(checkEmbedding(at(INPUT_CHECK_THRESHOLD - 1e-4)).passed).toBe(false);
  });

  it.each([
    ['too short', new Array(10).fill(1)],
    ['zero', new Array(EMBEDDING_DIM).fill(0)],
    ['NaN', new Array(EMBEDDING_DIM).fill(NaN)],
  ])('rejects a %s embedding', (_, e) => {
    expect(() => inputCheckScore(e)).toThrow();
  });
});

describe('input check at the worker boundary', () => {
  const check = { score: 0.9, threshold: INPUT_CHECK_THRESHOLD, passed: true, version: 'v' };
  const output = (patch: Partial<EngineOutput>): EngineOutput => ({
    inputCheck: check,
    inputCheckMs: 5,
    prediction: null,
    seen: new Uint8ClampedArray(224 * 224 * 4),
    ...patch,
  });

  it('accepts a rejected photo without a prediction', () => {
    const c = { ...check, score: 0.5, passed: false };
    expect(validateOutput(output({ inputCheck: c })).prediction).toBeNull();
  });

  it.each([
    ['a decision that contradicts the score', { ...check, score: 0.5 }],
    ['a score above 1', { ...check, score: 1.5, passed: true }],
    ['a NaN score', { ...check, score: NaN, passed: false }],
    ['no version', { ...check, version: '' }],
  ])('rejects %s', (_, c) => {
    expect(() => validateInputCheck(c)).toThrow(EngineError);
  });
});

describe('input check copy', () => {
  const all = [...Object.values(INPUT_CHECK_COPY), safeguardNote(), rejectionAnnouncement()];

  it('uses the agreed wording', () => {
    expect(INPUT_CHECK_COPY.heading).toBe("This doesn't look like a photo of poultry droppings.");
    expect(INPUT_CHECK_COPY.uncheckedLabel).toBe(
      'Unchecked input: this photo did not look like poultry droppings, so treat the result with extra caution.',
    );
    expect([INPUT_CHECK_COPY.retry, INPUT_CHECK_COPY.tips, INPUT_CHECK_COPY.anyway]).toEqual([
      'Try another photo',
      'Photo tips',
      'Analyse anyway',
    ]);
  });

  it('never implies a health verdict, never says diagnosis, no em-dashes', () => {
    for (const text of all) {
      expect(text).not.toMatch(/\b(healthy|diseased|sick|fake|pure|diagnos\w*)\b/i);
      expect(text).not.toContain('—');
    }
  });

  it('states the measured rates from the committed summary', () => {
    const note = safeguardNote();
    const p1 = (x: number) => `${(x * 100).toFixed(1)}%`;
    expect(note).toContain(p1(summary.false_rejection.test_internal.rate));
    expect(note).toContain(p1(summary.false_rejection.test_external.rate));
    expect(note).toContain(p1(summary.false_acceptance.near.rate));
    expect(note).toMatch(/not a guarantee/);
  });
});
