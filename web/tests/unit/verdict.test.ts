import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { THRESHOLDS } from '../../src/lib/metrics';
import { interpret, NCD_NOTICE, NOT_A_DIAGNOSIS, verdictCopy } from '../../src/lib/verdict';

const fixture = JSON.parse(readFileSync(join(__dirname, '..', 'fixtures', 'verdict-cases.json'), 'utf8')) as {
  thresholds: { minTop: number; minMargin: number };
  cases: {
    probabilities: number[];
    category: string;
    top: string;
    runnerUp: string;
    reasons: string[];
    notifiable: boolean;
  }[];
};

describe('interpret matches the Python rule (src/poultry_disease/verdict.py)', () => {
  it('uses the same thresholds', () => {
    expect(THRESHOLDS.minTopProbability).toBe(fixture.thresholds.minTop);
    expect(THRESHOLDS.minTop2Margin).toBe(fixture.thresholds.minMargin);
  });

  it(`agrees on all ${fixture.cases.length} recorded cases`, () => {
    for (const c of fixture.cases) {
      const v = interpret(c.probabilities);
      expect({
        category: v.category,
        top: v.top.id,
        runnerUp: v.runnerUp.id,
        reasons: v.reasons,
        notifiable: v.notifiable,
      }).toEqual({
        category: c.category,
        top: c.top,
        runnerUp: c.runnerUp,
        reasons: c.reasons,
        notifiable: c.notifiable,
      });
    }
  });

  it('rejects the wrong number of probabilities', () => {
    expect(() => interpret([0.5, 0.5])).toThrow();
  });
});

describe('verdict copy', () => {
  const all = (p: number[]) => {
    const c = verdictCopy(interpret(p));
    return [c.title, c.body, ...c.notes, c.notice ?? ''].join(' ');
  };

  it('never calls a result a diagnosis', () => {
    for (const p of [
      [0.99, 0.005, 0.003, 0.002],
      [0.002, 0.99, 0.005, 0.003],
      [0.4, 0.3, 0.2, 0.1],
      [0, 0, 0, 1],
    ]) {
      expect(all(p)).not.toMatch(/diagnos/i);
    }
    expect(NOT_A_DIAGNOSIS).toMatch(/not a veterinary diagnosis/);
  });

  it('says "pattern consistent with" for disease classes', () => {
    expect(verdictCopy(interpret([0.99, 0.004, 0.003, 0.003])).title).toBe(
      'Pattern consistent with Coccidiosis',
    );
  });

  it('does not treat Healthy as a clean bill of health and quotes the PCR-set precision', () => {
    const copy = verdictCopy(interpret([0.002, 0.99, 0.005, 0.003]));
    expect(copy.title).toBe('No disease pattern recognised');
    expect(copy.body).toMatch(/not a clean bill of health/);
    expect(copy.body).toMatch(/72%/);
  });

  it('adds the notifiable notice whenever NCD is most likely, even when uncertain', () => {
    expect(verdictCopy(interpret([0, 0, 0, 1])).notice).toBe(NCD_NOTICE);
    const unsure = verdictCopy(interpret([0.05, 0.35, 0.05, 0.55]));
    expect(unsure.title).toMatch(/Uncertain/);
    expect(unsure.notice).toBe(NCD_NOTICE);
    expect(verdictCopy(interpret([1, 0, 0, 0])).notice).toBeUndefined();
  });

  it('labels the thresholds a usability heuristic', () => {
    const copy = verdictCopy(interpret([0.4, 0.35, 0.15, 0.1]));
    expect(copy.notes.join(' ')).toMatch(/usability heuristic/);
    expect(copy.body).toMatch(/90 percentage points/);
  });
});
