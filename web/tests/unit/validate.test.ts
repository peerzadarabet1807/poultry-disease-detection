import { EngineError, type Prediction } from '../../src/engine/types';
import { validatePrediction } from '../../src/engine/validate';

const cam = () => Array.from({ length: 4 }, () => new Float32Array(49).fill(0.5));
const good = (): Prediction => ({
  probabilities: [0.1, 0.2, 0.3, 0.4],
  camMaps: cam(),
  seen: new Uint8ClampedArray(224 * 224 * 4),
  exactPreprocessing: true,
  inferenceMs: 12,
});

describe('validatePrediction (the boundary between the worker and the UI)', () => {
  it('accepts a well-formed prediction', () => {
    expect(validatePrediction(good())).toBeTruthy();
  });

  it.each([
    ['three probabilities', { probabilities: [0.3, 0.3, 0.4] }],
    ['NaN', { probabilities: [NaN, 0.2, 0.3, 0.5] }],
    ['not summing to one', { probabilities: [0.5, 0.5, 0.5, 0.5] }],
    ['wrong map size', { camMaps: [new Float32Array(10), ...cam().slice(1)] }],
    ['map values above one', { camMaps: [new Float32Array(49).fill(2), ...cam().slice(1)] }],
    ['wrong image size', { seen: new Uint8ClampedArray(10) }],
  ])('rejects %s', (_, patch) => {
    expect(() => validatePrediction({ ...good(), ...patch } as Prediction)).toThrow(EngineError);
  });
});
