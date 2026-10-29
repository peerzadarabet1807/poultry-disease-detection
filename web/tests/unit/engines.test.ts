import { ApiEngine } from '../../src/engine/apiEngine';
import { inferenceMode } from '../../src/engine';
import { EngineError, type Prediction } from '../../src/engine/types';
import { parseApiResponse, validatePrediction } from '../../src/engine/validate';

const cam = () => Array.from({ length: 4 }, () => new Float32Array(49).fill(0.5));
const good = (): Prediction => ({
  probabilities: [0.1, 0.2, 0.3, 0.4],
  camMaps: cam(),
  seen: new Uint8ClampedArray(224 * 224 * 4),
  exactPreprocessing: true,
  inferenceMs: 12,
});

describe('validatePrediction (the boundary both engines pass through)', () => {
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

describe('inferenceMode', () => {
  it('defaults to the browser engine', () => {
    expect(inferenceMode(undefined)).toBe('browser');
    expect(inferenceMode('nonsense')).toBe('browser');
    expect(inferenceMode('api')).toBe('api');
  });
});

/** A response shaped like backend/poultry_api/schemas.py PredictResponse. */
function apiBody() {
  const grid = Array.from({ length: 7 }, () => Array.from({ length: 7 }, () => 0.25));
  return {
    model_version: '1.0.0',
    predicted_class: 'Salmonella',
    confidence: 0.97,
    probabilities: { Coccidiosis: 0.01, Healthy: 0.01, Salmonella: 0.97, NCD: 0.01 },
    low_confidence: false,
    verdict: {},
    image: { format: 'jpeg', width: 10, height: 10 },
    cam: {
      target_class: 'Salmonella',
      maps: { Coccidiosis: grid, Healthy: grid, Salmonella: grid, NCD: grid },
      overlay_png_base64: '',
      input_png_base64: 'iVBORw0KGgo=',
    },
  };
}

describe('ApiEngine', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('needs a URL', () => {
    expect(() => new ApiEngine('')).toThrow(EngineError);
  });

  it('parses a backend response into the shared Prediction shape', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify(apiBody()), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    // jsdom has no createImageBitmap/canvas pixels: stand in for the PNG decode of the 224 x 224 input
    vi.stubGlobal('createImageBitmap', async () => ({ close() {} }));
    const getContext = vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      drawImage() {},
      getImageData: () => ({ data: new Uint8ClampedArray(224 * 224 * 4) }),
    } as unknown as ReturnType<HTMLCanvasElement['getContext']> as never);

    const engine = new ApiEngine('http://localhost:8000/');
    const bitmap = { close: vi.fn() } as unknown as ImageBitmap;
    const p = await engine.classify({
      file: new Blob(['x']),
      kind: 'jpeg',
      photo: { bitmap, orientation: 1 },
    });
    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8000/api/v1/predict?cam=true',
      expect.anything(),
    );
    expect(p.probabilities).toEqual([0.01, 0.01, 0.97, 0.01]);
    expect(p.camMaps).toHaveLength(4);
    expect(p.camMaps[2][0]).toBeCloseTo(0.25);
    expect((bitmap.close as unknown as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(1);
    getContext.mockRestore();
  });

  it('surfaces the API error message', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({ error: { code: 'unsupported_media_type', message: 'Upload a JPEG.' } }),
            {
              status: 415,
            },
          ),
      ),
    );
    const engine = new ApiEngine('http://localhost:8000');
    await expect(
      engine.classify({
        file: new Blob(['x']),
        kind: 'jpeg',
        photo: { bitmap: { close() {} } as unknown as ImageBitmap, orientation: 1 },
      }),
    ).rejects.toThrow('Upload a JPEG.');
  });

  it('rejects malformed responses', () => {
    expect(() => parseApiResponse({ probabilities: { Coccidiosis: 1 } })).toThrow(EngineError);
    expect(() => parseApiResponse(null)).toThrow(EngineError);
  });
});
