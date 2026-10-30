import { useEffect, useId, useRef, useState } from 'react';
import type { Prediction } from '../engine/types';
import { bilinearUpsample, CAM_SIZE, heatmapRGBA } from '../lib/cam';
import { CLASS_IDS, CLASS_INFO, type ClassId } from '../lib/classes';
import { MODEL_SIZE } from '../lib/resize';

interface Props {
  prediction: Prediction;
  initialClass: ClassId;
}

/**
 * The image exactly as the model saw it (224 x 224), with the class activation map of a chosen class on top.
 * Mount with a key per result, so the class selection starts from the prediction each time.
 */
export function CamViewer({ prediction, initialClass }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [cls, setCls] = useState<ClassId>(initialClass);
  const [opacity, setOpacity] = useState(55);
  const sliderId = useId();

  useEffect(() => {
    const ctx = canvas.current?.getContext('2d');
    if (!ctx) return;
    ctx.putImageData(new ImageData(new Uint8ClampedArray(prediction.seen), MODEL_SIZE, MODEL_SIZE), 0, 0);
    if (opacity === 0) return;
    const map = prediction.camMaps[CLASS_IDS.indexOf(cls)];
    const heat = heatmapRGBA(bilinearUpsample(map, CAM_SIZE, CAM_SIZE, MODEL_SIZE, MODEL_SIZE));
    const layer = document.createElement('canvas');
    layer.width = MODEL_SIZE;
    layer.height = MODEL_SIZE;
    layer.getContext('2d')?.putImageData(new ImageData(heat, MODEL_SIZE, MODEL_SIZE), 0, 0);
    ctx.globalAlpha = opacity / 100;
    ctx.drawImage(layer, 0, 0);
    ctx.globalAlpha = 1;
  }, [prediction, cls, opacity]);

  return (
    <figure className="space-y-3">
      <canvas
        ref={canvas}
        width={MODEL_SIZE}
        height={MODEL_SIZE}
        className="aspect-square w-full rounded-xl border border-line bg-surface-2"
        role="img"
        aria-label={`The 224 by 224 pixel image the model saw, with the ${CLASS_INFO[cls].name} heatmap at ${opacity}% opacity`}
        data-testid="cam-canvas"
      />
      <div className="space-y-3">
        <fieldset>
          <legend className="text-sm font-semibold text-ink">Heatmap for</legend>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {CLASS_IDS.map((id) => (
              <label
                key={id}
                className={`cursor-pointer rounded-lg border px-2.5 py-1.5 text-sm has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-focus ${
                  id === cls
                    ? 'border-accent bg-accent text-on-accent'
                    : 'border-line-strong text-ink-2 hover:border-ink-3'
                }`}
              >
                <input
                  type="radio"
                  name="cam-class"
                  value={id}
                  checked={id === cls}
                  onChange={() => setCls(id)}
                  className="sr-only"
                />
                {CLASS_INFO[id].name}
              </label>
            ))}
          </div>
        </fieldset>
        <div className="flex items-center gap-3">
          <label htmlFor={sliderId} className="shrink-0 text-sm font-semibold text-ink">
            Opacity
          </label>
          <input
            id={sliderId}
            type="range"
            min={0}
            max={100}
            step={5}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            className="w-full accent-[var(--accent)]"
            aria-valuetext={`${opacity}%`}
            data-testid="cam-opacity"
          />
          <span className="tabular w-10 text-right text-sm text-ink-3">{opacity}%</span>
        </div>
      </div>
      <figcaption className="text-xs text-ink-3">
        The model sees a 224 × 224 copy of the whole photo, squashed to a square and not rotated, exactly as
        in training. Warm colours mark where the features that raise the chosen class's score are strongest.
      </figcaption>
    </figure>
  );
}
