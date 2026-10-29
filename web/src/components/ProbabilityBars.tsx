import { CLASS_INFO } from '../lib/classes';
import { formatPercent } from '../lib/metrics';
import type { Ranked } from '../lib/verdict';

const pct = (p: number) => formatPercent(p, p >= 0.995 || p < 0.01 ? 1 : 0);

/** Horizontal bars, most likely class first, always labelled with the number. */
export function ProbabilityBars({ ranked }: { ranked: Ranked[] }) {
  return (
    <figure className="space-y-3">
      <figcaption className="text-sm font-semibold text-ink">Model output for each class</figcaption>
      <ol className="space-y-3" aria-label="Class probabilities, most likely first">
        {ranked.map(({ id, probability }) => {
          const info = CLASS_INFO[id];
          return (
            <li key={id} data-testid="prob-row" data-class={id} data-probability={probability.toFixed(6)}>
              <div className="flex items-baseline justify-between gap-3 text-sm">
                <span className="flex min-w-0 items-center gap-2">
                  <span
                    className="size-2.5 shrink-0 rounded-[3px]"
                    style={{ background: info.color }}
                    aria-hidden="true"
                  />
                  <span className="font-medium text-ink">{info.name}</span>
                </span>
                <span className="tabular shrink-0 font-semibold text-ink">{pct(probability)}</span>
              </div>
              <div className="mt-1.5 h-2.5 w-full overflow-hidden rounded-full bg-track" aria-hidden="true">
                <div
                  className="h-full origin-left rounded-full transition-[width] duration-700 ease-out"
                  style={{ width: `${Math.max(probability * 100, 0.8)}%`, background: info.color }}
                />
              </div>
            </li>
          );
        })}
      </ol>
      <p className="text-xs text-ink-3">
        These are softmax scores, not calibrated probabilities: on the PCR-confirmed set the model is more
        confident than it is accurate.
      </p>
    </figure>
  );
}
