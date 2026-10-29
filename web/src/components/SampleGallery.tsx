import type { SamplesState } from '../hooks/useSamples';
import { CLASS_INFO, type ClassId } from '../lib/classes';
import { SAMPLES_BASE, type SampleInfo } from '../lib/samples';
import { CheckIcon, CrossIcon } from './icons';

interface Props {
  samples: SamplesState;
  /** sample id -> the class the model picked, once classified (the dataset label is then revealed) */
  revealed: Record<string, ClassId>;
  activeId?: string;
  busy: boolean;
  onPick: (sample: SampleInfo, index: number) => void;
}

export function SampleGallery({ samples, revealed, activeId, busy, onPick }: Props) {
  return (
    <section aria-labelledby="samples-heading" className="space-y-4">
      <div className="space-y-1">
        <h2 id="samples-heading" className="text-2xl font-semibold">
          No photo at hand? Try a sample
        </h2>
        <p className="max-w-3xl text-ink-2">
          Real photos from the dataset, some from the internal test set and some from the PCR-confirmed set,
          including two the model gets wrong and one it is unsure about. The label is shown after you
          classify.
        </p>
      </div>

      {samples.status === 'error' && <p className="text-sm text-alert-ink">{samples.message}</p>}
      {samples.status === 'loading' && <p className="text-sm text-ink-3">Loading samples…</p>}
      {samples.status === 'ready' && (
        <>
          <ul className="grid grid-cols-3 gap-3 sm:grid-cols-4 lg:grid-cols-6" data-testid="sample-grid">
            {samples.manifest.samples.map((s, i) => {
              const picked = revealed[s.id];
              return (
                <li key={s.id}>
                  <button
                    type="button"
                    onClick={() => onPick(s, i)}
                    disabled={busy}
                    aria-pressed={activeId === s.id}
                    className={`group relative block w-full overflow-hidden rounded-xl border bg-surface-2 text-left transition-colors disabled:cursor-wait ${
                      activeId === s.id
                        ? 'border-accent ring-2 ring-accent'
                        : 'border-line hover:border-ink-3'
                    }`}
                    data-testid="sample"
                    data-sample={s.id}
                  >
                    <img
                      src={`${SAMPLES_BASE}${s.thumb}`}
                      alt={`Sample ${i + 1}`}
                      loading="lazy"
                      width={320}
                      height={320}
                      className="aspect-square w-full object-cover"
                    />
                    <span className="block px-2 py-1.5 text-xs">
                      <span className="font-medium text-ink">Sample {i + 1}</span>
                      {picked ? (
                        <span className="flex items-center gap-1 text-ink-2">
                          {picked === s.trueClass ? (
                            <CheckIcon className="shrink-0 text-healthy-ink" aria-label="model correct" />
                          ) : (
                            <CrossIcon className="shrink-0 text-disease-ink" aria-label="model wrong" />
                          )}
                          {CLASS_INFO[s.trueClass].name}
                        </span>
                      ) : (
                        <span className="block text-ink-3">
                          {s.set === 'internal' ? 'Internal test' : 'PCR-confirmed'}
                        </span>
                      )}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
          <p className="text-xs text-ink-3">{samples.manifest.credit}</p>
        </>
      )}
    </section>
  );
}
