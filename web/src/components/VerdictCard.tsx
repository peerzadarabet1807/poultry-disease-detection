import type { ClassificationResult } from '../hooks/useClassifier';
import { NOT_A_DIAGNOSIS, verdictCopy, type Category } from '../lib/verdict';
import { AlertIcon, InfoIcon } from './icons';

const TONE: Record<Category, { box: string; ink: string; label: string }> = {
  disease_pattern: {
    box: 'border-disease-line bg-disease-soft',
    ink: 'text-disease-ink',
    label: 'Pattern match',
  },
  no_disease_pattern: {
    box: 'border-healthy-line bg-healthy-soft',
    ink: 'text-healthy-ink',
    label: 'No disease pattern',
  },
  uncertain: {
    box: 'border-uncertain-line bg-uncertain-soft',
    ink: 'text-uncertain-ink',
    label: 'Uncertain',
  },
};

export function VerdictCard({ result }: { result: ClassificationResult }) {
  const { verdict } = result;
  const copy = verdictCopy(verdict);
  const tone = TONE[verdict.category];

  return (
    <div
      className="space-y-3"
      data-testid="verdict"
      data-category={verdict.category}
      data-top={verdict.top.id}
    >
      <div className={`rounded-2xl border-l-4 p-5 ${tone.box}`}>
        <p className={`text-xs font-semibold uppercase tracking-[0.12em] ${tone.ink}`}>{tone.label}</p>
        <h3 className={`mt-1 text-2xl font-semibold leading-tight ${tone.ink}`} data-testid="verdict-title">
          {copy.title}
        </h3>
        <p className="mt-2 text-ink-2">{copy.body}</p>
        {copy.notes.length > 0 && (
          <ul className="mt-3 space-y-1.5 text-sm text-ink-2">
            {copy.notes.map((note) => (
              <li key={note} className="flex gap-2">
                <InfoIcon className="mt-0.5 shrink-0 text-ink-3" />
                <span>{note}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {copy.notice && (
        <div
          role="note"
          className="flex gap-3 rounded-2xl border border-alert-line/60 bg-alert-soft p-4 text-sm text-alert-ink"
          data-testid="ncd-notice"
        >
          <AlertIcon className="mt-0.5 shrink-0 text-lg" />
          <p>{copy.notice}</p>
        </div>
      )}

      <p className="text-sm font-medium text-ink-2" data-testid="not-a-diagnosis">
        {NOT_A_DIAGNOSIS}
      </p>
    </div>
  );
}
