import { useEffect, useRef } from 'react';
import type { ClassificationResult, JobState, JobStep } from '../hooks/useClassifier';
import { CLASS_INFO } from '../lib/classes';
import { swapsAxes } from '../lib/exif';
import { CamViewer } from './CamViewer';
import { InfoIcon, RefreshIcon } from './icons';
import { ProbabilityBars } from './ProbabilityBars';
import { VerdictCard } from './VerdictCard';

const STEP_LABEL: Record<JobStep, string> = {
  reading: 'Reading the file…',
  decoding: 'Decoding the photo…',
  'waiting-for-model': 'Waiting for the model to finish loading…',
  classifying: 'Running the model…',
};

interface Props {
  job: JobState;
  onAnother: () => void;
}

export function ResultPanel({ job, onAnother }: Props) {
  const heading = useRef<HTMLHeadingElement>(null);
  const resultId = job.status === 'done' ? job.result.id : null;

  // Move focus to the result once it appears, so keyboard and screen-reader users land on it.
  useEffect(() => {
    if (resultId !== null) heading.current?.focus({ preventScroll: false });
  }, [resultId]);

  if (job.status === 'idle' || job.status === 'error') return null;

  if (job.status === 'working') {
    return (
      <div className="card flex items-center gap-4 p-5" aria-busy="true" data-testid="working">
        <span
          className="size-5 animate-spin rounded-full border-2 border-line-strong border-t-accent"
          aria-hidden="true"
        />
        <p className="text-ink-2">{STEP_LABEL[job.step]}</p>
      </div>
    );
  }

  const { result } = job;
  return (
    <section
      id="result"
      aria-labelledby="result-heading"
      className="card scroll-mt-20 p-4 sm:p-6"
      data-testid="result"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 id="result-heading" ref={heading} tabIndex={-1} className="text-xl font-semibold outline-none">
          Result{result.source.kind === 'sample' ? ` for ${result.source.name}` : ''}
        </h2>
        <button type="button" className="btn btn-secondary !min-h-9 !px-3 !text-sm" onClick={onAnother}>
          <RefreshIcon /> Classify another photo
        </button>
      </div>

      <div className="mt-5 grid gap-6 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] lg:gap-8">
        <div className="space-y-5">
          <figure className="space-y-2">
            <img
              src={result.displayUrl}
              alt={`What you submitted (${result.source.name}), shown upright`}
              className="max-h-80 w-full rounded-xl border border-line bg-surface-2 object-contain"
            />
            <figcaption className="text-xs text-ink-3">
              Your photo, {result.width} × {result.height} px
              {result.source.kind === 'sample' ? ' (sample)' : ''}
            </figcaption>
          </figure>
          <CamViewer key={result.id} prediction={result.prediction} initialClass={result.verdict.top.id} />
        </div>

        <div className="space-y-6">
          <VerdictCard result={result} />
          <SampleTruth result={result} />
          <ProbabilityBars ranked={result.verdict.ranked} />
          <InputNotes result={result} />
          <HeatmapExplainer />
        </div>
      </div>
    </section>
  );
}

function SampleTruth({ result }: { result: ClassificationResult }) {
  const sample = result.source.sample;
  if (!sample) return null;
  const correct = sample.trueClass === result.verdict.top.id;
  return (
    <p
      className="rounded-xl border border-line bg-surface-2 px-4 py-3 text-sm text-ink-2"
      data-testid="sample-truth"
    >
      <strong className="font-semibold text-ink">Dataset label: {CLASS_INFO[sample.trueClass].name}</strong> (
      {sample.set === 'internal' ? 'internal test set' : 'PCR-confirmed set'}). The model's most likely class{' '}
      {correct ? 'matches it' : 'does not match it'}.{sample.note && ` ${sample.note}`}
    </p>
  );
}

function InputNotes({ result }: { result: ClassificationResult }) {
  const notes: string[] = [];
  if (result.orientation !== 1) {
    notes.push(
      `This photo carries a rotation tag${swapsAxes(result.orientation) ? ' (portrait/landscape swap)' : ''}. ` +
        'The model reads the stored pixels without rotating them, exactly as in training, so its view above can look turned.',
    );
  }
  if (!result.prediction.exactPreprocessing) {
    notes.push(
      'This photo is larger than this browser can process at full size, so it was shrunk once before the model resize. ' +
        'The result can differ slightly from the Python package.',
    );
  }
  if (notes.length === 0) return null;
  return (
    <ul className="space-y-2 text-sm text-ink-2" data-testid="input-notes">
      {notes.map((n) => (
        <li key={n} className="flex gap-2">
          <InfoIcon className="mt-0.5 shrink-0 text-ink-3" />
          <span>{n}</span>
        </li>
      ))}
    </ul>
  );
}

function HeatmapExplainer() {
  return (
    <details className="group rounded-xl border border-line bg-surface-2/60 px-4 py-3 text-sm text-ink-2">
      <summary className="cursor-pointer font-semibold text-ink">
        What the heatmap does and does not mean
      </summary>
      <div className="prose-tight mt-3">
        <p>
          It is a class activation map, mathematically the same as the training notebook's Grad-CAM for this
          architecture. It shows which regions of the 224 × 224 image pushed the chosen class's score up, at a
          coarse 7 × 7 resolution.
        </p>
        <p>
          It is not a lesion map and not evidence of disease. A model can score highly for the wrong reasons,
          such as the floor, lighting or framing: in this dataset the Salmonella and Newcastle disease
          training photos came from experimentally infected birds, so their surroundings may differ from the
          farm photos of the other classes. If the warm areas sit on the background rather than the droppings,
          trust the result less.
        </p>
      </div>
    </details>
  );
}
