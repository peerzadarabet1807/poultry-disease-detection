import { useCallback, useState } from 'react';
import { AboutModel } from './components/AboutModel';
import { Footer } from './components/Footer';
import { Header } from './components/Header';
import { AlertIcon } from './components/icons';
import { ModelStatus } from './components/ModelStatus';
import { PhotoGuidelines } from './components/PhotoGuidelines';
import { ResultPanel } from './components/ResultPanel';
import { SampleGallery } from './components/SampleGallery';
import { UploadCard } from './components/UploadCard';
import { inferenceMode } from './engine';
import { useClassifier, type JobState, type SourceKind } from './hooks/useClassifier';
import { useSamples } from './hooks/useSamples';
import { useTheme } from './hooks/useTheme';
import { CLASS_INFO, type ClassId } from './lib/classes';
import { EXTERNAL, formatPercent, INTERNAL } from './lib/metrics';
import { fetchSampleBlob, type SampleInfo } from './lib/samples';
import { NOT_A_DIAGNOSIS, verdictCopy } from './lib/verdict';

/** Text for the polite live region, so screen readers hear progress and the outcome. */
function announcement(job: JobState): string {
  switch (job.status) {
    case 'working':
      return 'Analysing the photo…';
    case 'error':
      return job.message;
    case 'done': {
      const { verdict } = job.result;
      const copy = verdictCopy(verdict);
      const top = `${CLASS_INFO[verdict.top.id].name} ${formatPercent(verdict.top.probability)}`;
      return `Result: ${copy.title}. Most likely class ${top}.${copy.notice ? ` ${copy.notice}` : ''}`;
    }
    default:
      return '';
  }
}

export default function App() {
  const { theme, toggle } = useTheme();
  const { model, job, classify, reset, retryModel } = useClassifier();
  const samples = useSamples();
  const [guidelinesOpen, setGuidelinesOpen] = useState(true);
  const [revealed, setRevealed] = useState<Record<string, ClassId>>({});
  const [activeSample, setActiveSample] = useState<string>();

  const busy = job.status === 'working';
  const lastInferenceMs = job.status === 'done' ? job.result.prediction.inferenceMs : undefined;

  const run = useCallback(
    async (input: Blob | (() => Promise<Blob>), kind: SourceKind, name: string, sample?: SampleInfo) => {
      setGuidelinesOpen(false);
      setActiveSample(sample?.id);
      const result = await classify(input, { kind, name, sample });
      // reveal a sample's dataset label once it has been classified
      if (result && sample) setRevealed((r) => ({ ...r, [sample.id]: result.verdict.top.id }));
    },
    [classify],
  );

  const onFile = useCallback((file: File, kind: SourceKind) => void run(file, kind, file.name), [run]);

  const onSample = useCallback(
    (sample: SampleInfo, index: number) =>
      void run(() => fetchSampleBlob(sample), 'sample', `sample ${index + 1}`, sample),
    [run],
  );

  const onAnother = useCallback(() => {
    reset();
    setActiveSample(undefined);
    const upload = document.getElementById('upload');
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    upload?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'center' });
    upload?.querySelector<HTMLButtonElement>('button')?.focus({ preventScroll: true });
  }, [reset]);

  return (
    <>
      <a
        href="#main"
        className="sr-only z-50 rounded-lg bg-accent px-4 py-2 text-on-accent focus:not-sr-only focus:fixed focus:left-4 focus:top-4"
      >
        Skip to content
      </a>
      <div id="top" />
      <Header theme={theme} onToggleTheme={toggle} />

      <main id="main" className="mx-auto max-w-6xl space-y-16 px-4 pt-10 sm:px-6 sm:pt-14">
        <section id="classify" aria-labelledby="hero-heading" className="scroll-mt-20 space-y-8">
          <div className="grid items-start gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] lg:gap-12">
            <div className="space-y-6 lg:pt-2">
              <p className="eyebrow">01 · Classify a photo</p>
              <h1
                id="hero-heading"
                className="text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl"
              >
                What do these droppings <em className="font-medium text-accent-ink">resemble</em>?
              </h1>
              <p className="max-w-xl text-lg text-ink-2">
                A photo of chicken droppings goes in; a ResNet50 model says which of four classes it looks
                most like: Coccidiosis, Salmonella, Newcastle disease or Healthy, and shows where it looked.
                It is right on {formatPercent(INTERNAL.accuracy.value)} of held-out photos from its training
                data but only {formatPercent(EXTERNAL.accuracy.value)} of separately collected, PCR-confirmed
                ones.
              </p>
              <div
                className="flex gap-3 rounded-xl border border-uncertain-line/60 bg-uncertain-soft px-4 py-3 text-sm text-uncertain-ink"
                role="note"
              >
                <AlertIcon className="mt-0.5 shrink-0 text-base" />
                <p>
                  <strong className="font-semibold">Research prototype.</strong> {NOT_A_DIAGNOSIS}
                </p>
              </div>
              <PhotoGuidelines open={guidelinesOpen} onToggle={setGuidelinesOpen} />
            </div>
            <UploadCard
              busy={busy}
              remote={inferenceMode() === 'api'}
              error={job.status === 'error' ? job.message : undefined}
              onFile={onFile}
              status={<ModelStatus model={model} lastInferenceMs={lastInferenceMs} onRetry={retryModel} />}
            />
          </div>

          <ResultPanel job={job} onAnother={onAnother} />

          <SampleGallery
            samples={samples}
            revealed={revealed}
            activeId={activeSample}
            busy={busy}
            onPick={onSample}
          />
        </section>

        <AboutModel />
      </main>

      <Footer />
      <p aria-live="polite" className="sr-only" data-testid="live-region">
        {announcement(job)}
      </p>
    </>
  );
}
