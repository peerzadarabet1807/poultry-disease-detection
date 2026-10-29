import experiment from '../data/experiment_summary.json';
import confusions from '../data/external_confusions.json';
import ood from '../data/ood.json';
import { CLASS_IDS, CLASS_INFO } from '../lib/classes';
import { blob, PAPER_URL, ZENODO_V2, ZENODO_V3 } from '../lib/links';
import {
  EXTERNAL,
  formatInterval,
  formatPercent,
  INTERNAL,
  RUNS,
  THRESHOLDS,
  type SetSummary,
} from '../lib/metrics';
import { ConfusionMatrix } from './ConfusionMatrix';
import { Limitations } from './Limitations';

const p1 = (x: number) => formatPercent(x, 1);
const name = (id: string) => CLASS_INFO[id as keyof typeof CLASS_INFO].name;
const [worst, second] = confusions.confusions;

function EvalTable() {
  const sets: SetSummary[] = [INTERNAL, EXTERNAL];
  return (
    <div className="overflow-x-auto">
      <table className="tabular w-full min-w-[34rem] text-left text-sm">
        <caption className="sr-only">Evaluation of the published model on both test sets</caption>
        <thead className="border-b border-line-strong text-xs uppercase tracking-wide text-ink-3">
          <tr>
            <th scope="col" className="py-2 pr-3 font-semibold">
              Test set
            </th>
            <th scope="col" className="py-2 pr-3 font-semibold">
              Images
            </th>
            <th scope="col" className="py-2 pr-3 font-semibold">
              Accuracy (95% CI)
            </th>
            <th scope="col" className="py-2 pr-3 font-semibold">
              Macro-F1 (95% CI)
            </th>
            <th scope="col" className="py-2 font-semibold">
              Calibration error
            </th>
          </tr>
        </thead>
        <tbody>
          {sets.map((s) => (
            <tr key={s.key} className="border-b border-line align-top">
              <th scope="row" className="py-3 pr-3 font-semibold text-ink">
                {s.label}
                <span className="block text-xs font-normal text-ink-3">
                  {s.key === 'internal'
                    ? 'held-out photos from the training data (version 2)'
                    : 'collected separately, labelled by PCR (version 3)'}
                </span>
              </th>
              <td className="py-3 pr-3">{s.n.toLocaleString('en')}</td>
              <td className="py-3 pr-3">
                <strong className="text-ink">{p1(s.accuracy.value)}</strong>
                <span className="block text-xs text-ink-3">{formatInterval(s.accuracy)}</span>
              </td>
              <td className="py-3 pr-3">
                <strong className="text-ink">{p1(s.macroF1.value)}</strong>
                <span className="block text-xs text-ink-3">{formatInterval(s.macroF1)}</span>
              </td>
              <td className="py-3">{s.ece.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PerClassTable() {
  return (
    <div className="overflow-x-auto">
      <table className="tabular w-full min-w-[30rem] text-left text-sm">
        <caption className="sr-only">Recall and precision per class on both test sets</caption>
        <thead className="border-b border-line-strong text-xs uppercase tracking-wide text-ink-3">
          <tr>
            <th scope="col" className="py-2 pr-3 font-semibold">
              Class
            </th>
            <th scope="col" className="py-2 pr-3 font-semibold">
              Recall, internal
            </th>
            <th scope="col" className="py-2 pr-3 font-semibold">
              Recall, PCR set
            </th>
            <th scope="col" className="py-2 font-semibold">
              Precision, PCR set
            </th>
          </tr>
        </thead>
        <tbody>
          {CLASS_IDS.map((id) => (
            <tr key={id} className="border-b border-line">
              <th scope="row" className="py-2 pr-3 font-medium text-ink">
                {CLASS_INFO[id].name}
              </th>
              <td className="py-2 pr-3">{p1(INTERNAL.perClass[id].recall)}</td>
              <td className="py-2 pr-3">{p1(EXTERNAL.perClass[id].recall)}</td>
              <td className="py-2">{p1(EXTERNAL.perClass[id].precision)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function AboutModel() {
  const oodSummary = ood.summary;
  const otherDroppings = oodSummary.by_group['other droppings'];
  const farmShown = ood.photos.filter((x) => x.group === 'farm scene' && x.verdict_shown);
  return (
    <section id="about" aria-labelledby="about-heading" className="scroll-mt-20 space-y-10">
      <div className="max-w-3xl space-y-3">
        <p className="eyebrow">02 · About the model</p>
        <h2 id="about-heading" className="text-3xl font-semibold tracking-tight sm:text-4xl">
          Good on familiar photos, much weaker on new ones
        </h2>
        <div className="prose-tight text-lg text-ink-2">
          <p>
            A ResNet50 network, pretrained on ImageNet and fine-tuned on{' '}
            {experiment['Train samples'].toLocaleString('en')} photos from the dataset's version 2, classifies{' '}
            {p1(INTERNAL.accuracy.value)} of {INTERNAL.n.toLocaleString('en')} held-out photos from the same
            data correctly. On {EXTERNAL.n.toLocaleString('en')} photos collected separately and labelled by
            PCR, it gets {p1(EXTERNAL.accuracy.value)}. That second number is the one to keep in mind for real
            use.
          </p>
          <p>
            This run is the best of {RUNS.count} identical training runs on the PCR set; across all{' '}
            {RUNS.count}, accuracy there ranged from {p1(RUNS.externalAccuracy.min)} to{' '}
            {p1(RUNS.externalAccuracy.max)}.
          </p>
        </div>
      </div>

      <div className="card space-y-4 p-4 sm:p-6">
        <h3 className="text-xl font-semibold">Evaluation</h3>
        <EvalTable />
        <p className="text-sm text-ink-3">
          Intervals: stratified bootstrap of the saved predictions (10,000 resamples). Calibration error:
          expected calibration error (ECE), the gap between confidence and accuracy. On the PCR set the model
          is {p1(EXTERNAL.meanConfidence)} confident on average but {p1(EXTERNAL.accuracy.value)} accurate.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <div className="card space-y-4 p-4 sm:p-6">
          <h3 className="text-xl font-semibold">Where it goes wrong</h3>
          <div className="grid gap-6 md:grid-cols-2">
            <ConfusionMatrix set={INTERNAL} />
            <ConfusionMatrix set={EXTERNAL} />
          </div>
          <p className="text-sm text-ink-3">
            Rows are the dataset label, columns the model's answer. The most common mistakes on the PCR set:{' '}
            {name(worst.true)} called {name(worst.predicted)} ({p1(worst.share)} of {name(worst.true)} photos)
            and {name(second.true)} called {name(second.predicted)} ({p1(second.share)}).
          </p>
        </div>
        <div className="card space-y-4 p-4 sm:p-6">
          <h3 className="text-xl font-semibold">Per class</h3>
          <PerClassTable />
          <p className="text-sm text-ink-3">
            Precision on the PCR set is what a verdict is worth: when the model says Healthy, it is right{' '}
            {p1(EXTERNAL.perClass.Healthy.precision)} of the time.
          </p>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="card space-y-3 p-4 sm:p-6">
          <h3 className="text-xl font-semibold">Why some results say "uncertain"</h3>
          <p className="text-ink-2">
            A verdict is shown only when the top class reaches {formatPercent(THRESHOLDS.minTopProbability)}{' '}
            and leads the runner-up by {Math.round(THRESHOLDS.minTop2Margin * 100)} percentage points. The
            thresholds were chosen on the internal test set to hold back most of its errors, a usability
            heuristic rather than a validated operating point.
          </p>
          <p className="text-ink-2">
            On the PCR set this shows a verdict for {p1(THRESHOLDS.external.coverage)} of photos, and{' '}
            {p1(THRESHOLDS.external.accuracy_of_shown_verdicts)} of the shown verdicts are right; it holds
            back {THRESHOLDS.external.errors_withheld} of {THRESHOLDS.external.errors_total} errors.
          </p>
        </div>
        <div className="card space-y-3 p-4 sm:p-6" data-testid="ood-summary">
          <h3 className="text-xl font-semibold">It answers even when it should not</h3>
          <p className="text-ink-2">
            The model has no "this is not poultry droppings" option. On {oodSummary.n} openly licensed photos
            of other things, it would show a confident verdict for {oodSummary.verdict_shown}, including{' '}
            {otherDroppings.verdict_shown} of {otherDroppings.n} photos of other animals' droppings
            {farmShown.length > 0 &&
              `, and ${farmShown.map((x) => `${x.description.toLowerCase()} (${name(x.top_class)}, ${p1(x.top_probability)})`).join(' and ')}`}
            . Only use it on photos of poultry droppings.
          </p>
          <a className="link text-sm" href={blob('docs/ood/README.md')}>
            The photos and predictions
          </a>
        </div>
      </div>

      <Limitations />

      <div className="card space-y-3 p-4 text-sm text-ink-2 sm:p-6">
        <h3 className="text-xl font-semibold text-ink">Data and credit</h3>
        <p>
          Photos and labels come from Machuve, Nwankwo, Mduma, Mbelwa, Maguo, Munisi and Lyimo:{' '}
          <a className="link" href={ZENODO_V2}>
            Machine Learning Dataset for Poultry Diseases Diagnostics
          </a>{' '}
          (version 2) and its{' '}
          <a className="link" href={ZENODO_V3}>
            PCR-annotated version 3
          </a>
          , Zenodo, CC BY 4.0, funded by IDRC and OWSD. Related paper:{' '}
          <a className="link" href={PAPER_URL}>
            Machuve et al. 2022, Frontiers in Artificial Intelligence
          </a>
          . This app and model are independent work and not affiliated with the dataset authors.
        </p>
        <p>
          More:{' '}
          <a className="link" href={blob('models/MODEL_CARD.md')}>
            model card
          </a>
          ,{' '}
          <a className="link" href={blob('models/PARITY.md')}>
            parity report
          </a>{' '}
          (browser vs Python),{' '}
          <a className="link" href={blob('notebooks/poultry_disease_resnet50_training_executed.ipynb')}>
            training notebook
          </a>
          .
        </p>
      </div>
    </section>
  );
}
