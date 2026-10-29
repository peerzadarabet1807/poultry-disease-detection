import ood from '../data/ood.json';
import { EXTERNAL, formatPercent, INTERNAL, RUNS } from '../lib/metrics';

const p1 = (x: number) => formatPercent(x, 1);

export function Limitations() {
  const items: [string, string][] = [
    [
      'It is much weaker on new photos.',
      `${p1(INTERNAL.accuracy.value)} accuracy on held-out photos from the training data, ${p1(EXTERNAL.accuracy.value)} on the separately collected, PCR-confirmed set. Expect the lower number, or worse, on your own photos.`,
    ],
    [
      'The classes were photographed in different settings.',
      'In the training data, Healthy and Coccidiosis photos come from farms, while Salmonella and Newcastle disease photos come from experimentally infected birds. The model may have learnt the floor or framing as much as the droppings.',
    ],
    [
      'Most training labels are not laboratory results.',
      'Version 2 labels come from where and how the photos were collected; only the separate version 3 photos were confirmed by PCR, and they were never used for training.',
    ],
    [
      'One region, one season.',
      'All photos were taken with phones in the Arusha and Kilimanjaro regions of Tanzania between September 2020 and February 2021.',
    ],
    [
      'Newcastle disease is rare in the data.',
      `Only ${INTERNAL.perClass.NCD.support} internal test photos and ${EXTERNAL.perClass.NCD.support} PCR-confirmed photos show it, so its numbers are the least certain.`,
    ],
    [
      'This is the best of several runs.',
      `${RUNS.count} identical training runs differ only by GPU randomness; this one has the best PCR-set score, ${p1(RUNS.externalAccuracy.max)} accuracy against a mean of ${p1(RUNS.externalAccuracy.mean)}.`,
    ],
    [
      'Its percentages are overconfident.',
      `On the PCR set the average top probability is ${p1(EXTERNAL.meanConfidence)} but only ${p1(EXTERNAL.accuracy.value)} of answers are right.`,
    ],
    [
      'It always picks one of four classes.',
      `It cannot say "this is not poultry droppings": ${ood.summary.verdict_shown} of ${ood.summary.n} unrelated photos still got a confident verdict.`,
    ],
    [
      'No veterinary or field validation.',
      'It has not been tested by veterinarians, on farms, or against outcomes. It must not be used to decide on treatment, culling or reporting.',
    ],
  ];
  return (
    <div id="limitations" className="card scroll-mt-20 space-y-4 p-4 sm:p-6">
      <h3 className="text-xl font-semibold">Limitations</h3>
      <ol className="grid list-decimal gap-x-8 gap-y-3 pl-5 text-sm text-ink-2 marker:font-semibold marker:text-ink-3 md:grid-cols-2">
        {items.map(([title, body]) => (
          <li key={title}>
            <strong className="font-semibold text-ink">{title}</strong> {body}
          </li>
        ))}
      </ol>
    </div>
  );
}
