/**
 * Every sentence the app shows about the input check. The rates come from results/input_check/summary.json
 * (scripts/build_input_check.py, synced to src/data/); nothing here hard-codes a number.
 */
import summary from '../data/input_check_summary.json';
import { formatPercent } from './metrics';

const p1 = (x: number) => formatPercent(x, 1);

export const INPUT_CHECK_SUMMARY = summary;

export const INPUT_CHECK_COPY = {
  heading: "This doesn't look like a photo of poultry droppings.",
  why: 'The model only knows four kinds of chicken droppings, so its answer for this photo would not mean anything.',
  tip: 'Photograph one dropping close up from above, so it fills much of the frame, in daylight or bright, even light.',
  retry: 'Try another photo',
  tips: 'Photo tips',
  anyway: 'Analyse anyway',
  uncheckedLabel:
    'Unchecked input: this photo did not look like poultry droppings, so treat the result with extra caution.',
} as const;

/** Shown under the notice: the check's own error rates, so nobody takes it as a guarantee. */
export function safeguardNote(): string {
  const fr = summary.false_rejection;
  return (
    'The check is a safeguard, not a guarantee. In testing it turned away ' +
    `${p1(fr.test_internal.rate)} of held-out droppings photos and ${p1(fr.test_external.rate)} of ` +
    `PCR-confirmed ones, and let through ${p1(summary.false_acceptance.near.rate)} of look-alike photos such ` +
    "as other animals' droppings. If your photo does show poultry droppings, use Analyse anyway."
  );
}

/** Text for the polite live region when the check turns a photo away. */
export function rejectionAnnouncement(): string {
  return `${INPUT_CHECK_COPY.heading} ${INPUT_CHECK_COPY.why} No result is shown. Options: ${INPUT_CHECK_COPY.retry}, ${INPUT_CHECK_COPY.tips} or ${INPUT_CHECK_COPY.anyway}.`;
}
