/**
 * Probabilities -> a cautious verdict, the same rule as src/poultry_disease/verdict.py:
 *  - "uncertain" when the top probability or the top-2 margin is below the thresholds (a usability heuristic
 *    chosen on the internal test set, see results/posthoc/thresholds.json);
 *  - otherwise "disease_pattern" (Coccidiosis, Salmonella, NCD) or "no_disease_pattern" (Healthy).
 * None of this is a veterinary diagnosis; the copy says so.
 */
import { CLASS_IDS, CLASS_INFO, type ClassId } from './classes';
import { EXTERNAL, THRESHOLDS, formatPercent } from './metrics';

export type Category = 'disease_pattern' | 'no_disease_pattern' | 'uncertain';
export type Reason = 'low_top_probability' | 'small_margin';

export interface Ranked {
  id: ClassId;
  probability: number;
}

export interface Verdict {
  category: Category;
  top: Ranked;
  runnerUp: Ranked;
  /** all classes, most likely first (ties keep model order) */
  ranked: Ranked[];
  margin: number;
  reasons: Reason[];
  lowConfidence: boolean;
  /** Newcastle disease is the most likely class */
  notifiable: boolean;
}

export function interpret(
  probabilities: readonly number[],
  minTop = THRESHOLDS.minTopProbability,
  minMargin = THRESHOLDS.minTop2Margin,
): Verdict {
  if (probabilities.length !== CLASS_IDS.length) {
    throw new Error(`expected ${CLASS_IDS.length} probabilities, got ${probabilities.length}`);
  }
  const ranked = CLASS_IDS.map((id, i) => ({ id, probability: probabilities[i], i }))
    .sort((a, b) => b.probability - a.probability || a.i - b.i)
    .map(({ id, probability }) => ({ id, probability }));
  const [top, runnerUp] = ranked;
  const margin = top.probability - runnerUp.probability;
  const reasons: Reason[] = [];
  if (top.probability < minTop) reasons.push('low_top_probability');
  if (margin < minMargin) reasons.push('small_margin');
  const category: Category = reasons.length
    ? 'uncertain'
    : top.id === 'Healthy'
      ? 'no_disease_pattern'
      : 'disease_pattern';
  return {
    category,
    top,
    runnerUp,
    ranked,
    margin,
    reasons,
    lowConfidence: reasons.length > 0,
    notifiable: top.id === 'NCD',
  };
}

export interface VerdictCopy {
  /** short heading, never "Diagnosis" */
  title: string;
  /** one or two sentences under the heading */
  body: string;
  /** extra cautions shown as a list */
  notes: string[];
  /** shown prominently whenever Newcastle disease is the most likely class, whatever the category */
  notice?: string;
}

const healthyPrecision = formatPercent(EXTERNAL.perClass.Healthy.precision);
const pct = (p: number) => formatPercent(p, p > 0.995 || p < 0.005 ? 1 : 0);

export const NOT_A_DIAGNOSIS =
  'This is an image-based classification prototype, not a veterinary diagnosis. It does not replace a veterinarian or a laboratory test.';

export const NCD_NOTICE =
  'Newcastle disease is notifiable in many countries. If you suspect it, contact a veterinarian or your veterinary authority promptly.';

export function verdictCopy(v: Verdict): VerdictCopy {
  const top = CLASS_INFO[v.top.id];
  const second = CLASS_INFO[v.runnerUp.id];
  const notice = v.notifiable ? NCD_NOTICE : undefined;

  if (v.category === 'uncertain') {
    return {
      title: 'Uncertain: no verdict shown',
      body:
        `The model is split between ${top.name} (${pct(v.top.probability)}) and ${second.name} ` +
        `(${pct(v.runnerUp.probability)}). A verdict is shown only when the top class reaches ` +
        `${formatPercent(THRESHOLDS.minTopProbability)} and leads the runner-up by at least ` +
        `${Math.round(THRESHOLDS.minTop2Margin * 100)} percentage points.`,
      notes: [
        'These thresholds are a usability heuristic chosen on the internal test set, not a validated operating point.',
        'Try another photo of the same droppings: close up, in focus, filling most of the frame.',
      ],
      notice,
    };
  }

  if (v.category === 'no_disease_pattern') {
    return {
      title: 'No disease pattern recognised',
      body:
        `The photo most resembles the Healthy class (${pct(v.top.probability)}). That is not a clean bill of ` +
        `health: on the separately collected, PCR-confirmed photos, only ${healthyPrecision} of the images the model ` +
        `called Healthy really were.`,
      notes: ['If birds look or behave unwell, ask a veterinarian, whatever this result says.'],
      notice,
    };
  }

  return {
    title: `Pattern consistent with ${top.name}`,
    body:
      `The droppings in this photo most resemble the ${top.name} class (${pct(v.top.probability)}). ` +
      'Only a veterinarian, with a laboratory test, can confirm an infection.',
    notes: [],
    notice,
  };
}
