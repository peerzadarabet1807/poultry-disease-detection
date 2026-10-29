/**
 * Typed access to the committed results (synced into src/data by scripts/sync-results.mjs).
 * Every number the UI shows comes from here.
 */
import bootstrap from '../data/bootstrap_ci.json';
import calibration from '../data/calibration.json';
import metrics from '../data/metrics.json';
import runVariability from '../data/run_variability.json';
import thresholds from '../data/thresholds.json';
import { CLASS_IDS, type ClassId } from './classes';

export type SetKey = 'internal' | 'external';

interface Interval {
  value: number;
  ci95: [number, number];
}

export interface SetSummary {
  key: SetKey;
  label: string;
  n: number;
  accuracy: Interval;
  macroF1: Interval;
  /** rows = true class, columns = predicted class, model order */
  confusion: number[][];
  perClass: Record<ClassId, { precision: number; recall: number; f1: number; support: number }>;
  ece: number;
  meanConfidence: number;
  shareConfidenceGe099: number;
  accuracyWhenConfidenceGe099: number;
  rocAuc: number;
}

function summary(key: SetKey): SetSummary {
  const m = key === 'internal' ? metrics.test : metrics.pcr_evaluation;
  const b = bootstrap.sets[key];
  const c = calibration.sets[key];
  return {
    key,
    label: key === 'internal' ? 'Internal test' : 'External PCR-confirmed',
    n: m.n_samples,
    accuracy: { value: b.metrics.accuracy.value, ci95: b.metrics.accuracy.ci95 as [number, number] },
    macroF1: { value: b.metrics.macro_f1.value, ci95: b.metrics.macro_f1.ci95 as [number, number] },
    confusion: m.confusion_matrix,
    perClass: Object.fromEntries(CLASS_IDS.map((id) => [id, m.per_class[id]])) as SetSummary['perClass'],
    ece: c.ece,
    meanConfidence: c.mean_confidence,
    shareConfidenceGe099: c.share_confidence_ge_0_99,
    accuracyWhenConfidenceGe099: c.accuracy_when_confidence_ge_0_99,
    rocAuc: m.roc_auc_ovr_macro,
  };
}

export const INTERNAL = summary('internal');
export const EXTERNAL = summary('external');

export const THRESHOLDS = {
  minTopProbability: thresholds.min_top_probability,
  minTop2Margin: thresholds.min_top2_margin,
  rule: thresholds.rule,
  internal: thresholds.effect.internal,
  external: thresholds.effect.external,
};

export const RUNS = {
  count: runVariability.runs.length,
  rank: runVariability.published_run_external_macro_f1_rank,
  externalMacroF1: runVariability.summary.external_macro_f1,
  externalAccuracy: runVariability.summary.external_accuracy,
  externalAgreement: runVariability.prediction_agreement.external.all_runs_agree,
  internalAgreement: runVariability.prediction_agreement.internal.all_runs_agree,
};

/** 0.7222 -> "72%", with `digits` decimals. */
export function formatPercent(p: number, digits = 0): string {
  return `${(p * 100).toFixed(digits)}%`;
}

export const formatInterval = (i: Interval, digits = 1) =>
  `${formatPercent(i.ci95[0], digits)} to ${formatPercent(i.ci95[1], digits)}`;
