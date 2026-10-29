"""95% bootstrap confidence intervals for accuracy, macro-F1 and per-class recall/precision/F1.

Post-hoc analysis of the saved predictions (results/*_predictions.csv); the model is not re-run.

* Stratified bootstrap: each resample draws, within every true class, as many images as that class has
  (with replacement), so class counts stay fixed as in the evaluated set.
* 10,000 resamples, seed 42, percentile intervals (2.5th and 97.5th percentiles).
* The internal-minus-external gap uses independent resamples of the two sets (they share no images).

Outputs: results/posthoc/bootstrap_ci.json and results/posthoc/bootstrap_ci.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from posthoc_common import NUM_CLASSES, POSTHOC, SEED, SETS, CLASS_NAMES, load_predictions, save_json

N_RESAMPLES = 10_000
ALPHA = 0.05


def confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.bincount(y * NUM_CLASSES + pred, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def scores(cm: np.ndarray) -> dict[str, np.ndarray | float]:
    """Metrics from a confusion matrix, with sklearn's zero_division=0 convention."""
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {
        "accuracy": tp.sum() / cm.sum(),
        "macro_f1": f1.mean(),
        "weighted_f1": float((f1 * support).sum() / support.sum()),
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def bootstrap(y: np.ndarray, pred: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    by_class = [np.flatnonzero(y == c) for c in range(NUM_CLASSES)]
    out = {k: [] for k in ("accuracy", "macro_f1", "weighted_f1", "recall", "precision", "f1")}
    for _ in range(N_RESAMPLES):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_class])
        s = scores(confusion(y[idx], pred[idx]))
        for k in out:
            out[k].append(s[k])
    return {k: np.asarray(v) for k, v in out.items()}


def interval(samples: np.ndarray) -> list[float]:
    lo, hi = np.percentile(samples, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)], axis=0)
    return [float(lo), float(hi)] if np.ndim(lo) == 0 else [lo.tolist(), hi.tolist()]


def main() -> int:
    rng = np.random.default_rng(SEED)
    result = {
        "method": (
            f"Stratified bootstrap of saved predictions: {N_RESAMPLES:,} resamples, seed {SEED}, resampling "
            "with replacement within each true class; 95% percentile intervals. Post-hoc analysis, "
            "the model was not re-run."
        ),
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "sets": {},
    }
    draws, rows = {}, []
    for key, (label, path) in SETS.items():
        y, probs = load_predictions(path)
        pred = probs.argmax(axis=1)
        point = scores(confusion(y, pred))
        boot = bootstrap(y, pred, rng)
        draws[key] = boot
        entry = {"label": label, "n": int(len(y)), "metrics": {}, "per_class": {}}
        for m in ("accuracy", "macro_f1", "weighted_f1"):
            lo, hi = interval(boot[m])
            entry["metrics"][m] = {"value": float(point[m]), "ci95": [lo, hi]}
            rows.append({"set": key, "metric": m, "class": "", "value": float(point[m]), "ci_low": lo, "ci_high": hi})
        for m in ("precision", "recall", "f1"):
            lo, hi = interval(boot[m])
            for i, c in enumerate(CLASS_NAMES):
                entry["per_class"].setdefault(c, {})[m] = {"value": float(point[m][i]), "ci95": [lo[i], hi[i]]}
                rows.append(
                    {
                        "set": key,
                        "metric": m,
                        "class": c,
                        "value": float(point[m][i]),
                        "ci_low": lo[i],
                        "ci_high": hi[i],
                    }
                )
        result["sets"][key] = entry

    gap = {}
    for m in ("accuracy", "macro_f1"):
        diff = draws["internal"][m] - draws["external"][m]
        value = result["sets"]["internal"]["metrics"][m]["value"] - result["sets"]["external"]["metrics"][m]["value"]
        lo, hi = interval(diff)
        gap[m] = {"value": value, "ci95": [lo, hi]}
        rows.append(
            {"set": "internal_minus_external", "metric": m, "class": "", "value": value, "ci_low": lo, "ci_high": hi}
        )
    result["internal_minus_external"] = gap

    save_json(result, POSTHOC / "bootstrap_ci.json")
    pd.DataFrame(rows).to_csv(POSTHOC / "bootstrap_ci.csv", index=False, float_format="%.6f", lineterminator="\n")
    for key, entry in result["sets"].items():
        for m, v in entry["metrics"].items():
            print(f"{key:>9} {m:<12} {v['value']:.4f}  95% CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")
    for m, v in gap.items():
        print(f"      gap {m:<12} {v['value']:.4f}  95% CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
