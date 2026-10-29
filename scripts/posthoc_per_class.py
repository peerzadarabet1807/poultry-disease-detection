"""Per-class precision, recall and F1, internal vs external, with 95% bootstrap intervals.

Reads results/posthoc/bootstrap_ci.json (run posthoc_bootstrap.py first) and the notebook's confusion
matrices in results/metrics.json. Post-hoc analysis of saved predictions.

Outputs: results/posthoc/per_class_internal_vs_external.png, results/posthoc/external_confusions.json
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from posthoc_common import CLASS_NAMES, POSTHOC, RESULTS, SET_COLORS, save_figure, save_json, setup_style


def main() -> int:
    setup_style()
    ci = json.loads((POSTHOC / "bootstrap_ci.json").read_text("utf-8"))
    metrics = json.loads((RESULTS / "metrics.json").read_text("utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    x = np.arange(len(CLASS_NAMES))
    width = 0.38
    for ax, m in zip(axes, ("precision", "recall", "f1"), strict=False):
        for key, offset in (("internal", -width / 2), ("external", width / 2)):
            pc = ci["sets"][key]["per_class"]
            vals = np.array([pc[c][m]["value"] for c in CLASS_NAMES])
            lo = np.array([pc[c][m]["ci95"][0] for c in CLASS_NAMES])
            hi = np.array([pc[c][m]["ci95"][1] for c in CLASS_NAMES])
            bars = ax.bar(x + offset, vals, width, color=SET_COLORS[key], label=ci["sets"][key]["label"])
            ax.errorbar(x + offset, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor="black", lw=1, capsize=3)
            for b, v in zip(bars, vals, strict=False):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    0.03,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
        ax.set_xticks(x, ["NCD" if c == "NCD" else c for c in CLASS_NAMES])
        ax.set(title={"precision": "Precision", "recall": "Recall", "f1": "F1"}[m], ylim=(0, 1.12))
    axes[0].set_ylabel("Score (95% bootstrap CI)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Per-class performance: internal vs external (post-hoc, saved predictions)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False)
    save_figure(fig, "per_class_internal_vs_external.png")

    # Where the external errors go, as row-normalised shares (from the notebook's confusion matrix).
    cm = np.asarray(metrics["pcr_evaluation"]["confusion_matrix"], dtype=float)
    rows = cm / cm.sum(axis=1, keepdims=True)
    confusions = [
        {
            "true": CLASS_NAMES[t],
            "predicted": CLASS_NAMES[p],
            "count": int(cm[t, p]),
            "of_true_class": int(cm[t].sum()),
            "share": float(rows[t, p]),
        }
        for t in range(len(CLASS_NAMES))
        for p in range(len(CLASS_NAMES))
        if t != p and cm[t, p] > 0
    ]
    confusions.sort(key=lambda r: -r["share"])
    save_json(
        {"source": "results/metrics.json pcr_evaluation.confusion_matrix", "confusions": confusions},
        POSTHOC / "external_confusions.json",
    )
    for r in confusions[:5]:
        print(f"{r['true']:>11} -> {r['predicted']:<11} {r['count']:>3}/{r['of_true_class']} = {r['share']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
