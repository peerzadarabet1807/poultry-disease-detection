"""Calibration of the top-class confidence on the internal and external sets.

Post-hoc analysis of the saved predictions. For each set:
* Expected calibration error (ECE): 15 equal-width confidence bins on [0, 1],
  ECE = sum_b (n_b / n) * |accuracy_b - mean confidence_b|. Also the maximum calibration error (MCE).
* Multiclass Brier score and log loss (log loss matches the notebook's metrics.json).
* A 95% bootstrap interval for ECE (2,000 stratified resamples, seed 42).

Outputs: results/posthoc/calibration.json, results/posthoc/reliability_diagram.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from posthoc_common import (
    NUM_CLASSES,
    POSTHOC,
    SEED,
    SET_COLORS,
    SETS,
    load_predictions,
    save_figure,
    save_json,
    setup_style,
)

N_BINS = 15
N_BOOT = 2_000


def reliability(conf: np.ndarray, correct: np.ndarray) -> dict:
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    # right-closed bins, the first one also includes 0
    bins = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, N_BINS - 1)
    rows, ece, mce = [], 0.0, 0.0
    for b in range(N_BINS):
        m = bins == b
        if not m.any():
            rows.append({"lo": edges[b], "hi": edges[b + 1], "n": 0, "accuracy": None, "confidence": None})
            continue
        acc, cnf = float(correct[m].mean()), float(conf[m].mean())
        gap = abs(acc - cnf)
        ece += m.mean() * gap
        mce = max(mce, gap)
        rows.append({"lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()), "accuracy": acc, "confidence": cnf})
    return {"ece": float(ece), "mce": float(mce), "bins": rows}


def main() -> int:
    setup_style()
    rng = np.random.default_rng(SEED)
    out = {
        "method": (
            f"Top-label calibration of saved predictions, {N_BINS} equal-width bins. ECE interval: "
            f"{N_BOOT:,} stratified bootstrap resamples, seed {SEED}. Post-hoc analysis."
        ),
        "sets": {},
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for key, (label, path) in SETS.items():
        y, probs = load_predictions(path)
        conf, pred = probs.max(axis=1), probs.argmax(axis=1)
        correct = (pred == y).astype(float)
        rel = reliability(conf, correct)
        onehot = np.eye(NUM_CLASSES)[y]
        brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
        nll = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1)).mean())

        by_class = [np.flatnonzero(y == c) for c in range(NUM_CLASSES)]
        eces = []
        for _ in range(N_BOOT):
            idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_class])
            eces.append(reliability(conf[idx], correct[idx])["ece"])
        lo, hi = np.percentile(eces, [2.5, 97.5])

        out["sets"][key] = {
            "label": label,
            "n": int(len(y)),
            "accuracy": float(correct.mean()),
            "mean_confidence": float(conf.mean()),
            "overconfidence": float(conf.mean() - correct.mean()),
            "ece": rel["ece"],
            "ece_ci95": [float(lo), float(hi)],
            "mce": rel["mce"],
            "brier": brier,
            "log_loss": nll,
            "share_confidence_ge_0_99": float((conf >= 0.99).mean()),
            "accuracy_when_confidence_ge_0_99": float(correct[conf >= 0.99].mean()) if (conf >= 0.99).any() else None,
            "bins": rel["bins"],
        }

        ax = axes[0]
        pts = [(b["confidence"], b["accuracy"], b["n"]) for b in rel["bins"] if b["n"]]
        cx, ax_, n = (np.array(v) for v in zip(*pts, strict=False))
        ax.plot(cx, ax_, "-", color=SET_COLORS[key], lw=1, alpha=0.6)
        ax.scatter(
            cx,
            ax_,
            s=8 + 300 * np.sqrt(n / len(y)),
            color=SET_COLORS[key],
            edgecolor="white",
            lw=0.5,
            zorder=3,
            label=f"{label}\nECE {rel['ece']:.3f} [{lo:.3f}, {hi:.3f}]",
        )
        for xi, yi, ni in zip(cx, ax_, n, strict=False):
            if ni < 30:
                ax.annotate(
                    f"n={ni}", (xi, yi), textcoords="offset points", xytext=(4, -10), fontsize=6, color=SET_COLORS[key]
                )
        axes[1].hist(
            conf,
            bins=np.linspace(0, 1, N_BINS + 1),
            alpha=0.6,
            color=SET_COLORS[key],
            label=label,
            weights=np.full(len(conf), 1 / len(conf)),
        )

    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    axes[0].set(
        title="Reliability diagram (marker area ~ images in bin)",
        xlabel="Mean confidence in bin",
        ylabel="Accuracy in bin",
        xlim=(0.2, 1.02),
        ylim=(0, 1.05),
    )
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set_yscale("log")
    axes[1].set(title="Confidence distribution (log scale)", xlabel="Top-class probability", ylabel="Share of images")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Calibration: internal vs external (post-hoc, saved predictions)")
    fig.tight_layout()
    save_figure(fig, "reliability_diagram.png")
    save_json(out, POSTHOC / "calibration.json")
    for key, s in out["sets"].items():
        print(
            f"{key:>9}: ECE {s['ece']:.4f} [{s['ece_ci95'][0]:.4f}, {s['ece_ci95'][1]:.4f}]  MCE {s['mce']:.3f}  "
            f"Brier {s['brier']:.4f}  log loss {s['log_loss']:.4f}  mean conf {s['mean_confidence']:.3f} "
            f"vs acc {s['accuracy']:.3f}; conf>=0.99: {s['share_confidence_ge_0_99']:.1%} of images, "
            f"accuracy {s['accuracy_when_confidence_ge_0_99']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
