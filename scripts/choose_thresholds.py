"""Choose the "uncertain" thresholds of the app and the API from saved prediction distributions.

A usability heuristic, not a validated operating point. A prediction is shown as "uncertain" when the top
probability is below T or the gap between the two most likely classes is below M.

Rule (decided before looking at the external set's effect, and applied to the internal test set only):
    among T in {0.50, 0.55, ..., 0.95, 0.99} and M in {0.00, 0.05, ..., 0.95}, take the pair that withholds
    the most internal *errors* while withholding at most 5% of internal *correct* predictions;
    ties go to the smaller T, then the smaller M.
The 5% budget limits how often a photo like the training data is refused. The external PCR set is used only
to report what the chosen thresholds do under domain shift, never to choose them.

Outputs: src/poultry_disease/data/decision_thresholds.json (read by the package, the API and the web app),
results/posthoc/thresholds.json, results/posthoc/risk_coverage.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from posthoc_common import POSTHOC, ROOT, SET_COLORS, SETS, load_predictions, save_figure, save_json, setup_style

MAX_CORRECT_WITHHELD = 0.05
T_GRID = [round(t, 2) for t in np.arange(0.50, 0.951, 0.05)] + [0.99]
M_GRID = [round(m, 2) for m in np.arange(0.0, 0.951, 0.05)]


def top2(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.sort(probs, axis=1)
    return s[:, -1], s[:, -1] - s[:, -2]


def effect(top, margin, correct, t, m) -> dict:
    uncertain = (top < t) | (margin < m)
    kept = ~uncertain
    return {
        "coverage": float(kept.mean()),
        "accuracy_of_shown_verdicts": float(correct[kept].mean()) if kept.any() else None,
        "errors_withheld": int((~correct & uncertain).sum()),
        "errors_total": int((~correct).sum()),
        "share_of_errors_withheld": float((~correct & uncertain).sum() / max((~correct).sum(), 1)),
        "share_of_correct_withheld": float((correct & uncertain).sum() / max(correct.sum(), 1)),
    }


def main() -> int:
    setup_style()
    data = {}
    for key, (label, path) in SETS.items():
        y, probs = load_predictions(path)
        top, margin = top2(probs)
        data[key] = (label, top, margin, probs.argmax(axis=1) == y)

    _, top, margin, correct = data["internal"]
    best = None
    for t in T_GRID:
        for m in M_GRID:
            e = effect(top, margin, correct, t, m)
            if e["share_of_correct_withheld"] > MAX_CORRECT_WITHHELD:
                continue
            key = (e["errors_withheld"], -t, -m)
            if best is None or key > best[0]:
                best = (key, t, m)
    _, t_star, m_star = best

    effects = {k: {"label": v[0], **effect(v[1], v[2], v[3], t_star, m_star)} for k, v in data.items()}
    decision = {
        "min_top_probability": t_star,
        "min_top2_margin": m_star,
        "status": "usability heuristic, not a validated operating point",
        "chosen_on": "internal test set (version 2 held-out) saved predictions",
        "rule": (
            f"Withhold the most internal errors while withholding at most {MAX_CORRECT_WITHHELD:.0%} of internal "
            "correct predictions; grid T 0.50-0.95 step 0.05 plus 0.99, M 0.00-0.95 step 0.05."
        ),
        "source": "scripts/choose_thresholds.py",
    }
    save_json(decision, ROOT / "src" / "poultry_disease" / "data" / "decision_thresholds.json")
    save_json({**decision, "effect": effects}, POSTHOC / "thresholds.json")

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for key, (label, _, mg, ok) in data.items():
        order = np.argsort(-mg)
        cov = np.arange(1, len(mg) + 1) / len(mg)
        risk = 1 - np.cumsum(ok[order]) / np.arange(1, len(mg) + 1)
        ax.plot(cov, risk, color=SET_COLORS[key], label=label)
        e = effects[key]
        ax.plot(e["coverage"], 1 - e["accuracy_of_shown_verdicts"], "o", color=SET_COLORS[key], mec="k", ms=8)
        ax.annotate(
            f"M = {m_star:.2f}: shows {e['coverage']:.0%},\nerror {1 - e['accuracy_of_shown_verdicts']:.1%}",
            (e["coverage"], 1 - e["accuracy_of_shown_verdicts"]),
            textcoords="offset points",
            xytext=(-130, 12),
            fontsize=8,
        )
    ax.set(
        title="Risk-coverage (sorted by top-2 margin)",
        xlabel="Coverage (share of verdicts shown)",
        ylabel="Error rate among shown verdicts",
        xlim=(0, 1.02),
        ylim=(0, 0.3),
    )
    ax.legend(fontsize=8, loc="upper left")
    save_figure(fig, "risk_coverage.png")

    print(f"chosen: min_top_probability={t_star}, min_top2_margin={m_star}")
    for k, e in effects.items():
        print(
            f"{k:>9}: shows {e['coverage']:.1%}, accuracy of shown {e['accuracy_of_shown_verdicts']:.4f}, "
            f"errors withheld {e['errors_withheld']}/{e['errors_total']}, "
            f"correct withheld {e['share_of_correct_withheld']:.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
