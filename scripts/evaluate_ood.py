"""Run the model on the out-of-distribution photos in docs/ood/ and write predictions.json and README.md.

Uses the Python package (the training preprocessing) and the app's verdict rule, so "verdict shown" means
the web app and the API would display a disease or no-disease verdict instead of "uncertain".

Usage:
    python scripts/fetch_ood_images.py     # once
    python scripts/evaluate_ood.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OOD = ROOT / "docs" / "ood"


def pct(p: float) -> str:
    return f"{100 * p:.1f}%"


def main() -> int:
    from poultry_disease import CLASS_NAMES, Predictor
    from poultry_disease.verdict import MIN_TOP2_MARGIN, MIN_TOP_PROBABILITY, interpret
    from poultry_disease.weights import get_weights

    credits = json.loads((OOD / "credits.json").read_text("utf-8"))
    predictor = Predictor(get_weights(allow_download=False))
    rows = []
    for c in credits["photos"]:
        pred = predictor.predict_image(OOD / c["file"])
        v = interpret(pred.probs.tolist())
        rows.append(
            {
                "file": c["file"],
                "group": c["group"],
                "description": c["description"],
                "probabilities": {k: float(p) for k, p in zip(CLASS_NAMES, pred.probs, strict=True)},
                "top_class": v.top_class,
                "top_probability": v.top_probability,
                "margin": v.margin,
                "category": v.category,
                "verdict_shown": v.category != "uncertain",
            }
        )
        print(f"{c['file']:<26} {v.top_class:<12} {v.top_probability:.4f} {v.category}")

    groups = {}
    for g in dict.fromkeys(r["group"] for r in rows):
        sub = [r for r in rows if r["group"] == g]
        groups[g] = {"n": len(sub), "verdict_shown": sum(r["verdict_shown"] for r in sub)}
    summary = {
        "n": len(rows),
        "verdict_shown": sum(r["verdict_shown"] for r in rows),
        "top_probability_ge_0_9": sum(r["top_probability"] >= 0.9 for r in rows),
        "by_group": groups,
        "thresholds": {"min_top_probability": MIN_TOP_PROBABILITY, "min_top2_margin": MIN_TOP2_MARGIN},
    }
    doc = {
        "description": (
            "Predictions of the published model on openly licensed photos that are not poultry droppings "
            "(docs/ood/credits.json). The model has no 'not droppings' class. Post-hoc analysis."
        ),
        "summary": summary,
        "photos": rows,
    }
    (OOD / "predictions.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")

    by_file = {c["file"]: c for c in credits["photos"]}
    lines = [
        "# Photos that are not poultry droppings",
        "",
        'The classifier has no "not poultry droppings" class, so it always answers with Coccidiosis, Healthy, '
        "Salmonella or NCD. To see what it does with photos it was never meant for, it was run on "
        f"{len(rows)} openly licensed photos from Wikimedia Commons in three groups: poultry-farm scenes without "
        "droppings, droppings of other animals, and food that looks similar. Predictions come from "
        "[`scripts/evaluate_ood.py`](../../scripts/evaluate_ood.py) (the Python package, training "
        "preprocessing) and are stored in [`predictions.json`](predictions.json).",
        "",
        '"Verdict shown" means the app would display a verdict instead of "uncertain" (top probability at '
        f"least {MIN_TOP_PROBABILITY:.0%} and a lead of at least {MIN_TOP2_MARGIN * 100:.0f} percentage points "
        "over the runner-up).",
        "",
        "| # | Photo | Group | Top class | Top probability | Verdict shown |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        shown = "**yes**" if r["verdict_shown"] else "no (uncertain)"
        lines.append(
            f"| {i} | [{r['description']}]({r['file']}) | {r['group']} | {r['top_class']} | "
            f"{pct(r['top_probability'])} | {shown} |"
        )
    lines += [
        "",
        f"**Summary.** A verdict would be shown for {summary['verdict_shown']} of {summary['n']} photos; "
        f"{summary['top_probability_ge_0_9']} of {summary['n']} get a top probability of 90% or more. By group: "
        + "; ".join(f"{g}, {s['verdict_shown']} of {s['n']}" for g, s in groups.items())
        + ". The uncertain state catches some of these photos, but it is no substitute for a check that the photo "
        "shows poultry droppings at all; only the person taking the photo can make that check.",
        "",
        "## Credits",
        "",
        credits["note"],
        "",
        "| File | Author | License | Source |",
        "|---|---|---|---|",
    ]
    for r in rows:
        c = by_file[r["file"]]
        lic = f"[{c['license']}]({c['license_url']})" if c["license_url"] else c["license"]
        lines.append(f"| `{c['file']}` | {c['author']} | {lic} | [Wikimedia Commons]({c['source']}) |")
    lines += ["", "These photos keep their own licenses; the repository's MIT license does not apply to them.", ""]
    (OOD / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
