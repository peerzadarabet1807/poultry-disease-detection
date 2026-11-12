"""Run the model on the original out-of-distribution photos in docs/ood/ and write predictions.json, README.md
and CREDITS.md.

Uses the Python package (the training preprocessing) and the app's verdict rule, so "verdict shown" means
the web app would display a disease or no-disease verdict instead of "uncertain". predictions.json covers the
12 original photos (unchanged since the first evaluation). The README also summarises the full 251-photo set
and the input check's decisions, read from docs/ood/input_check.json (scripts/build_input_check.py).

Usage:
    python scripts/fetch_ood_images.py     # once: the 12 original photos
    python scripts/build_ood_set.py        # once: the photos under set/ (see its docstring)
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


def credit_row(c: dict) -> str:
    lic = f"[{c['license']}]({c['license_url']})" if c["license_url"] else c["license"]
    return f"| `{c['file']}` | {c['author'].replace('|', '/')} | {lic} | [Wikimedia Commons]({c['source']}) |"


def full_set_section(photos: list[dict], check: dict) -> list[str]:
    """The 251-photo set, its fixed split, and what the classifier and the input check do on ood-test."""
    rows = check["photos"]
    prox = {r["category"]: r["proximity"] for r in rows}
    cats = sorted(prox, key=lambda c: (prox[c] != "near", c))
    test = [r for r in rows if r["split"] == "test"]
    lines = [
        f"## The full set: {len(photos)} photos for the input check",
        "",
        f"To build and test the input check, the set was extended to {len(photos)} openly licensed photos from "
        "Wikimedia Commons (CC0, public domain, CC BY, CC BY-SA; no NC or ND licences). "
        "[`build_ood_set.py`](../../scripts/build_ood_set.py) found candidates with fixed search queries per "
        "category; the photos were then picked by hand from contact sheets, before any input check was scored "
        "([`selection.json`](selection.json)), and downscaled to 640 px on the long side. They are in "
        "[`set/`](set), one folder per category. "
        f"**Near** photos ({sum(r['proximity'] == 'near' for r in rows)}) look like the subject or come from the "
        f"same setting; **far** photos ({sum(r['proximity'] == 'far' for r in rows)}) are unrelated.",
        "",
        "[`split_ood.py`](../../scripts/split_ood.py) split the set once, before any tuning, into ood-tune "
        f"({sum(r['split'] == 'tune' for r in rows)}) and ood-test ({len(test)}), stratified by category with a "
        "fixed seed ([`split.json`](split.json)); the 12 original photos are all in ood-test. Only ood-tune was "
        "used while choosing the input check.",
        "",
        "On ood-test, per category: how often the classifier alone would show a verdict (not uncertain), and how "
        f"often the input check (`{check['version']}`, threshold {check['threshold']:.4f}) lets the photo through "
        "([`input_check.json`](input_check.json)):",
        "",
        "| Category | Near or far | Photos (tune / test) | Verdict shown, classifier alone | Passes the input check |",
        "|---|---|---|---|---|",
    ]
    for c in cats:
        sub = [r for r in rows if r["category"] == c]
        st = [r for r in sub if r["split"] == "test"]
        lines.append(
            f"| {c.replace('_', ' ')} | {prox[c]} | {len(sub) - len(st)} / {len(st)} | "
            f"{sum(r['classifier_verdict_shown'] for r in st)} of {len(st)} | "
            f"{sum(r['passed'] for r in st)} of {len(st)} |"
        )
    lines += [
        f"| **All** | | {len(rows) - len(test)} / {len(test)} | "
        f"{sum(r['classifier_verdict_shown'] for r in test)} of {len(test)} | "
        f"{sum(r['passed'] for r in test)} of {len(test)} |",
        "",
        "Rates with 95% intervals, and the false-rejection rates on real droppings photos, are in "
        "[`results/input_check/evaluation.md`](../../results/input_check/evaluation.md).",
        "",
    ]
    return lines


def write_credits(credits: dict) -> None:
    photos = credits["photos"]
    lines = [
        "# Credits: out-of-distribution photos",
        "",
        f"All {len(photos)} photos in this folder come from Wikimedia Commons. {credits['note']}",
        "",
        "| File | Category | Author | License | Source | Changes |",
        "|---|---|---|---|---|---|",
    ]
    for c in photos:
        lic = f"[{c['license']}]({c['license_url']})" if c["license_url"] else c["license"]
        changes = c.get("changes") or "Resized by Wikimedia Commons to 1024 px wide; no other changes."
        lines.append(
            f"| `{c['file']}` | {c.get('category', '').replace('_', ' ')} | {c['author'].replace('|', '/')} | "
            f"{lic} | [Wikimedia Commons]({c['source']}) | {changes} |"
        )
    lines.append("")
    (OOD / "CREDITS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    from poultry_disease import CLASS_NAMES, Predictor
    from poultry_disease.verdict import MIN_TOP2_MARGIN, MIN_TOP_PROBABILITY, interpret
    from poultry_disease.weights import get_weights

    credits = json.loads((OOD / "credits.json").read_text("utf-8"))
    predictor = Predictor(get_weights(allow_download=False))
    rows = []
    originals = [c for c in credits["photos"] if c.get("original")]
    check_path = OOD / "input_check.json"
    check = json.loads(check_path.read_text("utf-8")) if check_path.exists() else None
    check_by_file = {r["file"]: r for r in check["photos"]} if check else {}
    for c in originals:
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
        "The last column is what the app does now: the [input check](../../README.md#input-check) runs first, "
        "and a photo it turns away gets a notice instead of a verdict.",
        "",
        "| # | Photo | Group | Top class | Top probability | Verdict shown | Input check |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        shown = "**yes**" if r["verdict_shown"] else "no (uncertain)"
        ic = check_by_file.get(r["file"])
        gate = "n/a" if ic is None else ("**passes**" if ic["passed"] else "turned away")
        lines.append(
            f"| {i} | [{r['description']}]({r['file']}) | {r['group']} | {r['top_class']} | "
            f"{pct(r['top_probability'])} | {shown} | {gate} |"
        )
    lines += [
        "",
        f"**Summary.** A verdict would be shown for {summary['verdict_shown']} of {summary['n']} photos; "
        f"{summary['top_probability_ge_0_9']} of {summary['n']} get a top probability of 90% or more. By group: "
        + "; ".join(f"{g}, {s['verdict_shown']} of {s['n']}" for g, s in groups.items())
        + ". The uncertain state catches some of these photos, but it is no substitute for a check that the photo "
        "shows poultry droppings at all."
        + (
            f" The input check turns away {sum(not check_by_file[r['file']]['passed'] for r in rows)} of {len(rows)}."
            if check
            else ""
        ),
        "",
    ]
    if check:
        lines += full_set_section(credits["photos"], check)
    lines += [
        "## Credits",
        "",
        credits["note"],
        "",
        "| File | Author | License | Source |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(credit_row(by_file[r["file"]]))
    lines += [
        "",
        f"Credits for all {len(credits['photos'])} photos: [`CREDITS.md`](CREDITS.md) and "
        "[`credits.json`](credits.json). These photos keep their own licenses; the repository's MIT license does "
        "not apply to them.",
        "",
    ]
    (OOD / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    write_credits(credits)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
