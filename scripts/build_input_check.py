"""Fix the input check chosen at STOP 1 and evaluate it once on the test sets.

The chosen gate (results/input_check/candidate_study.md): the TinyCLIP ViT-8M/16 image encoder and an
image-prototype score. A photo passes when the cosine similarity between its unit image embedding and the
nearest of four class prototypes (the normalised mean embedding of 150 training photos per class, seed 0) is
at least the threshold. The threshold is the highest value that keeps false rejection on the validation split
at or below 1%; nothing else is tuned.

Inputs (written by scripts/gate_extract.py and scripts/gate_study.py, both gitignored under .work/gate/):
embeddings of every set, the classifier's probabilities for the same photos, and docs/ood/split.json.

Writes:
    src/poultry_disease/data/input_check.json   the rule: model, prototypes, threshold (package and web app)
    results/input_check/evaluation.json         every rate with a 95% Wilson interval, test sets only
    results/input_check/evaluation.md           the same as tables
    results/input_check/figures/input_check_roc.png
    docs/ood/input_check.json                   score and decision for every OOD photo

Usage (gate venv, after gate_extract.py and gate_study.py):
    .work/venv-gate/Scripts/python scripts/build_input_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gate_study import CANDIDATES, CREDITS, MAX_FRR, SPLIT, WORK, auroc, rate, threshold_for  # noqa: E402

NAME = "tinyclip-vit-8m-16-yfcc15m"
REVISION = "a2a8c6eaa2549ad66eb7c31b85022bf58273a26c"
VERSION = "tinyclip-8m16-proto-1"
DECIMALS = 8
CONFIG = ROOT / "src" / "poultry_disease" / "data" / "input_check.json"
CLASSES = json.loads((ROOT / "src" / "poultry_disease" / "data" / "class_names.json").read_text("utf-8"))
VERDICT = json.loads((ROOT / "src" / "poultry_disease" / "data" / "decision_thresholds.json").read_text("utf-8"))
OUT = ROOT / "results" / "input_check"
SETS = ("train_sample", "val", "test_internal", "test_external", "samples", "ood")
SET_LABEL = {
    "val": "validation split (tuning)",
    "test_internal": "internal test set",
    "test_external": "PCR-confirmed external test set",
}


def load() -> tuple[dict, dict, dict]:
    emb = {s: np.load(WORK / "emb" / NAME / f"{s}.npy").astype(np.float64) for s in SETS}
    meta = {s: json.loads((WORK / f"{s}.json").read_text("utf-8")) for s in SETS}
    probs = {s: np.load(WORK / f"{s}.npz")["probs"] for s in ("test_internal", "test_external", "ood")}
    for s in SETS:
        assert len(emb[s]) == len(meta[s]), s
    return emb, meta, probs


def prototypes(emb: np.ndarray, labels: list[str]) -> np.ndarray:
    protos = np.stack([emb[[lab == c for lab in labels]].mean(axis=0) for c in CLASSES])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    # stored rounded; every score below uses the stored values, as the app does
    return np.round(protos, DECIMALS)


def score(emb: np.ndarray, protos: np.ndarray) -> np.ndarray:
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return (unit @ protos.T).max(axis=1)


def verdict_shown(p: np.ndarray) -> bool:
    top2 = np.sort(p)[-2:]
    return bool(top2[1] >= VERDICT["min_top_probability"] and top2[1] - top2[0] >= VERDICT["min_top2_margin"])


def frr_block(s: np.ndarray, labels: list[str], t: float) -> dict:
    return {
        "overall": rate(int((s < t).sum()), len(s)),
        "by_class": {
            c: rate(int(sum(x < t for x, lab in zip(s, labels, strict=True) if lab == c)), labels.count(c))
            for c in CLASSES
        },
    }


def far_block(s: np.ndarray, rows: list[dict], t: float) -> dict:
    def acc(sel):
        idx = [i for i, r in enumerate(rows) if sel(r)]
        return rate(int(sum(s[i] >= t for i in idx)), len(idx))

    cats = sorted({r["category"] for r in rows}, key=lambda c: (rows_prox(rows, c) != "near", c))
    return {
        "overall": acc(lambda r: True),
        "near": acc(lambda r: r["proximity"] == "near"),
        "far": acc(lambda r: r["proximity"] == "far"),
        "by_category": {c: {"proximity": rows_prox(rows, c), **acc(lambda r, c=c: r["category"] == c)} for c in cats},
        "accepted": sorted(r["file"] for r, x in zip(rows, s, strict=True) if x >= t),
    }


def rows_prox(rows: list[dict], category: str) -> str:
    return next(r["proximity"] for r in rows if r["category"] == category)


def error_block(s: np.ndarray, p: np.ndarray, labels: list[str], t: float) -> dict:
    """Post-hoc: does the gate reject the photos the classifier gets wrong more often than the right ones?"""
    wrong = np.array([CLASSES[int(np.argmax(q))] != lab for q, lab in zip(p, labels, strict=True)])
    rej = s < t
    return {
        "classifier_wrong": int(wrong.sum()),
        "classifier_right": int((~wrong).sum()),
        "rejected_when_wrong": rate(int((rej & wrong).sum()), int(wrong.sum())),
        "rejected_when_right": rate(int((rej & ~wrong).sum()), int((~wrong).sum())),
    }


def pct(r: dict) -> str:
    lo, hi = r["ci95"]
    return f"{100 * r['rate']:.1f}% ({r['k']}/{r['n']}; {100 * lo:.1f} to {100 * hi:.1f})"


def figure(roc: dict, t: float, at_t: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    ax.plot(roc["far"], roc["frr"], label="all ood-test photos", color="#1f6f8b")
    ax.plot(roc["far_near"], roc["frr"], label="near ood-test photos", color="#b5651d", linestyle="--")
    ax.scatter([at_t["far_near"]], [at_t["frr"]], color="#b5651d", zorder=3)
    ax.scatter([at_t["far"]], [at_t["frr"]], color="#1f6f8b", zorder=3, label=f"shipped threshold {t:.4f}")
    ax.set_xlabel("false acceptance (OOD photos that pass)")
    ax.set_ylabel("false rejection (test droppings photos that fail)")
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 0.15)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Input check on the test sets (TinyCLIP 8M/16, prototype score)", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    emb, meta, probs = load()
    labels = {s: [m["label"] for m in meta[s]] for s in SETS}
    protos = prototypes(emb["train_sample"], labels["train_sample"])
    scores = {s: score(emb[s], protos) for s in SETS if s != "train_sample"}
    t = threshold_for(scores["val"], MAX_FRR)
    tuning = rate(int((scores["val"] < t).sum()), len(scores["val"]))

    config = {
        "description": (
            "Input check: is this a photo of poultry droppings? Pass when the cosine similarity between the "
            "unit image embedding and the nearest class prototype is at least the threshold. A safeguard, not a "
            "guarantee; measured rates are in results/input_check/evaluation.json."
        ),
        "version": VERSION,
        "model": {
            "name": NAME,
            "repo": CANDIDATES[NAME]["repo"],
            "revision": REVISION,
            "license": "MIT, Copyright (c) Microsoft Corporation (model card; microsoft/Cream TinyCLIP/LICENSE)",
            "input": "the 224 x 224 RGB image the classifier sees, float32 0..255, NHWC; normalised inside the graph",
            "output": "unit-length image embedding",
            "embedding_dim": int(protos.shape[1]),
            "browser_weights": "web/public/gate/ (TF.js graph model, float16 weights)",
        },
        "score": "max over classes of cosine(image embedding, class prototype)",
        "threshold": t,
        "rule": f"pass if score >= threshold; threshold = highest value with false rejection <= {MAX_FRR:.0%} on "
        "the validation split",
        "tuning": {"set": "validation split (dataset version 2)", "false_rejection": tuning},
        "prototypes": {
            "source": "normalised mean embedding of a seeded training sample, 150 photos per class "
            "(scripts/fetch_zenodo_images.py --split train --per-class 150 --seed 0), rounded to "
            f"{DECIMALS} decimals",
            "classes": CLASSES,
            "vectors": protos.tolist(),
        },
        "source": "scripts/build_input_check.py",
    }
    CONFIG.write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8", newline="\n")

    credits = {p["file"]: p for p in json.loads(CREDITS.read_text("utf-8"))["photos"]}
    split = json.loads(SPLIT.read_text("utf-8"))
    part = {f: "tune" for f in split["tune"]} | {f: "test" for f in split["test"]}
    ood_rows = []
    for m, s, p in zip(meta["ood"], scores["ood"], probs["ood"], strict=True):
        c = credits[m["id"]]
        ood_rows.append(
            {
                "file": m["id"],
                "category": c["category"],
                "proximity": c["proximity"],
                "split": part[m["id"]],
                "original": bool(c.get("original")),
                "score": round(float(s), 6),
                "passed": bool(s >= t),
                "classifier_top_class": CLASSES[int(np.argmax(p))],
                "classifier_top_probability": round(float(np.max(p)), 6),
                "classifier_verdict_shown": verdict_shown(p),
            }
        )
    test_idx = [i for i, r in enumerate(ood_rows) if r["split"] == "test"]
    tune_idx = [i for i, r in enumerate(ood_rows) if r["split"] == "tune"]
    ood_test = [ood_rows[i] for i in test_idx]
    s_test = scores["ood"][test_idx]
    in_test = np.concatenate([scores["test_internal"], scores["test_external"]])
    near_test = s_test[[r["proximity"] == "near" for r in ood_test]]

    grid = np.unique(np.quantile(np.concatenate([in_test, s_test]), np.linspace(0, 1, 401)))
    roc = {
        "threshold": grid.tolist(),
        "frr": [float((in_test < g).mean()) for g in grid],
        "far": [float((s_test >= g).mean()) for g in grid],
        "far_near": [float((near_test >= g).mean()) for g in grid],
    }

    shown = [r["classifier_verdict_shown"] for r in ood_test]
    evaluation = {
        "description": (
            "Evaluation of the shipped input check. The threshold comes from the validation split only; every "
            "rate below except 'tuning' is on the test sets (internal and PCR-confirmed test photos, ood-test). "
            "Rates carry 95% Wilson intervals."
        ),
        "version": VERSION,
        "model": NAME,
        "threshold": t,
        "tuning": {
            "false_rejection_val": frr_block(scores["val"], labels["val"], t),
            "false_acceptance_ood_tune": far_block(scores["ood"][tune_idx], [ood_rows[i] for i in tune_idx], t),
        },
        "false_rejection": {
            "test_internal": frr_block(scores["test_internal"], labels["test_internal"], t),
            "test_external": frr_block(scores["test_external"], labels["test_external"], t),
            "test_all": rate(int((in_test < t).sum()), len(in_test)),
        },
        "false_acceptance": far_block(s_test, ood_test, t),
        "auroc_test_vs_ood_test": auroc(in_test, s_test),
        "auroc_test_vs_near_ood_test": auroc(in_test, near_test),
        "samples": [
            {"id": m["id"], "label": m["label"], "score": round(float(s), 6), "passed": bool(s >= t)}
            for m, s in zip(meta["samples"], scores["samples"], strict=True)
        ],
        "posthoc": {
            "note": "Post-hoc observations, not used for any choice.",
            "classifier_errors": {
                s: error_block(scores[s], probs[s], labels[s], t) for s in ("test_internal", "test_external")
            },
            "ood_test_verdict_shown": {
                "rule": "the app's verdict rule (decision_thresholds.json): top >= "
                f"{VERDICT['min_top_probability']} and lead >= {VERDICT['min_top2_margin']}",
                "classifier_alone": rate(sum(shown), len(shown)),
                "after_input_check": rate(
                    sum(v and r["passed"] for v, r in zip(shown, ood_test, strict=True)), len(shown)
                ),
            },
        },
        "roc_test": roc,
        "source": "scripts/build_input_check.py",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation.json").write_text(json.dumps(evaluation, indent=1) + "\n", encoding="utf-8", newline="\n")
    (ROOT / "docs" / "ood" / "input_check.json").write_text(
        json.dumps(
            {
                "description": "Input check score and decision for every OOD photo, with the classifier's answer "
                "for comparison (scripts/build_input_check.py).",
                "version": VERSION,
                "threshold": t,
                "photos": ood_rows,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    at_t = {
        "frr": float((in_test < t).mean()),
        "far": float((s_test >= t).mean()),
        "far_near": float((near_test >= t).mean()),
    }
    figure(roc, t, at_t, OUT / "figures" / "input_check_roc.png")
    write_markdown(evaluation)
    fr, fa = evaluation["false_rejection"], evaluation["false_acceptance"]
    summary = {
        "description": "The numbers the web app shows about the input check (from evaluation.json).",
        "version": VERSION,
        "model": {"name": NAME, "repo": CANDIDATES[NAME]["repo"], "license": "MIT"},
        "threshold": t,
        "prototype_photos_per_class": {c: labels["train_sample"].count(c) for c in CLASSES},
        "false_rejection": {
            "val": tuning,
            "test_internal": fr["test_internal"]["overall"],
            "test_external": fr["test_external"]["overall"],
            "test_all": fr["test_all"],
        },
        "false_acceptance": {"near": fa["near"], "far": fa["far"], "overall": fa["overall"]},
        "samples": {
            "passed": sum(s["passed"] for s in evaluation["samples"]),
            "n": len(evaluation["samples"]),
        },
        "ood_test_verdict_shown": evaluation["posthoc"]["ood_test_verdict_shown"],
        "source": "scripts/build_input_check.py",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")

    print(f"threshold {t:.6f}  tuning FRR {pct(tuning)}")
    print(f"FRR internal {pct(fr['test_internal']['overall'])}  PCR {pct(fr['test_external']['overall'])}")
    print(f"FAR near {pct(fa['near'])}  far {pct(fa['far'])}")
    print(f"samples failing: {[s['id'] for s in evaluation['samples'] if not s['passed']]}")
    return 0


def write_markdown(ev: dict) -> None:
    fr, fa, ph = ev["false_rejection"], ev["false_acceptance"], ev["posthoc"]
    lines = [
        "# Input check: evaluation of the shipped gate",
        "",
        f"Written by [`scripts/build_input_check.py`](../../scripts/build_input_check.py). Gate `{ev['version']}`: "
        f"TinyCLIP ViT-8M/16 image embedding, cosine to the nearest of four class prototypes, threshold "
        f"{ev['threshold']:.6f} (highest value with at most 1% false rejection on the validation split). The test "
        "sets below played no part in any choice. Rates: percent (count; 95% Wilson interval).",
        "",
        "## False rejection (droppings photos the check turns away)",
        "",
        "| Class | Validation (tuning) | Internal test | PCR-confirmed test |",
        "|---|---|---|---|",
    ]
    val = ev["tuning"]["false_rejection_val"]
    for c in CLASSES:
        lines.append(
            f"| {c} | {pct(val['by_class'][c])} | {pct(fr['test_internal']['by_class'][c])} | "
            f"{pct(fr['test_external']['by_class'][c])} |"
        )
    lines.append(
        f"| **All** | {pct(val['overall'])} | {pct(fr['test_internal']['overall'])} | "
        f"{pct(fr['test_external']['overall'])} |"
    )
    lines += [
        "",
        f"Both test sets together: {pct(fr['test_all'])}.",
        "",
        "## False acceptance (ood-test photos that pass)",
        "",
        "| Category | Near or far | Passed |",
        "|---|---|---|",
    ]
    for c, r in fa["by_category"].items():
        lines.append(f"| {c.replace('_', ' ')} | {r['proximity']} | {pct(r)} |")
    tune = ev["tuning"]["false_acceptance_ood_tune"]
    lines += [
        f"| **All near** | near | {pct(fa['near'])} |",
        f"| **All far** | far | {pct(fa['far'])} |",
        f"| **All** | | {pct(fa['overall'])} |",
        "",
        f"On ood-tune (seen while choosing the gate): near {pct(tune['near'])}, far {pct(tune['far'])}.",
        "",
        f"Passing ood-test photos: {', '.join(f'`{f}`' for f in fa['accepted']) or 'none'}.",
        "",
        f"AUROC, test droppings photos against ood-test: {ev['auroc_test_vs_ood_test']:.4f} (near only: "
        f"{ev['auroc_test_vs_near_ood_test']:.4f}).",
        "",
        "![False rejection against false acceptance on the test sets](figures/input_check_roc.png)",
        "",
        "## Shipped sample photos",
        "",
        f"{sum(s['passed'] for s in ev['samples'])} of {len(ev['samples'])} pass.",
        "",
        "## Post-hoc observations",
        "",
        "Not used for any choice.",
        "",
        "| Test set | Classifier wrong | Rejected when wrong | Rejected when right |",
        "|---|---|---|---|",
    ]
    for s, b in ph["classifier_errors"].items():
        lines.append(
            f"| {SET_LABEL[s]} | {b['classifier_wrong']} of {b['classifier_wrong'] + b['classifier_right']} | "
            f"{pct(b['rejected_when_wrong'])} | {pct(b['rejected_when_right'])} |"
        )
    v = ph["ood_test_verdict_shown"]
    lines += [
        "",
        f"On ood-test photos the classifier alone would show a verdict (not 'uncertain') for "
        f"{pct(v['classifier_alone'])}; with the input check in front, for {pct(v['after_input_check'])}.",
        "",
    ]
    (OUT / "evaluation.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
