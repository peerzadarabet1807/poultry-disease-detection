"""Run-to-run variability: five Colab runs with the same configuration, seed and split.

The Drive exports hold five runs of the notebook in external-test mode. Their experiment configs differ only
in timestamps, output paths and environment (two used TF 2.20.0, three TF 2.21.0) and their dataset manifests are
byte-identical, so they differ only through non-deterministic GPU kernels (DETERMINISTIC_OPS = False).
That makes them a small, informal estimate of training noise, not a multi-seed study.

Post-hoc analysis of saved predictions and metrics. Outputs:
results/posthoc/run_variability.json, results/posthoc/run_variability.csv
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from posthoc_common import CLASS_NAMES, POSTHOC, RESULTS, save_json

PUBLISHED = "run_20261029_191001"


def run_dirs() -> dict[str, dict]:
    runs = {
        PUBLISHED: {
            "metrics": RESULTS / "metrics.json",
            "config": RESULTS / "experiment_config.json",
            "internal": RESULTS / "test_predictions.csv",
            "external": RESULTS / "pcr_predictions.csv",
        }
    }
    for d in sorted((RESULTS / "other_runs").iterdir()):
        runs[d.name] = {
            "metrics": d / "metrics.json",
            "config": d / "experiment_config.json",
            "internal": d / "test_predictions.csv",
            "external": d / "pcr_predictions.csv",
        }
    return dict(sorted(runs.items()))


def main() -> int:
    runs = run_dirs()
    rows, preds = [], {"internal": {}, "external": {}}
    for name, f in runs.items():
        m = json.loads(f["metrics"].read_text("utf-8"))
        cfg = json.loads(f["config"].read_text("utf-8"))
        ext = m["pcr_evaluation"]
        rows.append(
            {
                "run": name,
                "published": name == PUBLISHED,
                "tensorflow": cfg["environment"]["tensorflow"],
                "best_epoch": m["best_epoch"],
                "val_macro_f1": m["best_val_macro_f1"],
                "internal_accuracy": m["test"]["accuracy"],
                "internal_macro_f1": m["test"]["macro_f1"],
                "external_accuracy": ext["accuracy"],
                "external_macro_f1": ext["macro_f1"],
                "external_log_loss": ext["log_loss"],
                "external_roc_auc": ext["roc_auc_ovr_macro"],
                **{f"external_recall_{c}": ext["per_class"][c]["recall"] for c in CLASS_NAMES},
                **{f"external_precision_{c}": ext["per_class"][c]["precision"] for c in CLASS_NAMES},
            }
        )
        for key in preds:
            df = pd.read_csv(f[key])
            preds[key][name] = df.set_index("relpath")["predicted_class"]

    table = pd.DataFrame(rows)
    table.to_csv(POSTHOC / "run_variability.csv", index=False, float_format="%.6f", lineterminator="\n")

    numeric = table.drop(columns=["run", "published", "tensorflow"])
    summary = {
        c: {
            "mean": float(numeric[c].mean()),
            "sd": float(numeric[c].std(ddof=1)),
            "min": float(numeric[c].min()),
            "max": float(numeric[c].max()),
        }
        for c in numeric.columns
    }

    agreement = {}
    for key, by_run in preds.items():
        frame = pd.DataFrame(by_run)
        names = list(frame.columns)
        pair = [float((frame[a] == frame[b]).mean()) for i, a in enumerate(names) for b in names[i + 1 :]]
        agreement[key] = {
            "n_images": int(len(frame)),
            "all_runs_agree": float((frame.nunique(axis=1) == 1).mean()),
            "pairwise_agreement_min": float(min(pair)),
            "pairwise_agreement_mean": float(np.mean(pair)),
        }

    ext_rank = int((table["external_macro_f1"] > table.loc[table["published"], "external_macro_f1"].iloc[0]).sum()) + 1
    out = {
        "method": (
            "Five notebook runs with identical configuration, seed 42 and a byte-identical dataset manifest "
            "(same split); differences come from non-deterministic GPU kernels and, for two runs, TF 2.20 "
            "instead of 2.21. Informal estimate of training noise, not a multi-seed study. Post-hoc analysis "
            "of saved metrics and predictions."
        ),
        "published_run": PUBLISHED,
        "published_run_external_macro_f1_rank": f"{ext_rank} of {len(table)}",
        "runs": rows,
        "summary": summary,
        "prediction_agreement": agreement,
    }
    save_json(out, POSTHOC / "run_variability.json")
    cols = ["run", "tensorflow", "val_macro_f1", "internal_macro_f1", "external_accuracy", "external_macro_f1"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for k in ("internal_macro_f1", "external_macro_f1", "external_accuracy"):
        s = summary[k]
        print(f"{k}: mean {s['mean']:.4f} sd {s['sd']:.4f} range [{s['min']:.4f}, {s['max']:.4f}]")
    print("agreement:", json.dumps(agreement))
    print("published run external macro-F1 rank:", out["published_run_external_macro_f1_rank"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
