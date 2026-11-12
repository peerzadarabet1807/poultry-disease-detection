"""Write results/input_check/candidate_study.md and the ROC figures from the candidate study's result files.

Reads results/input_check/candidate_study.json (scripts/gate_study.py) and, if present,
results/input_check/browser_bench.json (web/scripts/bench-gate.ts). Writes no number of its own: every value
in the table is copied or formatted from those files. Rates are shown with their 95% Wilson interval.

Usage:
    .work/venv-gate/Scripts/python scripts/gate_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "input_check"


def pct(r: dict) -> str:
    lo, hi = r["ci95"]
    return f"{100 * r['rate']:.1f}% ({r['k']}/{r['n']}; {100 * lo:.1f} to {100 * hi:.1f})"


def short(r: dict) -> str:
    return f"{100 * r['rate']:.1f}% ({r['k']}/{r['n']})"


def mib(n: int) -> str:
    return f"{n / 2**20:.1f} MiB"


def browser_rows(bench: dict, name: str) -> dict:
    out = {}
    for r in bench.get("results", []):
        if r.get("name") != name or not r.get("ok"):
            continue
        out[(r["profile"], r["runtime"], r["backend"], r["variant"])] = r
    return out


def main() -> int:
    study = json.loads((RES / "candidate_study.json").read_text("utf-8"))
    bench_path = RES / "browser_bench.json"
    bench = json.loads(bench_path.read_text("utf-8")) if bench_path.exists() else {}
    cands = study["candidates"]

    lines = [
        "# Input check: candidate study (poultry droppings)",
        "",
        "Written by `scripts/gate_report.py` from `candidate_study.json` (`scripts/gate_study.py`) and "
        "`browser_bench.json` (`web/scripts/bench-gate.ts`).",
        "",
        "**Threshold rule (fixed before any test image was scored):** the highest score threshold that keeps the "
        "false-rejection rate (FRR) on the 973 validation images at or below 1%. A photo passes when its score is at "
        "or above the threshold. The false-acceptance rate (FAR) is the share of out-of-distribution (OOD) photos "
        "that pass. Rates are shown as value (count; 95% Wilson interval).",
        "",
        "**Tuning sets** (used to compare and choose): validation images and ood-tune. **Test sets** (reported, "
        "not used to choose): 974 internal and 1,201 PCR test images, and ood-test.",
        "",
        "## Tuning sets: the basis of the recommendation",
        "",
        "| Gate | Variant | AUROC (all OOD) | AUROC (near OOD) | FAR ood-tune, near | FAR ood-tune, far |",
        "|---|---|---|---|---|---|",
    ]
    rows = []
    for name, c in cands.items():
        variants = (
            [("feature distance", c["metrics"])]
            if "metrics" in c
            else [
                ("zero-shot", c["zero_shot"]),
                ("image prototypes", c["prototype"]),
                ("kNN (k=10)", c["knn10"]),
                ("fused zero-shot + kNN", c["fused"]),
            ]
        )
        for label, m in variants:
            rows.append((name, label, m))
            lines.append(
                f"| {name} | {label} | {m['auroc_val_vs_ood_tune']:.3f} | {m['auroc_near_tune']:.3f} | "
                f"{short(m['far_ood_tune']['near'])} | {short(m['far_ood_tune']['far'])} |"
            )

    lines += [
        "",
        "## Test sets (reported only)",
        "",
        "| Gate | Variant | FRR internal test | FRR PCR test | FAR ood-test, near | FAR ood-test, far |",
        "|---|---|---|---|---|---|",
    ]
    for name, label, m in rows:
        lines.append(
            f"| {name} | {label} | {short(m['frr_test_internal']['overall'])} | "
            f"{short(m['frr_test_external']['overall'])} | {short(m['far_ood_test']['near'])} | "
            f"{short(m['far_ood_test']['far'])} |"
        )

    lines += [
        "",
        "## Licence, size and speed",
        "",
        "Download sizes are the files the browser would fetch. Python latency: ONNX Runtime 1.20.1 on this "
        "machine's CPU, one image, one thread. Browser latency: median of 20 runs after warm-up; desktop GPU is "
        "headed Chromium on this machine's integrated GPU; phone is Pixel 7 emulation with 4x CPU throttling.",
        "",
        "| Gate | Licence | Image-encoder parameters | ONNX fp32 / fp16 / int8 | TF.js float16 / uint8 | "
        "Python CPU, 1 thread (fp32) | Browser, desktop GPU (TF.js WebGL) | Browser, desktop (TF.js WASM) | "
        "Browser, desktop (ORT WASM, int8) | Browser, phone (TF.js WASM) | Browser, phone (ORT WASM, int8) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, c in cands.items():
        if "onnx" not in c:
            if "metrics" in c:
                lines.append(
                    f"| {name} | {c['license']} | (main model) | none | none | n/a | n/a | n/a | n/a | n/a | n/a |"
                )
            continue
        o = c["onnx"]
        b = browser_rows(bench, name)
        tfjs_sizes = []
        for variant in ("float16", "uint8"):
            d = ROOT / "build" / "gate" / name / "tfjs" / variant
            tfjs_sizes.append(mib(sum(f.stat().st_size for f in d.iterdir())) if d.exists() else "not converted")

        def ms(key, b=b):
            r = b.get(key)
            return f"{r['medianMs']:.0f} ms" if r else "not measured"

        lines.append(
            f"| {name} | {c['license']} | {c['params_image_encoder'] / 1e6:.1f} M | "
            f"{mib(o['fp32']['bytes'])} / {mib(o['fp16']['bytes'])} / {mib(o['int8']['bytes'])} | "
            f"{tfjs_sizes[0]} / {tfjs_sizes[1]} | {o['fp32']['cpu_1t_median_ms']:.0f} ms | "
            f"{ms(('desktop-gpu', 'tfjs', 'webgl', 'float16'))} | {ms(('desktop-gpu', 'tfjs', 'wasm', 'float16'))} | "
            f"{ms(('desktop-headless', 'ort', 'wasm', 'int8'))} | {ms(('phone', 'tfjs', 'wasm', 'float16'))} | "
            f"{ms(('phone', 'ort', 'wasm', 'int8'))} |"
        )
    rt = bench.get("runtimeDownload", {})
    for k, v in rt.items():
        if v:
            lines += [
                "",
                f"ONNX Runtime Web adds its own WebAssembly binary: {k}, {mib(v['bytes'])} "
                f"({mib(v['gzipBytes'])} gzipped). TF.js is already loaded by the app.",
            ]

    lines += ["", "## Per-category false acceptance on ood-tune", ""]
    cats = sorted({c for _, _, m in rows for c in m["far_ood_tune"]["by_category"]})
    lines.append("| Gate | Variant | " + " | ".join(cats) + " |")
    lines.append("|---|---|" + "---|" * len(cats))
    for name, label, m in rows:
        by = m["far_ood_tune"]["by_category"]
        lines.append(
            f"| {name} | {label} | " + " | ".join(f"{by[c]['k']}/{by[c]['n']}" if c in by else "" for c in cats) + " |"
        )
    lines += ["", "Figures: `figures/roc_tune.png` (tuning sets) and `figures/roc_test.png` (test sets).", ""]
    (RES / "candidate_study.md").write_text("\n".join(lines), "utf-8", newline="\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (RES / "figures").mkdir(exist_ok=True)
    for part in ("tune", "test"):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
        for name, label, m in rows:
            roc = m[f"roc_{part}"]
            style = "--" if label != "zero-shot" else "-"
            for ax, key in zip(axes, ("far", "far_near"), strict=True):
                ax.plot(roc[key], [1 - x for x in roc["frr"]], style, lw=1.2, label=f"{name} ({label})")
        for ax, title in zip(axes, ("all OOD photos", "near OOD photos only"), strict=True):
            ax.set_xlabel(f"false-acceptance rate on ood-{part}")
            ax.set_title(title)
            ax.set_xlim(0, 0.5)
            ax.set_ylim(0, 1.01)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("in-domain pass rate (1 - FRR), " + ("validation" if part == "tune" else "test"))
        axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.suptitle(f"Input check candidates, {'tuning' if part == 'tune' else 'test'} sets (x axis cut at 0.5)")
        fig.tight_layout()
        fig.savefig(RES / "figures" / f"roc_{part}.png", dpi=130)
        plt.close(fig)
    print(f"wrote {RES / 'candidate_study.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
