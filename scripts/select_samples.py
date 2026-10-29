"""Pick the dataset images used as parity fixtures and as web-app samples, from the saved predictions only.

Deterministic (sorting by confidence, then path). Selection:
* internal test, per class: the 2 most confident correct predictions, preferring two different image sizes;
* external PCR, per class: the most confident correct prediction;
* high-confidence errors: the 2 most confident internal errors and the 2 most confident external errors
  of the two most common external confusions (NCD -> Healthy, Healthy -> Salmonella);
* low confidence: the internal and the external image with the smallest top-2 margin.

The Colab training probabilities (prob_* columns) become the parity oracle in tests/fixtures/fixtures.json.

Usage:
    python scripts/select_samples.py            # writes tests/fixtures/selection.json and prints a Colab cell
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CLASS_NAMES = json.loads((ROOT / "models" / "class_names.json").read_text("utf-8"))
DRIVE_ROOT = "/content/drive/MyDrive/poultry-disease-dataset-zenedo"
ZIP_NAME = "poultry_fixtures.zip"


def load(name: str, set_name: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / name)
    manifest = pd.read_csv(RESULTS / "dataset_manifest.csv", usecols=["relpath", "width", "height", "md5"])
    df = df.merge(manifest, on="relpath", how="left", validate="one_to_one")
    probs = df[[f"prob_{c}" for c in CLASS_NAMES]].to_numpy()
    probs.sort(axis=1)
    df["margin"] = probs[:, -1] - probs[:, -2]
    df["set"] = set_name
    return df.sort_values(["confidence", "relpath"], ascending=[False, True])


def pick_two_sizes(group: pd.DataFrame) -> pd.DataFrame:
    first = group.iloc[[0]]
    other = group[(group["width"] != first["width"].iloc[0]) | (group["height"] != first["height"].iloc[0])]
    second = other.iloc[[0]] if len(other) else group.iloc[[1]]
    return pd.concat([first, second])


def main() -> int:
    internal = load("test_predictions.csv", "internal")
    external = load("pcr_predictions.csv", "external")
    picks = []

    for c in CLASS_NAMES:
        ok = internal[(internal["true_class"] == c) & internal["correct"]]
        picks.append(pick_two_sizes(ok).assign(role="internal_correct"))
    for c in CLASS_NAMES:
        ok = external[(external["true_class"] == c) & external["correct"]]
        picks.append(ok.iloc[[0]].assign(role="external_correct"))
    picks.append(internal[~internal["correct"]].iloc[:2].assign(role="internal_error"))
    for t, p in (("NCD", "Healthy"), ("Healthy", "Salmonella")):
        err = external[(external["true_class"] == t) & (external["predicted_class"] == p)]
        picks.append(err.iloc[[0]].assign(role="external_error"))
    for df, role in ((internal, "internal_low_margin"), (external, "external_low_margin")):
        picks.append(df.sort_values(["margin", "relpath"]).iloc[[0]].assign(role=role))

    sel = pd.concat(picks).drop_duplicates("relpath").reset_index(drop=True)
    rows = []
    for i, r in sel.iterrows():
        rows.append(
            {
                "id": f"f{i + 1:02d}",
                "relpath": r["relpath"],
                "set": r["set"],
                "role": r["role"],
                "true_class": r["true_class"],
                "colab_predicted_class": r["predicted_class"],
                "colab_probabilities": {c: float(r[f"prob_{c}"]) for c in CLASS_NAMES},
                "width": int(r["width"]),
                "height": int(r["height"]),
                "md5": r["md5"],
            }
        )
    out = ROOT / "tests" / "fixtures" / "selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"images": rows}, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"{'id':<4} {'set':<9} {'role':<20} {'true':<12} {'colab pred':<12} {'conf':>7} {'size':>10}  relpath")
    for r in rows:
        conf = r["colab_probabilities"][r["colab_predicted_class"]]
        size = f"{r['width']}x{r['height']}"
        print(
            f"{r['id']:<4} {r['set']:<9} {r['role']:<20} {r['true_class']:<12} {r['colab_predicted_class']:<12} "
            f"{conf:7.4f} {size:>10}  {r['relpath']}"
        )

    relpaths = ",\n    ".join(json.dumps(r["relpath"]) for r in rows)
    print(
        f'''
# ---- Colab cell: copy the {len(rows)} fixture images into one zip on Drive ----
import hashlib, json, os, zipfile
ROOT = "{DRIVE_ROOT}"
FILES = [
    {relpaths},
]
out = os.path.join(ROOT, "{ZIP_NAME}")
md5s = {{}}
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:   # JPEGs: store, do not recompress
    for rel in FILES:
        data = open(os.path.join(ROOT, rel), "rb").read()
        md5s[rel] = hashlib.md5(data).hexdigest()
        zf.writestr(rel, data)
    zf.writestr("md5.json", json.dumps(md5s, indent=2))
print(f"{{len(FILES)}} files -> {{out}} ({{os.path.getsize(out) / 1e6:.1f}} MB)")
'''
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
