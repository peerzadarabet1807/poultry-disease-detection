"""Python reference embeddings for the input check's TF.js parity test (web/tests/parity.gate.node.test.ts).

Fixtures: the 18 dataset photos of the main parity test (tests/fixtures/images) and the 12 original OOD photos
(docs/ood/, all in ood-test). Two steps, each in the venv that has its framework:

1. ``seen`` (TensorFlow venv): decode with libjpeg's accurate integer IDCT (what browsers and sharp use) and
   apply the package resize, giving the 224 x 224 image both models see. Writes .work/gate/fixtures.npz and,
   for the OOD photos, web/tests/fixtures/input_check/<id>.seen.npy (the dataset photos already have theirs
   in web/tests/fixtures/reference/), so the parity test can feed TF.js exactly the Python image.
2. ``embed`` (gate venv): embed those images with the original PyTorch model (Hugging Face weights) and apply
   the rule in src/poultry_disease/data/input_check.json. Writes web/tests/fixtures/input_check/reference.json.

Usage:
    .work/venv-tf/Scripts/python   scripts/make_gate_references.py seen
    .work/venv-gate/Scripts/python scripts/make_gate_references.py embed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".work" / "gate"
OUT = ROOT / "web" / "tests" / "fixtures" / "input_check"
CONFIG = ROOT / "src" / "poultry_disease" / "data" / "input_check.json"


def fixtures() -> list[dict]:
    main = json.loads((ROOT / "tests" / "fixtures" / "fixtures.json").read_text("utf-8"))["images"]
    rows = [{"id": im["id"], "file": f"tests/fixtures/images/{im['file']}", "kind": "droppings"} for im in main]
    credits = json.loads((ROOT / "docs" / "ood" / "credits.json").read_text("utf-8"))["photos"]
    rows += [
        {"id": "ood_" + Path(p["file"]).stem, "file": f"docs/ood/{p['file']}", "kind": "ood"}
        for p in credits
        if p.get("original")
    ]
    return rows


def seen_step() -> int:
    import tensorflow as tf

    from poultry_disease.preprocess import resize_uint8

    rows, seen = [], []
    OUT.mkdir(parents=True, exist_ok=True)
    for r in fixtures():
        data = (ROOT / r["file"]).read_bytes()
        rgb = tf.io.decode_jpeg(data, channels=3, dct_method="INTEGER_ACCURATE").numpy()
        s = resize_uint8(rgb)
        seen.append(s)
        if r["kind"] == "droppings":
            npy = f"reference/{r['id']}.seen.npy"
            assert np.array_equal(np.load(OUT.parent / npy), s), f"{npy} differs from this decode"
        else:
            npy = f"input_check/{r['id']}.seen.npy"
            np.save(OUT.parent / npy, s)
        rows.append(
            {
                **r,
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "seen": npy,
                "seenSha256": hashlib.sha256(np.ascontiguousarray(s).tobytes()).hexdigest(),
            }
        )
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(WORK / "fixtures.npz", seen=np.stack(seen))
    (WORK / "fixtures.json").write_text(json.dumps(rows, indent=1), "utf-8")
    print(f"{len(rows)} fixtures -> {WORK / 'fixtures.npz'}")
    return 0


def embed_step() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    from gate_study import embed, load_candidate

    config = json.loads(CONFIG.read_text("utf-8"))
    rows = json.loads((WORK / "fixtures.json").read_text("utf-8"))
    seen = np.load(WORK / "fixtures.npz")["seen"]
    encoder, _, _ = load_candidate(config["model"]["name"])
    emb = embed(encoder, seen).astype(np.float64)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    protos = np.asarray(config["prototypes"]["vectors"], dtype=np.float64)
    scores = (emb @ protos.T).max(axis=1)
    t = config["threshold"]
    images = []
    for r, e, s in zip(rows, emb, scores, strict=True):
        images.append(
            {
                **r,
                "score": float(s),
                "passed": bool(s >= t),
                "margin": float(s - t),
                "embedding": np.round(e, 7).tolist(),
            }
        )
        print(f"{r['id']:<34} {s:.6f} {'pass' if s >= t else 'reject'}")
    doc = {
        "description": "PyTorch (Hugging Face) reference embeddings of the input check's image encoder, on the "
        "224 x 224 image the classifier sees (accurate-IDCT decode, package resize). scripts/make_gate_references.py",
        "model": config["model"]["name"],
        "revision": config["model"]["revision"],
        "version": config["version"],
        "threshold": t,
        "images": images,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reference.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"min |score - threshold| = {min(abs(i['margin']) for i in images):.6f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["seen", "embed"])
    args = ap.parse_args()
    return seen_step() if args.step == "seen" else embed_step()


if __name__ == "__main__":
    sys.exit(main())
