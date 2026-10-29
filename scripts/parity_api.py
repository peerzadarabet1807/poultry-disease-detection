"""Parity check through a running API (for example the Docker container) on the committed fixtures.

Usage:
    docker compose up -d
    python scripts/parity_api.py --url http://localhost:8000 --label docker

Writes results/parity/api_vs_colab_<label>.json.
"""

from __future__ import annotations

import argparse
import json
import uuid
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def post_image(url: str, name: str, data: bytes) -> dict:
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + data
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        f"{url}/api/v1/predict", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--label", default="docker")
    args = ap.parse_args()

    with urllib.request.urlopen(f"{args.url}/api/v1/model", timeout=30) as r:
        info = json.load(r)
    images = json.loads((FIXTURES / "fixtures.json").read_text("utf-8"))["images"]
    rows = []
    for im in images:
        body = post_image(args.url, im["file"], (FIXTURES / "images" / im["file"]).read_bytes())
        diff = max(abs(body["probabilities"][c] - p) for c, p in im["colab_probabilities"].items())
        rows.append(
            {
                "file": im["file"],
                "argmax_match": body["predicted_class"] == im["colab_predicted_class"],
                "max_abs_diff": diff,
            }
        )
    summary = {
        "description": "API (HTTP) vs the probabilities recorded by the training notebook in Colab, on the fixtures.",
        "label": args.label,
        "run": info["run_id"],
        "weights_sha256": info["weights"]["sha256"],
        "n_images": len(rows),
        "argmax_identical": sum(r["argmax_match"] for r in rows),
        "max_abs_prob_diff": max(r["max_abs_diff"] for r in rows),
        "images": rows,
    }
    out = ROOT / "results" / "parity" / f"api_vs_colab_{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    n_ok, worst = summary["argmax_identical"], summary["max_abs_prob_diff"]
    print(f"{args.label}: argmax {n_ok}/{len(rows)}, max abs diff {worst:.2e}")
    return 0 if summary["argmax_identical"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
