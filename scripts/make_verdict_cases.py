"""Cross-language test cases for the verdict rule (src/poultry_disease/verdict.py vs web/src/lib/verdict.ts).

Writes web/tests/fixtures/verdict-cases.json: seeded random probability vectors plus edge cases (ties,
values exactly at the thresholds), with the Python verdict for each. web/tests/unit/verdict.test.ts must
agree on every case.

Usage:
    python scripts/make_verdict_cases.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from poultry_disease.verdict import MIN_TOP2_MARGIN, MIN_TOP_PROBABILITY, interpret

    rng = np.random.default_rng(42)
    cases = [rng.dirichlet(np.full(4, a)).tolist() for a in (0.1, 0.3, 1.0) for _ in range(100)]
    t, m = MIN_TOP_PROBABILITY, MIN_TOP2_MARGIN
    cases += [
        [0.25, 0.25, 0.25, 0.25],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [t, 1 - t, 0.0, 0.0],
        [0.0, (1 + m) / 2, (1 - m) / 2, 0.0],
    ]
    out = []
    for probs in cases:
        v = interpret(probs)
        out.append(
            {
                "probabilities": probs,
                "category": v.category,
                "top": v.top_class,
                "runnerUp": v.runner_up,
                "reasons": list(v.reasons),
                "notifiable": v.notifiable,
            }
        )
    doc = {"thresholds": {"minTop": t, "minMargin": m}, "cases": out}
    path = ROOT / "web" / "tests" / "fixtures" / "verdict-cases.json"
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    counts = {c: sum(o["category"] == c for o in out) for c in ("disease_pattern", "no_disease_pattern", "uncertain")}
    print(f"{len(out)} cases -> {path.relative_to(ROOT)} {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
