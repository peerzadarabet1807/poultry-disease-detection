"""Download the released Keras weights into the cache and verify their SHA-256.

Usage:
    python scripts/fetch_weights.py        # prints the path of the verified poultry_resnet50.keras

The cache is $POULTRY_DISEASE_CACHE or ~/.cache/poultry-disease. Set $POULTRY_DISEASE_WEIGHTS to use a
local copy instead (it is still checksum-verified).
"""

from __future__ import annotations

from poultry_disease.weights import get_weights


def main() -> int:
    print(get_weights(allow_download=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
