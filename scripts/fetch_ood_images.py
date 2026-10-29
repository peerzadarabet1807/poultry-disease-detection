"""Download the out-of-distribution (OOD) photos in docs/ood/ from Wikimedia Commons, with their credits.

The model always answers with one of its four classes. To see what it does with photos it was never meant
for, docs/ood/ holds openly licensed photos in three groups: poultry-farm scenes without droppings, droppings
of other animals, and food that looks similar. Wikimedia serves each at 1024 px wide; author, license and
source come from the file's Commons metadata and are written to docs/ood/credits.json.

Usage:
    python scripts/fetch_ood_images.py
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ood"
UA = {"User-Agent": "poultry-disease-detection/1.0 (https://github.com/peerzadarabet1807/poultry-disease-detection)"}
API = "https://commons.wikimedia.org/w/api.php"

#: (local file, Commons title, group, short description)
PHOTOS = [
    ("hen_free_range.jpg", "Free Range Chicken.jpg", "farm scene", "A hen outdoors"),
    ("egg_tray.jpg", "A Tray Of Eggs.jpg", "farm scene", "A tray of eggs"),
    ("wood_shavings.jpg", "Wood Shavings ET5A7918 (30303840940).jpg", "farm scene", "Wood shavings (litter material)"),
    (
        "straw_bedding.jpg",
        "Straw-bedding heap at Haugh-head stables - geograph.org.uk - 5404366.jpg",
        "farm scene",
        "A heap of used straw bedding in a field",
    ),
    ("concrete_floor.jpg", "Cracked concrete floor in Brastad.jpg", "farm scene", "A bare concrete floor"),
    ("pigeon_droppings.jpg", "Pigeon droppings -or- Pigeon poop.JPG", "other droppings", "Pigeon droppings"),
    ("wood_pigeon_droppings.jpg", "NZ Wood pigeon droppings.jpg", "other droppings", "Wood pigeon droppings"),
    ("cow_dung.jpg", "Cow dung 34.jpg", "other droppings", "Cow dung"),
    ("horse_manure.jpg", "Horse manure in Amsterdam.jpg", "other droppings", "Horse manure on a street"),
    ("dog_feces.jpg", "Dog feces at SF Bay Trail.JPG", "other droppings", "Dog feces on a path"),
    ("porridge.jpg", "Porridge.jpg", "lookalike food", "A bowl of porridge"),
    ("oat_porridge.jpg", "Oat porridge in Ghana.jpg", "lookalike food", "Oat porridge"),
]


def plain(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def get_json(params: dict) -> dict:
    url = f"{API}?{urllib.parse.urlencode({'format': 'json', **params})}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:  # noqa: S310
        return json.load(r)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    credits = []
    for local, title, group, description in PHOTOS:
        data = get_json(
            {
                "action": "query",
                "titles": f"File:{title}",
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|mime",
                "iiurlwidth": 1024,
            }
        )
        page = next(iter(data["query"]["pages"].values()))
        info = page["imageinfo"][0]
        meta = info["extmetadata"]
        license_name = meta["LicenseShortName"]["value"]
        if not license_name.startswith(("CC BY", "CC0", "Public domain")):
            raise ValueError(f"{title}: license {license_name} is not open enough")
        with urllib.request.urlopen(urllib.request.Request(info["thumburl"], headers=UA), timeout=60) as r:  # noqa: S310
            (OUT / local).write_bytes(r.read())
        credits.append(
            {
                "file": local,
                "group": group,
                "description": description,
                "title": title,
                "author": plain(meta.get("Artist", {}).get("value", "unknown")),
                "license": license_name,
                "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                "source": info["descriptionurl"],
            }
        )
        print(f"{local:<26} {license_name:<14} {credits[-1]['author'][:50]}")
        time.sleep(0.5)
    doc = {
        "note": "Resized by Wikimedia Commons to 1024 px wide; no other changes. Each photo keeps its own license.",
        "photos": credits,
    }
    (OUT / "credits.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", "utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
