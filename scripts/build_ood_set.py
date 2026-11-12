"""Build the out-of-distribution (OOD) photo set for the input check from Wikimedia Commons.

Two steps:

1. ``--discover`` runs the Commons searches listed in CATEGORIES, keeps files whose licence is CC0, public
   domain, CC BY or CC BY-SA, and writes the candidates plus a numbered contact sheet per category to
   ``.work/ood-discover/`` (gitignored). A person looks at the sheets and writes the chosen Commons titles
   into ``docs/ood/selection.json``, with a reason for every candidate that was skipped (for example "shows
   a cow, not dung").
2. Without ``--discover`` the script downloads exactly the titles in ``docs/ood/selection.json``, downscales
   each to at most 640 px on the long side (JPEG quality 85), and writes ``docs/ood/set/<category>/`` and
   the credits in ``docs/ood/credits.json`` (title, author, licence, source URL and the change made).

The committed selection file, not the live search ranking, decides what is in the set, so the set can be
rebuilt byte for byte as long as the Commons files do not change. Licences are re-checked on download.

Usage:
    python scripts/build_ood_set.py --discover            # candidates and contact sheets for review
    python scripts/build_ood_set.py                       # download the committed selection
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OOD = ROOT / "docs" / "ood"
SET_DIR = OOD / "set"
SELECTION = OOD / "selection.json"
DISCOVER = ROOT / ".work" / "ood-discover"
UA = {"User-Agent": "poultry-disease-detection/1.0 (https://github.com/peerzadarabet1807/poultry-disease-detection)"}
API = "https://commons.wikimedia.org/w/api.php"
OPEN_LICENSES = ("CC0", "CC BY", "Public domain", "PD")
LONG_SIDE = 640

#: slug -> (proximity, description, Commons search queries). "near" categories are look-alikes or things a
#: user might plausibly photograph instead of chicken droppings; "far" categories are unrelated.
CATEGORIES: dict[str, tuple[str, str, list[str]]] = {
    "cow_dung": ("near", "Cow or cattle dung", ["cow dung", "cattle dung", "cow pat"]),
    "horse_manure": ("near", "Horse manure", ["horse manure", "horse dung", "horse droppings"]),
    "dog_feces": ("near", "Dog feces", ["dog feces", "dog poop", "dog excrement"]),
    "goose_duck_droppings": (
        "near",
        "Goose or duck droppings",
        ["goose droppings", "goose poop", "duck droppings", "Canada goose feces", "goose excrement"],
    ),
    "wild_bird_droppings": (
        "near",
        "Pigeon or other wild bird droppings",
        ["pigeon droppings", "bird droppings", "bird poop", "seagull droppings", "guano"],
    ),
    "other_animal_droppings": (
        "near",
        "Droppings of sheep, goats, rabbits, deer and other animals",
        ["sheep droppings", "goat droppings", "rabbit droppings", "deer droppings", "scat", "hare droppings"],
    ),
    "soil_mud": ("near", "Soil and mud", ["mud ground", "muddy soil", "soil closeup"]),
    "lookalike_food": (
        "near",
        "Food that looks similar (porridge, rice, beans, mash)",
        ["porridge bowl", "cooked rice", "refried beans", "mashed potatoes", "hummus", "lentil soup"],
    ),
    "eggs": ("near", "Eggs and egg trays", ["egg tray", "egg carton", "chicken eggs"]),
    "live_poultry": ("near", "Live hens, chickens and other poultry", ["hen chicken", "chickens farm", "rooster"]),
    "feathers": (
        "near",
        "Feathers",
        ["chicken feathers", "feathers on ground", "white feathers", "feather closeup", "down feathers"],
    ),
    "litter_bedding": (
        "near",
        "Litter, bedding and bare floors without droppings (straw, wood shavings, sawdust, concrete)",
        ["straw bedding", "wood shavings", "sawdust", "hay closeup", "barn floor", "concrete floor"],
    ),
    "people": ("far", "People", ["people walking street", "farmer portrait", "hands"]),
    "documents": (
        "far",
        "Documents and printed pages",
        ["document page scan", "printed letter", "receipt", "newspaper page", "book page", "form paper"],
    ),
    "screenshots": ("far", "Screenshots", ["screenshot Wikipedia", "screenshot website", "screenshot software"]),
    "everyday_objects": (
        "far",
        "Everyday objects",
        ["coffee mug", "computer keyboard", "bicycle", "chair", "car", "shoes", "smartphone"],
    ),
    "scenes": (
        "far",
        "Landscapes, buildings and streets",
        ["landscape", "city street", "building facade", "beach", "forest"],
    ),
    "other_animals": ("far", "Other animals", ["cat", "dog portrait", "cow field", "horse", "sheep", "songbird"]),
}


#: The original 12 photos (scripts/fetch_ood_images.py) in the categories above. They all go to ood-test.
ORIGINAL_CATEGORY = {
    "hen_free_range.jpg": "live_poultry",
    "egg_tray.jpg": "eggs",
    "wood_shavings.jpg": "litter_bedding",
    "straw_bedding.jpg": "litter_bedding",
    "concrete_floor.jpg": "litter_bedding",
    "pigeon_droppings.jpg": "wild_bird_droppings",
    "wood_pigeon_droppings.jpg": "wild_bird_droppings",
    "cow_dung.jpg": "cow_dung",
    "horse_manure.jpg": "horse_manure",
    "dog_feces.jpg": "dog_feces",
    "porridge.jpg": "lookalike_food",
    "oat_porridge.jpg": "lookalike_food",
}


def plain(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def http_get(url: str, *, retries: int = 6) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:  # noqa: S310
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            time.sleep(min(60, 5 * 2**attempt))
    raise AssertionError("unreachable")


def api(params: dict) -> dict:
    data = http_get(f"{API}?{urllib.parse.urlencode({'format': 'json', **params})}")
    time.sleep(1.0)
    return json.loads(data)


def license_ok(name: str) -> bool:
    return name.startswith(OPEN_LICENSES) and "NC" not in name and "ND" not in name


def describe(info: dict, title: str) -> dict:
    meta = info.get("extmetadata", {})
    return {
        "title": title,
        "author": plain(meta.get("Artist", {}).get("value", "unknown")),
        "license": meta.get("LicenseShortName", {}).get("value", ""),
        "license_url": meta.get("LicenseUrl", {}).get("value", ""),
        "source": info["descriptionurl"],
        "width": info.get("width"),
        "height": info.get("height"),
    }


def discover(limit: int, only: list[str] | None = None) -> int:
    from PIL import Image, ImageDraw

    DISCOVER.mkdir(parents=True, exist_ok=True)
    taken = {p["title"] for p in json.loads((OOD / "credits.json").read_text("utf-8"))["photos"]}
    for slug, (_, _, queries) in CATEGORIES.items():
        if only and slug not in only:
            continue
        seen: dict[str, dict] = {}
        per_query: list[list[str]] = []
        for query in queries:
            per_query.append([])
            data = api(
                {
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": f"{query} filetype:bitmap",
                    "gsrnamespace": 6,
                    "gsrlimit": 50,
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata|mime|size",
                    "iiurlwidth": 330,
                }
            )
            pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
            for page in pages:
                title = page["title"].removeprefix("File:")
                info = (page.get("imageinfo") or [{}])[0]
                if title in seen or title in taken or info.get("mime") not in ("image/jpeg", "image/png"):
                    continue
                if min(info.get("width", 0), info.get("height", 0)) < 300:
                    continue
                d = describe(info, title)
                if license_ok(d["license"]):
                    seen[title] = {**d, "query": query, "thumb": info["thumburl"]}
                    per_query[-1].append(title)
        # round-robin over the queries, so the first query cannot fill the whole list
        order = [t for group in zip(*[q + [None] * (50 - len(q)) for q in per_query]) for t in group if t]
        cands = [seen[t] for t in dict.fromkeys(order)][:limit]
        (DISCOVER / f"{slug}.json").write_text(json.dumps(cands, indent=2, ensure_ascii=False), "utf-8")
        tiles = []
        for i, c in enumerate(cands):
            try:
                im = Image.open(io.BytesIO(http_get(c["thumb"]))).convert("RGB")
            except Exception:  # noqa: BLE001 (a broken thumbnail just leaves a blank tile)
                im = Image.new("RGB", (200, 200), "grey")
            im.thumbnail((200, 200))
            tile = Image.new("RGB", (200, 220), "white")
            tile.paste(im, ((200 - im.width) // 2, 20))
            ImageDraw.Draw(tile).text((4, 2), str(i), fill="black")
            tiles.append(tile)
            time.sleep(0.3)
        cols = 8
        sheet = Image.new("RGB", (cols * 200, max(1, -(-len(tiles) // cols)) * 220), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * 200, (i // cols) * 220))
        sheet.save(DISCOVER / f"{slug}.jpg", quality=80)
        print(f"{slug:<24} {len(cands)} candidates", flush=True)
    return 0


def fetch() -> int:
    from PIL import Image

    selection = json.loads(SELECTION.read_text("utf-8"))
    credits = json.loads((OOD / "credits.json").read_text("utf-8"))
    original = [
        {**p, "category": ORIGINAL_CATEGORY[p["file"]], "proximity": CATEGORIES[ORIGINAL_CATEGORY[p["file"]]][0],
         "original": True}
        for p in credits["photos"]
        if "/" not in p["file"]
    ]
    photos = []
    for slug, entry in selection["categories"].items():
        proximity, description, _ = CATEGORIES[slug]
        out = SET_DIR / slug
        out.mkdir(parents=True, exist_ok=True)
        titles = entry["titles"]
        for i in range(0, len(titles), 25):
            batch = titles[i : i + 25]
            data = api(
                {
                    "action": "query",
                    "titles": "|".join(f"File:{t}" for t in batch),
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata|mime|size",
                    "iiurlwidth": 960,
                }
            )
            norm = {n["to"]: n["from"] for n in data["query"].get("normalized", [])}
            by_title = {}
            for page in data["query"]["pages"].values():
                if "imageinfo" not in page:
                    raise LookupError(f"{page['title']}: missing on Commons")
                by_title[norm.get(page["title"], page["title"]).removeprefix("File:")] = page["imageinfo"][0]
            for title in batch:
                info = by_title[title]
                d = describe(info, title)
                if not license_ok(d["license"]):
                    raise ValueError(f"{title}: licence {d['license']} is not open enough")
                im = Image.open(io.BytesIO(http_get(info.get("thumburl") or info["url"]))).convert("RGB")
                scale = LONG_SIDE / max(im.size)
                if scale < 1:
                    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
                stem = re.sub(r"[^a-z0-9]+", "_", Path(title).stem.lower()).strip("_")[:60]
                if len(stem) < 4:  # titles in other scripts: name the file after a hash of the title
                    stem = f"{slug}_{hashlib.sha1(title.encode()).hexdigest()[:10]}"
                path = out / f"{stem}.jpg"
                n = 2
                while path.relative_to(OOD).as_posix() in {q["file"] for q in photos}:
                    path, n = out / f"{stem}_{n}.jpg", n + 1
                im.save(path, quality=85, optimize=True)
                photos.append(
                    {
                        "file": path.relative_to(OOD).as_posix(),
                        "category": slug,
                        "proximity": proximity,
                        "description": description,
                        **{k: d[k] for k in ("title", "author", "license", "license_url", "source")},
                        "changes": f"Downscaled to {max(im.size)} px on the long side, saved as JPEG quality 85.",
                    }
                )
                print(f"{photos[-1]['file']:<70} {d['license']}", flush=True)
                time.sleep(0.5)
    doc = {
        "note": (
            "The original 12 photos (no 'category' prefix in 'file') were resized by Wikimedia Commons to 1024 px "
            "wide with no other changes. The photos under set/ were downscaled as stated in 'changes'. Each photo "
            "keeps its own licence; the repository's MIT licence does not apply to them."
        ),
        "photos": original + photos,
    }
    (OOD / "credits.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", "utf-8", newline="\n")
    print(f"{len(photos)} photos in set/, {len(original)} original photos kept")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discover", action="store_true", help="write candidates and contact sheets for review")
    ap.add_argument("--limit", type=int, default=40, help="candidates per category with --discover")
    ap.add_argument("--only", action="append", choices=sorted(CATEGORIES), help="with --discover: these only")
    args = ap.parse_args()
    return discover(args.limit, args.only) if args.discover else fetch()


if __name__ == "__main__":
    sys.exit(main())
