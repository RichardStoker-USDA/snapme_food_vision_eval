"""Build the Nutrition5k reference from overhead photos, ingredient names and weighed masses.

Source: Thames et al., CVPR 2021, doi:10.1109/CVPR46437.2021.00879.
Released by Google Research: github.com/google-research-datasets/Nutrition5k; CC BY 4.0.
Retain dishes with an overhead photo and preserve the authors' split labels.

fetch() downloads metadata, split files, overhead RGB images and raw depth images.
build() writes images.parquet, items.parquet and a build record; nutrient fields are not retained.
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import hashlib
import json
import pathlib
import time
import urllib.parse
import urllib.request

import pandas as pd

BUCKET = "nutrition5k_dataset"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
FILE = f"https://storage.googleapis.com/{BUCKET}/"
ROOT = "nutrition5k_dataset/"
OVERHEAD = ROOT + "imagery/realsense_overhead/"


def _list(prefix: str, delimiter: str | None = None) -> tuple[list[dict], list[str]]:
    items, prefixes, token = [], [], None
    while True:
        q = {"prefix": prefix, "maxResults": 1000, **({"delimiter": delimiter} if delimiter else {}),
             **({"pageToken": token} if token else {})}
        with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(q)}", timeout=120) as r:
            d = json.loads(r.read())
        items += d.get("items", [])
        prefixes += d.get("prefixes", [])
        token = d.get("nextPageToken")
        if not token:
            return items, prefixes


def _get(obj: dict, dest: pathlib.Path) -> str:
    rel = obj["name"][len(ROOT):]
    out = dest / rel
    if out.exists() and out.stat().st_size == int(obj["size"]):
        return rel
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(FILE + urllib.parse.quote(obj["name"]), timeout=300) as r, tmp.open("wb") as f:
                while b := r.read(1 << 20):
                    f.write(b)
            break
        except OSError:
            time.sleep(2 ** attempt)
    else:
        raise SystemExit(f"download failed: {obj['name']}")
    # Google Cloud Storage reports an MD5 per object (base64); check it.
    want = obj.get("md5Hash")
    if want:
        import base64
        got = base64.b64encode(hashlib.md5(tmp.read_bytes()).digest()).decode()
        if got != want:
            raise SystemExit(f"md5 mismatch: {obj['name']}")
    tmp.rename(out)
    return rel


def fetch(dest: pathlib.Path, workers: int = 16) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    objs = _list(ROOT + "metadata/")[0] + _list(ROOT + "dish_ids/")[0]
    overhead = [o for o in _list(OVERHEAD)[0] if o["name"].endswith(("/rgb.png", "/depth_raw.png"))]
    with cf.ThreadPoolExecutor(workers) as ex:
        done = list(ex.map(lambda o: _get(o, dest), objs + overhead))
    manifest = {"source": f"gs://{BUCKET}/{ROOT}", "license": "CC BY 4.0", "files": len(done),
                "dishes_overhead": len({o["name"].split("/")[-2] for o in overhead}),
                "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def build(src: pathlib.Path, out: pathlib.Path) -> dict:
    """Reference tables. Dish metadata rows: dish id, total calories, mass, fat, carbs, protein, then per
    ingredient: id, name, grams, calories, fat, carbs, protein. Only mass and ingredients are kept."""
    rows, items = [], []
    for f in ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"):
        with (src / "metadata" / f).open() as fh:
            for rec in csv.reader(fh):
                dish, mass = rec[0], float(rec[2])
                ing = [rec[i:i + 7] for i in range(6, len(rec) - 6, 7)]
                rows.append({"dish_id": dish, "cafe": 1 if f.endswith("cafe1.csv") else 2, "grams_total": mass,
                             "n_ingredients": len(ing)})
                for k, x in enumerate(ing):
                    items.append({"dish_id": dish, "line": k, "ingredient_id": x[0], "name": x[1].strip(),
                                  "grams": float(x[2])})
    split = {}
    for s in ("train", "test"):
        for line in (src / "dish_ids" / "splits" / f"rgb_{s}_ids.txt").read_text().split():
            split[line.strip()] = s
    img = pd.DataFrame(rows)
    img["split"] = img["dish_id"].map(split)
    img["path"] = "imagery/realsense_overhead/" + img["dish_id"] + "/rgb.png"
    img["image_id"] = img["filename"] = img["dish_id"]
    img["photo_type"], img["companion_path"], img["companion_exists"] = "overhead", None, False
    img = img[[(src / p).exists() for p in img["path"]]].reset_index(drop=True)
    it = pd.DataFrame(items)
    it = it[it["dish_id"].isin(img["dish_id"])].reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    img.to_parquet(out / "images.parquet", index=False)
    it.to_parquet(out / "items.parquet", index=False)
    info = {"benchmark": "nutrition5k", "dishes": len(img), "by_split": img["split"].value_counts().to_dict(),
            "ingredient_lines": len(it), "distinct_ingredients": int(it["ingredient_id"].nunique()),
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "build.json").write_text(json.dumps(info, indent=1))
    return info
