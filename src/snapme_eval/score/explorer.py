"""Write the JSON files read by the photo explorer in site/src/pages/explore.astro.

    photos.json      photo details, recorded grams, food-code count group and category with the most grams
    gt_items.json    food descriptions, grams, categories and ASA24 dish numbers for each photo
    runs.json        identifiers and summary results for each model and method
    matrix.json      each run's per-photo food-code F1 and matched-line shares; graded lists also supply
                     same-food F1, same-food-or-kind F1 and food-list correction counts
    items/<run>.json model food lists and matches: 2 same food code, 1 same category, 0 neither, null uncoded;
                     graded lists add [grade, record lines] for each food: 3 same food, 2 same kind,
                     1 mixed-dish part, 0 unpaired; line indices start at 0 in gt_items.json order

photos.json and matrix.json use the same photo order. Filtering takes place in the browser.
thumbs() writes reduced WebP copies of the photos separately.
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd
from PIL import Image, ImageOps

from snapme_eval import align
from snapme_eval.folds import fold_for
from snapme_eval.paths import data_dir, gt_dir, runs_dir
from snapme_eval.score.same_food import one_to_one
from snapme_eval.score.score_run import item_code, load_predictions, mapping_cache, paper_token_sets

GRADED = ("same_f1", "type_f1", "edits")


def _num(x, nd=1):
    return None if x is None or x != x else round(float(x), nd)


def build(out: pathlib.Path, rows: list[dict], gt_version: str = "snapme_v1") -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "items").mkdir(exist_ok=True)
    g = gt_dir(gt_version)
    images = pd.read_parquet(g / "images.parquet").sort_values("image_id").reset_index(drop=True)
    items = pd.read_parquet(g / "items.parquet")
    by_img = dict(tuple(items.groupby("image_id")))
    main_cat = {i: (d.groupby("wweia_desc")["grams"].sum().idxmax() if d["wweia_desc"].notna().any() else None)
                for i, d in by_img.items()}
    in_2023 = set(paper_token_sets() or {})
    cols = ["image_id", "subject_id", "study_day", "occasion", "photo_type", "grams", "n_items", "food_count",
            "main_category", "fold", "in_2023"]
    photos = [[r.image_id, r.subject_id, int(r.study_day), r.occasion, r.photo_type, _num(r.grams_total),
               int(r.n_items), r.food_code_bin, main_cat.get(r.image_id), fold_for(r.subject_id),
               int(r.filename in in_2023)] for r in images.itertuples()]
    (out / "photos.json").write_text(json.dumps({"columns": cols, "rows": photos}, separators=(",", ":")))
    # dish_id is "<image_id>#<FoodNum>"; the explorer keeps the dish number within the photo
    gti = {i: [[d.description, _num(d.grams), d.wweia_desc,
                int(str(d.dish_id).rsplit("#", 1)[-1]) if d.dish_id == d.dish_id else None]
               for d in grp.sort_values("line").itertuples()] for i, grp in by_img.items()}
    (out / "gt_items.json").write_text(json.dumps(gti, separators=(",", ":")))

    cache = mapping_cache()
    code_of = dict(zip(cache["name_norm"], cache["food_code"], strict=True)) if cache is not None else {}
    foods = pd.read_parquet(data_dir() / "fndds" / "fndds_2017_2018_foods.parquet").set_index("food_code")
    gt_codes = {i: set(d["food_code"].astype("int64")) for i, d in by_img.items()}
    gt_wweia = {i: set(d["wweia_code"].dropna().astype("int64")) for i, d in by_img.items()}
    order = images["image_id"].tolist()
    runs, mat = [], {"wweia_f1": [], "code_f1": [], "component_recall": [], **{c: [] for c in GRADED}}
    graded = align.load_cache()
    lines = {i: d.sort_values("line")["description"].tolist() for i, d in by_img.items()}
    for row in rows:
        key = f"{row['pipeline']}__{row['model']}"
        run = runs_dir() / gt_version / row["pipeline"] / row["model"]
        pp = pd.read_parquet(run / "per_photo.parquet").set_index("image_id")
        for col in mat:
            mat[col].append([_num(pp[col].get(i), 3) if col in pp else None for i in order])
        preds = load_predictions(run)
        det = {}
        for i in order:
            its = (preds.get(i) or {}).get("items")
            if its is None:
                det[i] = None
                continue
            lst = []
            # Assign each food a grade, giving one-to-one same/type pairs priority over mixed-dish parts.
            pairs = graded.get(align.key_for("gpt-oss-120b", lines[i], its)) if its and i in lines else None
            grade: dict[int, list] = {}
            if pairs is not None:
                oo, part = one_to_one([tuple(x) for x in pairs])
                for a, b in part:
                    grade.setdefault(a, [1]).append(b)
                for a, b, gg in oo:
                    grade[a] = [3 if gg == "same" else 2, b]
            for k, it in enumerate(its):
                c = item_code(it, code_of)
                if c is None:
                    rel = None
                elif c in gt_codes.get(i, set()):
                    rel = 2
                elif c in foods.index:
                    rel = int(int(foods.at[c, "wweia_code"]) in gt_wweia.get(i, set()))
                else:
                    rel = None
                ing = it.get("ingredients") or []
                lst.append([str(it.get("name", ""))[:90], str(it.get("preparation") or "")[:40],
                            ", ".join(map(str, ing))[:160], _num(it.get("grams"), 0), rel,
                            *([grade.get(k, [0])] if pairs is not None else [])])
            det[i] = lst
        (out / "items" / f"{key}.json").write_text(json.dumps(det, separators=(",", ":")))
        runs.append({"key": key, **{k: row.get(k) for k in ("pipeline", "model", "hf_repo", "developer", "country",
                                                           "params_total_b", "weights_dtype", "wweia_f1", "code_f1",
                                                           "token_f1_2023", "component_recall",
                                                           "preparation_agreement", "item_grams_median_ape",
                                                           "same_f1", "type_f1", "edits", "no_edits")}})
    # Write graded columns only if a run supplies values for them.
    mat = {c: v for c, v in mat.items() if c not in GRADED or any(x is not None for col in v for x in col)}
    (out / "runs.json").write_text(json.dumps(runs, separators=(",", ":")))
    (out / "matrix.json").write_text(json.dumps({"runs": [r["key"] for r in runs], **mat}, separators=(",", ":")))


def thumbs(out: pathlib.Path, release: pathlib.Path, gt_version: str = "snapme_v1", side: int = 480) -> int:
    """WebP copies of every photo for the explorer (SNAPMe photos are CC BY-SA 4.0; see the folder's README)."""
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    out.mkdir(parents=True, exist_ok=True)
    images = pd.read_parquet(gt_dir(gt_version) / "images.parquet")
    n = 0
    for r in images.itertuples():
        dst = out / f"{r.image_id}.webp"
        if dst.exists():
            continue
        with Image.open(release / r.path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((side, side), Image.Resampling.LANCZOS)
            im.save(dst, "WEBP", quality=72, method=6)
        n += 1
    return n
