"""Score a SNAPMe run against its linked ASA24 records.

Only photos present in predictions.jsonl are scored. An unreadable reply is treated as an empty list.
Identification F1 and matched-line shares are calculated per photo and averaged across photos, with
intervals from resampling participants. Preparation agreement and item-gram error pool eligible matched
pairs across photos. Only positive gram estimates enter amount errors; shares with an amount are reported
for eligible matched pairs and for photos. Photo totals sum supplied amounts, including unmatched foods.

Food identification uses exact FNDDS codes, WWEIA categories, subgroups and major groups. Precision counts
predicted items matching any recorded line; recall counts dishes with at least one matching item.
Item matching pairs predictions one-to-one with ASA24 lines, by exact code first and category second.

Token F1 is computed on food descriptions and separately against the 2023 paper's published reference sets.
The latter includes predicted ingredients. Item codes come from the method when supplied, otherwise from
the name-to-code cache. Calories and nutrients are not scored.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from snapme_eval import parse
from snapme_eval.foodterms import prep_terms
from snapme_eval.paths import data_dir, gt_dir
from snapme_eval.score import metrics, tokens

LEVELS = ("food_code", "wweia_code", "group2", "group1")


def load_predictions(run: pathlib.Path) -> dict[str, dict]:
    preds: dict[str, dict] = {}
    for line in (run / "predictions.jsonl").open():
        r = json.loads(line)
        # Re-parse the raw reply so every run is scored with the same parser version.
        if r.get("raw") and r.get("parse_error") != "request failed":
            r["items"], r["parse_error"] = parse.items(r["raw"])
        preds[r["image_id"]] = r  # retain the last recorded reply for a repeated photo
    return preds


def mapping_cache() -> pd.DataFrame | None:
    p = data_dir() / "mapping" / "name_to_fndds_v1.parquet"
    return pd.read_parquet(p) if p.exists() else None


def score(run: pathlib.Path, gt_version: str = "snapme_v1", reps: int = 1000) -> dict:
    g = gt_dir(gt_version)
    images = pd.read_parquet(g / "images.parquet")
    items = pd.read_parquet(g / "items.parquet")
    preds = load_predictions(run)
    images = images[images["image_id"].isin(preds)].copy()  # a --limit smoke test scores what it ran

    # Per-photo predicted totals
    rows = []
    for r in images.itertuples():
        p = preds[r.image_id]
        its = p.get("items")
        ok = its is not None
        rows.append({"image_id": r.image_id, "parsed": ok, "n_pred": len(its or []),
                     "grams_pred": photo_grams(its),
                     "latency_s": p.get("latency_s"),
                     "completion_tokens": (p.get("usage") or {}).get("completion_tokens")})
    df = images.merge(pd.DataFrame(rows), on="image_id")

    out: dict = {"run": str(run), "gt_version": gt_version, "n_images": int(len(df)),
                 "format_compliance": float(df["parsed"].mean()) if len(df) else 0.0}

    # Token F1 on descriptions (2023 paper formula)
    truth_tokens = items.groupby("image_id")["description"].apply(
        lambda s: set().union(*(tokens.clean(x) for x in s)))
    tf1 = []
    for r in df.itertuples():
        names = [it["name"] for it in (preds[r.image_id].get("items") or [])]
        pred_t = set().union(*(tokens.clean(n) for n in names)) if names else set()
        tf1.append(tokens.f1(truth_tokens.get(r.image_id, set()), pred_t))
    df["token_f1"] = tf1
    out["token_f1"] = {"mean": float(np.mean(tf1)) if tf1 else 0.0,
                       "by_food_code_bin": df.groupby("food_code_bin")["token_f1"].mean().to_dict(),
                       "by_photo_type": df.groupby("photo_type")["token_f1"].mean().to_dict()}
    if reps:
        out["token_f1"]["ci"] = metrics.cluster_bootstrap(df, "subject_id", lambda d: float(d["token_f1"].mean()), reps)

    # The 2023 comparison uses the published word sets for 1,463 before photos, including mixed-dish ingredients.
    # Predicted names and ingredients receive the same cleaning. Baselines are rescored from their own sets.
    ref = paper_token_sets()
    if ref is not None:
        rows = []
        for r in df.itertuples():
            if r.filename in ref:
                its = preds[r.image_id].get("items") or []
                words = [it["name"] for it in its] + [x for it in its for x in it.get("ingredients") or []]
                pred_t = set().union(*(tokens.clean(w) for w in words)) if words else set()
                rows.append((r.subject_id, r.food_code_bin, tokens.f1(ref[r.filename], pred_t)))
        p23 = pd.DataFrame(rows, columns=["subject_id", "food_code_bin", "f1"])
        out["token_f1_2023"] = {"mean": float(p23["f1"].mean()) if len(p23) else None, "n": len(p23),
                                "by_food_code_bin": p23.groupby("food_code_bin")["f1"].mean().to_dict()}
        if reps and len(p23):
            out["token_f1_2023"]["ci"] = metrics.cluster_bootstrap(p23, "subject_id", lambda d: float(d["f1"].mean()),
                                                                   reps)

    # Food identification against FNDDS codes (needs the mapping cache)
    cache = mapping_cache()
    if cache is not None:
        out.update(identification(df, preds, items, cache, reps))
        out["matched"] = amounts_and_preparation(df, preds, items, cache, reps)
    else:
        out["identification"] = "pending: mapping cache not built"

    # Same food and same kind of food, graded against the photo's record lines without codes (align.py)
    from snapme_eval.score import same_food
    sf = same_food.score(df, preds, items, reps)
    if sf:
        out["same_food"] = sf

    df.to_parquet(run / "per_photo.parquet", index=False)
    out["summary"] = summarize(out)
    (run / "scores.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def identification(df, preds, items, cache, reps) -> dict:
    code_of = dict(zip(cache["name_norm"], cache["food_code"], strict=True))
    foods = pd.read_parquet(data_dir() / "fndds" / "fndds_2017_2018_foods.parquet").set_index("food_code")
    gt_by_img = {k: v for k, v in items.groupby("image_id")}
    res = {lv: {"tp_pred": 0, "n_pred": 0, "dish_found": 0, "n_dish": 0, "f1_img": []} for lv in LEVELS}
    unmapped = 0
    for r in df.itertuples():
        gt = gt_by_img[r.image_id]
        dishes = gt.groupby("dish_id")
        pred_codes = []
        for it in preds[r.image_id].get("items") or []:
            c = item_code(it, code_of)
            if c is None or c not in foods.index:
                unmapped += 1
                pred_codes.append(None)
                continue
            pred_codes.append({"food_code": c, "wweia_code": foods.at[c, "wweia_code"],
                               "group2": c // 1_000_000, "group1": c // 10_000_000})
        for lv in LEVELS:
            gt_vals = set(gt[lv].dropna().astype("int64"))
            hits = [pc is not None and int(pc[lv]) in gt_vals for pc in pred_codes]
            tp = sum(hits)
            pred_vals = {int(pc[lv]) for pc in pred_codes if pc is not None}
            found = sum(any(int(v) in pred_vals for v in d[lv].dropna()) for _, d in dishes)
            nd = dishes.ngroups
            res[lv]["tp_pred"] += tp
            res[lv]["n_pred"] += len(pred_codes)
            res[lv]["dish_found"] += found
            res[lv]["n_dish"] += nd
            p = tp / len(pred_codes) if pred_codes else 0.0
            rc = found / nd if nd else 0.0
            res[lv]["f1_img"].append(2 * p * rc / (p + rc) if p + rc else 0.0)
    ident = {}
    for lv, v in res.items():
        p = v["tp_pred"] / v["n_pred"] if v["n_pred"] else 0.0
        rc = v["dish_found"] / v["n_dish"] if v["n_dish"] else 0.0
        ident[lv] = {"precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
                     "f1_photo_mean": float(np.mean(v["f1_img"]))}
    for lv, col in (("food_code", "code_f1"), ("wweia_code", "wweia_f1"), ("group2", "subgroup_f1"),
                    ("group1", "group_f1")):
        df[col] = res[lv]["f1_img"]
        ident[lv]["by_food_code_bin"] = df.groupby("food_code_bin")[col].mean().to_dict()
        ident[lv]["by_photo_type"] = df.groupby("photo_type")[col].mean().to_dict()
        if reps:
            ident[lv]["f1_photo_mean_ci"] = metrics.cluster_bootstrap(
                df, "subject_id", lambda d, c=col: float(d[c].mean()), reps)
    return {"identification": ident, "unmapped_names": unmapped}


def amount(it: dict) -> float | None:
    """Return a positive numeric gram estimate; zero, missing or invalid values mean no amount given."""
    g = it.get("grams")
    return float(g) if isinstance(g, (int, float)) and not isinstance(g, bool) and g > 0 else None


def photo_grams(its: list | None) -> float:
    """Sum positive item amounts, or NaN when none is supplied; foods without amounts do not contribute."""
    v = [g for g in (amount(it) for it in its or []) if g is not None]
    return float(sum(v)) if v else float("nan")


def item_code(it: dict, code_of: dict):
    """FNDDS code of a predicted item: the pipeline's own choice if it made one, else the mapping cache."""
    c = it.get("fndds_code")
    if isinstance(c, (int, float)) and c == c:
        return int(c)
    c = code_of.get(norm(it.get("name", "")))
    return int(c) if isinstance(c, (int, float)) and c == c else None


def norm(name: str) -> str:
    return " ".join(str(name).lower().split())


def summarize(o: dict) -> dict:
    s = {"n_images": o["n_images"], "format_compliance": round(o["format_compliance"], 4),
         "token_f1": round(o["token_f1"]["mean"], 3)}
    if o.get("token_f1_2023", {}).get("mean") is not None:
        s["token_f1_2023"] = round(o["token_f1_2023"]["mean"], 3)
    sf = o.get("same_food") or {}
    for k in ("same_f1", "type_f1", "same_found_lines", "same_correct", "edits", "no_edits"):
        if sf.get(k) is not None:
            s[k] = round(sf[k], 3)
    if isinstance(o.get("identification"), dict):
        for lv, key in (("food_code", "code"), ("wweia_code", "wweia"), ("group2", "subgroup")):
            s[f"{key}_f1"] = round(o["identification"][lv]["f1_photo_mean"], 3)
        m = o["matched"]
        s["component_recall"] = round(m["component_recall"], 3) if m["component_recall"] is not None else None
        s["preparation_agreement"] = (round(m["preparation_agreement"], 3)
                                      if m["preparation_agreement"] is not None else None)
        s["item_grams_median_ape"] = (round(m["item_grams_median_ape"], 3)
                                      if m["item_grams_median_ape"] is not None else None)
    return s


def paper_token_sets() -> dict[str, set[str]] | None:
    """Map filenames to the paper's reference word sets from the Inverse Cooking output file.

    Mixed-dish descriptions were manually converted to ingredients. The file is retrieved by fetch.
    """
    import ast
    f = data_dir() / "larke_2023" / "snapme_inv_cook_f1_score.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    return {fn: set(ast.literal_eval(cd)) for fn, cd in zip(d["filename"], d["clean_description"], strict=True)}


def match_items(gt: pd.DataFrame, pred: list[dict], code_of: dict, foods: pd.DataFrame) -> list[tuple[int, int]]:
    """One-to-one pairs (ASA24 line index, predicted item index): exact FNDDS food first, then same WWEIA
    category. Unmatched lines and items stay unpaired."""
    codes = [item_code(it, code_of) for it in pred]
    wweia = [int(foods.at[c, "wweia_code"]) if c is not None and c in foods.index else None for c in codes]
    pairs, used_g, used_p = [], set(), set()
    for level in ("food_code", "wweia_code"):
        for gi, row in gt.reset_index(drop=True).iterrows():
            if gi in used_g or pd.isna(row[level]):
                continue
            for pi in range(len(pred)):
                if pi in used_p:
                    continue
                v = codes[pi] if level == "food_code" else wweia[pi]
                if v is not None and int(v) == int(row[level]):
                    pairs.append((gi, pi))
                    used_g.add(gi)
                    used_p.add(pi)
                    break
    return pairs


def amounts_and_preparation(df: pd.DataFrame, preds: dict, items: pd.DataFrame, cache, reps: int) -> dict:
    """Score matched foods and photo totals, reporting amount errors and the shares with an amount.

    Item coverage uses matched pairs with recorded grams at or above MIN_FOR_PERCENT. Photo coverage uses
    all scored photos; a total is available when at least one predicted food has a positive gram estimate.
    Preparation agreement and matched-line recall do not require an amount estimate.
    """
    code_of = dict(zip(cache["name_norm"], cache["food_code"], strict=True)) if cache is not None else {}
    foods = pd.read_parquet(data_dir() / "fndds" / "fndds_2017_2018_foods.parquet").set_index("food_code")
    by_img = dict(tuple(items.groupby("image_id")))
    ape, prep_ok, rec, eligible = [], [], [], 0
    for r in df.itertuples():
        gt = by_img[r.image_id].reset_index(drop=True)
        pred = preds[r.image_id].get("items") or []
        pairs = match_items(gt, pred, code_of, foods)
        rec.append(len(pairs) / len(gt) if len(gt) else 0.0)
        for gi, pi in pairs:
            g_true, g_pred = gt.at[gi, "grams"], amount(pred[pi])
            if g_true >= metrics.MIN_FOR_PERCENT:
                eligible += 1
                if g_pred is not None:
                    ape.append(abs(g_pred - g_true) / g_true)
            t = prep_terms(gt.at[gi, "description"])
            if t:
                prep_ok.append(bool(t & prep_terms(pred[pi].get("preparation"), pred[pi].get("name"))))
    df["component_recall"] = rec
    out = {"component_recall": float(np.mean(rec)) if rec else None,
           "item_grams_median_ape": float(np.median(ape)) if ape else None, "item_grams_pairs": len(ape),
           "item_grams_given": len(ape) / eligible if eligible else None,
           "preparation_agreement": float(np.mean(prep_ok)) if prep_ok else None,
           "preparation_pairs": len(prep_ok)}
    if reps:
        out["component_recall_ci"] = metrics.cluster_bootstrap(
            df, "subject_id", lambda d: float(d["component_recall"].mean()), reps)
    has = df["grams_pred"].notna().to_numpy()
    g = metrics.agreement(df["grams_pred"].to_numpy()[has], df["grams_total"].to_numpy()[has])
    out["photo_grams_median_ape"], out["photo_grams_within_20pct"] = g["median_ape"], g["within_20pct"]
    out["photo_grams_given"] = float(has.mean()) if len(has) else None
    return out
