"""Score a run against Nutrition5k's weighed ingredients and dish masses.

Identification uses reference ingredients of at least 5 g. Predicted names and listed ingredients, and
reference ingredient names, use the shared FNDDS mapper. F1 compares sets of distinct mapped codes or
WWEIA categories. Token F1 compares cleaned word sets. Identification values are averaged over dishes,
with intervals from resampling dishes. Amount errors compare summed positive estimates with full dish mass,
omitting dishes with no amount supplied. dish_grams.given is the share of scored dishes with an amount.
names_not_in_cache counts predicted and reference names the name-to-code cache has never seen; such names get
no code, so the report accepts a run only when the count is 0 (run `map`, then `score` again).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from snapme_eval.paths import data_dir, gt_dir
from snapme_eval.score import metrics, tokens
from snapme_eval.score.score_run import load_predictions, mapping_cache, norm, photo_grams

MIN_GRAMS = 5.0
LEVELS = (("food_code", "code"), ("wweia_code", "wweia"))


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def score(run: pathlib.Path, gt_version: str = "nutrition5k_v1", reps: int = 1000) -> dict:
    g = gt_dir(gt_version)
    dishes = pd.read_parquet(g / "images.parquet")
    items = pd.read_parquet(g / "items.parquet")
    preds = load_predictions(run)
    dishes = dishes[dishes["dish_id"].isin(preds)].copy()
    ref = items[items["grams"] >= MIN_GRAMS]
    cache = mapping_cache()
    code_of = dict(zip(cache["name_norm"], cache["food_code"], strict=True)) if cache is not None else {}
    foods = pd.read_parquet(data_dir() / "fndds" / "fndds_2017_2018_foods.parquet").set_index("food_code")

    not_in_cache: set[str] = set()

    def codes(names) -> dict[str, set[int]]:
        keys = [norm(n) for n in names]
        not_in_cache.update(k for k in keys if k not in code_of)
        found = (code_of.get(k) for k in keys)
        cs = {int(v) for v in found if v is not None and v == v and int(v) in foods.index}   # v == v drops NaN
        return {"food_code": cs, "wweia_code": {int(foods.at[c, "wweia_code"]) for c in cs}}

    by_dish = dict(tuple(ref.groupby("dish_id")))
    rows = []
    for d in dishes.itertuples():
        p = preds[d.dish_id]
        its = p.get("items")
        names = [it.get("name", "") for it in its or []]
        ings = [x for it in its or [] for x in (it.get("ingredients") or [])]
        r = by_dish.get(d.dish_id)
        ref_names = list(r["name"]) if r is not None else []
        pc, rc = codes(names + ings), codes(ref_names)
        row = {"dish_id": d.dish_id, "parsed": its is not None, "n_pred": len(its or []), "n_ref": len(ref_names),
               "grams_total": d.grams_total,
               "grams_pred": photo_grams(its)}
        for lv, key in LEVELS:
            prec = len(pc[lv] & rc[lv]) / len(pc[lv]) if pc[lv] else 0.0
            rec = len(pc[lv] & rc[lv]) / len(rc[lv]) if rc[lv] else 0.0
            row[f"{key}_f1"] = _f1(prec, rec) if rc[lv] else np.nan
        truth = set().union(*(tokens.clean(n) for n in ref_names)) if ref_names else set()
        pred_t = set().union(*(tokens.clean(n) for n in names + ings)) if names or ings else set()
        row["token_f1"] = tokens.f1(truth, pred_t) if truth else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    out: dict = {"run": str(run), "gt_version": gt_version, "n_images": int(len(df)),
                 "format_compliance": float(df["parsed"].mean()) if len(df) else 0.0, "min_grams": MIN_GRAMS,
                 "names_not_in_cache": len(not_in_cache)}
    for col in ("wweia_f1", "code_f1", "token_f1"):
        out[col] = {"mean": float(df[col].mean()), "n": int(df[col].notna().sum())}
        if reps:
            out[col]["ci"] = metrics.cluster_bootstrap(df.dropna(subset=[col]), "dish_id",
                                                       lambda x, c=col: float(x[c].mean()), reps)
    has = df["grams_pred"].notna().to_numpy()
    a = metrics.agreement(df["grams_pred"].to_numpy()[has], df["grams_total"].to_numpy()[has])
    out["dish_grams"] = {**{k: a[k] for k in ("median_ape", "within_20pct", "ccc", "bias")},
                         "given": float(has.mean()) if len(has) else None}
    out["summary"] = {"n_images": out["n_images"], "format_compliance": round(out["format_compliance"], 4),
                      **{c: round(out[c]["mean"], 3) for c in ("wweia_f1", "code_f1", "token_f1")},
                      "dish_grams_median_ape": round(a["median_ape"], 3)}
    df.to_parquet(run / "per_photo.parquet", index=False)
    (run / "scores.json").write_text(json.dumps(out, indent=1, default=float))
    return out
