"""Compare two runs on shared photos and build the scale-card comparison table.

Read per-photo measures from each run's per_photo.parquet. Report run b minus run a, with a 95% interval from
participant resampling; per-photo measures share one set of resamples and amounts draw their own. For amounts,
use log(estimate / record): the median absolute value measures error; the signed median measures bias. Photo
totals require a positive estimate and at least 10 g recorded. Item amounts use record lines matched in both
runs, with a positive estimate in each; matching uses exact FNDDS food first, then WWEIA category.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pandas as pd

from snapme_eval.paths import data_dir, gt_dir
from snapme_eval.score.metrics import MIN_FOR_PERCENT
from snapme_eval.score.score_run import amount, load_predictions, mapping_cache, match_items

MEASURES = ("same_f1", "type_f1", "same_found_lines", "same_correct", "edits", "no_edits", "edits_amount",
            "wweia_f1", "code_f1", "component_recall", "token_f1", "n_pred")


def item_log_ratios(run: pathlib.Path, photos: set[str], gt_version: str = "snapme_v1") -> pd.DataFrame:
    preds = load_predictions(run)
    items = pd.read_parquet(gt_dir(gt_version) / "items.parquet")
    cache = mapping_cache()
    code_of = dict(zip(cache["name_norm"], cache["food_code"], strict=True)) if cache is not None else {}
    foods = pd.read_parquet(data_dir() / "fndds" / "fndds_2017_2018_foods.parquet").set_index("food_code")
    rows = []
    for img, gt in items[items["image_id"].isin(photos)].groupby("image_id"):
        pred = (preds.get(img) or {}).get("items") or []
        gt = gt.reset_index(drop=True)
        for gi, pi in match_items(gt, pred, code_of, foods):
            g, p = float(gt.at[gi, "grams"]), amount(pred[pi])
            if p is not None and g >= MIN_FOR_PERCENT:
                rows.append({"image_id": img, "line": int(gi), "lr": math.log(p / g)})
    return pd.DataFrame(rows, columns=["image_id", "line", "lr"])


def _amounts(lr_a: np.ndarray, lr_b: np.ndarray, subj: np.ndarray, rng, reps: int) -> dict:
    groups = {s: np.where(subj == s)[0] for s in np.unique(subj)}
    keys = list(groups)
    d = []
    for _ in range(reps):
        ix = np.concatenate([groups[s] for s in rng.choice(keys, len(keys))])
        d.append(np.median(np.abs(lr_b[ix])) - np.median(np.abs(lr_a[ix])))

    def bias(v):
        return round(float(100 * (math.exp(np.median(v)) - 1)), 2)
    return {"n": int(len(lr_a)), "a": round(float(np.median(np.abs(lr_a))), 4),
            "b": round(float(np.median(np.abs(lr_b))), 4),
            "b_minus_a": round(float(np.median(np.abs(lr_b)) - np.median(np.abs(lr_a))), 4),
            "ci": [round(float(np.percentile(d, q)), 4) for q in (2.5, 97.5)],
            "bias_pct_a": bias(lr_a), "bias_pct_b": bias(lr_b)}


def compare(a: pathlib.Path, b: pathlib.Path, photos: set[str] | None = None, reps: int = 2000,
            seed: int = 20261110, gt_version: str = "snapme_v1") -> dict:
    pa, pb = pd.read_parquet(a / "per_photo.parquet"), pd.read_parquet(b / "per_photo.parquet")
    both = set(pa["image_id"]) & set(pb["image_id"])
    both = sorted(both & photos if photos is not None else both)
    pa, pb = pa.set_index("image_id").loc[both], pb.set_index("image_id").loc[both]
    subj = pa["subject_id"].to_numpy()
    groups = {s: np.where(subj == s)[0] for s in np.unique(subj)}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([groups[s] for s in rng.choice(keys, len(keys))]) for _ in range(reps)]
    out: dict = {"photos": len(both), "participants": len(keys), "measures": {}}
    for m in MEASURES:
        if m not in pa or m not in pb:
            continue
        x, y = pa[m].to_numpy(float), pb[m].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            continue
        d = [np.mean(y[ix][ok[ix]]) - np.mean(x[ix][ok[ix]]) for ix in draws]
        out["measures"][m] = {"n": int(ok.sum()), "a": round(float(x[ok].mean()), 4),
                              "b": round(float(y[ok].mean()), 4),
                              "b_minus_a": round(float(y[ok].mean() - x[ok].mean()), 4),
                              "ci": [round(float(np.percentile(d, q)), 4) for q in (2.5, 97.5)]}
    out["parsed"] = {"a": round(float(pa["parsed"].mean()), 4), "b": round(float(pb["parsed"].mean()), 4)}
    # Compare total grams on eligible photos.
    t = pa["grams_total"].to_numpy(float)
    x, y = pa["grams_pred"].to_numpy(float), pb["grams_pred"].to_numpy(float)
    ok = (t >= MIN_FOR_PERCENT) & (x > 0) & (y > 0)
    if ok.sum() >= 20:
        out["photo_grams"] = _amounts(np.log(x[ok] / t[ok]), np.log(y[ok] / t[ok]), subj[ok], rng, reps)
    # Compare amounts for record lines matched in both runs.
    ia, ib = item_log_ratios(a, set(both), gt_version), item_log_ratios(b, set(both), gt_version)
    out["item_grams_pairs"] = {"a": len(ia), "b": len(ib)}
    pair = ia.merge(ib, on=["image_id", "line"], suffixes=("_a", "_b"))
    if len(pair) >= 20:
        out["item_grams"] = _amounts(pair["lr_a"].to_numpy(), pair["lr_b"].to_numpy(),
                                     pa.loc[pair["image_id"], "subject_id"].to_numpy(), rng, reps)
    return out


def public(model_id: str) -> bool:
    """Select registered models for reporting; private runs require SNAPME_REPORT_PRIVATE=1."""
    import os

    from snapme_eval import registry
    m = registry.models().get(model_id)
    return m is not None and (not m.get("private") or os.environ.get("SNAPME_REPORT_PRIVATE") == "1")


def marker_ablation(runs_root: pathlib.Path, gt_version: str = "snapme_v1") -> dict | None:
    """Compare each scale-card condition with the same model's standard run on the card table's photos."""
    from snapme_eval import marker
    from snapme_eval.systems.marker import ARMS
    if not marker.TABLE.exists():
        return None
    photos = set(marker.table().index)
    base = runs_root / gt_version / "direct_v2"
    models = []
    for ref in sorted(base.glob("*/per_photo.parquet")):
        if not public(ref.parent.name):
            continue
        arms = {}
        for arm in ARMS:
            run = runs_root / gt_version / f"marker_v1_{arm}" / ref.parent.name
            if (run / "per_photo.parquet").exists():
                c = compare(ref.parent, run, photos, gt_version=gt_version)
                if c["photos"] >= 0.98 * len(photos):   # Include completed conditions.
                    arms[arm] = c
        if arms:
            models.append({"model": ref.parent.name, "arms": arms})
    if not models:
        return None
    t = pd.read_csv(marker.TABLE)
    return {"photos": len(photos), "located": int(len(t)), "iou_min": marker.IOU_MIN,
            "checker_min": marker.CHECKER_MIN, "margin": marker.MARGIN,
            "arms": {k: {"hidden": v[0], "prompt_names_card": v[1]} for k, v in ARMS.items()}, "models": models}


def before_after(runs_root: pathlib.Path, gt_version: str = "snapme_v1") -> list[dict]:
    """Compare each model's two-photo run (b) with its one-photo run (a) on meal photos that have an after photo.
    Package fronts with a label photo are left out. ASA24 records grams eaten, so the second photo can help
    with the amount estimate."""
    images = pd.read_parquet(gt_dir(gt_version) / "images.parquet")
    meal = images["companion_exists"].astype(bool) & (images["photo_type"] == "before")
    photos = set(images.loc[meal, "image_id"])
    out = []
    for two in sorted((runs_root / gt_version / "consumed_v2").glob("*/per_photo.parquet")):
        one = runs_root / gt_version / "direct_v2" / two.parent.name
        if public(two.parent.name) and (one / "per_photo.parquet").exists():
            c = compare(one, two.parent, photos, gt_version=gt_version)
            if c["photos"] >= 0.98 * len(photos):
                out.append({"model": two.parent.name, **c})
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    x = ap.parse_args()
    print(json.dumps(compare(pathlib.Path(x.a), pathlib.Path(x.b)), indent=1))


if __name__ == "__main__":
    main()
