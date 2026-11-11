"""Collect scored runs into results/<gt_version>/ for the site and the release.

results/snapme_v1/leaderboard.json   one row per (pipeline, model): identification, matched-line, preparation and
                                     amount measures, available intervals and model metadata
results/snapme_v1/leaderboard.csv    the same table without intervals
results/snapme_v1/runs/<pipeline>__<model>.json   full scores and per-photo measures
results/snapme_v1/baselines_2023.json  Inverse Cooking and Im2Recipe as published with the SNAPMe paper, rescored
                                     here with the same code and participant intervals
results/snapme_v1/not_scored.json    registry models without a full direct run, with the reason
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import shutil

import pandas as pd

from snapme_eval import registry
from snapme_eval.paths import data_dir, gt_dir
from snapme_eval.score import metrics, tokens

PHOTO_COLS = ("image_id", "photo_type", "occasion", "food_code_bin", "split", "parsed", "n_pred", "code_f1",
              "wweia_f1", "subgroup_f1", "token_f1", "component_recall", "grams_pred", "grams_total",
              "same_f1", "same_correct", "same_found_lines", "same_found_dishes",
              "type_f1", "type_correct", "type_found_lines", "type_found_dishes",
              "edits", "edits_add", "edits_remove", "edits_rename", "edits_amount")


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def current(pipeline: str) -> bool:
    """True for runs of a pipeline's current version ("direct_v2", "check_v2_direct_reader", ...)."""
    import importlib
    system, _, rest = pipeline.partition("_")
    try:
        version = importlib.import_module(f"snapme_eval.systems.{system}").VERSION
    except ModuleNotFoundError:
        return False
    return rest == version or rest.startswith(version + "_")


def row_for(pipeline: str, model_id: str, m: dict, meta: dict, s: dict) -> dict:
    ident = s.get("identification") if isinstance(s.get("identification"), dict) else {}
    mt = s.get("matched") or {}
    t23 = s.get("token_f1_2023") or {}

    def lv(level, key):
        return (ident.get(level) or {}).get(key)
    return {
        "pipeline": pipeline, "model": model_id, "hf_repo": m["hf_repo"], "revision": m["revision"],
        "developer": m["developer"], "country": m["country"], "released": m["released"], "license": m["license"],
        "architecture": m["architecture"], "params_total_b": m["params_total_b"],
        "params_active_b": m.get("params_active_b"), "weights_dtype": m["weights_dtype"],
        "weights_gb": m.get("weights_gb"), "role": m.get("role", "generalist"), "precision_of": m.get("precision_of"),
        "engine": meta.get("engine"), "gpus": meta.get("gpus"), "partition": meta.get("partition"),
        "seconds": meta.get("seconds"), "finished_utc": meta.get("finished_utc"),
        "check_source": meta.get("check_source") or None, "ablation": meta.get("ablation_note"),
        "n_images": s["n_images"], "format_compliance": _r(s["format_compliance"]),
        "code_f1": _r(lv("food_code", "f1_photo_mean")), "code_f1_ci": lv("food_code", "f1_photo_mean_ci"),
        "wweia_f1": _r(lv("wweia_code", "f1_photo_mean")), "wweia_f1_ci": lv("wweia_code", "f1_photo_mean_ci"),
        "subgroup_f1": _r(lv("group2", "f1_photo_mean")), "subgroup_f1_ci": lv("group2", "f1_photo_mean_ci"),
        "group_f1": _r(lv("group1", "f1_photo_mean")),
        "wweia_precision": _r(lv("wweia_code", "precision")), "wweia_recall": _r(lv("wweia_code", "recall")),
        "wweia_f1_by_food_count": lv("wweia_code", "by_food_code_bin"),
        "wweia_f1_by_photo_type": lv("wweia_code", "by_photo_type"),
        "token_f1": _r(s["token_f1"]["mean"]), "token_f1_ci": s["token_f1"].get("ci"),
        "token_f1_2023": _r(t23.get("mean")), "token_f1_2023_ci": t23.get("ci"), "token_f1_2023_n": t23.get("n"),
        "token_f1_2023_by_food_count": t23.get("by_food_code_bin"),
        "component_recall": _r(mt.get("component_recall")), "component_recall_ci": mt.get("component_recall_ci"),
        "preparation_agreement": _r(mt.get("preparation_agreement")), "preparation_pairs": mt.get("preparation_pairs"),
        "item_grams_median_ape": _r(mt.get("item_grams_median_ape")), "item_grams_pairs": mt.get("item_grams_pairs"),
        "photo_grams_median_ape": _r(mt.get("photo_grams_median_ape")),
        "photo_grams_within_20pct": _r(mt.get("photo_grams_within_20pct")),
        "item_grams_given": _r(mt.get("item_grams_given")), "photo_grams_given": _r(mt.get("photo_grams_given")),
        "unmapped_names": s.get("unmapped_names"),
        # Same-food and same-kind measures from score/same_food.py, withheld until grading is complete.
        **same_food_fields(s.get("same_food") or {}),
    }


def same_food_fields(sf: dict) -> dict:
    """Return graded measures once grading covers at least 99.5% of the run's photos.
    Until then, return counts only to withhold means from partly graded runs."""
    done = (sf.get("photos_graded") or 0) >= 0.995 * (sf.get("photos") or 1) and "same_f1" in sf
    if not done:
        return {k: sf[k] for k in ("photos_graded", "photos") if k in sf}
    return {k: (_r(v) if isinstance(v, float) else v) for k, v in sf.items()}


def rank_intervals(out: pathlib.Path, rows: list[dict], gt_version: str, col: str = "wweia_f1",
                   reps: int = 2000, seed: int = 20261109) -> None:
    """Add intervals for ranks and paired differences to Table 1 runs (one photo, reference model weight format).

    Resample participants with replacement, keeping each person's photos together. Use the same sample for
    every run to calculate paired differences. Rank 1 has the highest mean across photos.
    Add <col>_rank, <col>_rank_ci and <col>_diff_top_ci, the interval for the difference from the top-ranked run.
    """
    import numpy as np
    table = [r for r in rows if r["pipeline"] == "direct_v2" and not r.get("precision_of") and r.get(col) is not None]
    if len(table) < 2:
        return
    subj = pd.read_parquet(gt_dir(gt_version) / "images.parquet").set_index("image_id")["subject_id"]
    per = {}
    for r in table:
        d = json.loads((out / "runs" / f"{r['pipeline']}__{r['model']}.json").read_text())
        per[r["model"]] = pd.DataFrame(d.get("photos") or []).set_index("image_id")[col]
    m = pd.DataFrame(per).dropna()                      # retain photos scored by all Table 1 runs
    g = subj.reindex(m.index).to_numpy()
    keys, inv = np.unique(g, return_inverse=True)
    sums = np.zeros((len(keys), m.shape[1]))
    np.add.at(sums, inv, m.to_numpy())
    counts = np.bincount(inv, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=reps).astype(float)
    means = (w @ sums) / (w @ counts)[:, None]           # rows are resamples; columns are runs
    ranks = (-means).argsort(axis=1).argsort(axis=1) + 1
    point = m.mean().to_numpy()
    top = int(point.argmax())
    order = (-point).argsort().argsort() + 1
    diff = means - means[:, [top]]
    for j, model in enumerate(m.columns):
        r = next(x for x in table if x["model"] == model)
        r[f"{col}_rank"] = int(order[j])
        r[f"{col}_rank_ci"] = [int(np.percentile(ranks[:, j], 2.5)), int(np.percentile(ranks[:, j], 97.5))]
        r[f"{col}_rank_first_share"] = round(float((ranks[:, j] == 1).mean()), 4)
        r[f"{col}_diff_top_ci"] = None if j == top else [round(float(np.percentile(diff[:, j], 2.5)), 4),
                                                           round(float(np.percentile(diff[:, j], 97.5)), 4)]


def baselines_2023(gt_version: str, reps: int = 1000) -> list[dict]:
    """The paper's two ingredient-prediction baselines, from the authors' per-photo outputs, with participant
    intervals and the same food-count strata as the models."""
    images = pd.read_parquet(gt_dir(gt_version) / "images.parquet")[["filename", "subject_id", "food_code_bin"]]
    out = []
    for name, f, col in (("Inverse Cooking", "snapme_inv_cook_f1_score.csv", "clean_inv_cook"),
                         ("Im2Recipe", "snapme_im2r_f1_score.csv", "clean_im2r")):
        p = data_dir() / "larke_2023" / f
        if not p.exists():
            continue
        d = pd.read_csv(p)
        pred_col = col if col in d else next(c for c in d.columns
                                             if c.startswith("clean_") and c != "clean_description")
        d["f1"] = [tokens.f1(set(ast.literal_eval(t)), set(ast.literal_eval(q)))
                   for t, q in zip(d["clean_description"], d[pred_col], strict=True)]
        d = d.merge(images, on="filename", how="left")
        row = {"method": name, "source": "Larke et al. 2023, Nutrients 15:4972; github.com/JulesLarke-USDA/SNAPMe",
               "n": int(len(d)), "token_f1_2023": _r(d["f1"].mean()),
               "published_mean": _r(d["F1_score"].mean()),
               "by_food_count": d.groupby("food_code_bin")["f1"].mean().round(4).to_dict()}
        if reps and d["subject_id"].notna().all():
            row["token_f1_2023_ci"] = metrics.cluster_bootstrap(d, "subject_id", lambda x: float(x["f1"].mean()), reps)
        out.append(row)
    return out


def collect_nutrition5k(runs_root: pathlib.Path, results: pathlib.Path) -> list[dict]:
    """Nutrition5k test-split runs: one row per pipeline and model (score_nutrition5k)."""
    rows = []
    for sp in sorted((runs_root / "nutrition5k_v1").glob("*/*/scores.json")):
        pipeline, model_id = sp.parent.parent.name, sp.parent.name
        if pipeline.endswith("_smoke") or not current(pipeline):
            continue
        try:
            m = registry.model(model_id)
        except SystemExit:
            continue
        if m.get("private") and os.environ.get("SNAPME_REPORT_PRIVATE") != "1":
            continue
        s = json.loads(sp.read_text())
        if s.get("names_not_in_cache") != 0:      # scored before food coding finished, or by an older scorer
            raise SystemExit(f"{sp}: names_not_in_cache={s.get('names_not_in_cache')}; run map, then "
                             f"score --run {sp.parent} --gt nutrition5k_v1")
        rows.append({"pipeline": pipeline, "model": model_id, "hf_repo": m["hf_repo"], "developer": m["developer"],
                     "n_images": s["n_images"], "format_compliance": _r(s["format_compliance"]),
                     **{f"{k}": _r(s[k]["mean"]) for k in ("wweia_f1", "code_f1", "token_f1")},
                     **{f"{k}_ci": s[k].get("ci") for k in ("wweia_f1", "code_f1", "token_f1")},
                     "dish_grams_median_ape": _r(s["dish_grams"]["median_ape"]),
                     "dish_grams_within_20pct": _r(s["dish_grams"]["within_20pct"]),
                     "dish_grams_given": _r(s["dish_grams"].get("given"))})
    rows.sort(key=lambda r: -(r["wweia_f1"] or 0))
    out = results / "nutrition5k_v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "leaderboard.json").write_text(json.dumps(rows, indent=1, default=float))
    return rows


def collect(runs_root: pathlib.Path, results: pathlib.Path, gt_version: str = "snapme_v1") -> None:
    out = results / gt_version
    # Every file under runs/ and site/public/data (including the explorer data) is rewritten here, so earlier versions'
    # files are cleared first rather than left behind.
    shutil.rmtree(out / "runs", ignore_errors=True)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    build = gt_dir(gt_version) / "build.json"
    if build.exists():
        (out / "gt_build.json").write_text(build.read_text())
    n_total = len(pd.read_parquet(gt_dir(gt_version) / "images.parquet"))
    rows, partial = [], []
    for sp in sorted((runs_root / gt_version).glob("*/*/scores.json")):
        pipeline, model_id = sp.parent.parent.name, sp.parent.name
        if pipeline.endswith("_smoke") or pipeline.startswith(".") or not current(pipeline):
            continue
        s = json.loads(sp.read_text())
        if "token_f1" not in s or "nutrients_photo" in s:   # scored by an earlier version of this module
            continue
        if s["n_images"] < n_total:   # a run still in progress (scored early by the mapper job)
            partial.append({"pipeline": pipeline, "model": model_id, "done": s["n_images"], "total": n_total})
            continue
        try:
            m = registry.model(model_id)
        except SystemExit:
            continue
        if m.get("private") and os.environ.get("SNAPME_REPORT_PRIVATE") != "1":   # models kept off the public site
            continue
        # Food-checking runs are named by their decision model; every other pipeline needs a food-record model.
        if m.get("role", "generalist") != "generalist" and not pipeline.startswith("check"):
            continue
        runs = sorted(sp.parent.glob("run_*.json"))
        meta = json.loads(runs[-1].read_text()) if runs else {}
        row = row_for(pipeline, model_id, m, meta, s)
        rows.append(row)
        # Published results name runs by pipeline/model, not by a cluster path.
        detail = {"row": row, "scores": {**s, "run": f"{gt_version}/{pipeline}/{model_id}"}}
        if "same_food" in s:
            detail["scores"]["same_food"] = same_food_fields(s["same_food"])
        pp = sp.parent / "per_photo.parquet"
        if pp.exists():
            d = pd.read_parquet(pp)
            # to_json writes missing values as null (json.dumps would write NaN, which is not valid JSON)
            detail["photos"] = json.loads(d[[c for c in PHOTO_COLS if c in d]].round(4).to_json(orient="records"))
        (out / "runs" / f"{pipeline}__{model_id}.json").write_text(json.dumps(detail, default=float))
    # Once Table 1 has graded results, sort by same-food F1, then ungraded runs by WWEIA category F1.
    # Rank each measure only among runs that supply it.
    table1 = [r for r in rows if r["pipeline"] == "direct_v2" and not r.get("precision_of")]
    primary = "same_f1" if any(r.get("same_f1") is not None for r in table1) else "wweia_f1"
    for r in rows:
        r["primary"] = primary
    rows.sort(key=lambda r: (r["pipeline"], r.get(primary) is None, -(r.get(primary) or 0), -(r["wweia_f1"] or 0)))
    for col in ("wweia_f1", "same_f1", "type_f1"):
        rank_intervals(out, rows, gt_version, col=col)
    direct = {r["model"] for r in rows if r["pipeline"].startswith("direct_")}
    missing = [{"model": k, "hf_repo": v["hf_repo"], "developer": v["developer"],
                "note": v.get("status_note") or next((f"run in progress ({p['done']:,} of {p['total']:,} photos)"
                                                       for p in partial if p["model"] == k
                                                       and p["pipeline"].startswith("direct_")), "run in progress")}
               for k, v in registry.models().items()
               if v.get("role", "generalist") == "generalist" and not v.get("private") and k not in direct]
    (out / "not_scored.json").write_text(json.dumps(missing, indent=1))
    (out / "leaderboard.json").write_text(json.dumps(rows, indent=1, default=float))
    (out / "baselines_2023.json").write_text(json.dumps(baselines_2023(gt_version), indent=1, default=float))
    flat = [{k: v for k, v in r.items() if not k.endswith("_ci") and not isinstance(v, dict)} for r in rows]
    pd.DataFrame(flat).to_csv(out / "leaderboard.csv", index=False)
    site_data = pathlib.Path(__file__).resolve().parents[3] / "site" / "public" / "data"
    shutil.rmtree(site_data, ignore_errors=True)
    site_data.mkdir(parents=True, exist_ok=True)
    # Explorer data for the site's Explore page (photos come from `thumbs` into site/public/photos)
    from snapme_eval.score import explorer
    explorer.build(site_data / "explorer", rows, gt_version)
    for f in ("leaderboard.json", "leaderboard.csv", "baselines_2023.json", "not_scored.json", "gt_build.json",
              "retrieval_embeddinggemma2.json"):
        if (out / f).exists():
            (site_data / f).write_text((out / f).read_text())
    from snapme_eval.score import mapcheck
    mc = mapcheck.check()
    if mc:
        (out / "mapping_check.json").write_text(json.dumps(mc, indent=1))
        (site_data / "mapping_check.json").write_text(json.dumps(mc, indent=1))
    from snapme_eval.score import paired
    ma = paired.marker_ablation(runs_root, gt_version)
    if ma:
        (out / "marker_ablation.json").write_text(json.dumps(ma, indent=1))
        (site_data / "marker_ablation.json").write_text(json.dumps(ma, indent=1))
    ba = paired.before_after(runs_root, gt_version)
    if ba:
        (out / "before_after.json").write_text(json.dumps(ba, indent=1))
        (site_data / "before_after.json").write_text(json.dumps(ba, indent=1))
    n5k = collect_nutrition5k(runs_root, results)
    (site_data / "nutrition5k.json").write_text(json.dumps(n5k, indent=1, default=float))
    print(f"{len(rows)} scored runs -> {out}")
