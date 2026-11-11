"""Score the food pairs returned by align.py, without using food codes.

For each photo, make "same" and "type" pairs one-to-one: use each list item and record line once, taking
"same" before "type", then list order. Keep "part" pairs as supplied. At each grading level:

    correct         share of list items in a pair at that level or a "part" pair
    found (lines)   share of record lines in such a pair
    found (dishes)  share of recorded dishes (ASA24 FoodNum) with at least one line found
    F1              harmonic mean of correct and found (lines)

The "type" level includes "same" pairs. Average results across photos and resample participants for intervals.

Food-list corrections per photo are the changes indicated by the grader:
    add      record lines in no pair
    remove   list items in no pair
    rename   one-to-one pairs graded "type" (same kind of food)
Report their sum and the share of photos needing none. Count amount corrections separately: one-to-one
"same" pairs whose estimate differs from recorded grams by more than 25%.

One-to-one "same" pairs also supply:
    raw or cooked   agreement where the record line says raw or names a cooking method
    grams           |log(estimate / recorded)| (median), the signed median log ratio (bias, as a percentage),
                    for foods and for drinks separately (WWEIA milk and beverage categories); recorded amounts
                    below 10 g and missing estimates are excluded.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from snapme_eval import align
from snapme_eval.score import metrics

LEVELS = {"same": ("same",), "type": ("same", "type")}
COOKED = re.compile(r"\b(cooked|boiled|baked|fried|roasted|grilled|broiled|steamed|stewed|braised|sauteed|sautéed|"
                    r"scrambled|poached|toasted|microwaved|barbecued|smoked|stir[- ]fried|deep[- ]fried|"
                    r"pan[- ]fried|simmered|seared|mashed)\b", re.I)
RAW = re.compile(r"\braw\b|\buncooked\b|(?<!from )\bfresh\b", re.I)   # "from fresh" describes origin, not preparation


def raw_or_cooked(*texts) -> str | None:
    t = " ".join(str(x) for x in texts if x)
    c, r = bool(COOKED.search(t)), bool(RAW.search(t))
    return "cooked" if c and not r else "raw" if r and not c else None


def is_drink(wweia_code) -> bool:
    try:
        w = int(wweia_code)
    except (TypeError, ValueError):
        return False
    return 1002 <= w <= 1208 or 7002 <= w <= 7802


def one_to_one(pairs: list) -> tuple[list[tuple[int, int, str]], list[tuple[int, int]]]:
    """Keep same/type pairs one-to-one and return part pairs unchanged."""
    used_i, used_j, oo = set(), set(), []
    for g in ("same", "type"):
        for i, j, gg in sorted(pairs, key=lambda p: (p[0], p[1])):
            if gg == g and i not in used_i and j not in used_j:
                oo.append((i, j, g))
                used_i.add(i)
                used_j.add(j)
    return oo, [(i, j) for i, j, g in pairs if g == "part"]


def photo(gt: pd.DataFrame, items: list[dict], pairs: list) -> dict:
    """Score one photo; gt contains its ASA24 lines in record order."""
    oo, part = one_to_one(pairs)
    dish = gt["dish_id"].fillna(gt["line"].astype(str)).tolist()
    out = {}
    for lv, grades in LEVELS.items():
        ok_i = {i for i, _, g in oo if g in grades} | {i for i, _ in part}
        ok_j = {j for _, j, g in oo if g in grades} | {j for _, j in part}
        p = len(ok_i) / len(items) if items else 0.0
        r = len(ok_j) / len(gt) if len(gt) else 0.0
        dishes = set(dish)
        out[f"{lv}_correct"] = p
        out[f"{lv}_found_lines"] = r
        out[f"{lv}_found_dishes"] = len({dish[j] for j in ok_j}) / len(dishes) if dishes else 0.0
        out[f"{lv}_f1"] = 2 * p * r / (p + r) if p + r else 0.0
    in_i = {i for i, _, _ in oo} | {i for i, _ in part}
    in_j = {j for _, j, _ in oo} | {j for _, j in part}
    out["edits_add"] = float(len(gt) - len(in_j))
    out["edits_remove"] = float(len(items) - len(in_i))
    out["edits_rename"] = float(sum(g == "type" for _, _, g in oo))
    out["edits"] = out["edits_add"] + out["edits_remove"] + out["edits_rename"]
    out["no_edits"] = float(out["edits"] == 0)
    out["pairs_same"] = [(i, j) for i, j, g in oo if g == "same"]
    return out


def score(df: pd.DataFrame, preds: dict, items: pd.DataFrame, reps: int, model: str = "gpt-oss-120b") -> dict:
    """Add graded measures to df, one row per photo, and summarize; return empty if no lists are graded."""
    cache = align.load_cache()
    if not cache:
        return {}
    by_img = {k: g.sort_values("line").reset_index(drop=True) for k, g in items.groupby("image_id")}
    rows, prep, grams = [], [], []
    graded = 0
    for r in df.itertuples():
        gt = by_img[r.image_id]
        its = preds[r.image_id].get("items") or []
        if its:
            pairs = cache.get(align.key_for(model, gt["description"].tolist(), its))
            if pairs is None:
                rows.append(None)      # No saved grade for this list yet.
                continue
            graded += 1
        else:
            pairs = []                 # Empty or unreadable lists find no recorded foods.
        m = photo(gt, its, [tuple(p) for p in pairs])
        m["edits_amount"] = 0.0
        for i, j in m.pop("pairs_same"):
            rec = raw_or_cooked(gt.at[j, "description"])
            if rec:
                est = raw_or_cooked(its[i].get("preparation"), its[i].get("name"))
                prep.append(est == rec)
            g_true, g_pred = gt.at[j, "grams"], its[i].get("grams")
            if (isinstance(g_pred, (int, float)) and not isinstance(g_pred, bool) and g_pred > 0
                    and g_true >= metrics.MIN_FOR_PERCENT):
                grams.append((math.log(g_pred / g_true), is_drink(gt.at[j, "wweia_code"])))
                m["edits_amount"] += float(abs(g_pred / g_true - 1) > 0.25)
        rows.append(m)
    if graded == 0:
        return {}
    keep = [i for i, m in enumerate(rows) if m is not None]
    cols = [k for k in next(m for m in rows if m is not None)]
    sub = df.iloc[keep].copy()
    for c in cols:
        sub[c] = [rows[i][c] for i in keep]
        df[c] = [rows[i][c] if rows[i] is not None else np.nan for i in range(len(rows))]
    out = {"grader": align.VERSION, "grader_model": model, "photos_graded": len(keep), "photos": int(len(df))}
    for c in cols:
        out[c] = float(sub[c].mean())
        if reps and (c.endswith(("_f1", "_found_lines", "_found_dishes", "_correct")) or c in ("edits", "no_edits")):
            out[f"{c}_ci"] = metrics.cluster_bootstrap(sub, "subject_id", lambda d, c=c: float(d[c].mean()), reps)
    out["raw_or_cooked_agreement"] = float(np.mean(prep)) if prep else None
    out["raw_or_cooked_pairs"] = len(prep)
    for name, sel in (("foods", False), ("drinks", True)):
        v = np.array([x for x, d in grams if d == sel])
        out[f"grams_{name}_pairs"] = int(len(v))
        out[f"grams_{name}_median_abs_log"] = float(np.median(np.abs(v))) if len(v) else None
        # Convert the signed median log ratio to a percentage; +20 means estimates are 20% above the record.
        out[f"grams_{name}_bias_pct"] = float(100 * (math.exp(np.median(v)) - 1)) if len(v) else None
        out[f"grams_{name}_within_25pct"] = float(np.mean(np.abs(np.exp(v) - 1) <= 0.25)) if len(v) else None
    return out
