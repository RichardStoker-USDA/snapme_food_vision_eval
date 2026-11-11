"""Compare mapped ASA24 descriptions with their recorded food codes.

Only descriptions already in the mapping cache are checked. Exact-code agreement uses lines whose
recorded code is in FNDDS 2017-2018; category agreement uses all checked lines, including those without
a recorded WWEIA category. This check does not estimate or bound accuracy on model-written names.
"""
from __future__ import annotations

import pandas as pd

from snapme_eval.paths import data_dir, gt_dir
from snapme_eval.score.score_run import mapping_cache, norm


def check(gt_version: str = "snapme_v1") -> dict | None:
    cache = mapping_cache()
    if cache is None:
        return None
    code_of = dict(zip(cache["name_norm"], cache["food_code"], strict=True))
    foods = pd.read_parquet(data_dir() / "fndds" / "fndds_2017_2018_foods.parquet").set_index("food_code")
    items = pd.read_parquet(gt_dir(gt_version) / "items.parquet")
    items = items.assign(name_norm=items["description"].map(norm))
    items = items[items["name_norm"].isin(code_of)]
    if items.empty:
        return None
    got = items["name_norm"].map(code_of)
    in_fndds = items["food_code"].isin(foods.index)
    got_w = got.map(lambda c: foods.at[int(c), "wweia_code"] if c == c and int(c) in foods.index else None)
    same_code = (got == items["food_code"]) & in_fndds
    same_w = got_w == items["wweia_code"]
    d = items.drop_duplicates("name_norm")
    dg = d["name_norm"].map(code_of)
    return {"lines": int(len(items)), "lines_fndds_code": int(in_fndds.sum()),
            "same_code_share": round(float(same_code[in_fndds].mean()), 4),
            "same_wweia_share": round(float(same_w.mean()), 4),
            "no_match_share": round(float(got.isna().mean()), 4),
            "descriptions": int(len(d)),
            "descriptions_same_code_share": round(float(((dg == d["food_code"]) & d["food_code"].isin(foods.index))[
                d["food_code"].isin(foods.index)].mean()), 4)}
