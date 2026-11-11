"""Evaluate food descriptions retrieved by photo-to-text vector similarity.

For each k in 5, 15 and 50, food-code recall is the share of all ASA24 lines whose code is retrieved.
Category recall uses only lines with a recorded WWEIA category. Values are averaged over photos;
intervals resample participants. Reference records are used for scoring, not ranking descriptions.

    python -m snapme_eval.score.retrieval [--model embeddinggemma2]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from snapme_eval.paths import REPO, data_dir, gt_dir
from snapme_eval.score import metrics

KS = (5, 15, 50)


def evaluate(model: str = "embeddinggemma2", gt_version: str = "snapme_v1", reps: int = 1000) -> dict:
    e = data_dir() / "embeddings"
    f = np.load(e / f"{model}_fndds.npz")
    p = np.load(e / f"{model}_photos_{gt_version}.npz", allow_pickle=True)
    codes, fv = f["codes"], f["vectors"]
    foods = pd.read_parquet(data_dir() / "fndds" / "fndds_2017_2018_foods.parquet").set_index("food_code")
    wweia = foods.loc[codes, "wweia_code"].to_numpy()
    items = pd.read_parquet(gt_dir(gt_version) / "items.parquet")
    images = pd.read_parquet(gt_dir(gt_version) / "images.parquet").set_index("image_id")
    by_img = dict(tuple(items.groupby("image_id")))
    order = np.argsort(-(p["vectors"] @ fv.T), axis=1)[:, :max(KS)]
    rows = []
    for img, top in zip(p["image_ids"], order, strict=True):
        gt = by_img.get(img)
        if gt is None:
            continue
        r = {"image_id": img, "subject_id": images.at[img, "subject_id"], "photo_type": images.at[img, "photo_type"]}
        for k in KS:
            tc, tw = set(codes[top[:k]].tolist()), set(wweia[top[:k]].tolist())
            r[f"code@{k}"] = float(np.mean([c in tc for c in gt["food_code"]]))
            ws = gt["wweia_code"].dropna()
            r[f"wweia@{k}"] = float(np.mean([w in tw for w in ws])) if len(ws) else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    out = {"model": model, "gt_version": gt_version, "n_photos": len(df), "k": list(KS), "recall": {}}
    for col in [c for c in df.columns if "@" in c]:
        d = df.dropna(subset=[col])
        out["recall"][col] = {"mean": round(float(d[col].mean()), 4),
                              "by_photo_type": d.groupby("photo_type")[col].mean().round(4).to_dict()}
        if reps:
            out["recall"][col]["ci"] = metrics.cluster_bootstrap(d, "subject_id",
                                                                 lambda x, c=col: float(x[c].mean()), reps)
    res = REPO / "results" / gt_version
    res.mkdir(parents=True, exist_ok=True)
    (res / f"retrieval_{model}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="embeddinggemma2")
    a = ap.parse_args()
    print(json.dumps(evaluate(a.model)["recall"], indent=1))
