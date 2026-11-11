"""Agreement statistics between predicted and recorded intake."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MIN_FOR_PERCENT = 10.0  # percent errors only where the recorded amount is at least this large (g)


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    d = vx + vy + (mx - my) ** 2
    return float(2 * cov / d) if d > 0 else float("nan")


def agreement(pred: np.ndarray, true: np.ndarray) -> dict:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    err = pred - true
    big = true >= MIN_FOR_PERCENT
    ape = np.abs(err[big]) / true[big]
    sd = err.std(ddof=1) if len(err) > 1 else float("nan")
    return {
        "n": int(len(true)),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "loa_low": float(err.mean() - 1.96 * sd),   # Bland-Altman limits of agreement
        "loa_high": float(err.mean() + 1.96 * sd),
        "median_ape": float(np.median(ape)) if len(ape) else float("nan"),
        "within_20pct": float((ape <= 0.20).mean()) if len(ape) else float("nan"),
        "ccc": lins_ccc(pred, true),
        "spearman": float(stats.spearmanr(pred, true).statistic) if len(true) > 2 else float("nan"),
    }


def cluster_bootstrap(df: pd.DataFrame, cluster: str, fn, reps: int = 1000, seed: int = 20261108) -> tuple:
    """Return a 95% percentile interval for fn(df), resampling whole clusters with replacement."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = np.array(list(groups))
    vals = []
    for _ in range(reps):
        pick = rng.choice(keys, size=len(keys), replace=True)
        vals.append(fn(pd.concat([groups[k] for k in pick], ignore_index=True)))
    vals = np.array(vals, float)
    vals = vals[~np.isnan(vals)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (None, None)
