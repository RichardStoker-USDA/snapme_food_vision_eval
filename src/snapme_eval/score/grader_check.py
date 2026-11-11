"""Prepare a human review of the food-list grader and calculate agreement after review.

sheet() samples graded photo lists from the supplied runs, then individual item-line decisions. These are
pairs graded same, type or part, or unpaired items and lines from the same photo. Unpaired cases alternate
between ones that share a word, which are harder to distinguish, and ones that do not, where a list has both.
Sampling balances the four outcomes and takes lists from the runs in random order. It writes:

    grader_sheet.csv   id, photo, record line, list item, the photo's full record and list, and an empty grade column
    grader_key.json    id -> the grader's grade and where the decision came from (kept away from the person grading)

The reviewer enters same, type, part or different in grade, following the grader's rules from rules().
agreement() compares the completed sheet with the key. It reports a table of grades, the share agreeing and
Cohen's kappa for the four outcomes and for same versus the rest, with intervals from resampling photos.
"""
from __future__ import annotations

import csv
import json
import pathlib
import random
import re

import numpy as np
import pandas as pd

from snapme_eval import align

OUTCOMES = ("same", "type", "part", "different")
STOP = {"with", "and", "or", "of", "the", "a", "in", "on", "from", "ns", "nfs", "as", "to", "raw", "cooked", "not"}


def rules() -> str:
    """Extract the unchanged grading rules between the prompt's lists and reply format."""
    p = align.PROMPT
    return p[p.index("Give a pair for every"):p.index("Reply with JSON only")].strip()


def words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", str(t).lower()) if w not in STOP}


def decisions(lines: list[str], items: list[str], pairs: list) -> dict[str, list[tuple[int, int, bool]]]:
    """Group a list's decisions by outcome as (item, line, shares a word)."""
    out: dict[str, list] = {o: [] for o in OUTCOMES}
    paired = set()
    for i, j, g in pairs:
        out[g].append((int(i), int(j), True))
        paired.add((int(i), int(j)))
    for i, it in enumerate(items):
        for j, ln in enumerate(lines):
            if (i, j) not in paired:
                out["different"].append((i, j, bool(words(it) & words(ln))))
    return out


def draw(lists: list[dict], n: int, seed: int = 20261110) -> list[dict]:
    """Draw n decisions across outcomes, at most two per list. lists: {run, image_id, lines, items, pairs}."""
    rng = random.Random(seed)
    order = list(range(len(lists)))
    rng.shuffle(order)
    want = {o: n // len(OUTCOMES) + (k < n % len(OUTCOMES)) for k, o in enumerate(OUTCOMES)}
    used: dict[int, int] = {}
    rows = []
    for o in OUTCOMES:
        hard_next = True
        for k in order:
            if want[o] == 0:
                break
            if used.get(k, 0) >= 2:
                continue
            L = lists[k]
            pool = decisions(L["lines"], L["items"], L["pairs"])[o]
            if o == "different":   # Alternate between pairs with and without a shared word.
                pick = [d for d in pool if d[2] == hard_next] or pool
                hard_next = not hard_next
            else:
                pick = pool
            if not pick:
                continue
            i, j, _ = rng.choice(pick)
            rows.append({"run": L["run"], "image_id": L["image_id"], "item": i, "line": j, "grader": o,
                         "record_line": L["lines"][j], "list_item": L["items"][i],
                         "full_record": " | ".join(L["lines"]), "full_list": " | ".join(L["items"])})
            used[k] = used.get(k, 0) + 1
            want[o] -= 1
    rng.shuffle(rows)
    for k, r in enumerate(rows, 1):
        r["id"] = f"D{k:04d}"
    return rows


def graded_lists(runs: list[pathlib.Path], model: str = "gpt-oss-120b", gt_version: str = "snapme_v1") -> list[dict]:
    from snapme_eval.score.score_run import load_predictions
    cache, lines = align.load_cache(), align.record_lines(gt_version)
    out = []
    for run in runs:
        for image_id, p in sorted(load_predictions(run).items()):
            its = p.get("items")
            if not its or image_id not in lines:
                continue
            pairs = cache.get(align.key_for(model, lines[image_id], its))
            if pairs is not None:
                out.append({"run": f"{run.parent.name}/{run.name}", "image_id": image_id, "lines": lines[image_id],
                            "items": [align.item_text(i) for i in its], "pairs": pairs})
    return out


def sheet(runs: list[pathlib.Path], out: pathlib.Path, n: int = 400, seed: int = 20261110) -> dict:
    rows = draw(graded_lists(runs), n, seed)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["id", "image_id", "record_line", "list_item", "grade", "full_record", "full_list"]
    with (out / "grader_sheet.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "grade": ""})
    key = {r["id"]: {k: r[k] for k in ("grader", "run", "image_id", "item", "line")} for r in rows}
    (out / "grader_key.json").write_text(json.dumps(key, indent=1))
    (out / "grader_rules.txt").write_text(rules() + "\n")
    counts = pd.Series([r["grader"] for r in rows]).value_counts().to_dict()
    return {"decisions": len(rows), "by_grader_grade": counts, "runs": len({r["run"] for r in rows})}


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Cohen's kappa from two raters' labels."""
    cats = sorted(set(a) | set(b))
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in cats))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def agreement(sheet_csv: pathlib.Path, key_json: pathlib.Path, reps: int = 2000, seed: int = 20261110) -> dict:
    key = json.loads(key_json.read_text())
    d = pd.read_csv(sheet_csv, dtype=str).fillna("")
    d["person"] = d["grade"].str.strip().str.lower().replace({"kind": "type", "diff": "different", "none": "different"})
    d = d[d["person"].isin(OUTCOMES)].copy()
    d["grader"] = [key[i]["grader"] for i in d["id"]]
    a, b = d["person"].to_numpy(), d["grader"].to_numpy()
    photo = d["image_id"].to_numpy()
    groups = {p: np.where(photo == p)[0] for p in np.unique(photo)}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([groups[p] for p in rng.choice(keys, len(keys))]) for _ in range(reps)]

    def with_ci(fn) -> dict:
        v = [fn(a[ix], b[ix]) for ix in draws]
        return {"value": round(float(fn(a, b)), 4),
                "ci": [round(float(np.nanpercentile(v, q)), 4) for q in (2.5, 97.5)]}
    table = pd.crosstab(pd.Categorical(a, OUTCOMES), pd.Categorical(b, OUTCOMES), dropna=False)
    return {"decisions": int(len(d)), "photos": len(keys),
            "table_person_rows_grader_columns": {r: {c: int(table.at[r, c]) for c in OUTCOMES} for r in OUTCOMES},
            "agree": with_ci(lambda x, y: float(np.mean(x == y))),
            "kappa": with_ci(kappa),
            "kappa_same_or_not": with_ci(lambda x, y: kappa(x == "same", y == "same")),
            "kappa_paired_or_not": with_ci(lambda x, y: kappa(x != "different", y != "different"))}
