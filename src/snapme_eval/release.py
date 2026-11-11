"""Package result tables, reference records and predictions into a dated release.

The package contains leaderboard.csv, per_photo.csv, models.csv, available 2023 baseline scores,
reference tables under ground_truth/, and raw predictions under predictions/<pipeline>/<model>.jsonl.
A generated README identifies the packaging commit. Raw photo files are not copied.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import time

import pandas as pd

from snapme_eval import registry
from snapme_eval.paths import REPO, gt_dir, project, runs_dir


def package(out: pathlib.Path, gt_version: str = "snapme_v1") -> pathlib.Path:
    res = REPO / "results" / gt_version
    rows = json.loads((res / "leaderboard.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(res / "leaderboard.csv", out / "leaderboard.csv")

    long = []
    for r in rows:
        d = json.loads((res / "runs" / f"{r['pipeline']}__{r['model']}.json").read_text())
        for ph in d.get("photos", []):
            long.append({"pipeline": r["pipeline"], "model": r["model"], **ph})
    pd.DataFrame(long).to_csv(out / "per_photo.csv", index=False)
    if (res / "baselines_2023.json").exists():
        shutil.copy(res / "baselines_2023.json", out / "baselines_2023.json")

    models = []
    for r in rows:
        m = registry.model(r["model"])
        models.append({k: m.get(k) for k in ("id", "hf_repo", "revision", "developer", "country", "released",
                                              "license", "architecture", "params_total_b", "params_active_b",
                                              "weights_dtype", "engine")})
    pd.DataFrame(models).drop_duplicates("id").to_csv(out / "models.csv", index=False)

    g = out / "ground_truth"
    g.mkdir(exist_ok=True)
    # The reference as the benchmark uses it: foods, codes, categories and grams.
    for t in ("images", "items"):
        pd.read_parquet(gt_dir(gt_version) / f"{t}.parquet").to_csv(g / f"{t}.csv", index=False)
    shutil.copy(gt_dir(gt_version) / "build.json", g / "build.json")

    for r in rows:
        src = runs_dir() / gt_version / r["pipeline"] / r["model"] / "predictions.jsonl"
        dst = out / "predictions" / r["pipeline"] / f"{r['model']}.jsonl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    p = project()["project"]
    pipelines = sorted({r["pipeline"] for r in rows})
    n_models, n_photos = len({r["model"] for r in rows}), rows[0]["n_images"] if rows else 0
    (out / "README.md").write_text(f"""# {p['name']} results, {time.strftime('%Y-%m-%d')}

{n_models} open-weight vision-language models scored on {n_photos} SNAPMe photos against the
participants' ASA24 records: which foods were named, how many recorded components were found, preparation, and
amount. Pipelines: {', '.join(pipelines)}. Code commit {commit}.

| File | Contents |
|---|---|
| leaderboard.csv | one row per pipeline and model (intervals in the site's leaderboard.json) |
| per_photo.csv | per-photo measures for every pipeline and model |
| baselines_2023.json | Inverse Cooking and Im2Recipe as published with the SNAPMe paper, rescored here |
| models.csv | Hugging Face repo, pinned commit, license and engine for each model |
| ground_truth/ | photos and ASA24 lines used as the reference (derived from SNAPMe, CC BY-SA 4.0) |
| predictions/ | raw model replies and parsed items, one JSON line per photo |

Column notes: `code_f1`, `wweia_f1` and `subgroup_f1` are per-photo F1 at three levels of the FNDDS hierarchy,
averaged over photos; `token_f1_2023` follows the SNAPMe paper's protocol on its 1,463 photos;
`component_recall` is the share of ASA24 lines matched one-to-one; `preparation_agreement` and
`item_grams_median_ape` are computed on matched lines; `format_compliance` is the share of replies that could be
read. ASA24 grams are the amount eaten, so amounts from a single before photo include any leftovers.

Data: SNAPMe DB v1.0 (Chin, Larke et al., Ag Data Commons, doi:10.15482/USDA.ADC/1528346), CC BY-SA 4.0.
""")
    return out
