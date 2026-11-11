"""snapme_eval command line.

  snapme-eval fetch                                  download SNAPMe and FNDDS, check pinned hashes
  snapme-eval fndds   --src DIR                      FNDDS workbooks -> parquet in $SNAPME_ROOT/data/fndds
  snapme-eval gt      --version snapme_v1            build ground truth from the SNAPMe release
  snapme-eval serve-cmd --model ID --gpus N --port P print the engine command for a model
  snapme-eval run     --system direct --model ID ... run a pipeline against a served model
  snapme-eval score   --run DIR                      score one run (writes scores.json)
  snapme-eval map     --embed-url U --judge-url U    map new predicted food names to FNDDS codes
  snapme-eval report                                 collect scored runs into results/
  snapme-eval align --url URL [--shard I/N]         grade food lists, optionally one non-overlapping share
  snapme-eval grader-sheet --out DIR                prepare a sheet for human grading
  snapme-eval grader-agreement --sheet F --key K    compare human and model grades
  snapme-eval marker-table --run R --run R           check scale-card boxes from geometry runs (systems/marker.py)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import shlex
import sys

from snapme_eval import paths, registry

RELEASE_SUBDIR = "snapme/snapme_db_09Dec2022"


def image_root(gt_version: str) -> pathlib.Path:
    """Folder that a reference's image paths are relative to."""
    if gt_version.startswith("nutrition5k"):
        return paths.data_dir() / "nutrition5k"
    return paths.data_dir() / RELEASE_SUBDIR


def cmd_fndds(a) -> None:
    from snapme_eval import fndds
    out = paths.data_dir() / "fndds"
    fndds.write_parquet(fndds.load(pathlib.Path(a.src)), out)
    print(f"wrote FNDDS parquet to {out}")


def cmd_gt(a) -> None:
    from snapme_eval import fndds
    from snapme_eval.gt import snapme
    f = fndds.read_parquet(paths.data_dir() / "fndds")
    info = snapme.build(paths.data_dir() / RELEASE_SUBDIR, f["foods"], paths.gt_dir(a.version),
                        check_files=not a.no_check_files)
    print(json.dumps(info["counts"], indent=1))


def serve_cmd(model_id: str, gpus: int, port: int) -> list[str]:
    """Engine command line for a registry model. Model weights load from the pinned local snapshot."""
    import os
    m = registry.model(model_id)
    eng = registry.engines()[m["engine"]]
    snap = registry.snapshot_path(m, pathlib.Path(os.environ["HF_HUB_CACHE"]))
    max_len = m.get("max_model_len", 16384)
    if m.get("reasoning") == "always":
        max_len = max(max_len, 32768)
    eargs = dict(m.get("engine_args", {}))
    # Pipeline stages split whole layers across GPUs; tensor parallel splits each layer (the default, pp = 1).
    pp = int(eargs.pop("pipeline_parallel_size", 1))
    if pp > 1:
        eargs["pipeline_parallel_size"] = pp
    # Data-parallel replicas (with expert parallel, the dense layers stay whole on each GPU and experts are split).
    dp = int(eargs.pop("data_parallel_size", 1))
    if dp > 1:
        eargs["data_parallel_size"] = dp
    if eng["kind"] == "vllm":
        cmd = ["vllm", "serve", str(snap), "--served-model-name", model_id, "--host", "127.0.0.1",
               "--port", str(port), "--tensor-parallel-size", str(gpus // (pp * dp)), "--max-model-len", str(max_len),
               "--gpu-memory-utilization", str(eargs.pop("gpu_memory_utilization", 0.90)),
               "--limit-mm-per-prompt", json.dumps({"image": int(os.environ.get("SNAPME_MAX_IMAGES", 2))}),
               "--trust-remote-code",
               "--seed", str(m.get("sampling", {}).get("seed", 0))]
        if eargs.pop("enforce_eager", False):
            cmd.append("--enforce-eager")
    elif eng["kind"] == "sglang":
        cmd = ["python3", "-m", "sglang.launch_server", "--model-path", str(snap), "--served-model-name",
               model_id, "--host", "127.0.0.1", "--port", str(port), "--tp", str(gpus),
               "--context-length", str(max_len), "--trust-remote-code",
               "--mem-fraction-static", str(eargs.pop("mem_fraction_static", 0.85))]
    elif eng["kind"] == "lmdeploy":
        cmd = ["lmdeploy", "serve", "api_server", str(snap), "--server-name", "127.0.0.1",
               "--server-port", str(port), "--tp", str(gpus), "--backend", "pytorch",
               "--model-name", model_id, "--session-len", str(max_len), "--trust-remote-code"]
    elif eng["kind"] == "systemone_py":
        cmd = ["python3", str(paths.REPO / "src" / "snapme_eval" / "systemone_server.py"), "--checkpoint", str(snap),
               "--host", "127.0.0.1", "--port", str(port), "--model-name", model_id]
    else:
        raise SystemExit(f"unsupported engine kind {eng['kind']}")
    sub = {"{snapshot}": str(snap), "{root}": os.environ.get("SNAPME_ROOT", "")}
    for k, v in eargs.items():
        flag = "--" + k.replace("_", "-")
        vals = v if isinstance(v, list) else [v]
        for a, b in sub.items():
            vals = [str(x).replace(a, b) for x in vals]
        cmd += [flag] if v is True else [flag, *vals]
    return cmd


def compose_files(model_id: str, gpus: int, out: pathlib.Path) -> None:
    """Write out/.env (model, GPU count, engine image) and out/engine.sh (the same serve command as on a
    cluster, model weights from /data/models/hf) for containers/compose.yaml."""
    import os
    m = registry.model(model_id)
    eng = registry.engines()[m["engine"]]
    os.environ["HF_HUB_CACHE"] = "/data/models/hf/hub"
    src = str(paths.REPO / "src")
    cmd = " ".join(shlex.quote("0.0.0.0" if x == "127.0.0.1" else x.replace(src, "/snapme_src"))
                   for x in serve_cmd(model_id, gpus, 8000))
    lines = ["#!/bin/sh", f"# {model_id}: written by `snapme-eval serve-cmd --format compose`", "set -e"]
    pp = []
    if m.get("python_shim"):
        pp.append(f"/shims/{m['python_shim']}")
    if m.get("extra_pip"):
        pp.append("/tmp/extras")
        lines.append("python3 -m pip install --quiet --no-deps --target /tmp/extras "
                     + " ".join(shlex.quote(x) for x in m["extra_pip"]))
    if pp:
        lines.append(f'export PYTHONPATH="{":".join(pp)}${{PYTHONPATH:+:$PYTHONPATH}}"')
    lines.append(f"exec {cmd}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "engine.sh").write_text("\n".join(lines) + "\n")
    (out / ".env").write_text(f"SNAPME_MODEL={model_id}\nSNAPME_GPUS={gpus}\n"
                              f"SNAPME_ENGINE_IMAGE={eng['image'].removeprefix('docker://')}\n")
    print(f"wrote {out / '.env'} and {out / 'engine.sh'}")


def cmd_serve_cmd(a) -> None:
    if a.format == "compose":
        compose_files(a.model, a.gpus, pathlib.Path(a.out))
        return
    m = registry.model(a.model)
    eng = registry.engines()[m["engine"]]
    print(f"SNAPME_SIF={shlex.quote(eng['sif'])}")
    print(f"SNAPME_ENGINE={shlex.quote(m['engine'])}")
    print(f"SNAPME_ENGINE_IMAGE={shlex.quote(eng['image'])}")
    print(f"SNAPME_EXTRA_PIP={shlex.quote(' '.join(m.get('extra_pip', [])))}")
    shim = paths.REPO / "hpc" / "atlas" / "shims" / m["python_shim"] if m.get("python_shim") else ""
    print(f"SNAPME_SHIM={shlex.quote(str(shim))}")
    print("SNAPME_SERVE_CMD=(" + " ".join(shlex.quote(x) for x in serve_cmd(a.model, a.gpus, a.port)) + ")")


def cmd_run(a) -> None:
    from snapme_eval import runner
    meta = asyncio.run(runner.run(a.system, a.model, a.base_url, a.gt, a.split, pathlib.Path(a.out),
                                  image_root(a.gt), a.concurrency, a.limit))
    print(json.dumps({k: meta[k] for k in ("n_total", "n_ok", "n_fail", "seconds")}))


def cmd_score(a) -> None:
    if a.gt.startswith("nutrition5k"):
        from snapme_eval.score import score_nutrition5k as score_run
    else:
        from snapme_eval.score import score_run
    s = score_run.score(pathlib.Path(a.run), a.gt)
    print(json.dumps(s["summary"], indent=1))


def cmd_map(a) -> None:
    from snapme_eval import mapping
    print(json.dumps(mapping.update_cache(pathlib.Path(a.runs) if a.runs else paths.runs_dir(),
                                          a.embed_url, a.embed_model, a.judge_url, a.judge_model)))


def cmd_align(a) -> None:
    from snapme_eval import align
    from snapme_eval.score.report import current
    also = set(a.pipeline or [])
    runs = [pathlib.Path(r) for r in a.run] if a.run else [
        p.parent for p in sorted(paths.runs_dir().glob("*/*/*/predictions.jsonl"))
        if not p.parent.parent.name.startswith(".") and (current(p.parent.parent.name) or p.parent.parent.name in also)]
    shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
    print(json.dumps(align.update(runs, a.url, a.model, limit=a.limit, shard=shard)))


def cmd_grader_sheet(a) -> None:
    from snapme_eval.score import grader_check
    from snapme_eval.score.report import current
    runs = [pathlib.Path(r) for r in a.run] if a.run else [
        p.parent for p in sorted(paths.runs_dir().glob("snapme_v1/direct_v2/*/predictions.jsonl"))
        if current(p.parent.parent.name)]
    print(json.dumps(grader_check.sheet(runs, pathlib.Path(a.out), a.n)))


def cmd_grader_agreement(a) -> None:
    from snapme_eval.score import grader_check
    print(json.dumps(grader_check.agreement(pathlib.Path(a.sheet), pathlib.Path(a.key)), indent=1))


def cmd_marker_table(a) -> None:
    import pandas as pd

    from snapme_eval import marker
    runs = [paths.runs_dir() / "snapme_v1" / r for r in a.run]
    d = marker.build(runs, pd.read_parquet(paths.gt_dir("snapme_v1") / "images.parquet"), image_root("snapme_v1"))
    d.to_csv(a.out, index=False)
    print(json.dumps({"located": len(d), "kept": int(d["kept"].sum()),
                      "two_locators": int((d["locators"] >= 2).sum()),
                      "agree": int((d["iou"] >= marker.IOU_MIN).sum()),
                      "box_from": d.loc[d["kept"], "box_from"].value_counts().to_dict(), "out": a.out}))


def cmd_report(a) -> None:
    from snapme_eval.score import report
    report.collect(pathlib.Path(a.runs) if a.runs else paths.runs_dir(), paths.REPO / "results")


def cmd_download(a) -> None:
    import os
    os.environ.setdefault("HF_HOME", str(paths.root() / "models" / "hf"))
    from snapme_eval.download import download
    download([str(registry.model_path(m)) for m in a.model])


def cmd_ablate(a) -> None:
    from snapme_eval.systems.check import ablate
    run = pathlib.Path(a.run)
    out = run.parent.parent / f"{run.parent.name}_{a.variant}" / run.name
    ablate(run, a.variant, out)
    print(out)


def cmd_nutrition5k(a) -> None:
    from snapme_eval.gt import nutrition5k
    src = paths.data_dir() / "nutrition5k"
    if a.step == "fetch":
        print(nutrition5k.fetch(src))
    else:
        print(nutrition5k.build(src, paths.gt_dir("nutrition5k_v1")))


def cmd_fetch(a) -> None:
    from snapme_eval import fetch
    fetch.main()


def cmd_thumbs(a) -> None:
    from snapme_eval.score import explorer
    print(explorer.thumbs(pathlib.Path(a.out), paths.data_dir() / RELEASE_SUBDIR), "photos written")


def cmd_release(a) -> None:
    from snapme_eval import release
    print(release.package(pathlib.Path(a.out)))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="snapme-eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fndds", help="convert FNDDS 2017-2018 workbooks to parquet")
    s.add_argument("--src", required=True)
    s.set_defaults(f=cmd_fndds)

    s = sub.add_parser("gt", help="build the ground truth tables")
    s.add_argument("--version", default="snapme_v1")
    s.add_argument("--no-check-files", action="store_true")
    s.set_defaults(f=cmd_gt)

    s = sub.add_parser("serve-cmd", help="print the engine command for a registry model")
    s.add_argument("--model", required=True)
    s.add_argument("--gpus", type=int, required=True)
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--format", choices=["shell", "compose"], default="shell",
                   help="shell: variables for the Slurm job; compose: .env and engine.sh for containers/compose.yaml")
    s.add_argument("--out", default=str(paths.REPO / "containers"), help="folder for the compose files")
    s.set_defaults(f=cmd_serve_cmd)

    s = sub.add_parser("run", help="run a pipeline against a served model")
    s.add_argument("--system", default="direct")
    s.add_argument("--model", required=True)
    s.add_argument("--base-url", required=True)
    s.add_argument("--gt", default="snapme_v1")
    s.add_argument("--split", default="all", choices=["all", "dev", "test", "train"])
    s.add_argument("--out", required=True)
    s.add_argument("--concurrency", type=int, default=48)
    s.add_argument("--limit", type=int)
    s.set_defaults(f=cmd_run)

    s = sub.add_parser("score", help="score one run folder")
    s.add_argument("--run", required=True)
    s.add_argument("--gt", default="snapme_v1")
    s.set_defaults(f=cmd_score)

    s = sub.add_parser("report", help="collect scored runs into results/")
    s.add_argument("--runs")
    s.set_defaults(f=cmd_report)

    s = sub.add_parser("release", help="package results, predictions and ground truth for sharing")
    s.add_argument("--out", required=True)
    s.set_defaults(f=cmd_release)

    s = sub.add_parser("download", help="download model weights at the registry's pinned revision")
    s.add_argument("--model", action="append", required=True)
    s.set_defaults(f=cmd_download)

    s = sub.add_parser("ablate", help="food-checking run with one step omitted, using existing predictions")
    s.add_argument("--run", required=True)
    s.add_argument("--variant", required=True, choices=["keep_all", "reader_grams"])
    s.set_defaults(f=cmd_ablate)

    s = sub.add_parser("fetch", help="download SNAPMe and FNDDS 2017-2018 and check them against pinned hashes")
    s.set_defaults(f=cmd_fetch)

    s = sub.add_parser("nutrition5k", help="Nutrition5k overhead dishes: fetch (1.7 GB), then build the reference")
    s.add_argument("step", choices=["fetch", "build"])
    s.set_defaults(f=cmd_nutrition5k)

    s = sub.add_parser("thumbs", help="write smaller WebP photo copies for the Explore page (site/public/photos)")
    s.add_argument("--out", default=str(paths.REPO / "site" / "public" / "photos"))
    s.set_defaults(f=cmd_thumbs)

    s = sub.add_parser("grader-sheet", help="sample food-list pairs for a person to grade")
    s.add_argument("--out", required=True, help="write grader_sheet.csv, grader_key.json and grader_rules.txt here")
    s.add_argument("--n", type=int, default=400)
    s.add_argument("--run", action="append", help="sample this run; repeat as needed (default: every one-photo run)")
    s.set_defaults(f=cmd_grader_sheet)

    s = sub.add_parser("grader-agreement", help="compare a completed human grading sheet with the model grader")
    s.add_argument("--sheet", required=True)
    s.add_argument("--key", required=True)
    s.set_defaults(f=cmd_grader_agreement)

    s = sub.add_parser("align", help="pair food lists with record lines as the same food, same kind or mixed-dish part")
    s.add_argument("--url", action="append", required=True, help="grader server address; repeat to use several servers")
    s.add_argument("--model", default="gpt-oss-120b")
    s.add_argument("--run", action="append", help="grade this run; repeat as needed (default: every current run)")
    s.add_argument("--pipeline", action="append", help="add this method folder to current runs; repeat as needed")
    s.add_argument("--limit", type=int, help="limit the pilot to this many photo lists")
    s.add_argument("--shard", help="I/N: share I of N non-overlapping shares, numbered from 0, for concurrent jobs")
    s.set_defaults(f=cmd_align)

    s = sub.add_parser("marker-table", help="check geometry runs' card locations for the scale-card comparisons")
    s.add_argument("--run", action="append", required=True, help="run, e.g. geometry_v2/gemma-4-31b-it; repeat")
    s.add_argument("--out", default=str(paths.REPO / "src" / "snapme_eval" / "tables" / "marker_v1.csv"))
    s.set_defaults(f=cmd_marker_table)

    s = sub.add_parser("map", help="map predicted food names to FNDDS codes (mapper v1)")
    s.add_argument("--runs")
    s.add_argument("--embed-url", required=True)
    s.add_argument("--embed-model", default="qwen3-embedding-8b")
    s.add_argument("--judge-url", required=True)
    s.add_argument("--judge-model", default="qwen3-8-27b")
    s.set_defaults(f=cmd_map)

    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main(sys.argv[1:])
