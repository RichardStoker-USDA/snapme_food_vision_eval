"""Submit serve-and-run jobs for registry models on a Slurm cluster (written for SCINet Atlas).

    python hpc/atlas/submit.py --system direct qwen3-5-9b gemma-4-31b-it ...
    python hpc/atlas/submit.py --system direct --all --prefer auto

Partition choice per model: the model file's GPU plans ([serve.<gpu>] blocks), mapped to partitions by
SNAPME_PARTITIONS in site.env ("a100=gpu-a100,l40s=gpu-l40s" on Atlas; the GPU name is also the Slurm GRES
type), then the partition with enough free GPUs right now (smaller GPUs first for jobs of up to 2 GPUs, larger
GPUs first for bigger jobs, which need fast links between GPUs). Every job is appended to
$SNAPME_ROOT/control/jobs.tsv, the ledger of jobs this project owns.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import pandas as pd

from snapme_eval import registry, systems

REPO = pathlib.Path(__file__).resolve().parents[2]
# GPU type (model file [serve.<gpu>]) -> Slurm partition, from site.env. Order: smaller GPUs first.
PARTS = dict(kv.split("=", 1) for kv in os.environ.get("SNAPME_PARTITIONS", "l40s=gpu-l40s,a100=gpu-a100")
             .split(",") if kv)
GRES = {part: gpu for gpu, part in PARTS.items()}
# Nodes to avoid (for example ones where CUDA fails to initialize), from SNAPME_EXCLUDE in site.env.
EXCLUDE = os.environ.get("SNAPME_EXCLUDE", "")


def free_gpus() -> dict[str, int]:
    fields = "partition,gres:40,gresused:60,statelong"
    out = subprocess.run(["sinfo", "-h", "-p", ",".join(GRES), "-N", "-O", fields],
                         capture_output=True, text=True).stdout
    free: dict[str, int] = dict.fromkeys(GRES, 0)
    for line in out.splitlines():
        part, gres, used, state = line.split()[:4]
        if any(s in state for s in ("drain", "down", "maint", "reserved")):
            continue
        tot = int(gres.split(":")[2].split("(")[0])
        u = int(used.split(":")[2].split("(")[0])
        free[part] = free.get(part, 0) + max(0, tot - u)
    return free


def pick(m: dict, prefer: str, free: dict[str, int]) -> tuple[str, int] | None:
    serve = m.get("serve", {})
    options = []
    for rank, (gpu, part) in enumerate(PARTS.items()):
        if gpu in serve:
            options.append((part, int(serve[gpu]["gpus"]), rank))
    if prefer != "auto":
        options = [o for o in options if GRES[o[0]] == prefer] or options
    elif options:
        # auto: jobs of up to 2 GPUs on the smaller GPU type, larger jobs on the larger type, then free capacity
        options.sort(key=lambda o: (o[1] > 2 and o[2] == 0, -free.get(o[0], 0) + o[1]))
    options = [(p, n) for p, n, _ in options]
    return options[0] if options else None


def needs_retry(out: pathlib.Path, n_photos: int) -> str:
    """Why a finished run folder needs another pass, or "" when it is complete."""
    pred = out / "predictions.jsonl"
    if not pred.exists():
        return "no predictions"
    seen, retry = set(), set()
    for line in pred.open():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        err = str(r.get("parse_error") or "")
        (retry.add if err.startswith(("exception:", "request failed")) else seen.add)(r["image_id"])
    retry -= seen
    missing = n_photos - len(seen | retry)
    return f"{len(retry)} unreached, {missing} missing" if retry or missing > 0 else ""


def queued_names() -> set[str]:
    r = subprocess.run(["squeue", "--me", "-h", "-o", "%j"], capture_output=True, text=True)
    return set(r.stdout.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--system", default="direct")
    ap.add_argument("--prefer", default="auto", help="auto, or a GPU type from SNAPME_PARTITIONS")
    ap.add_argument("--split", default="all")
    ap.add_argument("--gt", default="snapme_v1", help="reference: snapme_v1 or nutrition5k_v1 (use --split test)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--time", default="08:00:00")
    ap.add_argument("--out-suffix", default="", help="separate output folder, e.g. for a smoke test")
    ap.add_argument("--members", default="", help="models whose lists are combined, as comma-separated registry ids")
    ap.add_argument("--source", default="", help="check: run that supplies the items, e.g. direct_v2/gemma-4-31b-it")
    ap.add_argument("--dependency", default="", help="Slurm dependency, e.g. afterok:12345")
    ap.add_argument("--retry", action="store_true",
                    help="only models whose run is incomplete or has photos that never reached the model")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    root = pathlib.Path(os.environ["SNAPME_ROOT"])
    # --all covers models that write food records; perception-only models (role = "perception") are run by name.
    ids = sorted(k for k, v in registry.models().items() if v.get("role", "generalist") == "generalist") if a.all \
        else a.models
    queued = queued_names() if a.retry else set()
    n_photos = len(pd.read_parquet(root / "gt" / a.gt / "images.parquet")) if a.retry else 0
    free = free_gpus()
    print(f"free GPUs now: {free}")
    ledger = root / "control" / "jobs.tsv"
    logs = root / "logs"
    for mid in ids:
        m = registry.model(mid)
        # A tuned model or a settings variant of a model uses the weights (and download receipt) of its base.
        receipt = root / "models" / "receipts" / f"{m.get('finetuned_from') or m.get('variant_of') or mid}.json"
        if not receipt.exists():
            print(f"skip {mid}: weights not downloaded yet")
            continue
        choice = pick(m, a.prefer, free)
        if not choice:
            print(f"skip {mid}: no GPU plan")
            continue
        part, n = choice
        gpu_mem_cpu = {"cpus": 8 * n, "mem": f"{64 * n + 32}G"}
        several = "+" in a.system           # Joined methods share an engine and keep separate run folders.
        mod = systems.load(a.system.split("+")[0])
        label = f"{mod.NAME}_set" if several else a.system.replace(":", "-")
        out = root / "runs" / a.gt / f"{mod.NAME}_{mod.VERSION}{a.out_suffix}" / mid
        if a.retry:
            if f"snapme_{label}_{mid}"[:60] in queued:
                continue
            why = needs_retry(out, n_photos)
            if not why:
                continue
            print(f"retry {mid}: {why}")
        if a.members:
            os.environ["SNAPME_COMBINE_MEMBERS"] = a.members
        if a.source:
            os.environ["SNAPME_CHECK_SOURCE"] = a.source
        env = (f"ALL,SNAPME_REPO={REPO},SNAPME_MODEL={mid},SNAPME_SYSTEM={a.system},SNAPME_GPUS={n},"
               f"SNAPME_SPLIT={a.split},SNAPME_GT={a.gt}" + ("" if several else f",SNAPME_OUT={out}")
               + (f",SNAPME_LIMIT={a.limit}" if a.limit else ""))
        cmd = ["sbatch", "--parsable", "-p", part, f"--gres=gpu:{GRES[part]}:{n}",
               f"--cpus-per-task={gpu_mem_cpu['cpus']}", f"--mem={gpu_mem_cpu['mem']}", f"--time={a.time}",
               f"--job-name=snapme_{label}_{mid}"[:60], f"--output={logs}/%x_%j.out", f"--export={env}",
               *([f"--exclude={EXCLUDE}"] if EXCLUDE else []),
               *([f"--dependency={a.dependency}"] if a.dependency else []),
               str(REPO / "hpc/atlas/serve_and_run.sbatch")]
        if a.dry_run:
            print(" ".join(cmd))
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print(f"submit failed for {mid}: {r.stderr.strip()}", file=sys.stderr)
            continue
        jid = r.stdout.strip()
        free[part] = free.get(part, 0) - n
        with ledger.open("a") as fh:
            fh.write(f"{jid}\tfpb_{label}_{mid}\t{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t"
                     f"{a.system} {mid} on {part} x{n}{' limit ' + str(a.limit) if a.limit else ''}\n")
        print(f"{jid}  {mid:40s} {part} x{n}")


if __name__ == "__main__":
    main()
