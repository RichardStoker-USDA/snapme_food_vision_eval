"""Send reference photos to a method and record one JSON line per reply.

With the same output path, completed photos are skipped and request failures are retried.
Unreadable model replies are retained for scoring as empty lists. Reference food values are excluded
from the fields passed to the method.
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import platform
import subprocess
import time

import pandas as pd

from snapme_eval import client, registry, systems
from snapme_eval.client import Chat
from snapme_eval.paths import REPO, gt_dir

# Only image identifiers, paths and photo types reach the method; reference food values stay with the scorer.
PIPELINE_FIELDS = ("image_id", "filename", "photo_type", "path", "companion_path", "companion_exists")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def sampling_for(m: dict) -> tuple[dict, dict]:
    s = dict(m.get("sampling", {}))
    extra: dict = {}
    if m.get("reasoning") == "optional":
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    if m.get("reasoning") == "always":
        s["max_tokens"] = max(s.get("max_tokens", 4096), 12288)
    # Model-specific template switches (lowest reasoning effort where thinking cannot be turned off).
    if m.get("chat_template_kwargs"):
        extra["chat_template_kwargs"] = {**extra.get("chat_template_kwargs", {}), **m["chat_template_kwargs"]}
    # Other request fields a model file asks for (a [request] table), e.g. a scheduling priority on a shared server.
    extra.update(m.get("request", {}))
    return s, extra


async def run(system: str, model_id: str, base_url: str, gt_version: str, split: str, out: pathlib.Path,
              release_dir: pathlib.Path, concurrency: int, limit: int | None) -> dict:
    mod = systems.load(system)
    m = registry.model(model_id)
    sampling, extra = sampling_for(m)
    images = pd.read_parquet(gt_dir(gt_version) / "images.parquet")
    if split != "all":
        images = images[images["split"] == split]
    if hasattr(mod, "select"):           # Let the method select eligible photos.
        images = mod.select(images)
    if os.environ.get("SNAPME_FOLDS"):   # retain selected participant groups, e.g. "0,2"
        from snapme_eval.folds import fold_for
        keep = {int(f) for f in os.environ["SNAPME_FOLDS"].split(",")}
        images = images[images["subject_id"].map(fold_for).isin(keep)]
    rows = images.sort_values("image_id").to_dict("records")
    if limit:
        rows = rows[:limit]

    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        for line in pred_path.open():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            # A photo that never reached the model (unreadable file, request error) is tried again.
            if not str(r.get("parse_error") or "").startswith(("exception:", "request failed")):
                done.add(r["image_id"])
    todo = [r for r in rows if r["image_id"] not in done]
    meta = {
        "system": f"{mod.NAME}_{mod.VERSION}", "model": model_id, "model_registry": m,
        "prompt_hash": mod.prompt_hash(), "gt_version": gt_version, "split": split,
        "folds": os.environ.get("SNAPME_FOLDS", "all"),
        "engine": os.environ.get("SNAPME_ENGINE", "unknown"), "engine_image": os.environ.get("SNAPME_ENGINE_IMAGE", ""),
        "gpus": os.environ.get("SNAPME_GPUS", ""), "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""), "host": platform.node(), "code_commit": git_commit(),
        "sampling": sampling, "extra_body": extra, "concurrency": concurrency,
        "photo_pixels": m.get("photo_pixels"), "engine_args": m.get("engine_args", {}),
        "combine_members": os.environ.get("SNAPME_COMBINE_MEMBERS", ""),
        "check_source": os.environ.get("SNAPME_CHECK_SOURCE", ""),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_total": len(rows), "n_previously_done": len(done),
    }
    (out / f"run_{meta['started_utc'].replace(':', '')}.json").write_text(json.dumps(meta, indent=1, default=str))

    client.PIXELS = m.get("photo_pixels")   # A model file can ask for photos at a set pixel count.
    # served_name: the name a server that the benchmark did not start gives the model.
    chat = Chat(base_url, m.get("served_name") or model_id, sampling, extra, info=m)
    # When per-fold model IDs are configured, route each participant to its assigned fold.
    fold_chats = {}
    if m.get("lora_per_fold"):
        from snapme_eval.folds import K, fold_for
        fold_chats = {f: Chat(base_url, m["lora_per_fold"].format(fold=f), sampling, extra, info=m)
                      for f in range(K)}
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    t0 = time.time()
    n_ok = n_fail = 0

    async def one(r: dict) -> None:
        nonlocal n_ok, n_fail
        async with sem:
            try:
                c = fold_chats[fold_for(r["subject_id"])] if fold_chats else chat
                res = await mod.predict({k: r[k] for k in PIPELINE_FIELDS if k in r}, c, release_dir)
            except Exception as e:  # keep the run going; the failure is recorded for scoring
                res = {"items": None, "parse_error": f"exception: {type(e).__name__}: {e}"}
        rec = {"image_id": r["image_id"], "photo_type": r["photo_type"], **res}
        async with lock:
            with pred_path.open("a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            if rec.get("items") is None:
                n_fail += 1
            else:
                n_ok += 1
            n = n_ok + n_fail
            if n % 100 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} ok={n_ok} fail={n_fail} {time.time()-t0:.0f}s", flush=True)

    await asyncio.gather(*(one(r) for r in todo))
    await chat.close()
    for c in fold_chats.values():
        await c.close()
    meta.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                seconds=round(time.time() - t0, 1), n_ok=n_ok, n_fail=n_fail)
    (out / f"run_{meta['started_utc'].replace(':', '')}.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta
