"""System One API (POST /v1/systemone) for decision models that ship their own Python inference code instead of
a head an engine can read (Intern-Decision: `inference.DecisionEngine` in the model folder).

Standalone, so it runs inside an engine image (torch, transformers, fastapi, uvicorn, pillow):

    python3 systemone_server.py --checkpoint <model folder> --port 8000 --model-name intern-decision-4b

Images arrive as data URLs and are written to temporary files, since DecisionEngine reads image paths.
"""
from __future__ import annotations

import argparse
import base64
import pathlib
import sys
import tempfile
import threading

import uvicorn
from fastapi import FastAPI, HTTPException


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model-name", default="decision")
    a = ap.parse_args()
    sys.path.insert(0, a.checkpoint)
    from inference import DecisionEngine  # the model's own code, from its pinned snapshot

    engine = DecisionEngine(a.checkpoint, device="cuda")
    # One request at a time: the model's PyTorch fallback kernels are not safe across threads ("lazy wrapper
    # should be called at most once"); a warm-up call initializes them before the first photo request.
    lock = threading.Lock()
    engine.predict({"state": "warm-up", "questions": {"q": {"type": "noul", "instructions": "Is this a test?"}}})
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="systemone_"))
    app = FastAPI()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": a.model_name, "object": "model"}]}

    @app.post("/v1/systemone")
    def systemone(body: dict) -> dict:
        paths = []
        for i, img in enumerate(body.get("images") or []):
            if not str(img).startswith("data:"):
                raise HTTPException(400, "images must be data URLs")
            p = tmp / f"{id(body)}_{i}.jpg"
            p.write_bytes(base64.b64decode(str(img).split(",", 1)[1]))
            paths.append(str(p))
        try:
            req = {"state": body["state"], "questions": body["questions"], **({"images": paths} if paths else {})}
            with lock:
                out = engine.predict(req)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        finally:
            for p in paths:
                pathlib.Path(p).unlink(missing_ok=True)
        return {"model": a.model_name, **out}

    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
