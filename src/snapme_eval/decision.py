"""Clients for models that return probabilities for specified answers.

The shared interface is await decider.ask(state, image_url, questions). System One sends question sets
to POST /v1/systemone. JEV requests option-token log probabilities from vLLM and applies the supplied
bias and temperature. The JEV client sends each question separately.

Questions use {id: {"type": "noul", "instructions": ...}} for yes/no or
{id: {"type": "choice", "instructions": ..., "criteria": {option_id: description}}} for choices.
Responses use {id: {option_id: probability}}; yes/no options are "false" and "true".
"""
from __future__ import annotations

import json
import math
import os
import pathlib
from functools import cache

import httpx

from snapme_eval import registry

# Raw decision template from the JEV card: message text and images in order, nothing else.
JEV_TEMPLATE = ("{%- for m in messages -%}{%- for c in m['content'] -%}{%- if c['type'] == 'text' -%}"
                "{{ c['text'] }}{%- else -%}<|vision_start|><|image_pad|><|vision_end|>{%- endif -%}"
                "{%- endfor -%}{%- endfor -%}")
LABELS = "ABCDEFGHIJKLMNOP"   # the 16 option labels supported by JEV's decision head


class SystemOne:
    def __init__(self, base_url: str, model: str):
        self.model = model
        self.http = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=600, trust_env=False)

    async def ask(self, state: str, image_url: str | None, questions: dict) -> dict:
        body = {"model": self.model, "state": state, "questions": questions,
                "images": [image_url] if image_url else []}
        r = await self.http.post("/v1/systemone", json=body)
        r.raise_for_status()
        out = {}
        for qid, a in r.json()["answers"].items():
            if a["type"] == "noul":
                out[qid] = {"false": 1.0 - a["noul"], "true": a["noul"]}
            else:
                out[qid] = dict(a["probabilities"])
        return out

    async def close(self) -> None:
        await self.http.aclose()


@cache
def _jev_files(model_id: str) -> tuple[dict, dict]:
    snap = registry.snapshot_path(registry.model(model_id), pathlib.Path(os.environ["HF_HUB_CACHE"]))
    head = json.loads((snap / "adapter_vllm" / "decision_head.json").read_text())
    temps = json.loads((snap / "calibration.json").read_text())["per_kind"]
    return head, temps


class JevVllm:
    """JEV System 1 on vLLM, served with --enable-lora --lora-modules jev-decision=<snapshot>/adapter_vllm."""

    def __init__(self, base_url: str, model: str, lora: str = "jev-decision"):
        self.model, self.lora = model, lora
        self.http = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=600, trust_env=False)

    async def _one(self, kind: str, state: str, image_url: str | None, question: str,
                   options: list[str], texts: list[str]) -> list[float]:
        head, temps = _jev_files(self.model)
        lines = texts if kind != "choice" else [f"{LABELS[i]}) {t}" for i, t in enumerate(texts)]
        content = [{"type": "text", "text": f"[kind] {kind}\n[state] "}]
        if image_url:
            content += [{"type": "text", "text": "Photo: "}, {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "\n"}]
        content.append({"type": "text", "text": f"{state}\n[question] {question}\n[options]\n"
                        + "\n".join(lines) + "\n[decision]:"})
        start = head["slots"]["ranges"][kind][0]
        ids = head["verbalizer_ids"][start:start + len(options)]
        r = await self.http.post("/v1/chat/completions", json={
            "model": self.lora, "messages": [{"role": "user", "content": content}], "chat_template": JEV_TEMPLATE,
            "add_generation_prompt": False, "add_special_tokens": False, "max_tokens": 1, "temperature": 1.0,
            "top_k": 0, "top_p": 1.0, "logprobs": True, "top_logprobs": len(options), "allowed_token_ids": ids,
            "return_tokens_as_token_ids": True})
        r.raise_for_status()
        top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        lp = {int(t["token"].split(":")[1]): t["logprob"] for t in top}
        z = [(lp.get(t, -1e9) + head["bias"][start + i]) / temps[kind] for i, t in enumerate(ids)]
        e = [math.exp(x - max(z)) for x in z]
        return [x / sum(e) for x in e]

    async def ask(self, state: str, image_url: str | None, questions: dict) -> dict:
        out = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                p = await self._one("noul", state, image_url, q["instructions"], ["false", "true"],
                                    ["false", "true"])
                out[qid] = {"false": p[0], "true": p[1]}
            else:
                opts = list(q["criteria"])
                if len(opts) > len(LABELS):
                    raise ValueError(f"JEV choice limited to {len(LABELS)} options here, got {len(opts)}")
                p = await self._one("choice", state, image_url, q["instructions"], opts,
                                    [q["criteria"][o] or o for o in opts])
                out[qid] = dict(zip(opts, p, strict=True))
        return out

    async def close(self) -> None:
        await self.http.aclose()


def make(model_id: str, base_url: str):
    """base_url is the server root; a trailing /v1 (the OpenAI base the runner uses) is dropped."""
    base_url = base_url.rstrip("/")
    base_url = base_url[:-3] if base_url.endswith("/v1") else base_url
    api = registry.model(model_id).get("decision_api", "systemone")
    if api == "jev_vllm":
        return JevVllm(base_url, model_id)
    if api == "systemone":
        return SystemOne(base_url, model_id)
    raise ValueError(f"unknown decision_api {api!r} for {model_id}")
