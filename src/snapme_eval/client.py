"""Send image-and-text requests to an OpenAI-compatible chat endpoint.

Photos are converted to JPEG data URLs. Server and rate-limit errors are retried; other HTTP errors are
returned to the caller. The serving endpoint must support the registered model's request fields.

An endpoint that the benchmark did not start may want a key and its own model name: SNAPME_API_KEY_FILE names a
file holding the key (read here, never logged), and a model file's served_name is the name the runner sends as
"model".
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import pathlib
import time

import httpx
import pillow_heif
from PIL import Image, ImageOps

# 17 SNAPMe photos are HEIC files with a .jpeg name; Pillow opens them once this opener is registered.
pillow_heif.register_heif_opener()

MAX_SIDE = 1536  # longest image side sent to a model, in pixels (most SNAPMe photos are 800 x 800)
# A model file may set photo_pixels (the runner copies it here): every photo is then resized, larger or smaller, to
# about that many pixels, so the model's image-token setting is filled. Unset, photos are only capped at MAX_SIDE.
PIXELS: int | None = None


def image_data_url(path: pathlib.Path, max_side: int = MAX_SIDE) -> tuple[str, dict]:
    """JPEG data URL with EXIF rotation applied and the long side capped (or the photo_pixels size applied)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        scale = (PIXELS / (w * h)) ** 0.5 if PIXELS else min(1.0, max_side / max(w, h))
        if abs(scale - 1.0) > 1e-3:
            im = im.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
    meta = {"orig_w": w, "orig_h": h, "sent_w": im.size[0], "sent_h": im.size[1]}
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), meta


def user_message(text: str, image_urls: list[str]) -> dict:
    parts: list[dict] = [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
    parts.append({"type": "text", "text": text})
    return {"role": "user", "content": parts}


class Chat:
    def __init__(self, base_url: str, model: str, sampling: dict, extra_body: dict | None = None,
                 timeout: float = 900.0, api_key: str = "none", info: dict | None = None):
        self.model = model
        self.info = info or {}   # Model settings from the registry.
        if api_key == "none" and os.environ.get("SNAPME_API_KEY_FILE"):
            api_key = pathlib.Path(os.environ["SNAPME_API_KEY_FILE"]).expanduser().read_text().strip()
        self.sampling = sampling
        self.extra = extra_body or {}
        local = any(h in base_url for h in ("127.0.0.1", "localhost"))
        self.http = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout, trust_env=not local,
                                      headers={"Authorization": f"Bearer {api_key}"})

    async def __call__(self, messages: list[dict], **override) -> dict:
        body = {"model": self.model, "messages": messages, **self.sampling, **self.extra, **override}
        delay = 2.0
        for attempt in range(5):
            t0 = time.perf_counter()
            try:
                r = await self.http.post("/chat/completions", json=body)
                if r.status_code >= 500 or r.status_code == 429:
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                r.raise_for_status()
                d = r.json()
                msg = d["choices"][0]["message"]
                out = {
                    "content": msg.get("content") or "",
                    "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                    "finish_reason": d["choices"][0].get("finish_reason"),
                    "usage": d.get("usage", {}),
                    "latency_s": round(time.perf_counter() - t0, 3),
                    "attempts": attempt + 1,
                }
                if d["choices"][0].get("logprobs"):   # Keep token log probabilities when the method requests them.
                    out["logprobs"] = d["choices"][0]["logprobs"]
                return out
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500 \
                        and e.response.status_code != 429:
                    return {"error": f"HTTP {e.response.status_code}: {e.response.text[:500]}",
                            "attempts": attempt + 1}
                if attempt == 4:
                    return {"error": f"{type(e).__name__}: {e}", "attempts": attempt + 1}
                await asyncio.sleep(delay)
                delay *= 2
        return {"error": "unreachable"}

    async def close(self) -> None:
        await self.http.aclose()
