"""Model and engine registry (configs/models/*.toml, configs/engines/engines.toml)."""
from __future__ import annotations

import os
import pathlib
import tomllib

from snapme_eval.paths import configs


def model_dirs() -> list[pathlib.Path]:
    """configs/models plus any folders in SNAPME_MODEL_DIRS (colon-separated), so users can add models
    without editing the repository. A file in SNAPME_MODEL_DIRS wins over a registry file of the same id."""
    extra = [pathlib.Path(d) for d in os.environ.get("SNAPME_MODEL_DIRS", "").split(":") if d]
    return [*extra, configs() / "models"]


def model_path(model_id: str) -> pathlib.Path:
    """The model's file; supporting models (mapper embedding, depth) are found in configs/aux_models."""
    for d in [*model_dirs(), configs() / "aux_models"]:
        p = d / f"{model_id}.toml"
        if p.exists():
            return p
    raise SystemExit(f"unknown model id {model_id!r}: not in {', '.join(map(str, model_dirs()))}")


def model(model_id: str) -> dict:
    return tomllib.loads(model_path(model_id).read_text())


def models() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for d in reversed(model_dirs()):
        out.update({p.stem: tomllib.loads(p.read_text()) for p in sorted(d.glob("*.toml"))})
    return dict(sorted(out.items()))


def engines() -> dict[str, dict]:
    return tomllib.loads((configs() / "engines" / "engines.toml").read_text())


def snapshot_path(m: dict, hf_hub_cache: pathlib.Path) -> pathlib.Path:
    org, name = m["hf_repo"].split("/")
    return hf_hub_cache / f"models--{org}--{name}" / "snapshots" / m["revision"]
