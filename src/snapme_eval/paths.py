"""Project identity and storage locations.

`project.toml` holds the display name and slug. Data locations come from the
environment (SNAPME_ROOT) so the same code runs on Atlas, another cluster or a laptop.
"""
from __future__ import annotations

import os
import pathlib
import tomllib
from functools import cache

REPO = pathlib.Path(__file__).resolve().parents[2]


@cache
def project() -> dict:
    return tomllib.loads((REPO / "project.toml").read_text())


def root() -> pathlib.Path:
    """Return the required SNAPME_ROOT directory for models, datasets and runs."""
    if "SNAPME_ROOT" not in os.environ:
        raise SystemExit("set SNAPME_ROOT to the folder that holds data/, gt/ and runs/")
    return pathlib.Path(os.environ["SNAPME_ROOT"])


def data_dir() -> pathlib.Path:
    return root() / "data"


def gt_dir(version: str) -> pathlib.Path:
    return root() / "gt" / version


def runs_dir() -> pathlib.Path:
    return root() / "runs"


def configs() -> pathlib.Path:
    return REPO / "configs"
