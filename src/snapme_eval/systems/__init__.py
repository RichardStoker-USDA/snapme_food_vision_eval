"""Ways of running the task. Each module has NAME, VERSION, prompt_hash() and predict()."""
from __future__ import annotations

import importlib
from types import ModuleType


def load(name: str) -> ModuleType:
    """Load a local module by short name ("direct") or another importable module by full path.
    Use "name:arm" to select a module variant, such as "marker:masked"."""
    name, _, arm = name.partition(":")
    mod = importlib.import_module(name if "." in name else f"snapme_eval.systems.{name}")
    return mod.variant(arm) if arm else mod
