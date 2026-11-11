"""Assign participants to stable groups using fixed hashes.

All photos of a participant receive the same fold. A second hash marks an inner subset of participants.
Changing either salt changes the assignments.
"""
from __future__ import annotations

import hashlib

K = 5
SALT = "snapme-finetune-cv5"


def _h(subject: str, salt: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}:{subject}".encode()).digest()[:8], "big")


def fold_for(subject: str, k: int = K) -> int:
    return _h(subject, SALT) % k


def is_inner_val(subject: str) -> bool:
    return _h(subject, SALT + ":inner") % 10 == 0
