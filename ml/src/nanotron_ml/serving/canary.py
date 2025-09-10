"""Canary router for staged model rollouts.

Routes a fraction of requests to a candidate model and the rest to the
production model.  Decision is keyed on a string (e.g. user_id, symbol)
so the same caller is consistently routed to the same variant within
a rollout — no thrashing for a single client.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class CanaryRouter:
    """A deterministic, smooth-rollout canary splitter.

    Parameters
    ----------
    candidate_fraction : float in [0, 1]
        Fraction of routing keys directed to the candidate.
    salt : str
        Any constant — bumping it shuffles which keys belong to the
        candidate set without changing the fraction.
    """

    candidate_fraction: float
    salt: str = "nanotron"

    def __post_init__(self) -> None:
        if not 0.0 <= self.candidate_fraction <= 1.0:
            raise ValueError("candidate_fraction must be in [0, 1]")

    def is_candidate(self, key: str) -> bool:
        h = hashlib.blake2b(
            f"{self.salt}::{key}".encode(), digest_size=8
        ).digest()
        # uniform in [0, 1)
        u = int.from_bytes(h, "big") / 2**64
        return u < self.candidate_fraction
