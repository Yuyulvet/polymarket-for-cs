"""Shared history coverage policy; this alone is NOT trade eligibility."""
from __future__ import annotations

import math

MIN_ROSTER_HISTORY = 3


def roster_history_eligibility(coverage, min_history=MIN_ROSTER_HISTORY):
    if isinstance(min_history, bool) or not isinstance(min_history, int) or min_history < 0:
        raise ValueError("invalid_minimum_roster_history")
    try:
        counts = [float(coverage[f"history_all_{side}"]) for side in ("a", "b")]
        if any(not math.isfinite(x) or x < 0 or not x.is_integer() for x in counts):
            raise ValueError("invalid_count")
    except (KeyError, TypeError, ValueError, OverflowError):
        return {"eligible": False, "reason": "invalid_roster_history_coverage"}
    eligible = min(counts) >= min_history
    return {"eligible": eligible, "reason": "eligible" if eligible else "insufficient_roster_history"}
