from __future__ import annotations

import random
import numpy as np


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)


def time_to_minutes(x):
    """
    Accepts:
      - "HH:MM" string
      - int minutes
      - float minutes
    Returns int minutes.
    """
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return int(x)
    if isinstance(x, str):
        hh, mm = x.split(":")
        return int(hh) * 60 + int(mm)
    raise TypeError(f"time_to_minutes: unsupported type {type(x)} for value={x}")


def in_window(t_min: int, start_hhmm, end_hhmm) -> bool:
    """
    Accepts start/end either as "HH:MM" strings or minutes (int).
    """
    if start_hhmm is None or end_hhmm is None:
        return False
    start = time_to_minutes(start_hhmm)
    end = time_to_minutes(end_hhmm)
    if start is None or end is None:
        return False
    return start <= int(t_min) <= end


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))
