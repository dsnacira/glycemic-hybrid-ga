from __future__ import annotations
import numpy as np
from .utils import in_window

def _smooth_bump(dt: float, peak: float, width: float) -> float:
    """
    Smooth bump on [0,width]. Returns ~[0,1] after normalization.
    """
    if dt < 0 or dt > width:
        return 0.0
    x = dt / width
    # beta-like shape
    a = max(1.0, (peak / width) * 10.0)
    b = max(1.0, (1.0 - peak / width) * 10.0)
    y = (x ** a) * ((1.0 - x) ** b)
    # normalize roughly (avoid division by 0)
    return float(y * 12.0)

def meal_effect_mgdl_per_step(
    cho_g: float,
    t_min: int,
    meal_time_min: int,
    width_min: int = 150,
    peak_min: int = 60,
    gain: float = 1.2
) -> float:
    dt = t_min - meal_time_min
    shape = _smooth_bump(dt, peak=float(peak_min), width=float(width_min))
    return float(gain * cho_g * shape / 100.0)

def stress_effect_mgdl_per_step(stress_level: float) -> float:
    return float(0.2 * stress_level)

def activity_effect_mgdl_per_step(activity_level: float, t_min: int,
                                  start_hhmm: str | None, end_hhmm: str | None) -> float:
    if start_hhmm is None or end_hhmm is None:
        return 0.0
    if in_window(t_min, start_hhmm, end_hhmm):
        return float(-1.0 * activity_level)
    return 0.0
