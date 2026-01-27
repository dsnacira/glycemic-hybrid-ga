from __future__ import annotations
from dataclasses import dataclass
from typing import Any

def time_to_minutes(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)

@dataclass
class Scenario:
    name: str
    meals: list
    stress_level: float = 0.0
    activity_level: float = 0.0
    noise_sigma: float = 2.0
    fasting: bool = False
    fasting_window: Any = None          # can be dict OR list OR None
    activity_window: Any = None         # can be dict OR list OR None

    def _parse_window(self, win):
        """
        Accept:
          - {"start":"10:00","end":"14:00"}
          - ["10:00","14:00"]
          - None
        Return: (start_min, end_min) or (None, None)
        """
        if not win:
            return None, None

        # dict format
        if isinstance(win, dict):
            s = win.get("start", None)
            e = win.get("end", None)
            return (time_to_minutes(s) if s else None,
                    time_to_minutes(e) if e else None)

        # list/tuple format: ["10:00","14:00"]
        if isinstance(win, (list, tuple)) and len(win) >= 2:
            return time_to_minutes(str(win[0])), time_to_minutes(str(win[1]))

        # fallback
        return None, None

    def inputs_at(self, t_min: int) -> dict:
        act_start, act_end = self._parse_window(self.activity_window)
        fast_start, fast_end = self._parse_window(self.fasting_window)

        meals_now = [{"time_min": time_to_minutes(m["time"]), "cho_g": float(m["cho_g"])} for m in self.meals]

        # simple “flags” (optional)
        in_activity = False
        if act_start is not None and act_end is not None:
            in_activity = (act_start <= t_min <= act_end)

        in_fasting = False
        if self.fasting and fast_start is not None and fast_end is not None:
            in_fasting = (fast_start <= t_min <= fast_end)

        return {
            "meals": meals_now,
            "stress": float(self.stress_level),
            "activity": float(self.activity_level),
            "noise_sigma": float(self.noise_sigma),

            "activity_start": act_start,
            "activity_end": act_end,
            "fasting_start": fast_start,
            "fasting_end": fast_end,

            "in_activity": in_activity,
            "in_fasting": in_fasting,
        }
