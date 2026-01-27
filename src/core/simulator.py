from __future__ import annotations
import numpy as np

from .metrics import tir_tbr_tar, compute_fitness
from .utils import clamp
from ..safety.rules import apply_fitness_penalties, clamp_actions
from .dynamics import (
    meal_effect_mgdl_per_step,
    stress_effect_mgdl_per_step,
    activity_effect_mgdl_per_step,
)

def simulate_day(
    dose_basal: float,
    insulin_duration_h: float,
    scenario,
    dt_minutes: int = 5,
    horizon_hours: int = 24,
    g0: float = 110.0,
    gmin: float = 40.0,
    gmax: float = 400.0,
    # --- robust tuning ---
    kI: float = 0.025,     # insulin effect (reduced to avoid chronic hypo)
    kB: float = 0.006,    # basal homeostasis (pulls back toward g0)
    safety_action_level: bool = False,
    safety_fitness_level: bool = True,
) -> dict:
    """
    Robust simplified simulator (stable over 24h):
      - IOB proxy (exponential decay)
      - Meal: smooth bump via dynamics.py
      - Stress/activity: small drifts
      - Homeostatic term: prevents drift to hypo/hyper
      - Noise: Gaussian
    """
    n_steps = int(horizon_hours * 60 / dt_minutes)
    G = np.zeros(n_steps + 1, dtype=float)
    G[0] = float(g0)

    # Insulin-on-board proxy: exponential decay
    beta = np.exp(-dt_minutes / (60.0 * float(insulin_duration_h)))
    iob = 0.0

    for k in range(n_steps):
        t_min = k * dt_minutes
        inp = scenario.inputs_at(t_min)

        # Basal only (bolus = 0 in this project)
        ub = float(dose_basal)
        ubol = 0.0

        if safety_action_level:
            ub, ubol = clamp_actions(ub, ubol, G[k], iob, scenario.name)

        # Update IOB
        iob = beta * iob + ub + ubol

        # Meal contribution
        dG_meal = 0.0
        for m in inp["meals"]:
            dG_meal += meal_effect_mgdl_per_step(
                cho_g=float(m["cho_g"]),
                t_min=t_min,
                meal_time_min=int(m["time_min"]),
                width_min=150,
                peak_min=60,
                gain=1.2,    # stronger meal effect (robust against hypo drift)
            )

        dG_stress = stress_effect_mgdl_per_step(float(inp["stress"]))
        dG_act = activity_effect_mgdl_per_step(
            float(inp["activity"]),
            t_min,
            inp["activity_start"],
            inp["activity_end"],
        )

        eps = np.random.normal(0.0, float(inp["noise_sigma"]))

        # Core update
        g_next = G[k] + dG_meal - kI * iob + dG_stress + dG_act + eps

        # Homeostatic pull to baseline
        g_next += kB * (g0 - G[k])

        G[k + 1] = clamp(g_next, gmin, gmax)

    mets = tir_tbr_tar(G)
    fitness = compute_fitness(mets)

    if safety_fitness_level:
        fitness = apply_fitness_penalties(fitness, G, scenario.name)

    return {"G": G, "metrics": mets, "fitness": float(fitness)}
