from __future__ import annotations
import numpy as np

def apply_fitness_penalties(
    fitness: float,
    glucose: np.ndarray,
    scenario_name: str,
    # robust penalties
    lambda_severe_hypo: float = 300.0,
    lambda_hypo_time: float = 3.0,
    lambda_extreme: float = 600.0,
) -> float:
    """
    Robust safety penalties:
    - Severe hypo event (min < 54): big penalty
    - Hypo duration (time < 70): additional penalty per time step
    - Extreme hypo (min < 40): very big penalty
    """
    gmin = float(np.min(glucose))
    t_hypo = int(np.sum(glucose < 70.0))     # number of steps under 70
    t_severe = int(np.sum(glucose < 54.0))  # severe steps

    # Extreme hypo (should almost never happen)
    if gmin < 40.0:
        fitness -= lambda_extreme

    # Severe hypo event
    if gmin < 54.0:
        fitness -= lambda_severe_hypo

    # Penalize duration in hypo range (keeps TBR low)
    fitness -= lambda_hypo_time * (t_hypo / max(1, len(glucose))) * 100.0

    # Extra penalty if severe lasts long
    if t_severe > 0:
        fitness -= 1.5 * (t_severe / max(1, len(glucose))) * 100.0

    return float(fitness)

def clamp_actions(
    ub: float,
    ubol: float,
    glucose_now: float,
    iob: float,
    scenario_name: str
) -> tuple[float, float]:
    """
    Optional action-level safety clamps.
    You can keep it minimal since you optimize basal/duration.
    """
    # Avoid adding bolus if already low or IOB high
    if glucose_now < 90.0 or iob > 5.0:
        ubol = 0.0

    ub = max(0.0, float(ub))
    ubol = max(0.0, float(ubol))
    return ub, ubol
