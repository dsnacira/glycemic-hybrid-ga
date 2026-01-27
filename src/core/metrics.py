from __future__ import annotations
import numpy as np

def tir_tbr_tar(glucose: np.ndarray) -> dict:
    g = glucose
    n = len(g)
    tir = 100.0 * np.sum((g >= 70) & (g <= 180)) / n
    tbr = 100.0 * np.sum(g < 70) / n
    tar = 100.0 * np.sum(g > 180) / n
    return {"TIR": tir, "TBR": tbr, "TAR": tar}

def compute_fitness(metrics: dict) -> float:
    return metrics["TIR"] - 2.0 * metrics["TBR"] - 0.5 * metrics["TAR"]

def glucose_stats(glucose: np.ndarray) -> dict:
    g = glucose
    mean = float(np.mean(g))
    std = float(np.std(g))
    cv = float(100.0 * std / mean) if mean > 1e-9 else float("nan")
    return {"mean": mean, "std": std, "CV": cv, "min": float(np.min(g)), "max": float(np.max(g))}
