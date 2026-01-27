# src/analysis/plotting.py
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_convergence_two(
    hist_ga: np.ndarray,
    hist_hybrid: np.ndarray,
    scenario_name: str,
    out_path: str,
) -> None:
    """
    Plot best TRUE fitness per generation for GA-only vs Hybrid.
    """
    ensure_dir(os.path.dirname(out_path))
    h1 = np.asarray(hist_ga, dtype=float)
    h2 = np.asarray(hist_hybrid, dtype=float)

    plt.figure(figsize=(6.2, 4.0))
    plt.plot(h1, label="GA-only")
    plt.plot(h2, label="Hybrid")
    plt.xlabel("Generation")
    plt.ylabel("Best true fitness")
    plt.title(f"Convergence — {scenario_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_glucose_two(
    G_ga: np.ndarray,
    G_hybrid: np.ndarray,
    scenario_name: str,
    out_path: str,
) -> None:
    """
    Plot glucose trajectories for GA-only vs Hybrid, with 70/180 lines.
    """
    ensure_dir(os.path.dirname(out_path))
    g1 = np.asarray(G_ga, dtype=float)
    g2 = np.asarray(G_hybrid, dtype=float)

    plt.figure(figsize=(6.2, 4.0))
    plt.plot(g1, label="GA-only")
    plt.plot(g2, label="Hybrid")
    plt.axhline(70, linestyle="--")
    plt.axhline(180, linestyle="--")
    plt.xlabel("Time step (dt)")
    plt.ylabel("Glucose (mg/dL)")
    plt.title(f"Glucose trajectory — {scenario_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_surrogate_scatter(
    pred: np.ndarray,
    true: np.ndarray,
    title: str,
    out_path: str,
) -> None:
    """
    Scatter of predicted fitness vs true (simulated) fitness.
    """
    ensure_dir(os.path.dirname(out_path))
    p = np.asarray(pred, dtype=float).ravel()
    t = np.asarray(true, dtype=float).ravel()

    m = np.isfinite(p) & np.isfinite(t)
    p, t = p[m], t[m]

    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(p, t, s=14)
    if len(p) > 0:
        lo = float(min(p.min(), t.min()))
        hi = float(max(p.max(), t.max()))
        plt.plot([lo, hi], [lo, hi], linestyle="--")  # y=x
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)

    plt.xlabel("Predicted fitness (surrogate)")
    plt.ylabel("True fitness (simulation)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
