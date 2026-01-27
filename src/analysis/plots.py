from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(hist_a: np.ndarray, hist_b: np.ndarray, label_a: str, label_b: str, outpath: str):
    plt.figure()
    plt.plot(hist_a, label=label_a)
    plt.plot(hist_b, label=label_b)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.legend()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
