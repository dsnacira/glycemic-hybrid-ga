from __future__ import annotations
import numpy as np

def tournament_select(pop: np.ndarray, fitness: np.ndarray, k: int) -> int:
    idx = np.random.choice(len(pop), size=k, replace=False)
    best = idx[np.argmax(fitness[idx])]
    return int(best)

def arithmetic_crossover(p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.random.rand()
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = (1 - alpha) * p1 + alpha * p2
    return c1, c2

def gaussian_mutation(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return x + np.random.normal(0.0, sigma, size=x.shape)
