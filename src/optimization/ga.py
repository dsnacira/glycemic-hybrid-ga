from __future__ import annotations
import numpy as np
from .operators import tournament_select, arithmetic_crossover, gaussian_mutation

def clip_to_bounds(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)

def init_population(n_pop: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return lo + (hi - lo) * np.random.rand(n_pop, len(lo))

def evolve_population(pop: np.ndarray, fitness: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                      pc: float, pm: float, elitism: int, k: int) -> np.ndarray:
    n_pop, dim = pop.shape
    new_pop = []

    elite_idx = np.argsort(fitness)[-elitism:]
    elites = pop[elite_idx].copy()
    new_pop.extend(list(elites))

    sigma = 0.1 * (hi - lo)

    while len(new_pop) < n_pop:
        i1 = tournament_select(pop, fitness, k)
        i2 = tournament_select(pop, fitness, k)
        p1, p2 = pop[i1], pop[i2]

        if np.random.rand() < pc:
            c1, c2 = arithmetic_crossover(p1, p2)
        else:
            c1, c2 = p1.copy(), p2.copy()

        if np.random.rand() < pm:
            c1 = gaussian_mutation(c1, sigma)
        if np.random.rand() < pm:
            c2 = gaussian_mutation(c2, sigma)

        c1 = clip_to_bounds(c1, lo, hi)
        c2 = clip_to_bounds(c2, lo, hi)

        new_pop.append(c1)
        if len(new_pop) < n_pop:
            new_pop.append(c2)

    return np.array(new_pop)
