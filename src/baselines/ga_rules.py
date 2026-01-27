from __future__ import annotations
import numpy as np

from ..optimization.ga import init_population, evolve_population
from ..core.simulator import simulate_day


def run_ga_rules(
    scenario,
    bounds_lo,
    bounds_hi,
    n_pop,
    n_gen,
    pc,
    pm,
    elitism,
    tournament_k,
    sim_kwargs=None,
) -> dict:
    """
    GA + Rules robust:
      - sim_calls
      - best_out (no re-simulation)
      - best_true_history (monotone best-so-far) for AUC/AUCregret
    """
    sim_kwargs = dict(sim_kwargs or {})

    pop = init_population(int(n_pop), bounds_lo, bounds_hi)

    best_hist_gen: list[float] = []

    best_solution = None
    best_fitness = -1e18
    best_out = None

    sim_calls = 0

    for _g in range(int(n_gen)):
        fitness = np.zeros(int(n_pop), dtype=float)
        outs = [None] * int(n_pop)

        for i in range(int(n_pop)):
            x = pop[i]
            out = simulate_day(
                float(x[0]), float(x[1]), scenario,
                safety_action_level=True,
                safety_fitness_level=True,
                **sim_kwargs,
            )
            sim_calls += 1

            f = float(out["fitness"])
            fitness[i] = f
            outs[i] = out

        gen_best_idx = int(np.argmax(fitness))
        gen_best_fit = float(fitness[gen_best_idx])

        best_hist_gen.append(gen_best_fit)

        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_solution = pop[gen_best_idx].copy()
            best_out = outs[gen_best_idx]  # IMPORTANT: no re-sim

        pop = evolve_population(
            pop, fitness, bounds_lo, bounds_hi,
            pc, pm, elitism, tournament_k
        )

    best_history = np.asarray(best_hist_gen, dtype=float)
    best_true_history = np.maximum.accumulate(best_history)  # MONOTONE (paper/AUC)

    return {
        # legacy
        "best_solution": best_solution,
        "best_fitness": float(best_fitness),
        "best_history": best_history,

        # paper keys
        "best_true_fitness": float(best_fitness),
        "best_true_history": best_true_history,

        "best_out": best_out,
        "sim_calls": int(sim_calls),
    }
