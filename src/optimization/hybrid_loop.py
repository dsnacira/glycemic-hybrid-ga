from __future__ import annotations
import numpy as np

from .ga import init_population, evolve_population
from ..core.simulator import simulate_day
from ..surrogate.model import SurrogateModel


def run_hybrid(
    scenario,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    n_pop: int,
    n_gen: int,
    pc: float,
    pm: float,
    elitism: int,
    tournament_k: int,
    alpha: float = 0.7,
    rho: float = 0.3,
    warmup_generations: int = 2,
    retrain_period: int = 5,
    surrogate_type: str = "random_forest",
    sim_kwargs: dict | None = None,
) -> dict:
    """
    Hybrid GA–Surrogate:
      - selection uses prediction for all, and alpha-mix on simulated subset
      - TRUE-best tracking uses ONLY simulated individuals
      - best_true_history is MONOTONE best-so-far (stable AUC metrics)
      - best_out stored from TRUE sim (no re-sim)
      - returns (pred,true) pairs for surrogate accuracy plot (reproducible order)
    """
    sim_kwargs = dict(sim_kwargs or {})

    n_pop = int(n_pop)
    n_gen = int(n_gen)
    warmup_generations = int(warmup_generations)
    retrain_period = int(retrain_period)

    pop = init_population(n_pop, bounds_lo, bounds_hi)

    # dataset for surrogate (TRUE simulated fitness only)
    D_X: list[np.ndarray] = []
    D_y: list[float] = []
    sim_calls = 0

    surrogate = SurrogateModel(model_type=surrogate_type)

    # histories
    best_sel_history: list[float] = []
    best_true_history: list[float] = []

    # best trackers
    best_sel_fitness = -1e18
    best_sel_solution = None

    best_true_fitness = -1e18
    best_true_solution = None
    best_out = None

    # store (pred,true) only when pred exists and we simulated that individual
    pred_true_pairs: list[tuple[float, float]] = []

    for g in range(n_gen):
        fitness_sel = np.zeros(n_pop, dtype=float)
        true_fits_this_gen: list[float] = []

        # -------------------------
        # Warmup: simulate all
        # -------------------------
        if g < warmup_generations or getattr(surrogate, "model", None) is None:
            for i in range(n_pop):
                x = pop[i]
                out = simulate_day(float(x[0]), float(x[1]), scenario, **sim_kwargs)
                sim_calls += 1

                f_true = float(out["fitness"])
                fitness_sel[i] = f_true

                D_X.append(x.copy())
                D_y.append(f_true)

                true_fits_this_gen.append(f_true)

                if f_true > best_true_fitness:
                    best_true_fitness = f_true
                    best_true_solution = x.copy()
                    best_out = out

        # -------------------------
        # Surrogate-guided generation
        # -------------------------
        else:
            pred = np.asarray(surrogate.predict(pop), dtype=float)
            fitness_sel[:] = pred

            n_top = int(np.ceil(float(rho) * n_pop))
            n_top = max(1, min(n_pop, n_top))
            top_idx = np.argsort(pred, kind="mergesort")[-n_top:]

            for i in top_idx:
                x = pop[i]
                out = simulate_day(float(x[0]), float(x[1]), scenario, **sim_kwargs)
                sim_calls += 1

                f_true = float(out["fitness"])
                D_X.append(x.copy())
                D_y.append(f_true)

                true_fits_this_gen.append(f_true)

                # store (pred,true) for accuracy plot
                pred_true_pairs.append((float(pred[i]), float(f_true)))

                # alpha-mix for selection on simulated subset
                fitness_sel[i] = float(alpha) * float(pred[i]) + (1.0 - float(alpha)) * f_true

                if f_true > best_true_fitness:
                    best_true_fitness = f_true
                    best_true_solution = x.copy()
                    best_out = out

            if retrain_period > 0 and (g % retrain_period == 0) and len(D_y) >= 5:
                surrogate.fit(np.vstack(D_X), np.asarray(D_y, dtype=float))

        # train right after warmup
        if g == warmup_generations - 1 and len(D_y) >= 5:
            surrogate.fit(np.vstack(D_X), np.asarray(D_y, dtype=float))

        # selection history
        gen_best_sel = float(np.max(fitness_sel))
        best_sel_history.append(gen_best_sel)

        if gen_best_sel > best_sel_fitness:
            best_sel_fitness = gen_best_sel
            best_sel_solution = pop[int(np.argmax(fitness_sel))].copy()

        # TRUE monotone best-so-far history
        gen_best_true = float(np.max(true_fits_this_gen)) if len(true_fits_this_gen) > 0 else -1e18
        prev = best_true_history[-1] if best_true_history else -1e18
        best_true_history.append(max(prev, gen_best_true))

        # evolve
        pop = evolve_population(pop, fitness_sel, bounds_lo, bounds_hi, pc, pm, elitism, tournament_k)

    best_true_history_arr = np.asarray(best_true_history, dtype=float)
    best_true_history_arr = np.maximum.accumulate(best_true_history_arr)

    pairs = np.asarray(pred_true_pairs, dtype=float)
    if pairs.size == 0:
        surrogate_pred = np.asarray([], dtype=float)
        surrogate_true = np.asarray([], dtype=float)
        pairs = pairs.reshape(0, 2)
    else:
        surrogate_pred = pairs[:, 0].copy()
        surrogate_true = pairs[:, 1].copy()

    return {
        # selection side
        "best_solution": best_sel_solution,
        "best_fitness": float(best_sel_fitness),
        "best_history": np.asarray(best_sel_history, dtype=float),

        # TRUE side
        "best_true_solution": best_true_solution,
        "best_true_fitness": float(best_true_fitness),
        "best_true_history": best_true_history_arr,
        "best_out": best_out,

        "dataset_size": int(len(D_y)),
        "sim_calls": int(sim_calls),

        # accuracy plot outputs
        "pred_true_pairs": pairs,              # shape (N,2)
        "surrogate_pred": surrogate_pred,      # shape (N,)
        "surrogate_true": surrogate_true,      # shape (N,)
    }
