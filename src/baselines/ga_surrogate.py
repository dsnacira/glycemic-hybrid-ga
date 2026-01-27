from __future__ import annotations
import numpy as np

from ..optimization.ga import init_population, evolve_population
from ..core.simulator import simulate_day
from ..surrogate.model import SurrogateModel


def run_ga_surrogate(
    scenario,
    bounds_lo,
    bounds_hi,
    n_pop,
    n_gen,
    pc,
    pm,
    elitism,
    tournament_k,
    surrogate_type: str = "random_forest",
    warmup_generations: int = 2,
    retrain_period: int = 5,
    rho: float = 0.30,
    sim_kwargs=None,
) -> dict:
    """
    GA + Surrogate (no rules) with TRUE-best tracking (paper-ready).

    Robust points:
      - sim_calls counter
      - best_out stored from TRUE simulation (no re-sim)
      - best_true_history is MONOTONE best-so-far (stable AUC metrics)
      - returns (pred,true) pairs for surrogate accuracy plots (reproducible order)
    """
    sim_kwargs = dict(sim_kwargs or {})

    n_pop = int(n_pop)
    n_gen = int(n_gen)
    warmup_generations = int(warmup_generations)
    retrain_period = int(retrain_period)

    pop = init_population(n_pop, bounds_lo, bounds_hi)
    surrogate = SurrogateModel(model_type=surrogate_type)

    # training data (TRUE simulated only)
    X_data: list[np.ndarray] = []
    y_data: list[float] = []

    sim_calls = 0

    # histories
    best_hist_sel: list[float] = []   # selection best (may include predictions)
    best_hist_true: list[float] = []  # TRUE monotone best-so-far

    # global TRUE best
    best_solution_true = None
    best_true_fitness = -1e18
    best_true_out = None

    # store (pred,true) only when pred exists and we simulated that individual
    pred_true_pairs: list[tuple[float, float]] = []

    for g in range(n_gen):
        fitness_sel = np.zeros(n_pop, dtype=float)                  # used by GA evolution
        fitness_true = np.full(n_pop, np.nan, dtype=float)          # true only for simulated inds
        outs = [None] * n_pop

        # -------------------------
        # Warmup: simulate all
        # -------------------------
        if g < warmup_generations or getattr(surrogate, "model", None) is None:
            for i in range(n_pop):
                x = pop[i]
                out = simulate_day(float(x[0]), float(x[1]), scenario, **sim_kwargs)
                sim_calls += 1

                f = float(out["fitness"])
                fitness_sel[i] = f
                fitness_true[i] = f
                outs[i] = out

                X_data.append(x.copy())
                y_data.append(f)

            # train right after warmup ends
            if g == warmup_generations - 1 and len(y_data) >= 5:
                surrogate.fit(np.asarray(X_data, float), np.asarray(y_data, float))

        # -------------------------
        # Surrogate-guided generation
        # -------------------------
        else:
            pred = np.asarray(surrogate.predict(pop), dtype=float)
            fitness_sel[:] = pred

            # simulate top rho predicted (stable tie-breaking)
            n_sim = int(np.ceil(float(rho) * n_pop))
            n_sim = max(1, min(n_pop, n_sim))
            top_idx = np.argsort(pred, kind="mergesort")[-n_sim:]

            for i in top_idx:
                x = pop[i]
                out = simulate_day(float(x[0]), float(x[1]), scenario, **sim_kwargs)
                sim_calls += 1

                f = float(out["fitness"])
                fitness_true[i] = f
                outs[i] = out

                # store (pred,true) for accuracy plot
                pred_true_pairs.append((float(pred[i]), float(f)))

                # replace predicted by true on those simulated (selection uses true there)
                fitness_sel[i] = f

                X_data.append(x.copy())
                y_data.append(f)

            # periodic retrain
            if retrain_period > 0 and (g % retrain_period == 0) and len(y_data) >= 5:
                surrogate.fit(np.asarray(X_data, float), np.asarray(y_data, float))

        # -------------------------
        # Track selection-best
        # -------------------------
        best_hist_sel.append(float(np.max(fitness_sel)))

        # -------------------------
        # Track TRUE best this generation
        # -------------------------
        if np.any(np.isfinite(fitness_true)):
            gen_best_true_idx = int(np.nanargmax(fitness_true))
            gen_best_true = float(fitness_true[gen_best_true_idx])
        else:
            gen_best_true_idx = None
            gen_best_true = -1e18

        # update global TRUE best
        if gen_best_true_idx is not None and gen_best_true > best_true_fitness:
            best_true_fitness = gen_best_true
            best_solution_true = pop[gen_best_true_idx].copy()
            best_true_out = outs[gen_best_true_idx]  # IMPORTANT: no re-sim

        # monotone best-so-far history
        prev = best_hist_true[-1] if best_hist_true else -1e18
        best_hist_true.append(max(prev, gen_best_true))

        # evolve
        pop = evolve_population(pop, fitness_sel, bounds_lo, bounds_hi, pc, pm, elitism, tournament_k)

    best_true_history = np.asarray(best_hist_true, dtype=float)
    best_true_history = np.maximum.accumulate(best_true_history)

    pairs = np.asarray(pred_true_pairs, dtype=float)
    if pairs.size == 0:
        surrogate_pred = np.asarray([], dtype=float)
        surrogate_true = np.asarray([], dtype=float)
        pairs = pairs.reshape(0, 2)
    else:
        surrogate_pred = pairs[:, 0].copy()
        surrogate_true = pairs[:, 1].copy()

    return {
        # selection side (GA behavior)
        "best_solution": best_solution_true,
        "best_fitness": float(np.max(best_hist_sel)) if best_hist_sel else float("nan"),
        "best_history": np.asarray(best_hist_sel, dtype=float),

        # TRUE side (paper)
        "best_solution_true": best_solution_true,
        "best_true_fitness": float(best_true_fitness),
        "best_true_history": best_true_history,
        "best_out": best_true_out,

        "sim_calls": int(sim_calls),
        "dataset_size": int(len(y_data)),

        # accuracy plot outputs
        "pred_true_pairs": pairs,              # shape (N,2)
        "surrogate_pred": surrogate_pred,      # shape (N,)
        "surrogate_true": surrogate_true,      # shape (N,)
    }
