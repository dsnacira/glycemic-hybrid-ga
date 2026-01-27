from __future__ import annotations

import os
import sys
import argparse
import yaml
import numpy as np

# ----------------------------- Robust import setup ----------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src
PROJECT_ROOT = os.path.dirname(THIS_DIR)                # project root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.utils import set_seed
from src.core.scenarios import Scenario

from src.baselines.ga_only import run_ga_only
from src.baselines.ga_rules import run_ga_rules
from src.baselines.ga_surrogate import run_ga_surrogate
from src.optimization.hybrid_loop import run_hybrid

from src.analysis.convergence_auc import auc_sum, auc_mean, auc_regret

# Matplotlib (for file export)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_METHODS = ["ga_only", "ga_rules", "ga_surrogate", "hybrid"]


# ----------------------------- Helpers ----------------------------------------
def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def pretty_metrics(m: dict) -> str:
    return f"TIR={m['TIR']:5.1f}% TBR={m['TBR']:5.1f}% TAR={m['TAR']:5.1f}%"


def stable_seed(base_seed: int, scenario_name: str, method: str) -> int:
    """
    Stable deterministic seed per (scenario, method), independent of run order.
    Avoid Python's hash() (not stable across processes).
    """
    s = f"{scenario_name}::{method}"
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) % 2_147_483_647
    return int((int(base_seed) + h) % 2_147_483_647)


def _as_hhmm(x):
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        m = int(x)
        hh, mm = m // 60, m % 60
        return f"{hh:02d}:{mm:02d}"
    return None


def normalize_window(w):
    if w is None:
        return None
    if isinstance(w, dict):
        return {"start": _as_hhmm(w.get("start")), "end": _as_hhmm(w.get("end"))}
    if isinstance(w, (list, tuple)) and len(w) >= 2:
        return {"start": _as_hhmm(w[0]), "end": _as_hhmm(w[1])}
    return None


def ensure_true_history(res: dict) -> dict:
    """
    Make all methods expose:
      - best_true_history (monotone best-so-far)
      - best_true_fitness
    For ga_only/ga_rules, best_history is already TRUE per generation best;
    we convert it into monotone best-so-far for consistency.
    """
    if "best_true_history" in res and "best_true_fitness" in res:
        return res

    if "best_history" in res:
        h = np.asarray(res["best_history"], dtype=float)
        h_mono = np.maximum.accumulate(h)
        res["best_true_history"] = h_mono
        res["best_true_fitness"] = float(np.max(h_mono)) if h_mono.size else float("nan")
        return res

    raise RuntimeError("Result dict missing both best_true_history and best_history.")


def run_method(method: str, scenario: Scenario, lo: np.ndarray, hi: np.ndarray,
               cfg_ga: dict, cfg_ml: dict, sim_kwargs: dict) -> dict:

    if method == "ga_only":
        return run_ga_only(
            scenario, lo, hi,
            cfg_ga["n_pop"], cfg_ga["n_gen"],
            cfg_ga["pc"], cfg_ga["pm"],
            cfg_ga["elitism"], cfg_ga["tournament_k"],
            sim_kwargs=sim_kwargs,
        )

    if method == "ga_rules":
        return run_ga_rules(
            scenario, lo, hi,
            cfg_ga["n_pop"], cfg_ga["n_gen"],
            cfg_ga["pc"], cfg_ga["pm"],
            cfg_ga["elitism"], cfg_ga["tournament_k"],
            sim_kwargs=sim_kwargs,
        )

    if method == "ga_surrogate":
        return run_ga_surrogate(
            scenario, lo, hi,
            cfg_ga["n_pop"], cfg_ga["n_gen"],
            cfg_ga["pc"], cfg_ga["pm"],
            cfg_ga["elitism"], cfg_ga["tournament_k"],
            warmup_generations=cfg_ml["warmup_generations"],
            retrain_period=cfg_ml["retrain_period"],
            rho=cfg_ml.get("rho", 0.30),
            surrogate_type=cfg_ml["model_type"],
            sim_kwargs=sim_kwargs,
        )

    if method == "hybrid":
        return run_hybrid(
            scenario, lo, hi,
            cfg_ga["n_pop"], cfg_ga["n_gen"],
            cfg_ga["pc"], cfg_ga["pm"],
            cfg_ga["elitism"], cfg_ga["tournament_k"],
            alpha=cfg_ml["alpha"],
            rho=cfg_ml["rho"],
            warmup_generations=cfg_ml["warmup_generations"],
            retrain_period=cfg_ml["retrain_period"],
            surrogate_type=cfg_ml["model_type"],
            sim_kwargs=sim_kwargs,
        )

    raise ValueError(f"Unknown method: {method}")


# ----------------------------- Plotting ---------------------------------------
def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def plot_convergence(scenario_name: str, res_ga: dict, res_h: dict, out_path: str) -> None:
    h1 = np.asarray(res_ga["best_true_history"], dtype=float)
    h2 = np.asarray(res_h["best_true_history"], dtype=float)

    plt.figure()
    plt.plot(np.arange(1, len(h1) + 1), h1, label="GA-only")
    plt.plot(np.arange(1, len(h2) + 1), h2, label="Hybrid")
    plt.xlabel("Generation")
    plt.ylabel("Best true fitness (best-so-far)")
    plt.title(f"Convergence ({scenario_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_glucose_ga_vs_hybrid(res_ga: dict, res_h: dict, dt_minutes: int, out_path: str) -> None:
    out_a = res_ga.get("best_out", None)
    out_b = res_h.get("best_out", None)
    if out_a is None or out_b is None:
        raise RuntimeError("Missing best_out (cannot plot glucose).")

    G_a = np.asarray(out_a["G"], dtype=float)
    G_b = np.asarray(out_b["G"], dtype=float)

    t_h = (np.arange(len(G_a)) * float(dt_minutes)) / 60.0

    plt.figure()
    plt.plot(t_h, G_a, label="GA-only")
    plt.plot(t_h, G_b, label="Hybrid")
    plt.axhline(70.0, linestyle="--")
    plt.axhline(180.0, linestyle="--")
    plt.xlabel("Time (h)")
    plt.ylabel("Glucose (mg/dL)")
    plt.title("Glucose trajectory G(t) — S4_exercise_lowcarb")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pred_vs_true_for_method(all_res_by_scenario: dict[str, dict[str, dict]],
                                 method: str,
                                 out_path: str,
                                 title: str) -> bool:
    xs, ys = [], []
    for _sc, res_by_method in all_res_by_scenario.items():
        res = res_by_method.get(method, None)
        if res is None:
            continue
        pairs = res.get("pred_true_pairs", None)
        if pairs is None:
            continue
        arr = np.asarray(pairs, dtype=float).reshape(-1, 2)
        if arr.size == 0:
            continue
        xs.append(arr[:, 0])
        ys.append(arr[:, 1])

    if len(xs) == 0:
        return False

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    lo = float(np.min([x.min(), y.min()]))
    hi = float(np.max([x.max(), y.max()]))

    plt.figure()
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Predicted fitness")
    plt.ylabel("True fitness")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


# ----------------------------- Main -------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="ALL",
                        help="Scenario name (e.g., S4_exercise_lowcarb) or ALL")
    parser.add_argument("--methods", type=str, default="ALL",
                        help="Comma list: ga_only,ga_rules,ga_surrogate,hybrid or ALL")

    parser.add_argument("--save_figs", action="store_true",
                        help="Save convergence + glucose(S4) + surrogate scatter plots as PDFs.")
    parser.add_argument("--fig_dir", type=str, default="figures",
                        help="Directory where figures are saved (default: figures).")

    args = parser.parse_args()

    base = PROJECT_ROOT
    cfg_default = load_yaml(os.path.join(base, "config", "default.yaml"))
    cfg_ga = load_yaml(os.path.join(base, "config", "ga.yaml"))
    cfg_ml = load_yaml(os.path.join(base, "config", "ml.yaml"))
    cfg_sc = load_yaml(os.path.join(base, "config", "scenarios.yaml"))

    base_seed = int(cfg_default.get("seed", 0))

    lo = np.array([cfg_ga["dose_basal_min"], cfg_ga["insulin_duration_min"]], dtype=float)
    hi = np.array([cfg_ga["dose_basal_max"], cfg_ga["insulin_duration_max"]], dtype=float)

    dt_minutes = int(cfg_default["dt_minutes"])

    sim_kwargs = dict(
        dt_minutes=dt_minutes,
        horizon_hours=int(cfg_default["horizon_hours"]),
        g0=float(cfg_default["g0_mgdl"]),
        gmin=float(cfg_default["gmin_mgdl"]),
        gmax=float(cfg_default["gmax_mgdl"]),
    )

    # methods selection
    if args.methods.strip().upper() == "ALL":
        methods = list(DEFAULT_METHODS)
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        for m in methods:
            if m not in DEFAULT_METHODS:
                raise ValueError(f"Unknown method in --methods: {m}")

    # scenario selection
    if args.scenario.strip().upper() == "ALL":
        scenario_names = list(cfg_sc.keys())
    else:
        scenario_names = [args.scenario.strip()]
        if scenario_names[0] not in cfg_sc:
            raise KeyError(f"Scenario '{scenario_names[0]}' not found in scenarios.yaml")

    all_res_by_scenario: dict[str, dict[str, dict]] = {}

    for name in scenario_names:
        sc = cfg_sc[name]
        scenario = Scenario(
            name=name,
            meals=sc.get("meals", []),
            stress_level=sc.get("stress_level", 0.0),
            activity_level=sc.get("activity_level", 0.0),
            noise_sigma=sc.get("noise_sigma", 2.0),
            fasting=bool(sc.get("fasting", False)),
            fasting_window=normalize_window(sc.get("fasting_window", None)),
            activity_window=normalize_window(sc.get("activity_window", None)),
        )

        print(f"\n=== Running scenario: {scenario.name} ===")

        # PASS 1: run methods and compute f_star for regret
        all_res: dict[str, dict] = {}
        best_final_list: list[float] = []

        for method in methods:
            set_seed(stable_seed(base_seed, scenario.name, method))
            res = run_method(method, scenario, lo, hi, cfg_ga, cfg_ml, sim_kwargs)

            if res.get("best_out", None) is None:
                raise RuntimeError(f"[{method}] best_out missing. Fix runner to return it.")

            res = ensure_true_history(res)

            best_final = float(res.get("best_true_fitness", float(res["best_out"]["fitness"])))
            best_final_list.append(best_final)
            all_res[method] = res

        all_res_by_scenario[scenario.name] = all_res
        f_star = float(np.max(best_final_list))

        # PASS 2: print
        for method in methods:
            res = all_res[method]
            best_out = res["best_out"]

            G = np.asarray(best_out["G"], dtype=float)
            mets = best_out["metrics"]
            fit_out = float(best_out["fitness"])
            best_final = float(res.get("best_true_fitness", fit_out))

            hist_true = np.asarray(res["best_true_history"], dtype=float)

            AUCsum = float(auc_sum(hist_true))
            AUCmean = float(auc_mean(hist_true))
            AUCreg = float(auc_regret(hist_true, f_star=f_star))

            minG = float(np.min(G))
            maxG = float(np.max(G))

            sim_calls = int(res.get("sim_calls", -1))
            dataset = res.get("dataset_size", None)

            extra = ""
            if dataset is not None:
                extra += f" | dataset={int(dataset)}"
            if sim_calls >= 0:
                extra += f" | sim_calls={sim_calls}"

            print(
                f"{method:<11} | best={best_final:6.2f} | fit_out={fit_out:7.2f} "
                f"| AUCsum={AUCsum:8.1f} | AUCmean={AUCmean:6.2f} | AUCregret={AUCreg:8.1f} "
                f"| {pretty_metrics(mets)} | minG={minG:6.1f} | maxG={maxG:6.1f}{extra}"
            )

    # ------------------- FIGURES -------------------
    if args.save_figs:
        fig_dir = os.path.join(PROJECT_ROOT, args.fig_dir)
        ensure_dir(fig_dir)

        # (1) Convergence figures: GA-only vs Hybrid
        for sc_name, res_by_method in all_res_by_scenario.items():
            if "ga_only" in res_by_method and "hybrid" in res_by_method:
                out_path = os.path.join(fig_dir, f"convergence_{sc_name}.pdf")
                plot_convergence(sc_name, res_by_method["ga_only"], res_by_method["hybrid"], out_path)

        # (2) Glucose S4: GA-only vs Hybrid
        s4 = "S4_exercise_lowcarb"
        if s4 in all_res_by_scenario:
            rbm = all_res_by_scenario[s4]
            if "ga_only" in rbm and "hybrid" in rbm:
                out_path = os.path.join(fig_dir, "glucose_S4_ga_vs_hybrid.pdf")
                plot_glucose_ga_vs_hybrid(rbm["ga_only"], rbm["hybrid"], dt_minutes, out_path)

        # (3) Scatter pred vs true (separate files per method)
        ok1 = plot_pred_vs_true_for_method(
            all_res_by_scenario,
            method="ga_surrogate",
            out_path=os.path.join(fig_dir, "ml_pred_vs_true_ga_surrogate.pdf"),
            title="GA+Surrogate: predicted vs true fitness",
        )
        ok2 = plot_pred_vs_true_for_method(
            all_res_by_scenario,
            method="hybrid",
            out_path=os.path.join(fig_dir, "ml_pred_vs_true_hybrid.pdf"),
            title="Hybrid: predicted vs true fitness",
        )

        if not ok1:
            print("[WARN] No pred_true_pairs found for ga_surrogate -> scatter not generated.")
        if not ok2:
            print("[WARN] No pred_true_pairs found for hybrid -> scatter not generated.")

        print(f"\n[OK] Figures saved in: {fig_dir}")


if __name__ == "__main__":
    main()
