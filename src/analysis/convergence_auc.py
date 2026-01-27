import numpy as np

def auc_sum(best_history: np.ndarray) -> float:
    """
    AUC classique = somme(best_history).
    Plus grand = meilleur (mais sensible au niveau absolu).
    """
    h = np.asarray(best_history, dtype=float)
    if h.size == 0:
        return float("nan")
    # auto-flip si l'historique ressemble à une loss (moyenne < 0)
    if float(np.mean(h)) < 0.0:
        h = -h
    return float(np.sum(h))


def auc_mean(best_history: np.ndarray) -> float:
    """
    AUC normalisée = moyenne(best_history) = auc_sum / n_gen.
    Plus grand = meilleur.
    """
    h = np.asarray(best_history, dtype=float)
    if h.size == 0:
        return float("nan")
    return float(auc_sum(h) / len(h))


def auc_regret(best_history: np.ndarray, f_star: float) -> float:
    """
    AUC_regret = somme( f_star - best_history[t] ).
    Plus petit = meilleur (convergence plus rapide vers l'optimum).

    - f_star doit être un niveau de référence (ex: meilleur best_final parmi méthodes).
    - auto-flip si historique est une loss.
    """
    h = np.asarray(best_history, dtype=float)
    if h.size == 0:
        return float("nan")

    if float(np.mean(h)) < 0.0:
        h = -h

    regret = float(f_star) - h
    # clamp sécurité (si une valeur dépasse f_star à cause du bruit)
    regret = np.maximum(regret, 0.0)
    return float(np.sum(regret))
