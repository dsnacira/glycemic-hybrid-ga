import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(histories: dict, title: str, out_path: str):
    """
    histories = {
        "GA-only": np.array(...),
        "Hybrid": np.array(...)
    }
    """
    plt.figure(figsize=(6,4))
    for name, h in histories.items():
        plt.plot(h, label=name)
    plt.xlabel("Generation")
    plt.ylabel("Best true fitness")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
