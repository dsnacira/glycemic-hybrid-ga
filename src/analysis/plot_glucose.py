def plot_glucose(G, title, out_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.plot(G)
    plt.axhline(70, linestyle="--", color="gray")
    plt.axhline(180, linestyle="--", color="gray")
    plt.xlabel("Time step")
    plt.ylabel("Glucose (mg/dL)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
