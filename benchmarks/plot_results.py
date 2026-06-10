import csv
import matplotlib.pyplot as plt

configs = {"Baseline (HNSW float32)": [], "GAFAM (HNSW int8 + re-ranking)": [], "Quorex (HNSW int8 + Cortex)": []}

with open("results/benchmark_bio_v1.csv") as f:
    for row in csv.DictReader(f):
        configs[row["config"]].append(row)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = {"Baseline (HNSW float32)": "#64748B", "GAFAM (HNSW int8 + re-ranking)": "#3B82F6", "Quorex (HNSW int8 + Cortex)": "#6366F1"}

for name, rows in configs.items():
    x = [int(r["n_vectors"]) for r in rows]
    recall = [float(r["recall_at_10"]) * 100 for r in rows]
    latency = [float(r["p50_ms"]) for r in rows]
    ram = [float(r["ram_mb"]) for r in rows]
    c = colors[name]

    axes[0].plot(x, recall, marker="o", label=name, color=c)
    axes[1].plot(x, latency, marker="o", label=name, color=c)
    axes[2].plot(x, ram, marker="o", label=name, color=c)

axes[0].set_title("Recall@10 (%)")
axes[1].set_title("Latence p50 (ms)")
axes[2].set_title("RAM (MB)")

for ax in axes:
    ax.set_xlabel("N vecteurs")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/benchmark_curves.png", dpi=150)
print("Saved → results/benchmark_curves.png")