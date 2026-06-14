"""
benchmarks/plot_results.py
───────────────────────────
Génère les courbes Recall / Latence / RAM depuis un CSV de benchmark.

Usage :
  python benchmarks/plot_results.py
  python benchmarks/plot_results.py --csv results/benchmark_glove.csv --out results/benchmark_glove.png
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Couleurs par engine
COLORS = {
    "Quorex-Baseline (HNSW float32)":                    "#94a3b8",
    "Baseline (HNSW float32)":                            "#94a3b8",
    "Quorex (int8 + Piste1 + Piste2)":                   "#6366f1",
    "Quorex (HNSW int8 + Cortex)":                       "#6366f1",
    "Quorex-Optimized (int8 + Piste1+2 + ef=80 + R=16)": "#a855f7",
    "hnswlib (C++ float32)":                              "#22c55e",
    "FAISS IndexHNSWFlat (Meta, C++)":                    "#3b82f6",
    "FAISS IVF+SQ8 (Meta, C++)":                          "#f59e0b",
    "GAFAM (HNSW int8 + re-ranking)":                     "#3b82f6",
    "GAFAM (int8 basique + reranking float32, RAM réelle)": "#3b82f6",
}

STYLES = {
    "C++":    "-",
    "Python": "--",
}

DEFAULT_COLOR = "#e2e8f0"


def load_csv(path: str) -> dict[str, list[dict]]:
    data = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # supporte "config" (ancien format) et "engine" (nouveau format)
            key = row.get("engine") or row.get("config", "unknown")
            data[key].append(row)
    return dict(data)


def plot(csv_path: str, out_path: str) -> None:
    data = load_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#f1f5f9")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.grid(True, alpha=0.2, color="#475569")

    for name, rows in data.items():
        rows_sorted = sorted(rows, key=lambda r: int(r["n_vectors"]))
        x       = [int(r["n_vectors"])              for r in rows_sorted]
        recall  = [float(r["recall_at_10"]) * 100   for r in rows_sorted]
        latency = [float(r["p50_ms"])               for r in rows_sorted]
        ram     = [float(r["ram_mb"])               for r in rows_sorted]

        impl    = rows_sorted[0].get("implementation", "Python")
        color   = COLORS.get(name, DEFAULT_COLOR)
        ls      = STYLES.get(impl, "--")
        lw      = 2.0 if impl == "C++" else 1.6
        label   = f"{name} [{impl}]"

        axes[0].plot(x, recall,  marker="o", markersize=4, label=label, color=color, linestyle=ls, linewidth=lw)
        axes[1].plot(x, latency, marker="o", markersize=4, label=label, color=color, linestyle=ls, linewidth=lw)
        axes[2].plot(x, ram,     marker="o", markersize=4, label=label, color=color, linestyle=ls, linewidth=lw)

    titles  = ["Recall@10 (%)", "Latence p50 (ms)", "RAM (MB)"]
    ylabels = ["Recall (%)",    "p50 (ms)",          "RAM (MB)"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel("N vecteurs", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Légende commune en bas
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(data), 3),
        fontsize=8,
        facecolor="#1e293b",
        labelcolor="#cbd5e1",
        edgecolor="#334155",
        bbox_to_anchor=(0.5, -0.18),
    )

    title_text = Path(csv_path).stem.replace("_", " ").title()
    fig.suptitle(f"Quorex — {title_text}", fontsize=15, color="#f1f5f9", y=1.02)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="results/benchmark_real_v2.csv")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    out = args.out or args.csv.replace(".csv", ".png")
    plot(args.csv, out)