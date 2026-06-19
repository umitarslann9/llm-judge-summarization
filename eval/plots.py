"""Stage 5 - figures for the report.

(1) cost_quality.png: cost/speed (mean latency per summary, log x) vs. human
    alignment (average Spearman) - the central cost-quality trade-off.
(2) correlation_bars.png: average human Spearman per method.

Reads results/correlation.csv and results/cost.csv (run those first).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import load_config, project_paths

# Map correlation-table method labels -> cost-table method labels.
COST_LABEL = {
    "Zero-shot": "Zero-shot (local)",
    "Small judge": "Small judge (local)",
    "Large judge": "Large judge (Gemini)",
}


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)
    rdir = paths["results_dir"]

    corr = pd.read_csv(rdir / "correlation.csv")
    avg = corr[corr["dimension"] == "average"][["method", "spearman"]]
    avg = avg.set_index("method")["spearman"].to_dict()

    # ---- (2) bar chart of average Spearman -------------------------------
    order = [m for m in ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore-F",
                         "Zero-shot", "Small judge", "Large judge"] if m in avg]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = [avg[m] for m in order]
    colors = ["#9aa0a6"] * len(order)
    for i, m in enumerate(order):
        if m == "Small judge":
            colors[i] = "#1a73e8"
        elif m == "Large judge":
            colors[i] = "#d93025"
    ax.bar(order, vals, color=colors)
    ax.set_ylabel("Avg. Spearman with human scores")
    ax.set_title("Human alignment by method")
    ax.axhline(0, color="black", linewidth=0.6)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(rdir / "correlation_bars.png", dpi=160)
    print(f"[plots] wrote {rdir / 'correlation_bars.png'}")

    # ---- (1) cost vs quality (model-based methods) -----------------------
    cost_path = rdir / "cost.csv"
    if cost_path.exists():
        cost = pd.read_csv(cost_path).set_index("method")
        fig, ax = plt.subplots(figsize=(7, 5))
        for m, clabel in COST_LABEL.items():
            if m in avg and clabel in cost.index:
                x = cost.loc[clabel, "mean_latency_s"]
                y = avg[m]
                ax.scatter(x, y, s=90)
                ax.annotate(m, (x, y), textcoords="offset points", xytext=(8, 4))
        ax.set_xscale("log")
        ax.set_xlabel("Mean latency per summary (s, log scale)")
        ax.set_ylabel("Avg. Spearman with human scores")
        ax.set_title("Cost-quality trade-off")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        plt.tight_layout()
        fig.savefig(rdir / "cost_quality.png", dpi=160)
        print(f"[plots] wrote {rdir / 'cost_quality.png'}")
    else:
        print("[plots] cost.csv not found - skipping cost-quality figure")


if __name__ == "__main__":
    main()
