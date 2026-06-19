"""Generate LaTeX (booktabs) tables for the report directly from the result CSVs.

Producing the tables programmatically guarantees the numbers in the paper match
the computed results exactly (no manual transcription). Writes report/tables.tex,
which main.tex \\input's.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import pandas as pd

from common import HUMAN_DIMS, ROOT, load_config, project_paths

METHOD_ORDER = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore-F",
                "Zero-shot", "Small judge", "Large judge"]
DIM_LABEL = {"coherence": "Coh.", "consistency": "Cons.",
             "fluency": "Flu.", "relevance": "Rel."}


def f(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def correlation_table(corr: pd.DataFrame) -> str:
    sp = corr.pivot(index="method", columns="dimension", values="spearman")
    kt = corr[corr["dimension"] != "average"].pivot(
        index="method", columns="dimension", values="kendall")
    rows = []
    for m in METHOD_ORDER:
        if m not in sp.index:
            continue
        cells = [f(sp.loc[m, d]) for d in HUMAN_DIMS]
        avg_sp = f(sp.loc[m, "average"]) if "average" in sp.columns else "--"
        avg_kt = f(kt.loc[m, HUMAN_DIMS].mean()) if m in kt.index else "--"
        bold = m in ("Small judge", "Large judge")
        name = f"\\textbf{{{m}}}" if bold else m
        rows.append(f"{name} & " + " & ".join(cells) +
                    f" & {avg_sp} & {avg_kt} \\\\")
    head = " & ".join(DIM_LABEL[d] for d in HUMAN_DIMS)
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Spearman correlation of each method with SummEval human ratings, "
        "per dimension. Avg.\\ columns report the mean Spearman ($\\rho$) and Kendall "
        "($\\tau$) across the four dimensions. LLM judges in bold.}\n"
        "\\label{tab:correlation}\n"
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        f"Method & {head} & Avg.\\,$\\rho$ & Avg.\\,$\\tau$ \\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def bootstrap_table(boot: pd.DataFrame) -> str:
    boot = boot.set_index("method")
    rows = []
    for m in METHOD_ORDER:
        if m not in boot.index:
            continue
        r = boot.loc[m]
        rows.append(f"{m} & {f(r['avg_spearman'])} & "
                    f"[{f(r['ci95_low'])}, {f(r['ci95_high'])}] \\\\")
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Average human Spearman correlation with 95\\% bootstrap "
        "confidence intervals (1000 resamples over the evaluation records).}\n"
        "\\label{tab:bootstrap}\n"
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "Method & Avg.\\ Spearman & 95\\% CI \\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def agreement_table(agr: pd.DataFrame) -> str:
    rows = []
    for _, r in agr.iterrows():
        label = str(r["dimension"]).capitalize()
        rows.append(f"{label} & {f(r['pearson'])} & "
                    f"{f(r['mean_diff_small_minus_large'])} & {f(r['mean_abs_diff'])} \\\\")
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Agreement between the small and large judge per dimension: "
        "Pearson correlation, mean signed difference (small$-$large; negative means "
        "the small judge is stricter) and mean absolute difference.}\n"
        "\\label{tab:agreement}\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Dimension & Pearson & Mean diff. & Mean abs.\\ diff. \\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def cost_table(cost: pd.DataFrame) -> str:
    rows = []
    for _, r in cost.iterrows():
        mon = (str(r["monetary_cost"]).replace("$", r"\$")
               .replace("~", r"$\sim$"))
        rows.append(
            f"{r['method']} & {int(r['n_summaries'])} & {int(r['tokens_per_summary'])} & "
            f"{f(r['mean_latency_s'], 2)} & {f(r['throughput_summ_per_min'], 1)} & "
            f"{mon} \\\\")
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Cost and speed of the model-based methods on the evaluation subset.}\n"
        "\\label{tab:cost}\n"
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "Method & $N$ & Tok./summ. & Lat.\\ (s) & Summ./min & Monetary \\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)
    rdir = paths["results_dir"]
    parts = []

    parts.append(correlation_table(pd.read_csv(rdir / "correlation.csv")))
    if (rdir / "bootstrap.csv").exists():
        parts.append(bootstrap_table(pd.read_csv(rdir / "bootstrap.csv")))
    if (rdir / "agreement.csv").exists():
        parts.append(agreement_table(pd.read_csv(rdir / "agreement.csv")))
    if (rdir / "cost.csv").exists():
        parts.append(cost_table(pd.read_csv(rdir / "cost.csv")))

    out = ROOT / "report" / "tables.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"[make_tables] wrote {out}")


if __name__ == "__main__":
    main()
