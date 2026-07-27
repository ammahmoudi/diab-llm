#!/usr/bin/env python3
"""Generate an auditable ECG fairness figure from locked seed-level results.

The plot reports sex-specific ectopy recall alongside the per-seed equal-
opportunity (EO) gaps for the binary MIT-BIH ectopy protocol. The input summary
is produced by ``aggregate_mitbih_multiseed_classification.py`` and stores the
sex-specific recalls used to calculate each selected EO value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("experiments/mitbih_binary_ectopy_five_seed/multiseed_per_seed_summary.csv")
DEFAULT_OUTPUT = Path("fairness_article/figures/generated/fig_ecg_fairness_summary")
VARIANT_NAMES = {
    "KD": "Standard KD",
    "T1": "EBTD",
    "O2": "GCOA",
    "T1+O2": "EBTD+GCOA",
}
METHODS = list(VARIANT_NAMES.values())
GROUP_COLORS = {"Female": "#2A7F9E", "Male": "#789D3C"}
SEED_COLOR = "#455A64"
MEAN_COLOR = "#C7511F"
GRID_COLOR = "#D7E0E5"
ERROR_COLOR = "#263238"


def load_results(path: Path) -> pd.DataFrame:
    """Extract per-seed sex recall and EO values from the locked summary."""
    rows: list[dict[str, float | int | str]] = []
    source = pd.read_csv(path)
    for _, record in source.iterrows():
        if record["variant"] not in VARIANT_NAMES:
            continue
        details = json.loads(record["eo_details"])
        ectopy = details["1"]
        for sex_code, label in (("F", "Female"), ("M", "Male")):
            rows.append(
                {
                    "seed": int(record["seed"]),
                    "method": VARIANT_NAMES[record["variant"]],
                    "sex": label,
                    "ectopy_support": int(ectopy["supports"][sex_code]),
                    "ectopy_recall": float(ectopy["recalls"][sex_code]),
                    "raw_eo_gap": float(record["selected_eo"]),
                }
            )
    output = pd.DataFrame(rows)
    expected = len(METHODS) * 5 * 2
    if len(output) != expected:
        raise ValueError(f"Expected {expected} sex-specific rows, found {len(output)}")
    return output


def plot_fairness_summary(results: pd.DataFrame, output_prefix: Path) -> pd.DataFrame:
    """Plot absolute sex-specific recall and matched EO gaps across seeds."""
    summary = (
        results.groupby(["method", "sex"], observed=True)["ectopy_recall"]
        .agg(mean="mean", sd=lambda values: values.std(ddof=1))
        .reset_index()
    )
    eo_by_seed = (
        results[["seed", "method", "raw_eo_gap"]]
        .drop_duplicates()
        .pivot(index="seed", columns="method", values="raw_eo_gap")
        .reindex(columns=METHODS)
    )

    figure, (recall_axis, eo_axis) = plt.subplots(1, 2, figsize=(7.35, 3.35), constrained_layout=True)
    method_positions = np.arange(len(METHODS))
    offsets = {"Female": -0.18, "Male": 0.18}
    for sex in GROUP_COLORS:
        subset = summary[summary["sex"] == sex].set_index("method").loc[METHODS]
        positions = method_positions + offsets[sex]
        recall_axis.bar(
            positions,
            subset["mean"],
            width=0.32,
            yerr=subset["sd"],
            capsize=2.5,
            color=GROUP_COLORS[sex],
            edgecolor=ERROR_COLOR,
            linewidth=0.6,
            error_kw={"elinewidth": 0.85, "ecolor": ERROR_COLOR},
            label=sex,
        )
        points = results[results["sex"] == sex]
        for index, method in enumerate(METHODS):
            values = points.loc[points["method"] == method, "ectopy_recall"].to_numpy()
            recall_axis.scatter(
                np.full(values.size, positions[index]),
                values,
                s=16,
                color=SEED_COLOR,
                alpha=0.78,
                edgecolors="white",
                linewidths=0.35,
                zorder=4,
            )

    recall_axis.set_title("(a) Group-specific ectopy detection", fontsize=10, fontweight="bold", pad=12)
    recall_axis.set_xticks(method_positions, METHODS, fontsize=8)
    recall_axis.set_ylabel("Ectopy recall (higher better)", fontsize=9)
    recall_axis.set_ylim(0, 1)
    recall_axis.set_yticks(np.arange(0, 1.01, 0.2))
    recall_axis.tick_params(axis="y", labelsize=8)
    recall_axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
    recall_axis.set_axisbelow(True)
    recall_axis.spines[["top", "right"]].set_visible(False)
    recall_axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8,
        frameon=True, facecolor="white", edgecolor="#90A4AE", framealpha=1,
        handlelength=1.1, columnspacing=1.0, borderpad=0.45,
    )

    eo_positions = np.arange(len(METHODS))
    for _, values in eo_by_seed.iterrows():
        eo_axis.plot(eo_positions, values.to_numpy(), color="#B0BEC5", linewidth=0.9, alpha=0.9, zorder=1)
        eo_axis.scatter(eo_positions, values.to_numpy(), color=SEED_COLOR, s=18, edgecolors="white", linewidths=0.35, zorder=2)
    means = eo_by_seed.mean(axis=0).to_numpy()
    sds = eo_by_seed.std(axis=0, ddof=1).to_numpy()
    eo_axis.errorbar(
        eo_positions,
        means,
        yerr=sds,
        fmt="D",
        markersize=5.5,
        color=MEAN_COLOR,
        ecolor=MEAN_COLOR,
        capsize=3,
        linewidth=1.1,
        zorder=3,
        label="Mean +/- SD",
    )
    eo_axis.set_title("(b) Ectopy EO endpoint", fontsize=10, fontweight="bold", pad=12)
    eo_axis.set_xticks(eo_positions, METHODS, fontsize=8)
    eo_axis.set_ylabel("Female/male EO gap (lower better)", fontsize=9)
    eo_axis.set_ylim(0, 0.28)
    eo_axis.set_yticks(np.arange(0, 0.281, 0.05))
    eo_axis.tick_params(axis="y", labelsize=8)
    eo_axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
    eo_axis.set_axisbelow(True)
    eo_axis.spines[["top", "right"]].set_visible(False)
    eo_axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8,
        frameon=True, facecolor="white", edgecolor="#90A4AE", framealpha=1,
        borderpad=0.45,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a locked-result ECG fairness figure.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    results = load_results(args.input)
    summary = plot_fairness_summary(results, args.output_prefix)
    results.to_csv(args.output_prefix.parent / "ecg_ectopy_recall_by_seed.csv", index=False)
    summary.to_csv(args.output_prefix.parent / "ecg_ectopy_recall_summary.csv", index=False)
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
