#!/usr/bin/env python3
"""Plot the easy-control success and natural-control transport failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from design import SEED


def cluster_interval(frame: pd.DataFrame, n_boot: int = 3000) -> tuple[float, float, float]:
    values = frame.groupby("conversation_id").score.mean().to_numpy()
    rng = np.random.default_rng(SEED)
    means = np.empty(n_boot)
    for index in range(n_boot):
        means[index] = rng.choice(values, len(values), replace=True).mean()
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result_dir = args.work_dir / "random_control_results"
    oof = pd.read_csv(result_dir / "oof_scores.csv")
    hard = pd.read_csv(result_dir / "natural_hard_control_scores.csv")
    trajectory = pd.read_csv(result_dir / "trajectory_scores.csv")
    results = json.loads((result_dir / "results.json").read_text())
    positives = oof[oof.label == 1].copy()

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    colors = {"Real endpoints": "#D95F59", "Random controls": "#4C78A8"}
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.35), gridspec_kw={"wspace": 0.34})

    oof["group"] = np.where(oof.label == 1, "Real endpoints", "Random controls")
    sns.violinplot(
        data=oof,
        x="group",
        y="score",
        hue="group",
        palette=colors,
        legend=False,
        inner="quart",
        cut=0,
        linewidth=0.8,
        ax=axes[0],
    )
    axes[0].set(xlabel="", ylabel="DelusionScore analog", title="Easy controls")
    axes[0].text(
        0.97, 0.95, "OOF AUC 0.979", transform=axes[0].transAxes, ha="right", va="top",
        fontsize=9, weight="bold", bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"}
    )

    ecdf = pd.concat(
        [positives.assign(group="Real endpoints"), hard.assign(group="Natural near misses")],
        ignore_index=True,
    )
    sns.ecdfplot(
        data=ecdf,
        x="score",
        hue="group",
        palette=["#D95F59", "#54A24B"],
        linewidth=2,
        ax=axes[1],
    )
    axes[1].axvline(0.5, color="#555555", linestyle="--", linewidth=1)
    axes[1].set(xlabel="DelusionScore analog", ylabel="Cumulative share", title="Natural near misses")
    axes[1].text(
        0.97, 0.05, "AUC 0.670  |  FPR 84.5%", transform=axes[1].transAxes,
        ha="right", va="bottom", fontsize=9, weight="bold",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"}
    )
    axes[1].get_legend().set_title("")

    prior = trajectory[~trajectory.is_target].copy()
    prior["relative_position"] = prior.user_turn_index / np.maximum(prior.n_user_turns - 1, 1)
    prior["bin"] = pd.cut(prior.relative_position, [-0.001, .2, .4, .6, .8, 1.001], labels=False)
    x_values, estimates, lower, upper = [], [], [], []
    for bin_index, group in prior.groupby("bin", observed=True):
        estimate, low, high = cluster_interval(group)
        x_values.append((float(bin_index) + 0.5) / 5)
        estimates.append(estimate)
        lower.append(estimate - low)
        upper.append(high - estimate)
    target_estimate, target_low, target_high = cluster_interval(trajectory[trajectory.is_target])
    x_values.append(1.08)
    estimates.append(target_estimate)
    lower.append(target_estimate - target_low)
    upper.append(target_high - target_estimate)
    axes[2].errorbar(
        x_values,
        estimates,
        yerr=[lower, upper],
        color="#F58518",
        marker="o",
        linewidth=2,
        capsize=3,
    )
    axes[2].axvline(1.0, color="#777777", linewidth=1, linestyle=":")
    axes[2].set(
        xlabel="Progress toward selected endpoint",
        ylabel="Mean score (95% cluster CI)",
        title="Real user-turn trajectories",
        xlim=(0, 1.16),
    )
    p_value = results["trajectory_cluster_robust_regressions"]["prior_only_conversation_fe"]["relative_position_p"]
    axes[2].text(
        0.04, 0.94, f"Pre-endpoint adjusted p={p_value:.3f}", transform=axes[2].transAxes,
        va="top", fontsize=9, weight="bold",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"}
    )
    axes[2].text(1.08, axes[2].get_ylim()[0], "selected\nendpoint", ha="center", va="bottom", fontsize=8)

    for label, axis in zip("ABC", axes):
        axis.text(-0.15, 1.05, label, transform=axis.transAxes, fontsize=13, weight="bold")
    sns.despine(fig=fig)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
