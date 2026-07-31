#!/usr/bin/env python3
"""Analyze Psychosis-Bench claims on paired real WildDelusion histories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import read_jsonl, write_manifest  # noqa: E402


PAPER_EFFECTS = {
    "DCS": 1.07 - 0.76,
    "HES": 0.82 - 0.56,
    # Paper reports intervention counts over six applicable turns.
    "SIS": (1.55 - 2.89) / 6,
}
MODEL_LABELS = {
    "allenai/Olmo-3-7B-Instruct": "OLMo-3-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
}
COLORS = {"paper": "#7B8794", "real": "#D05A47"}


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def paired_results(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, model_frame in frame.groupby("model"):
        for metric in ("DCS", "HES", "SIS"):
            eligible = (
                model_frame
                if metric == "DCS"
                else model_frame[model_frame["harm_pair_valid"] == True]  # noqa: E712
            )
            wide = eligible.pivot(
                index="pair_id", columns="condition", values=metric
            ).dropna(subset=["explicit", "implicit"])
            differences = (wide["implicit"] - wide["explicit"]).to_numpy()
            low, high = bootstrap_ci(differences, seed=250910970)
            p_value = (
                1.0
                if not np.any(differences)
                else float(
                    wilcoxon(
                        differences, zero_method="pratt", method="auto"
                    ).pvalue
                )
            )
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "paired_n": len(wide),
                    "explicit_mean": wide["explicit"].mean(),
                    "implicit_mean": wide["implicit"].mean(),
                    "real_delta_implicit_minus_explicit": differences.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "paired_p": p_value,
                    "paper_delta": PAPER_EFFECTS[metric],
                    "same_direction_as_paper": (
                        np.sign(differences.mean())
                        == np.sign(PAPER_EFFECTS[metric])
                    ),
                }
            )
    return pd.DataFrame(rows)


def correlation_results(frame: pd.DataFrame) -> pd.DataFrame:
    harm = frame[frame["harm_pair_valid"] == True].copy()  # noqa: E712
    groups = [("pooled", harm)]
    groups.extend((model, group) for model, group in harm.groupby("model"))
    rows = []
    for name, group in groups:
        result = spearmanr(group["DCS"], group["HES"])
        rows.append(
            {
                "group": name,
                "n_responses": len(group),
                "spearman_dcs_hes": float(result.statistic),
                "p_value": float(result.pvalue),
                "paper_spearman": 0.77,
            }
        )
    return pd.DataFrame(rows)


def theme_results(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, condition, theme), group in frame.groupby(
        ["model", "condition", "theme"]
    ):
        rows.append(
            {
                "model": model,
                "condition": condition,
                "theme": theme,
                "n": len(group),
                "mean_dcs": group["DCS"].mean(),
                "mean_hes": group.loc[
                    group["harm_pair_valid"] == True, "HES"  # noqa: E712
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_results(paired: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    models = list(MODEL_LABELS)
    metrics = ["DCS", "HES", "SIS"]
    titles = [
        "Delusion confirmation",
        "Harm enablement",
        "Safety intervention",
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 3.5))
    x = np.arange(len(models))
    width = 0.28
    for axis, metric, title in zip(axes, metrics, titles, strict=True):
        selected = paired[paired["metric"] == metric].set_index("model")
        paper = np.array([selected.loc[model, "paper_delta"] for model in models])
        real = np.array(
            [
                selected.loc[
                    model, "real_delta_implicit_minus_explicit"
                ]
                for model in models
            ]
        )
        low = np.array([selected.loc[model, "ci95_low"] for model in models])
        high = np.array([selected.loc[model, "ci95_high"] for model in models])
        axis.bar(
            x - width / 2,
            paper,
            width,
            color=COLORS["paper"],
            label="Synthetic paper",
        )
        axis.bar(
            x + width / 2,
            real,
            width,
            color=COLORS["real"],
            label="WildDelusion",
            yerr=np.array([real - low, high - real]),
            capsize=3,
        )
        axis.axhline(0, color="#28323C", linewidth=0.8)
        axis.set_title(title, weight="bold")
        axis.set_ylabel("Implicit minus explicit")
        axis.set_xticks(x)
        axis.set_xticklabels([MODEL_LABELS[model] for model in models])
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=2, loc="upper center")
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output_dir / "implicit_explicit_transport.png", dpi=240)
    figure.savefig(output_dir / "implicit_explicit_transport.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judgments", type=Path, default=Path("results/judgments.jsonl")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/analysis")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        row for row in read_jsonl(args.judgments) if not row.get("judge_error")
    )
    paired = paired_results(frame)
    correlations = correlation_results(frame)
    themes = theme_results(frame)
    paired.to_csv(args.output_dir / "paired_effects.csv", index=False)
    correlations.to_csv(args.output_dir / "dcs_hes_correlations.csv", index=False)
    themes.to_csv(args.output_dir / "theme_results.csv", index=False)
    plot_results(paired, args.output_dir)
    write_manifest(
        args.output_dir / "summary.json",
        {
            "paper": "arXiv:2509.10970",
            "design": (
                "target-level paired explicit/implicit variants inserted into "
                "unchanged real conversation histories"
            ),
            "judged_responses": len(frame),
            "pairs": int(frame["pair_id"].nunique()),
            "harm_valid_pairs": int(
                frame.loc[frame["harm_pair_valid"] == True, "pair_id"].nunique()  # noqa: E712
            ),
            "paper_effects": PAPER_EFFECTS,
            "paired_effects": json.loads(paired.to_json(orient="records")),
            "correlations": json.loads(correlations.to_json(orient="records")),
        },
    )
    print(paired.to_string(index=False))
    print("\nDCS-HES correlations:")
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
