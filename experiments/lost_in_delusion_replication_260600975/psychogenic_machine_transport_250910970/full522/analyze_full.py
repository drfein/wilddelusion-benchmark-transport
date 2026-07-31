#!/usr/bin/env python3
"""Cluster-aware analysis for the full WildDelusion transport experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from io_utils import read_jsonl, write_manifest  # noqa: E402


PAPER_EFFECTS = {
    "DCS": 1.07 - 0.76,
    "HES": 0.82 - 0.56,
    "SIS": (1.55 - 2.89) / 6,
}
MODEL_LABELS = {
    "allenai/Olmo-3-7B-Instruct": "OLMo-3-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "Qwen/Qwen3-4B": "Qwen3-4B",
    "Qwen/Qwen3-14B": "Qwen3-14B",
    "Qwen/Qwen3-30B-A3B": "Qwen3-30B-A3B",
}
METRIC_LABELS = {
    "DCS": "Delusion confirmation",
    "HES": "Harm enablement",
    "SIS": "Safety intervention",
}
EXPECTED_JUDGE_PROTOCOL = "Psychosis-Bench original separate prompts"


def validate_judgments(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject mixed judge protocols or nonnumeric paper-scale scores."""
    protocols = set(frame["judge_protocol"].dropna().unique())
    if protocols != {EXPECTED_JUDGE_PROTOCOL}:
        raise ValueError(
            f"Expected only {EXPECTED_JUDGE_PROTOCOL!r}, found {sorted(protocols)!r}"
        )
    score_ranges = {"DCS": (0, 2), "HES": (0, 2), "SIS": (0, 1)}
    for metric, (minimum, maximum) in score_ranges.items():
        numeric = pd.to_numeric(frame[metric], errors="coerce")
        if numeric.isna().any() or not numeric.between(minimum, maximum).all():
            raise ValueError(f"{metric} contains invalid paper-scale scores")
        frame[metric] = numeric.astype(int)
    return frame


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * p_values[order]
    )
    adjusted = np.empty_like(adjusted_sorted, dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def cluster_bootstrap(
    differences: pd.DataFrame, seed: int, draws: int
) -> tuple[float, float, float, float]:
    """Bootstrap source conversations, retaining every target within a draw."""
    by_cluster = {
        cluster: group["difference"].to_numpy(dtype=float)
        for cluster, group in differences.groupby("cluster_id", sort=False)
    }
    cluster_ids = np.array(list(by_cluster), dtype=object)
    cluster_sums = np.array([by_cluster[key].sum() for key in cluster_ids])
    cluster_sizes = np.array([len(by_cluster[key]) for key in cluster_ids])
    cluster_means = cluster_sums / cluster_sizes
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    cluster_equal_estimates = np.empty(draws)
    chunk = 2_000
    for start in range(0, draws, chunk):
        count = min(chunk, draws - start)
        picks = rng.integers(0, len(cluster_ids), size=(count, len(cluster_ids)))
        estimates[start : start + count] = (
            cluster_sums[picks].sum(axis=1) / cluster_sizes[picks].sum(axis=1)
        )
        cluster_equal_estimates[start : start + count] = cluster_means[picks].mean(
            axis=1
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    equal_low, equal_high = np.quantile(
        cluster_equal_estimates, [0.025, 0.975]
    )
    return float(low), float(high), float(equal_low), float(equal_high)


def cluster_randomization_p(
    differences: pd.DataFrame, seed: int, draws: int
) -> float:
    """Flip all paired effects from one source conversation together."""
    cluster_sums = (
        differences.groupby("cluster_id", sort=False)["difference"].sum().to_numpy()
    )
    observed = abs(float(cluster_sums.sum()))
    n_clusters = len(cluster_sums)
    if n_clusters <= 16:
        assignments = np.arange(1 << n_clusters, dtype=np.uint64)[:, None]
        bit_positions = np.arange(n_clusters, dtype=np.uint64)[None, :]
        signs = 1.0 - 2.0 * ((assignments >> bit_positions) & 1)
        randomized = np.abs(signs @ cluster_sums)
        return float(np.mean(randomized >= observed - 1e-12))
    rng = np.random.default_rng(seed)
    extreme = 0
    seen = 0
    chunk = 10_000
    while seen < draws:
        count = min(chunk, draws - seen)
        signs = rng.choice((-1.0, 1.0), size=(count, n_clusters))
        extreme += int(np.sum(np.abs(signs @ cluster_sums) >= observed - 1e-12))
        seen += count
    return float((extreme + 1) / (draws + 1))


def make_paired(
    frame: pd.DataFrame, metric: str, condition_a: str, condition_b: str
) -> pd.DataFrame:
    metadata = (
        frame.groupby("pair_id", as_index=False)
        .agg(
            cluster_id=("cluster_id", "first"),
            theme=("theme", "first"),
            source=("source", "first"),
        )
        .set_index("pair_id")
    )
    wide = frame.pivot_table(
        index="pair_id", columns="condition", values=metric, aggfunc="first"
    ).dropna(subset=[condition_a, condition_b])
    paired = wide.join(metadata, how="inner")
    paired["difference"] = paired[condition_a] - paired[condition_b]
    return paired.reset_index()


def paired_results(
    frame: pd.DataFrame, bootstrap_draws: int, permutation_draws: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    pair_rows: list[pd.DataFrame] = []
    for model, model_frame in frame.groupby("model", sort=False):
        for metric in ("DCS", "HES", "SIS"):
            eligible = (
                model_frame
                if metric == "DCS"
                else model_frame[model_frame["harm_pair_valid"] == True]  # noqa: E712
            )
            paired = make_paired(eligible, metric, "implicit", "explicit")
            if paired.empty:
                continue
            paired["model"] = model
            paired["metric"] = metric
            pair_rows.append(paired)
            low, high, equal_low, equal_high = cluster_bootstrap(
                paired, seed=250910970, draws=bootstrap_draws
            )
            cluster_means = paired.groupby("cluster_id")["difference"].mean()
            delta = float(paired["difference"].mean())
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "paired_n": len(paired),
                    "cluster_n": paired["cluster_id"].nunique(),
                    "explicit_mean": paired["explicit"].mean(),
                    "implicit_mean": paired["implicit"].mean(),
                    "real_delta_implicit_minus_explicit": delta,
                    "cluster_bootstrap_ci95_low": low,
                    "cluster_bootstrap_ci95_high": high,
                    "cluster_randomization_p": cluster_randomization_p(
                        paired, seed=250910970, draws=permutation_draws
                    ),
                    "cluster_equal_delta": cluster_means.mean(),
                    "cluster_equal_ci95_low": equal_low,
                    "cluster_equal_ci95_high": equal_high,
                    "paper_delta": PAPER_EFFECTS[metric],
                    "same_direction_as_paper": (
                        np.sign(delta) == np.sign(PAPER_EFFECTS[metric])
                    ),
                }
            )
    return pd.DataFrame(rows), pd.concat(pair_rows, ignore_index=True)


def correlation_results(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eligibility_frames = {
        "strict high-stakes action": frame[
            frame["harm_pair_valid"] == True  # noqa: E712
        ],
        "broad possible/clear harm": frame[
            frame["original_harm"].isin(["possible", "clear"])
        ],
    }
    for eligibility, harm in eligibility_frames.items():
        groups = [("pooled", harm)]
        groups.extend((model, group) for model, group in harm.groupby("model"))
        for name, group in groups:
            result = spearmanr(group["DCS"], group["HES"])
            rows.append(
                {
                    "eligibility": eligibility,
                    "group": name,
                    "n_responses": len(group),
                    "n_clusters": group["cluster_id"].nunique(),
                    "spearman_dcs_hes": float(result.statistic),
                    "p_value_unclustered_descriptive": float(result.pvalue),
                    "paper_spearman": 0.77,
                }
            )
    return pd.DataFrame(rows)


def theme_results(pair_level: pd.DataFrame) -> pd.DataFrame:
    return (
        pair_level.groupby(["model", "metric", "theme"], as_index=False)
        .agg(
            paired_n=("pair_id", "size"),
            cluster_n=("cluster_id", "nunique"),
            mean_difference=("difference", "mean"),
        )
        .sort_values(["model", "metric", "paired_n"], ascending=[True, True, False])
    )


def plot_results(paired: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Avenir",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
        }
    )
    models = [model for model in MODEL_LABELS if model in set(paired["model"])]
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.25), facecolor="white")
    x = np.arange(len(models))
    width = 0.29
    for axis, metric in zip(axes, ("DCS", "HES", "SIS"), strict=True):
        selected = paired[paired["metric"] == metric].set_index("model")
        paper = np.array([PAPER_EFFECTS[metric]] * len(models))
        real = np.array(
            [selected.loc[model, "real_delta_implicit_minus_explicit"] for model in models]
        )
        low = np.array(
            [selected.loc[model, "cluster_bootstrap_ci95_low"] for model in models]
        )
        high = np.array(
            [selected.loc[model, "cluster_bootstrap_ci95_high"] for model in models]
        )
        axis.bar(
            x - width / 2,
            paper,
            width,
            color="#9AA5B1",
            label="Synthetic pooled",
        )
        axis.bar(
            x + width / 2,
            real,
            width,
            color="#D8734A",
            label="WildDelusion",
            yerr=np.vstack((real - low, high - real)),
            capsize=3,
            error_kw={"elinewidth": 1.1},
        )
        axis.axhline(0, color="#263238", linewidth=0.8)
        axis.set_title(METRIC_LABELS[metric], weight="bold")
        axis.set_ylabel("Implicit minus explicit")
        axis.set_xticks(x)
        axis.set_xticklabels([MODEL_LABELS.get(model, model) for model in models])
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=2, loc="upper center")
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    figure.savefig(output_dir / "implicit_explicit_transport_clustered.png", dpi=300)
    figure.savefig(output_dir / "implicit_explicit_transport_clustered.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judgments", type=Path, default=Path("results/judgments.jsonl")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/analysis_clustered")
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--permutation-draws", type=int, default=100_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = validate_judgments(
        pd.DataFrame(
            row for row in read_jsonl(args.judgments) if not row.get("judge_error")
        )
    )
    paired, pair_level = paired_results(
        frame, args.bootstrap_draws, args.permutation_draws
    )
    paired["holm_p_across_primary_tests"] = holm_adjust(
        paired["cluster_randomization_p"].to_numpy(dtype=float)
    )
    correlations = correlation_results(frame)
    themes = theme_results(pair_level)
    broad_rows: list[dict[str, Any]] = []
    for model, model_frame in frame.groupby("model", sort=False):
        broad = model_frame[model_frame["original_harm"].isin(["possible", "clear"])]
        for metric in ("HES", "SIS"):
            paired_broad = make_paired(
                broad, metric, "implicit", "explicit"
            )
            low, high, equal_low, equal_high = cluster_bootstrap(
                paired_broad, seed=250910970, draws=args.bootstrap_draws
            )
            broad_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "paired_n": len(paired_broad),
                    "cluster_n": paired_broad["cluster_id"].nunique(),
                    "explicit_mean": paired_broad["explicit"].mean(),
                    "implicit_mean": paired_broad["implicit"].mean(),
                    "delta_implicit_minus_explicit": paired_broad[
                        "difference"
                    ].mean(),
                    "cluster_bootstrap_ci95_low": low,
                    "cluster_bootstrap_ci95_high": high,
                    "cluster_randomization_p": cluster_randomization_p(
                        paired_broad,
                        seed=250910970,
                        draws=args.permutation_draws,
                    ),
                    "cluster_equal_ci95_low": equal_low,
                    "cluster_equal_ci95_high": equal_high,
                    "eligibility": "original_harm in {possible, clear}",
                }
            )
    broad_sensitivity = pd.DataFrame(broad_rows)
    paired.to_csv(args.output_dir / "paired_effects_clustered.csv", index=False)
    pair_level.to_csv(args.output_dir / "pair_level_effects.csv", index=False)
    correlations.to_csv(args.output_dir / "dcs_hes_correlations.csv", index=False)
    themes.to_csv(args.output_dir / "theme_effects_descriptive.csv", index=False)
    broad_sensitivity.to_csv(
        args.output_dir / "broad_harm_sensitivity.csv", index=False
    )
    plot_results(paired, args.output_dir)
    write_manifest(
        args.output_dir / "summary.json",
        {
            "paper": "arXiv:2509.10970",
            "design": (
                "paired explicit/implicit target rewrites in unchanged real histories; "
                "source-conversation-clustered inference"
            ),
            "judged_responses": len(frame),
            "pairs": int(frame["pair_id"].nunique()),
            "clusters": int(frame["cluster_id"].nunique()),
            "harm_valid_pairs": int(
                frame.loc[frame["harm_pair_valid"] == True, "pair_id"].nunique()  # noqa: E712
            ),
            "paper_effects": PAPER_EFFECTS,
            "paired_effects": json.loads(paired.to_json(orient="records")),
        },
    )
    print(paired.to_string(index=False))
    print("\nDCS-HES correlations (descriptive p-values are unclustered):")
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
