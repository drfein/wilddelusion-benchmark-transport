from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def clustered_mean_interval(
    frame: pd.DataFrame, value: str, draws: int, seed: int
) -> tuple[float, float, float]:
    grouped = frame.groupby("conversation_hash")[value].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(grouped), size=(draws, len(grouped)))
    boot = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return (
        float(frame[value].mean()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def model_summary(frame: pd.DataFrame, draws: int, seed: int) -> dict[str, Any]:
    effect, low, high = clustered_mean_interval(frame, "difference", draws, seed)
    positive_flips = int(
        (
            (frame["target_only_endorse"] == 0) & (frame["full_context_endorse"] == 1)
        ).sum()
    )
    negative_flips = int(
        (
            (frame["target_only_endorse"] == 1) & (frame["full_context_endorse"] == 0)
        ).sum()
    )
    return {
        "model": str(frame["model"].iloc[0]),
        "targets": len(frame),
        "conversations": int(frame["conversation_hash"].nunique()),
        "target_only_endorsements": int(frame["target_only_endorse"].sum()),
        "target_only_rate": float(frame["target_only_endorse"].mean()),
        "full_context_endorsements": int(frame["full_context_endorse"].sum()),
        "full_context_rate": float(frame["full_context_endorse"].mean()),
        "positive_flips": positive_flips,
        "negative_flips": negative_flips,
        "paired_effect": effect,
        "paired_effect_cluster_ci_low": low,
        "paired_effect_cluster_ci_high": high,
    }


def compare(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    draws: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if reference["model"].nunique() != 1 or comparison["model"].nunique() != 1:
        raise ValueError("Each input must contain exactly one model")
    keys = ["original_row_idx", "conversation_hash", "message_hash"]
    reference_ids = set(reference["original_row_idx"])
    comparison_ids = set(comparison["original_row_idx"])
    if reference_ids != comparison_ids:
        raise ValueError(
            "Inputs do not contain the same target IDs: "
            f"reference_only={len(reference_ids - comparison_ids)}, "
            f"comparison_only={len(comparison_ids - reference_ids)}"
        )

    columns = [
        *keys,
        "target_only_score",
        "full_context_score",
        "target_only_endorse",
        "full_context_endorse",
        "difference",
    ]
    merged = reference[columns].merge(
        comparison[columns],
        on="original_row_idx",
        suffixes=("_reference", "_comparison"),
        validate="one_to_one",
    )
    for key in ("conversation_hash", "message_hash"):
        if not merged[f"{key}_reference"].equals(merged[f"{key}_comparison"]):
            raise ValueError(f"Cohort integrity mismatch in {key}")
        merged[key] = merged.pop(f"{key}_reference")
        merged = merged.drop(columns=f"{key}_comparison")

    merged["effect_difference"] = (
        merged["difference_comparison"] - merged["difference_reference"]
    )
    effect_difference, low, high = clustered_mean_interval(
        merged, "effect_difference", draws, seed + 2
    )
    reference_positive = set(
        merged.loc[
            (merged["target_only_endorse_reference"] == 0)
            & (merged["full_context_endorse_reference"] == 1),
            "original_row_idx",
        ]
    )
    comparison_positive = set(
        merged.loc[
            (merged["target_only_endorse_comparison"] == 0)
            & (merged["full_context_endorse_comparison"] == 1),
            "original_row_idx",
        ]
    )
    summary = {
        "design": "same-target, same-prompt, same-judge matched model comparison",
        "threshold": 7,
        "reference": model_summary(reference, draws, seed),
        "comparison": model_summary(comparison, draws, seed + 1),
        "comparison_minus_reference_paired_effect": effect_difference,
        "comparison_minus_reference_cluster_ci_low": low,
        "comparison_minus_reference_cluster_ci_high": high,
        "positive_flip_overlap": {
            "reference": len(reference_positive),
            "comparison": len(comparison_positive),
            "intersection": len(reference_positive & comparison_positive),
            "union": len(reference_positive | comparison_positive),
        },
        "uncertainty": (
            "Percentile bootstrap resampling source conversations; model and judge "
            "responses are one temperature-zero API sample per arm."
        ),
    }
    output_columns = [
        "original_row_idx",
        "conversation_hash",
        "message_hash",
        "target_only_score_reference",
        "full_context_score_reference",
        "target_only_endorse_reference",
        "full_context_endorse_reference",
        "difference_reference",
        "target_only_score_comparison",
        "full_context_score_comparison",
        "target_only_endorse_comparison",
        "full_context_endorse_comparison",
        "difference_comparison",
        "effect_difference",
    ]
    return summary, merged[output_columns].sort_values("original_row_idx")


def plot_comparison(summary: dict[str, Any], output: Path) -> None:
    models = [summary["reference"], summary["comparison"]]
    labels = ["GPT-5.4 mini", "GPT-4.1 mini"]
    colors = ["#287271", "#D2643F"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)

    x = np.arange(len(models))
    width = 0.31
    for arm_index, (arm, label, offset, marker) in enumerate(
        (
            ("target_only", "Target only", -width / 2, "o"),
            ("full_context", "Full history", width / 2, "s"),
        )
    ):
        rates = np.array([model[f"{arm}_rate"] for model in models])
        counts = np.array([model[f"{arm}_endorsements"] for model in models])
        totals = np.array([model["targets"] for model in models])
        intervals = np.array(
            [
                proportion_confint(int(k), int(n), method="wilson")
                for k, n in zip(counts, totals, strict=True)
            ]
        )
        errors = np.maximum(
            0,
            np.vstack(
                (100 * (rates - intervals[:, 0]), 100 * (intervals[:, 1] - rates))
            ),
        )
        axes[0].errorbar(
            x + offset,
            100 * rates,
            yerr=errors,
            fmt=marker,
            markersize=7,
            capsize=3,
            linewidth=1.7,
            color="#344E5C" if arm_index == 0 else "#D2643F",
            label=label,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Endorsement (%)")
    axes[0].set_title("Matched response rates", loc="left", weight="bold")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(axis="y", alpha=0.22)

    for index, (model, label, color) in enumerate(
        zip(models, labels, colors, strict=True)
    ):
        effect = 100 * model["paired_effect"]
        low = 100 * model["paired_effect_cluster_ci_low"]
        high = 100 * model["paired_effect_cluster_ci_high"]
        axes[1].errorbar(
            effect,
            index,
            xerr=[[effect - low], [high - effect]],
            fmt="o",
            markersize=8,
            capsize=3,
            linewidth=2,
            color=color,
        )
    axes[1].axvline(0, color="#88959B", linewidth=1)
    axes[1].set_yticks(range(len(labels)), labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Full-history minus target-only (pp)")
    axes[1].set_title("Paired context effect", loc="left", weight="bold")
    axes[1].grid(axis="x", alpha=0.22)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two matched context studies.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    summary, rows = compare(
        pd.read_csv(args.reference),
        pd.read_csv(args.comparison),
        args.bootstrap_draws,
        args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out_dir / "matched_model_comparison.csv", index=False)
    (args.out_dir / "model_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_comparison(summary, args.out_dir / "model_comparison.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
