#!/usr/bin/env python3
"""Compare synthetic-paper effects with strict real-data counterfactuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from io_utils import read_jsonl, write_manifest


ORDINAL = {"N/A": 0.0, "1": 0.0, "2": 1.0, "3": 2.0}
MODEL_LABELS = {
    "allenai/Olmo-3-7B-Instruct": "OLMo-3-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
}
SYNTHETIC_DELTAS = {
    "allenai/Olmo-3-7B-Instruct": {
        "DCS ordinal": 0.96,
        "HES ordinal": 0.60,
        "Safety intervention": -0.30,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "DCS ordinal": 0.58,
        "HES ordinal": 0.18,
        "Safety intervention": -0.09,
    },
}
COLORS = {
    "synthetic": "#7B8794",
    "real": "#D05A47",
    "target_only": "#D9A441",
    "assistant_neutralized": "#3E8E8B",
    "full_history": "#284B63",
}


def load_judgments(path: Path, scope: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        row for row in read_jsonl(path) if not row.get("judge_error")
    )
    frame["scope"] = scope
    frame["DCS ordinal"] = frame["DCS"].map(ORDINAL)
    frame["HES ordinal"] = frame["HES"].map(ORDINAL)
    frame["Safety intervention"] = frame["SIS"].astype(float)
    frame["DCS validate"] = (frame["DCS"] == "3").astype(float)
    frame["DCS positive"] = frame["DCS"].isin(["2", "3"]).astype(float)
    return frame


def bootstrap_ci(
    values: np.ndarray, seed: int, draws: int = 20_000
) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def paired_p_value(differences: np.ndarray, binary: bool) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 1.0
    if binary:
        positive = int((nonzero > 0).sum())
        return float(
            binomtest(min(positive, len(nonzero) - positive), len(nonzero), 0.5).pvalue
        )
    return float(wilcoxon(differences, zero_method="pratt", method="auto").pvalue)


def paired_condition_results(target: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ["DCS ordinal", "HES ordinal", "Safety intervention"]
    for model, model_frame in target.groupby("model"):
        for metric in metrics:
            wide = model_frame.pivot(
                index="pair_id", columns="condition", values=metric
            ).dropna(subset=["delusion", "grounded_control"])
            differences = (
                wide["delusion"] - wide["grounded_control"]
            ).to_numpy()
            low, high = bootstrap_ci(differences, seed=260600975)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "paired_n": len(wide),
                    "control_mean": wide["grounded_control"].mean(),
                    "delusion_mean": wide["delusion"].mean(),
                    "real_delta": differences.mean(),
                    "real_ci95_low": low,
                    "real_ci95_high": high,
                    "paired_p": paired_p_value(
                        differences, binary=metric == "Safety intervention"
                    ),
                    "synthetic_delta": SYNTHETIC_DELTAS[model][metric],
                    "delta_gap_real_minus_synthetic": (
                        differences.mean() - SYNTHETIC_DELTAS[model][metric]
                    ),
                }
            )
    return pd.DataFrame(rows)


def harm_subset_results(
    target: pd.DataFrame, validation_path: Path
) -> pd.DataFrame:
    validation = pd.DataFrame(read_jsonl(validation_path))[
        ["pair_id", "original_harm", "control_harm", "original_distress", "control_distress"]
    ]
    joined = target.drop(
        columns=["original_harm", "control_harm"], errors="ignore"
    ).merge(validation, on="pair_id", how="left")
    subsets = {
        "possible_or_clear_harm": joined[
            joined["original_harm"].isin(["possible", "clear"])
            & joined["control_harm"].isin(["possible", "clear"])
        ],
        "moderate_or_severe_distress": joined[
            joined["original_distress"].isin(["moderate", "severe"])
            & joined["control_distress"].isin(["moderate", "severe"])
        ],
    }
    rows: list[dict[str, Any]] = []
    for subset_name, subset in subsets.items():
        for model, model_frame in subset.groupby("model"):
            wide = model_frame.pivot(
                index="pair_id",
                columns="condition",
                values="Safety intervention",
            ).dropna(subset=["delusion", "grounded_control"])
            differences = (
                wide["delusion"] - wide["grounded_control"]
            ).to_numpy()
            low, high = bootstrap_ci(differences, seed=260600976)
            rows.append(
                {
                    "subset": subset_name,
                    "model": model,
                    "paired_n": len(wide),
                    "control_sis": wide["grounded_control"].mean(),
                    "delusion_sis": wide["delusion"].mean(),
                    "sis_delta": differences.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "paired_p": paired_p_value(differences, binary=True),
                }
            )
    return pd.DataFrame(rows)


def context_results(
    target: pd.DataFrame,
    neutralized: pd.DataFrame,
    grounded: pd.DataFrame,
    full: pd.DataFrame,
    cohort_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_rows = read_jsonl(cohort_path)
    cohort = pd.DataFrame(
        {
            "pair_id": row["pair_id"],
            "escalated_observed": row["escalated_observed"],
            "retained_message_count": row["retained_message_count"],
            "prior_assistant_turns": sum(
                message["role"] == "assistant"
                for message in row["history_messages"][:-1]
            ),
        }
        for row in cohort_rows
    )
    pieces = [
        target[target["condition"] == "delusion"].assign(arm="target_only"),
        neutralized.assign(arm="assistant_neutralized"),
        grounded.assign(arm="assistant_grounded"),
        full.assign(arm="full_history"),
    ]
    combined = pd.concat(pieces, ignore_index=True).merge(
        cohort, on="pair_id", how="left"
    )
    metrics = [
        "DCS ordinal",
        "DCS validate",
        "DCS positive",
        "HES ordinal",
        "Safety intervention",
    ]
    contrasts = {
        "user_history_placeholder": ("assistant_neutralized", "target_only"),
        "assistant_semantics_placeholder": (
            "full_history",
            "assistant_neutralized",
        ),
        "grounded_context": ("assistant_grounded", "target_only"),
        "assistant_semantics_grounded": ("full_history", "assistant_grounded"),
        "all_context": ("full_history", "target_only"),
    }
    rows: list[dict[str, Any]] = []
    for model, model_frame in combined.groupby("model"):
        for escalation_group in ("all", False, True):
            group = (
                model_frame
                if escalation_group == "all"
                else model_frame[
                    model_frame["escalated_observed"] == escalation_group
                ]
            )
            for metric in metrics:
                wide = group.pivot(index="pair_id", columns="arm", values=metric)
                assistant_pair_ids = set(
                    group.loc[
                        group["prior_assistant_turns"] > 0, "pair_id"
                    ].tolist()
                )
                for contrast, (left, right) in contrasts.items():
                    paired = wide.dropna(subset=[left, right])
                    if contrast.startswith("assistant_semantics"):
                        paired = paired.loc[
                            paired.index.isin(assistant_pair_ids)
                        ]
                    differences = (paired[left] - paired[right]).to_numpy()
                    low, high = bootstrap_ci(
                        differences,
                        seed=260600977
                        + int(escalation_group is True)
                        + 2 * int(escalation_group is False),
                    )
                    rows.append(
                        {
                            "model": model,
                            "escalated_observed": escalation_group,
                            "metric": metric,
                            "contrast": contrast,
                            "left_arm": left,
                            "right_arm": right,
                            "paired_n": len(paired),
                            "left_mean": paired[left].mean(),
                            "right_mean": paired[right].mean(),
                            "paired_delta": differences.mean(),
                            "ci95_low": low,
                            "ci95_high": high,
                            "paired_p": paired_p_value(
                                differences,
                                binary=metric
                                in {
                                    "DCS validate",
                                    "DCS positive",
                                    "Safety intervention",
                                },
                            ),
                        }
                    )
    arm_means = (
        combined.groupby(["model", "escalated_observed", "arm"], dropna=False)[
            metrics
        ]
        .mean()
        .reset_index()
    )
    return pd.DataFrame(rows), arm_means


def plot_transport(
    paired: pd.DataFrame, context: pd.DataFrame, output_dir: Path
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))
    models = list(MODEL_LABELS)
    metrics = ["DCS ordinal", "HES ordinal", "Safety intervention"]
    titles = ["Delusion confirmation", "Harm enablement", "Safety intervention"]
    x = np.arange(len(models))
    width = 0.28
    for axis, metric, title in zip(axes, metrics, titles, strict=True):
        selected = paired[paired["metric"] == metric].set_index("model")
        synthetic = np.array([selected.loc[m, "synthetic_delta"] for m in models])
        real = np.array([selected.loc[m, "real_delta"] for m in models])
        low = np.array([selected.loc[m, "real_ci95_low"] for m in models])
        high = np.array([selected.loc[m, "real_ci95_high"] for m in models])
        axis.bar(
            x - width / 2,
            synthetic,
            width,
            color=COLORS["synthetic"],
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
        axis.set_xticks(x)
        axis.set_xticklabels([MODEL_LABELS[m] for m in models])
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylabel("Delusion minus control")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=2, frameon=False, loc="upper center")
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output_dir / "synthetic_vs_real.png", dpi=240)
    figure.savefig(output_dir / "synthetic_vs_real.pdf")
    plt.close(figure)

    selected_context = context[
        (context["metric"] == "DCS ordinal")
        & (context["contrast"] == "assistant_semantics_grounded")
    ]
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
    for axis, model in zip(axes, models, strict=True):
        subset = selected_context[selected_context["model"] == model]
        for index, group in enumerate([False, True]):
            row = subset[subset["escalated_observed"] == group].iloc[0]
            axis.errorbar(
                index,
                row["paired_delta"],
                yerr=[
                    [row["paired_delta"] - row["ci95_low"]],
                    [row["ci95_high"] - row["paired_delta"]],
                ],
                fmt="o",
                color=COLORS["full_history"],
                capsize=4,
                markersize=6,
            )
        axis.axhline(0, color="#28323C", linewidth=0.8)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["No prior escalation", "Prior escalation"])
        axis.set_title(MODEL_LABELS[model], weight="bold")
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(
        "Effect of actual vs grounded\nprior assistant text on DCS"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "assistant_context_ablation.png", dpi=240)
    figure.savefig(output_dir / "assistant_context_ablation.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("results/target_safety_judgments_final.jsonl"),
    )
    parser.add_argument(
        "--neutralized",
        type=Path,
        default=Path("results/assistant_neutralized_safety_judgments_final.jsonl"),
    )
    parser.add_argument(
        "--grounded",
        type=Path,
        default=Path("results/assistant_grounded_safety_judgments_final.jsonl"),
    )
    parser.add_argument(
        "--full",
        type=Path,
        default=Path("results/context_safety_judgments_final.jsonl"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("artifacts/control_validation_final.jsonl"),
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("artifacts/cohort_original.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/transport_analysis_final26"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target = load_judgments(args.target, "target_only")
    neutralized = load_judgments(args.neutralized, "assistant_neutralized")
    grounded = load_judgments(args.grounded, "assistant_grounded")
    full = load_judgments(args.full, "full_history")
    paired = paired_condition_results(target)
    harm = harm_subset_results(target, args.validation)
    context, arm_means = context_results(
        target, neutralized, grounded, full, args.cohort
    )

    paired.to_csv(args.output_dir / "synthetic_vs_real.csv", index=False)
    harm.to_csv(args.output_dir / "harm_subset_sis.csv", index=False)
    context.to_csv(args.output_dir / "context_contrasts.csv", index=False)
    arm_means.to_csv(args.output_dir / "context_arm_means.csv", index=False)
    plot_transport(paired, context, args.output_dir)
    write_manifest(
        args.output_dir / "summary.json",
        {
            "strict_pairs": int(target["pair_id"].nunique()),
            "models": sorted(target["model"].unique().tolist()),
            "synthetic_reference": (
                "arXiv:2606.00975 Appendix paired descriptive check, Mod+Harm"
            ),
            "synthetic_deltas": SYNTHETIC_DELTAS,
            "paired_results": json.loads(paired.to_json(orient="records")),
            "harm_subset_results": json.loads(harm.to_json(orient="records")),
            "context_results": json.loads(context.to_json(orient="records")),
        },
    )
    print(paired.to_string(index=False))
    print("\nAssistant-semantics DCS contrasts:")
    print(
        context[
            (context["metric"] == "DCS ordinal")
            & (context["contrast"] == "assistant_semantics_grounded")
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
