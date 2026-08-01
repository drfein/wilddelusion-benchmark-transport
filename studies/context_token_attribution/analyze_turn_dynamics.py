from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from config import SEED
from io_utils import read_jsonl
from sklearn.metrics import average_precision_score, roc_auc_score

OFFSETS = tuple(range(-5, 1))
GROUPS = ("prior_assistant_endorsing", "other")


def bootstrap_mean(values: np.ndarray, iterations: int, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    for index in range(iterations):
        estimates[index] = values[
            rng.integers(0, len(values), len(values))
        ].mean()
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def bootstrap_difference(
    first: np.ndarray, second: np.ndarray, iterations: int
) -> list[float]:
    rng = np.random.default_rng(SEED)
    estimates = np.empty(iterations)
    for index in range(iterations):
        first_sample = first[rng.integers(0, len(first), len(first))]
        second_sample = second[rng.integers(0, len(second), len(second))]
        estimates[index] = first_sample.mean() - second_sample.mean()
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def bootstrap_auc(y: np.ndarray, score: np.ndarray, iterations: int) -> list[float]:
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(iterations):
        sample = rng.integers(0, len(y), len(y))
        if len(np.unique(y[sample])) == 2:
            estimates.append(roc_auc_score(y[sample], score[sample]))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def group_summary(matrix: np.ndarray, iterations: int) -> dict:
    means = matrix.mean(axis=0)
    intervals = np.asarray(
        [bootstrap_mean(matrix[:, index], iterations) for index in range(matrix.shape[1])]
    )
    pre_ramp = matrix[:, -2] - matrix[:, 0]
    event_step = matrix[:, -1] - matrix[:, -2]
    slopes = np.asarray(
        [np.polyfit(np.arange(matrix.shape[1]), row, 1)[0] for row in matrix]
    )
    return {
        "conversations": len(matrix),
        "mean_by_offset": {str(offset): float(value) for offset, value in zip(OFFSETS, means)},
        "bootstrap_95_ci_by_offset": {
            str(offset): [float(value) for value in interval]
            for offset, interval in zip(OFFSETS, intervals)
        },
        "pre_latest_assistant_ramp_minus5_to_minus1": {
            "mean": float(pre_ramp.mean()),
            "bootstrap_95_ci": bootstrap_mean(pre_ramp, iterations),
            "fraction_positive": float((pre_ramp > 0).mean()),
        },
        "latest_assistant_plus_target_step_minus1_to_0": {
            "mean": float(event_step.mean()),
            "bootstrap_95_ci": bootstrap_mean(event_step, iterations),
            "fraction_positive": float((event_step > 0).mean()),
        },
        "six_state_linear_slope": {
            "mean": float(slopes.mean()),
            "bootstrap_95_ci": bootstrap_mean(slopes, iterations),
            "fraction_positive": float((slopes > 0).mean()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    rows = read_jsonl(args.rows)
    by_conversation: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_conversation[row["conversation_hash"]].append(row)
    trajectories = {
        key: {row["user_turns_before_target"]: row for row in value}
        for key, value in by_conversation.items()
    }
    fixed = {
        group: [
            key
            for key, value in trajectories.items()
            if value[0]["latest_prior_assistant_group"] == group
            and all(offset in value for offset in OFFSETS)
        ]
        for group in GROUPS
    }
    matrices = {
        group: np.asarray(
            [
                [trajectories[key][offset]["probe_probability"] for offset in OFFSETS]
                for key in keys
            ]
        )
        for group, keys in fixed.items()
    }
    summary = {
        "design": (
            "Fixed-composition event study over the final six exact assistant-decision "
            "states. Offset -1 is immediately before the source assistant response "
            "whose endorsement label defines the group; offset 0 is after that response "
            "and the final target user message."
        ),
        "offsets": list(OFFSETS),
        "groups": {
            group: group_summary(matrix, args.bootstrap)
            for group, matrix in matrices.items()
        },
    }
    prior_ramp = matrices["prior_assistant_endorsing"][:, -2] - matrices[
        "prior_assistant_endorsing"
    ][:, 0]
    other_ramp = matrices["other"][:, -2] - matrices["other"][:, 0]
    prior_event = matrices["prior_assistant_endorsing"][:, -1] - matrices[
        "prior_assistant_endorsing"
    ][:, -2]
    other_event = matrices["other"][:, -1] - matrices["other"][:, -2]
    summary["between_group_differences"] = {
        "pre_latest_assistant_ramp": {
            "mean": float(prior_ramp.mean() - other_ramp.mean()),
            "bootstrap_95_ci": bootstrap_difference(
                prior_ramp, other_ramp, args.bootstrap
            ),
        },
        "latest_assistant_plus_target_step": {
            "mean": float(prior_event.mean() - other_event.mean()),
            "bootstrap_95_ci": bootstrap_difference(
                prior_event, other_event, args.bootstrap
            ),
        },
    }

    previous_states = [
        value[-1]
        for value in trajectories.values()
        if -1 in value
    ]
    for name, selected in (
        ("all_sources", previous_states),
        (
            "sharechat_chatgpt",
            [row for row in previous_states if row["source"] == "sharechat_chatgpt"],
        ),
    ):
        y = np.asarray(
            [row["latest_prior_assistant_group"] == "prior_assistant_endorsing" for row in selected]
        )
        scores = np.asarray([row["probe_probability"] for row in selected])
        summary.setdefault("cross_model_prediction", {})[name] = {
            "conversations": len(y),
            "endorsing_source_assistant_responses": int(y.sum()),
            "auroc": float(roc_auc_score(y, scores)),
            "auroc_bootstrap_95_ci": bootstrap_auc(y, scores, args.bootstrap),
            "average_precision": float(average_precision_score(y, scores)),
            "interpretation": (
                "Held-out Qwen susceptibility immediately before the source assistant "
                "response predicts whether that source assistant endorses."
            ),
        }

    # Source-restricted sensitivity keeps the same six-state requirement.
    source_matrices = {}
    for group in GROUPS:
        keys = [
            key
            for key in fixed[group]
            if trajectories[key][0]["source"] == "sharechat_chatgpt"
        ]
        source_matrices[group] = np.asarray(
            [
                [trajectories[key][offset]["probe_probability"] for offset in OFFSETS]
                for key in keys
            ]
        )
    source_prior_ramp = source_matrices["prior_assistant_endorsing"][:, -2] - source_matrices[
        "prior_assistant_endorsing"
    ][:, 0]
    source_other_ramp = source_matrices["other"][:, -2] - source_matrices["other"][:, 0]
    summary["sharechat_chatgpt_sensitivity"] = {
        group: group_summary(matrix, args.bootstrap)
        for group, matrix in source_matrices.items()
    }
    summary["sharechat_chatgpt_sensitivity"]["pre_ramp_difference"] = {
        "mean": float(source_prior_ramp.mean() - source_other_ramp.mean()),
        "bootstrap_95_ci": bootstrap_difference(
            source_prior_ramp, source_other_ramp, args.bootstrap
        ),
    }
    summary["scope_warning"] = (
        "The pre-event ramp is predictive and temporally prior but not a randomized "
        "effect. The step from -1 to 0 combines the source assistant response and the "
        "next user message. Separate assistant deletion/relocation experiments establish "
        "that the latest endorsing assistant content has a causal persistent effect."
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {
        "prior_assistant_endorsing": "#D85A4A",
        "other": "#287D98",
    }
    labels = {
        "prior_assistant_endorsing": "Next source-assistant response endorses",
        "other": "Next source-assistant response does not endorse",
    }
    fig = plt.figure(figsize=(11.2, 4.2), constrained_layout=True, facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.55, 1])
    axis = fig.add_subplot(grid[0, 0])
    x = np.arange(len(OFFSETS))
    for group in GROUPS:
        matrix = matrices[group]
        means = matrix.mean(axis=0)
        intervals = np.asarray(
            [
                bootstrap_mean(matrix[:, index], args.bootstrap)
                for index in range(matrix.shape[1])
            ]
        )
        axis.plot(
            x,
            means,
            color=colors[group],
            marker="o",
            linewidth=2.5,
            markersize=5,
            label=f"{labels[group]} (n={len(matrix)})",
        )
        axis.fill_between(
            x, intervals[:, 0], intervals[:, 1], color=colors[group], alpha=0.16
        )
    axis.axvspan(4.5, 5.0, color="#7A6C5D", alpha=0.10)
    axis.annotate(
        "latest assistant response\n+ target user turn",
        xy=(4.75, 0.02),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#5F574F",
    )
    axis.set_title("A  Susceptibility rises before endorsement", loc="left")
    axis.set_ylabel("Held-out Qwen endorsement projection")
    axis.set_xlabel("Assistant-decision states relative to the flagged target")
    axis.set_xticks(x, [str(offset) for offset in OFFSETS])
    axis.set_ylim(0, 0.68)
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.grid(axis="y", alpha=0.18)
    axis.legend(frameon=False, loc="upper left", fontsize=8.7)

    bar_axis = fig.add_subplot(grid[0, 1])
    measures = ("Pre-endorsement ramp\n−5 to −1", "Final transition\n−1 to 0")
    positions = np.arange(2)
    width = 0.34
    for index, group in enumerate(GROUPS):
        values = np.asarray(
            [
                (matrices[group][:, -2] - matrices[group][:, 0]).mean(),
                (matrices[group][:, -1] - matrices[group][:, -2]).mean(),
            ]
        )
        intervals = np.asarray(
            [
                bootstrap_mean(
                    matrices[group][:, -2] - matrices[group][:, 0], args.bootstrap
                ),
                bootstrap_mean(
                    matrices[group][:, -1] - matrices[group][:, -2], args.bootstrap
                ),
            ]
        )
        bar_axis.bar(
            positions + (index - 0.5) * width,
            values,
            width,
            color=colors[group],
            alpha=0.92,
            label=labels[group],
        )
        bar_axis.errorbar(
            positions + (index - 0.5) * width,
            values,
            yerr=np.vstack((values - intervals[:, 0], intervals[:, 1] - values)),
            fmt="none",
            ecolor="#3D3935",
            capsize=3,
            linewidth=1.2,
        )
        for xpos, value in zip(positions + (index - 0.5) * width, values):
            bar_axis.text(
                xpos,
                value + 0.018,
                f"{value:+.0%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    bar_axis.axhline(0, color="#777777", linewidth=0.8)
    bar_axis.set_title("B  The separation precedes the event", loc="left")
    bar_axis.set_ylabel("Change in probe projection")
    bar_axis.set_xticks(positions, measures)
    bar_axis.set_ylim(-0.05, 0.38)
    bar_axis.yaxis.set_major_formatter(lambda value, _: f"{value:+.0%}")
    bar_axis.grid(axis="y", alpha=0.18)
    fig.suptitle(
        "A cross-model risk state builds over multiple conversational turns",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(args.figure, dpi=240, facecolor="white")
    fig.savefig(args.figure.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
