from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from config import ENDORSEMENT_THRESHOLD, SEED
from io_utils import read_jsonl, write_jsonl

STAGES = ("target_only", "last_exchange", "full")
STAGE_LABELS = ("Target turn only", "+ latest exchange", "+ earlier history")


def bootstrap_mean(values: np.ndarray, iterations: int) -> list[float]:
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(values), size=(iterations, len(values)))
    estimates = values[samples].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def load_rates(path: Path, eligible: set[str]) -> dict[str, float]:
    labels: dict[str, list[int]] = defaultdict(list)
    for row in read_jsonl(path):
        conversation_hash = row.get("conversation_hash")
        if (
            conversation_hash in eligible
            and isinstance(row.get("annotation_score"), int)
            and not row.get("judge_error")
        ):
            labels[conversation_hash].append(
                int(row["annotation_score"] >= ENDORSEMENT_THRESHOLD)
            )
    return {key: float(np.mean(value)) for key, value in labels.items()}


def summarize(rows: list[dict], key: str, bootstrap: int) -> dict:
    result = {}
    for group in ("all", "prior_assistant_endorsing", "other"):
        selected = rows if group == "all" else [row for row in rows if row["group"] == group]
        values = np.asarray([row[key] for row in selected], dtype=np.float64)
        result[group] = {
            "conversations": len(selected),
            "mean": float(values.mean()),
            "bootstrap_95_ci": bootstrap_mean(values, bootstrap),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--full-judgments", type=Path, required=True)
    parser.add_argument("--last-judgments", type=Path, required=True)
    parser.add_argument("--target-judgments", type=Path, required=True)
    parser.add_argument("--prior-assistant-judgments", type=Path, required=True)
    parser.add_argument("--full-activations", type=Path, required=True)
    parser.add_argument("--last-activations", type=Path, required=True)
    parser.add_argument("--target-activations", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    cohort = {row["conversation_hash"]: row for row in read_jsonl(args.cohort)}
    eligible = set(cohort)
    prior_scores = {
        row["conversation_hash"]: int(row["annotation_score"])
        for row in read_jsonl(args.prior_assistant_judgments)
        if row.get("conversation_hash") in eligible
        and isinstance(row.get("annotation_score"), int)
        and not row.get("judge_error")
    }
    rates = {
        "full": load_rates(args.full_judgments, eligible),
        "last_exchange": load_rates(args.last_judgments, eligible),
        "target_only": load_rates(args.target_judgments, eligible),
    }
    activation_dirs = {
        "full": args.full_activations,
        "last_exchange": args.last_activations,
        "target_only": args.target_activations,
    }

    rows = []
    for conversation_hash, cohort_row in sorted(cohort.items()):
        checkpoint = np.load(
            args.checkpoints / f"outer_fold_{int(cohort_row['fold'])}.npz"
        )
        layer = str(checkpoint["layer"].item())
        direction = checkpoint["raw_direction"].astype(np.float64)
        intercept = float(checkpoint["raw_intercept"].item())
        prior_score = prior_scores[conversation_hash]
        group = (
            "prior_assistant_endorsing"
            if prior_score >= ENDORSEMENT_THRESHOLD
            else "other"
        )
        row = {
            "conversation_hash": conversation_hash,
            "fold": int(cohort_row["fold"]),
            "selected_layer": layer,
            "prior_assistant_score": prior_score,
            "group": group,
        }
        for stage in STAGES:
            path = activation_dirs[stage] / f"{conversation_hash}.npz"
            activation = np.load(path)[layer].astype(np.float64)
            logit = float(activation @ direction + intercept)
            row[f"{stage}_probe_logit"] = logit
            row[f"{stage}_probe_probability"] = float(expit(logit))
            row[f"{stage}_endorsement_rate"] = rates[stage][conversation_hash]
        rows.append(row)

    write_jsonl(args.rows, rows)
    summary = {
        "design": (
            "Paired context-depth trajectory through the same outer-fold-held-out "
            "dense endorsement probe, with independently generated behavioral rates"
        ),
        "stage_order": list(STAGES),
        "stage_labels": list(STAGE_LABELS),
        "probe_warning": (
            "Projected probabilities at ablated prompts are probe scores under "
            "distribution shift, not calibrated causal probabilities. Behavioral "
            "endorsement rates are the causal regeneration endpoint."
        ),
        "stages": {},
    }
    for stage in STAGES:
        summary["stages"][stage] = {
            "probe_probability": summarize(
                rows, f"{stage}_probe_probability", args.bootstrap
            ),
            "endorsement_rate": summarize(
                rows, f"{stage}_endorsement_rate", args.bootstrap
            ),
        }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    groups = (
        ("all", "All conversations", "#34495e", "o"),
        ("prior_assistant_endorsing", "Prior assistant endorsed", "#d3544a", "o"),
        ("other", "Prior assistant did not endorse", "#2a7f9e", "s"),
    )
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.65), constrained_layout=True)
    for axis, metric, title, ylabel in (
        (
            axes[0],
            "probe_probability",
            "Latent endorsement susceptibility",
            "Held-out probe projection",
        ),
        (
            axes[1],
            "endorsement_rate",
            "Generated endorsement",
            "Endorsing responses",
        ),
    ):
        for group, label, color, marker in groups:
            means = np.asarray(
                [summary["stages"][stage][metric][group]["mean"] for stage in STAGES]
            )
            intervals = np.asarray(
                [
                    summary["stages"][stage][metric][group]["bootstrap_95_ci"]
                    for stage in STAGES
                ]
            )
            axis.errorbar(
                x,
                means,
                yerr=np.vstack((means - intervals[:, 0], intervals[:, 1] - means)),
                color=color,
                marker=marker,
                linewidth=2.1,
                markersize=5.5,
                capsize=3,
                label=label,
            )
        axis.set_title(title, loc="left", weight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, STAGE_LABELS)
        axis.set_ylim(-0.02, 1.02)
        axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.suptitle(
        "Context accumulates a persistent endorsement state",
        x=0.02,
        ha="left",
        weight="bold",
    )
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=220, facecolor="white")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
