from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "llama31_8b": "Llama 3.1 8B",
    "qwen3_8b": "Qwen3 8B",
    "gemma3_12b": "Gemma 3 12B",
}
COLORS = {
    "llama31_8b": "#D55E00",
    "qwen3_8b": "#0072B2",
    "gemma3_12b": "#009E73",
}


def load_results(root: Path, result_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    turn_rows = []
    for model_key in MODEL_LABELS:
        run_dir = root / "runs" / model_key
        summaries = json.loads((run_dir / result_dir / "summary.json").read_text())
        for row in summaries:
            flat = {
                "model_key": model_key,
                "model": MODEL_LABELS[model_key],
                "horizon": row["horizon"],
                "midpoint_layer": row["midpoint_layer"],
                "n": row["probe"]["n"],
                "positives": row["probe"]["positives"],
                "prevalence": row["probe"]["prevalence"],
            }
            for name, source in [
                ("probe", "probe"),
                ("schedule", "schedule_baseline"),
                ("history", "history_baseline"),
                ("user_text", "user_text_baseline"),
            ]:
                flat[f"{name}_auroc"] = row[source]["auroc"]
                flat[f"{name}_ap"] = row[source]["average_precision"]
            flat["probe_auc_ci_low"] = row["cluster_bootstrap_95"][
                "probe_prediction"
            ]["lower"]
            flat["probe_auc_ci_high"] = row["cluster_bootstrap_95"][
                "probe_prediction"
            ]["upper"]
            flat["probe_minus_history_ci_low"] = row["cluster_bootstrap_95"][
                "probe_minus_history"
            ]["lower"]
            flat["probe_minus_history_ci_high"] = row["cluster_bootstrap_95"][
                "probe_minus_history"
            ]["upper"]
            metric_rows.append(flat)

        oof = pd.read_json(run_dir / result_dir / "oof_predictions.jsonl", lines=True)
        current = oof[oof["horizon"] == 0]
        for turn, group in current.groupby("turn_number"):
            turn_rows.append(
                {
                    "model_key": model_key,
                    "model": MODEL_LABELS[model_key],
                    "turn_number": int(turn),
                    "endorsement_rate": float(group["target"].mean()),
                    "n": len(group),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(turn_rows)


def make_figure(metrics: pd.DataFrame, turns: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), constrained_layout=True)
    for model_key, label in MODEL_LABELS.items():
        subset = metrics[metrics["model_key"] == model_key].sort_values("horizon")
        color = COLORS[model_key]
        axes[0].plot(
            subset["horizon"], subset["probe_auroc"], marker="o", color=color, label=label
        )
        axes[0].fill_between(
            subset["horizon"],
            subset["probe_auc_ci_low"],
            subset["probe_auc_ci_high"],
            color=color,
            alpha=0.13,
            linewidth=0,
        )
        strongest = subset[["schedule_auroc", "history_auroc", "user_text_auroc"]].max(
            axis=1
        )
        axes[1].plot(
            subset["horizon"],
            subset["probe_auroc"] - strongest,
            marker="o",
            color=color,
        )
        turn_subset = turns[turns["model_key"] == model_key]
        axes[2].plot(
            turn_subset["turn_number"],
            turn_subset["endorsement_rate"],
            marker="o",
            color=color,
        )

    axes[0].axhline(0.5, color="#999999", linestyle="--", linewidth=1)
    axes[0].set(title="Forecasting held-out scenarios", xlabel="Turns ahead", ylabel="AUROC")
    axes[0].set_xticks(range(5))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0, color="#777777", linestyle="--", linewidth=1)
    axes[1].set(
        title="Probe advantage over best baseline",
        xlabel="Turns ahead",
        ylabel="AUROC difference",
    )
    axes[1].set_xticks(range(5))
    axes[2].set(title="Observed endorsement", xlabel="Conversation turn", ylabel="Rate")
    axes[2].set_xticks(range(3, 13, 2))
    for axis in axes:
        axis.grid(axis="y", color="#E6E3DE", linewidth=0.8)
    figure.savefig(output, dpi=240, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-dir", default="probe_results")
    parser.add_argument("--judge-suffix", default="primary_qwen")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, turns = load_results(args.root, args.result_dir)
    metrics.to_csv(args.output_dir / f"metrics_{args.judge_suffix}.csv", index=False)
    turns.to_csv(args.output_dir / f"turn_rates_{args.judge_suffix}.csv", index=False)
    make_figure(
        metrics,
        turns,
        args.output_dir / f"trajectory_forecasting_{args.judge_suffix}.png",
    )
    summary = {
        "judge": args.judge_suffix,
        "models": {},
    }
    for model_key, label in MODEL_LABELS.items():
        subset = metrics[metrics["model_key"] == model_key].set_index("horizon")
        summary["models"][model_key] = {
            "label": label,
            "midpoint_layer": int(subset["midpoint_layer"].iloc[0]),
            "current_turn_probe_auroc": float(subset.loc[0, "probe_auroc"]),
            "one_turn_ahead_probe_auroc": float(subset.loc[1, "probe_auroc"]),
            "four_turn_ahead_probe_auroc": float(subset.loc[4, "probe_auroc"]),
            "current_turn_best_baseline_auroc": float(
                subset.loc[0, ["schedule_auroc", "history_auroc", "user_text_auroc"]].max()
            ),
        }
    (args.output_dir / f"summary_{args.judge_suffix}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
