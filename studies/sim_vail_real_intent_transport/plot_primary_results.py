#!/usr/bin/env python3
"""Plot paired primary effects with 99% source-conversation intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INTENTS = ("dependence", "glorification", "risky_action")
LABELS = {
    "dependence": "Dependence request → boundary risk",
    "glorification": "Special-insight request → romanticization",
    "risky_action": "Action request → risky-action enablement",
}
COLORS = {
    "Qwen/Qwen3-4B-AWQ": "#2A9D8F",
    "Qwen/Qwen3-14B-AWQ": "#E76F51",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    lookup = {(row["model"], row["intent"]): row for row in rows}
    models = sorted({row["model"] for row in rows}, key=lambda value: "14B" in value)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#2F3437",
            "axes.labelcolor": "#2F3437",
            "xtick.color": "#4A4F52",
            "ytick.color": "#2F3437",
        }
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.8), facecolor="white")
    ax.set_facecolor("white")
    base_y = np.arange(len(INTENTS))[::-1]
    offsets = (-0.11, 0.11)
    for offset, model in zip(offsets, models, strict=True):
        values = np.array([lookup[(model, intent)]["target_minus_control_mean"] for intent in INTENTS])
        lows = np.array([lookup[(model, intent)]["ci99_low"] for intent in INTENTS])
        highs = np.array([lookup[(model, intent)]["ci99_high"] for intent in INTENTS])
        label = "Qwen3-14B" if "14B" in model else "Qwen3-4B"
        ax.errorbar(
            values,
            base_y + offset,
            xerr=np.vstack([values - lows, highs - values]),
            fmt="o",
            markersize=7,
            capsize=4,
            elinewidth=2,
            color=COLORS[model],
            label=label,
            zorder=3,
        )
        for value, y in zip(values, base_y + offset, strict=True):
            ax.text(value + 0.035, y, f"{value:+.2f}", va="center", fontsize=9, color=COLORS[model])

    ax.axvline(0, color="#5E666B", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_yticks(base_y, [LABELS[intent] for intent in INTENTS])
    ax.set_xlabel("Target-intent minus matched-control score (1–10 scale)")
    ax.set_title("Causal transport of user intent into response risk", loc="left", fontsize=15, pad=22)
    ax.text(
        0,
        1.025,
        "80 paired real conversations per model and intent · 99% bootstrap intervals",
        transform=ax.transAxes,
        fontsize=10,
        color="#62696D",
    )
    ax.grid(axis="x", color="#E6E8E9", linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        columnspacing=2.2,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(args.output)


if __name__ == "__main__":
    main()
