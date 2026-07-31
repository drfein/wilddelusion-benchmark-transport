#!/usr/bin/env python3
"""Create a compact overview of the merged, theme-labeled WildDelusion dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator

SHARECHAT_MODEL_LABELS = {
    "sharechat_chatgpt": "ChatGPT · ShareChat",
    "sharechat_grok": "Grok · ShareChat",
    "sharechat_gemini": "Gemini · ShareChat",
    "sharechat_claude": "Claude · ShareChat",
}

TYPE_LABELS = {
    "grandiose": "Grandiose",
    "persecutory": "Persecutory",
    "bizarre": "Bizarre",
    "religious_or_spiritual": "Religious / spiritual",
    "mixed": "Mixed",
    "thought_broadcasting": "Thought broadcast",
    "referential": "Referential",
    "unclear_or_no_delusional_theme": "Unclear / no theme",
    "somatic": "Somatic",
    "thought_insertion": "Thought insertion",
    "nihilistic": "Nihilistic",
    "delusional_jealousy": "Jealousy",
    "erotomanic": "Erotomanic",
}


def add_value_labels(ax: plt.Axes, bars: object, *, offset: float = 4) -> None:
    for bar in bars:
        value = round(bar.get_width())
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            ha="left",
            fontsize=6.8,
            color="#25323A",
        )


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, colors="#25323A")
    ax.tick_params(axis="x", colors="#667079")
    ax.grid(axis="x", color="#D9D5CD", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def model_family(model: str) -> str:
    value = str(model or "").lower()
    if value == "wildchat_original_openai_assistant":
        return "Original OpenAI"
    if value.startswith("gpt-3.5"):
        return "GPT-3.5 Turbo"
    if value.startswith("gpt-4o-mini"):
        return "GPT-4o mini"
    if value.startswith("gpt-4o"):
        return "GPT-4o"
    if value.startswith("gpt-4.1-mini"):
        return "GPT-4.1 mini"
    if value.startswith("gpt-4-turbo") or "gpt-4-1106" in value or "gpt-4-0125" in value:
        return "GPT-4 Turbo"
    if value.startswith("gpt-4"):
        return "GPT-4"
    if value.startswith("o1"):
        return "o1 preview"
    return str(model or "Unknown")


def draw_example(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Example", loc="left")
    ax.text(
        0.0,
        0.87,
        "GRANDIOSE  ·  o1 preview",
        fontsize=5.9,
        color="#B7463B",
        fontweight=600,
        va="center",
    )
    bubbles = [
        (
            0.00,
            0.46,
            0.94,
            0.26,
            "#F4D9D2",
            "USER",
            (
                "i died and came back to life. ...\n"
                "then i started gaining control over the sun."
            ),
        ),
        (
            0.08,
            0.14,
            0.92,
            0.22,
            "#D8E8E5",
            "ASSISTANT",
            (
                "I'm sorry to hear that you're experiencing this. ...\n"
                "Please consider reaching out ... for support."
            ),
        ),
    ]
    for x, y, width, height, color, role, content in bubbles:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.014,rounding_size=0.022",
            linewidth=0,
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(
            x,
            y + height + 0.018,
            role,
            fontsize=4.8,
            fontweight=600,
            color="#647078",
            va="bottom",
        )
        ax.text(
            x + 0.03,
            y + height - 0.055,
            content,
            fontsize=4.7,
            color="#25323A",
            va="top",
            linespacing=1.1,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/merged_delusion_taxonomy_v2/"
            "wilddelusion_combined_with_themes.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/iclr2026/figures/wilddelusion_merged_dataset_overview.png"),
    )
    parser.add_argument(
        "--wildchat-models",
        type=Path,
        default=Path("results/merged_delusion_taxonomy_v2/wildchat_models.json"),
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 522:
        raise ValueError(f"Expected 522 merged target turns, found {len(rows)}")

    conversation_rows = {}
    for row in rows:
        key = (row["source"], row["split"], str(row["conversation_id"]))
        conversation_rows.setdefault(key, row)

    lengths = np.asarray(
        [len(row.get("messages") or []) for row in conversation_rows.values()],
        dtype=int,
    )
    type_counts = Counter(row["delusion_type"] for row in rows)

    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial"],
            "font.size": 7.4,
            "axes.titlesize": 9.2,
            "axes.titleweight": 600,
            "axes.labelsize": 7.3,
            "text.color": "#25323A",
            "axes.labelcolor": "#59636B",
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 7.0,
        }
    )

    figure = plt.figure(figsize=(7.2, 2.65), facecolor="#FFFFFF")
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[1.15, 1.0],
        wspace=0.48,
    )
    type_ax = figure.add_subplot(grid[0, 0])
    length_ax = figure.add_subplot(grid[0, 1])

    bins = np.arange(0, 155, 5)
    clipped = np.minimum(lengths, 154)
    length_ax.hist(
        clipped,
        bins=bins,
        color="#2F7F78",
        edgecolor="#FFFFFF",
        linewidth=1.2,
    )
    median = float(np.median(lengths))
    length_ax.axvline(median, color="#D25B4D", linewidth=2)
    length_ax.text(
        median + 1.5,
        length_ax.get_ylim()[1] * 0.9,
        f"median {median:.0f}",
        color="#B7463B",
        fontsize=7,
        va="top",
    )
    length_ax.set_title("Conversation length", loc="left")
    length_ax.set_xlabel("Messages in source conversation")
    length_ax.set_ylabel("Conversations")
    length_ax.set_xticks([0, 25, 50, 75, 100, 125, 150])
    length_ax.set_xticklabels(["0", "25", "50", "75", "100", "125", "150+"])
    length_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    style_axis(length_ax)

    ordered_types = sorted(
        ((TYPE_LABELS.get(key, key.replace("_", " ").title()), value) for key, value in type_counts.items()),
        key=lambda item: item[1],
    )
    type_labels = [label for label, _ in ordered_types]
    type_values = [value for _, value in ordered_types]
    type_y = np.arange(len(type_labels))
    type_bars = type_ax.barh(
        type_y,
        type_values,
        color="#4B70A6",
        height=0.65,
    )
    type_ax.set_yticks(type_y, type_labels)
    type_ax.set_xlim(0, max(type_values) * 1.2)
    type_ax.set_title("Delusional-content type", loc="left")
    type_ax.set_xlabel("Contextualized target turns")
    add_value_labels(type_ax, type_bars, offset=2.5)
    style_axis(type_ax)

    figure.subplots_adjust(left=0.20, right=0.97, bottom=0.20, top=0.83)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=260, facecolor=figure.get_facecolor())
    figure.savefig(args.output.with_suffix(".pdf"), facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
