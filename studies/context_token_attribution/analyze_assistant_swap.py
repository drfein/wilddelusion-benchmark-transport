from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_behavior_interventions import (
    bootstrap,
    directional_sign_flip_p_value,
    sign_flip_p_value,
)
from config import ENDORSEMENT_THRESHOLD
from io_utils import read_jsonl, sha256_file, write_jsonl


def valid(path: Path) -> list[dict]:
    return [
        row
        for row in read_jsonl(path)
        if isinstance(row.get("annotation_score"), int) and not row.get("judge_error")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-judgments", type=Path, required=True)
    parser.add_argument("--swap-judgments", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--deletion-rows", type=Path)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--randomization-iterations", type=int, default=200_000)
    args = parser.parse_args()

    selections = {row["conversation_hash"]: row for row in read_jsonl(args.selections)}
    deletion = (
        {row["conversation_hash"]: row for row in read_jsonl(args.deletion_rows)}
        if args.deletion_rows
        else {}
    )
    original: dict[str, dict[int, tuple[int, int]]] = defaultdict(dict)
    swapped: dict[str, dict[int, tuple[int, int]]] = defaultdict(dict)
    for row in valid(args.original_judgments):
        if row["conversation_hash"] in selections:
            original[row["conversation_hash"]][int(row["repetition"])] = (
                int(row["annotation_score"] >= ENDORSEMENT_THRESHOLD),
                int(row["annotation_score"]),
            )
    for row in valid(args.swap_judgments):
        if (
            row["conversation_hash"] in selections
            and row.get("condition") == "assistant_positions_swapped"
        ):
            swapped[row["conversation_hash"]][int(row["repetition"])] = (
                int(row["annotation_score"] >= ENDORSEMENT_THRESHOLD),
                int(row["annotation_score"]),
            )
    rows = []
    expected = set(range(5))
    for conversation_hash, selection in sorted(selections.items()):
        if not (
            set(original[conversation_hash]) == set(swapped[conversation_hash]) == expected
        ):
            raise ValueError(f"Incomplete paired labels for {conversation_hash}")
        latest_score = int(selection["prior_assistant_score"])
        control_score = int(selection["control_assistant_score"])
        original_rate = float(np.mean([value[0] for value in original[conversation_hash].values()]))
        swapped_rate = float(np.mean([value[0] for value in swapped[conversation_hash].values()]))
        original_score = float(np.mean([value[1] for value in original[conversation_hash].values()]))
        swapped_score = float(np.mean([value[1] for value in swapped[conversation_hash].values()]))
        rows.append(
            {
                **selection,
                "latest_assistant_score": latest_score,
                "matched_earlier_assistant_score": control_score,
                "matched_earlier_endorses": int(control_score >= ENDORSEMENT_THRESHOLD),
                "original_rate": original_rate,
                "swapped_rate": swapped_rate,
                "swap_minus_original": swapped_rate - original_rate,
                "original_mean_score": original_score,
                "swapped_mean_score": swapped_score,
                "swap_minus_original_score": swapped_score - original_score,
                **(
                    {
                        "latest_deleted_rate": deletion[conversation_hash][
                            "top_deleted_rate"
                        ],
                        "latest_deletion_minus_original": deletion[conversation_hash][
                            "top_minus_original"
                        ],
                        "latest_deletion_minus_swap": deletion[conversation_hash][
                            "top_deleted_rate"
                        ]
                        - swapped_rate,
                        "latest_deleted_mean_score": deletion[conversation_hash][
                            "top_deleted_mean_score"
                        ],
                        "latest_deletion_minus_original_score": deletion[
                            conversation_hash
                        ]["top_minus_original_score"],
                        "latest_deletion_minus_swap_score": deletion[
                            conversation_hash
                        ]["top_deleted_mean_score"]
                        - swapped_score,
                    }
                    if conversation_hash in deletion
                    else {}
                ),
            }
        )

    def summarize(subset: list[dict]) -> dict:
        binary = np.asarray([row["swap_minus_original"] for row in subset])
        ordinal = np.asarray([row["swap_minus_original_score"] for row in subset])
        return {
            "n_conversations": len(subset),
            "original_endorsement_rate": float(
                np.mean([row["original_rate"] for row in subset])
            ),
            "swapped_endorsement_rate": float(
                np.mean([row["swapped_rate"] for row in subset])
            ),
            "binary_effect": {
                "mean": float(binary.mean()),
                "bootstrap_95_ci_by_conversation": bootstrap(binary, args.bootstrap),
                "two_sided_cluster_sign_flip_p": sign_flip_p_value(
                    binary, args.randomization_iterations
                ),
                "directional_cluster_sign_flip_p": directional_sign_flip_p_value(
                    binary, args.randomization_iterations
                ),
            },
            "ordinal_effect": {
                "mean": float(ordinal.mean()),
                "bootstrap_95_ci_by_conversation": bootstrap(ordinal, args.bootstrap),
                "two_sided_cluster_sign_flip_p": sign_flip_p_value(
                    ordinal, args.randomization_iterations
                ),
                "directional_cluster_sign_flip_p": directional_sign_flip_p_value(
                    ordinal, args.randomization_iterations
                ),
            },
        }

    control_nonendorsing = [row for row in rows if not row["matched_earlier_endorses"]]
    removal_vs_relocation = {}
    if deletion and all("latest_deletion_minus_swap" in row for row in rows):
        binary = np.asarray(
            [row["latest_deletion_minus_swap"] for row in rows], dtype=np.float64
        )
        ordinal = np.asarray(
            [row["latest_deletion_minus_swap_score"] for row in rows],
            dtype=np.float64,
        )
        removal_vs_relocation = {
            "latest_deleted_endorsement_rate": float(
                np.mean([row["latest_deleted_rate"] for row in rows])
            ),
            "latest_deletion_minus_original": {
                "mean": float(
                    np.mean([row["latest_deletion_minus_original"] for row in rows])
                ),
                "bootstrap_95_ci_by_conversation": bootstrap(
                    np.asarray(
                        [row["latest_deletion_minus_original"] for row in rows]
                    ),
                    args.bootstrap,
                ),
            },
            "latest_deletion_minus_relocation": {
                "mean": float(binary.mean()),
                "bootstrap_95_ci_by_conversation": bootstrap(binary, args.bootstrap),
                "two_sided_cluster_sign_flip_p": sign_flip_p_value(
                    binary, args.randomization_iterations
                ),
            },
            "latest_deletion_minus_relocation_ordinal_score": {
                "mean": float(ordinal.mean()),
                "bootstrap_95_ci_by_conversation": bootstrap(ordinal, args.bootstrap),
                "two_sided_cluster_sign_flip_p": sign_flip_p_value(
                    ordinal, args.randomization_iterations
                ),
            },
        }
    summary = {
        "original_judgments_sha256": sha256_file(args.original_judgments),
        "swap_judgments_sha256": sha256_file(args.swap_judgments),
        "selections_sha256": sha256_file(args.selections),
        "estimand": (
            "Effect of moving the latest endorsing assistant content to an earlier "
            "assistant position while moving matched earlier content to the latest "
            "position, preserving all tokens and dialogue structure."
        ),
        "all_pairs": summarize(rows),
        "matched_earlier_nonendorsing_pairs": summarize(control_nonendorsing),
        "matched_earlier_endorsing_conversations": len(rows)
        - len(control_nonendorsing),
        "removal_vs_relocation": removal_vs_relocation,
    }
    write_jsonl(args.rows_output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
