from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from analyze import build_features, canonical_labels

ROLE_FEATURES = [
    "log_prefix_words",
    "user_delusion_density",
    "assistant_delusion_density",
    "rapport_density",
    "self_disclosure_density",
]


def add_transition_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["score_difference"] = frame["full_context_score"] - frame["target_only_score"]
    frame["transition"] = np.select(
        [
            frame["target_only_endorse"].eq(0) & frame["full_context_endorse"].eq(1),
            frame["target_only_endorse"].eq(1) & frame["full_context_endorse"].eq(0),
            frame["target_only_endorse"].eq(0) & frame["full_context_endorse"].eq(0),
        ],
        ["increase_0_to_1", "decrease_1_to_0", "stable_0_to_0"],
        default="stable_1_to_1",
    )
    frame["score_direction"] = np.select(
        [frame["score_difference"].gt(0), frame["score_difference"].lt(0)],
        ["increase", "decrease"],
        default="same",
    )
    return frame


def role_specific_regression(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    data = frame.copy()
    z_features = []
    for feature in ROLE_FEATURES:
        name = f"z_{feature}"
        data[name] = (data[feature] - data[feature].mean()) / data[feature].std(ddof=0)
        z_features.append(name)
    fitted = smf.ols(f"{outcome} ~ {' + '.join(z_features)}", data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["conversation_hash"], "use_correction": True},
        use_t=True,
    )
    intervals = fitted.conf_int()
    return pd.DataFrame(
        [
            {
                "outcome": outcome,
                "feature": feature,
                "coefficient": float(fitted.params[z_feature]),
                "ci_low": float(intervals.loc[z_feature, 0]),
                "ci_high": float(intervals.loc[z_feature, 1]),
                "p_value": float(fitted.pvalues[z_feature]),
                "unit": "outcome units per 1 SD; exploratory",
            }
            for feature, z_feature in zip(ROLE_FEATURES, z_features, strict=True)
        ]
    )


def last_message_features(
    chunks_path: Path, membership_path: Path, labels_path: Path
) -> pd.DataFrame:
    chunks = pd.read_parquet(chunks_path).merge(
        canonical_labels(labels_path), on="item_id", validate="one_to_one"
    )
    for feature in ("delusion_content", "rapport", "self_disclosure"):
        chunks[f"{feature}_weighted"] = chunks[feature] * chunks["word_count"]
    messages = chunks.groupby(
        ["message_key", "message_index", "role"], as_index=False
    ).agg(
        words=("word_count", "sum"),
        delusion_weighted=("delusion_content_weighted", "sum"),
        rapport_weighted=("rapport_weighted", "sum"),
        self_disclosure_weighted=("self_disclosure_weighted", "sum"),
    )
    for feature in ("delusion", "rapport", "self_disclosure"):
        messages[f"{feature}_density"] = messages[f"{feature}_weighted"] / (
            4 * messages["words"]
        )
    membership = pd.read_parquet(membership_path).merge(
        messages,
        on=["message_key", "message_index"],
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for original_row_idx, group in membership.groupby("original_row_idx"):
        record: dict[str, Any] = {"original_row_idx": int(original_row_idx)}
        for role in ("user", "assistant"):
            last = group[group["role"].eq(role)].nlargest(1, "message_index")
            for feature in (
                "delusion_density",
                "rapport_density",
                "self_disclosure_density",
            ):
                record[f"last_{role}_{feature}"] = (
                    float(last[feature].iloc[0]) if len(last) else np.nan
                )
        rows.append(record)
    return pd.DataFrame(rows)


def transition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    features = [
        "prefix_words",
        "prefix_messages",
        "delusion_density",
        "user_delusion_density",
        "assistant_delusion_density",
        "rapport_density",
        "self_disclosure_density",
        "last_user_delusion_density",
        "last_assistant_delusion_density",
        "score_difference",
    ]
    rows = []
    for transition, group in frame.groupby("transition"):
        for feature in features:
            rows.append(
                {
                    "transition": transition,
                    "feature": feature,
                    "count": int(group[feature].count()),
                    "mean": float(group[feature].mean()),
                    "median": float(group[feature].median()),
                }
            )
    return pd.DataFrame(rows)


def last_assistant_bins(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame[frame["target_only_endorse"].eq(0)].copy()
    labels = ["0", ">0-.25", ">.25-.50", ">.50-.75", ">.75-1"]
    eligible["last_assistant_delusion_bin"] = pd.cut(
        eligible["last_assistant_delusion_density"],
        [-0.001, 0, 0.25, 0.5, 0.75, 1],
        labels=labels,
        include_lowest=True,
    )
    grouped = eligible.groupby(
        "last_assistant_delusion_bin", observed=True, as_index=False
    )["full_context_endorse"].agg(["sum", "count", "mean"])
    return grouped.rename(
        columns={
            "sum": "full_context_endorsements",
            "count": "targets",
            "mean": "full_context_endorsement_rate",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exploratory analysis of contexts increasing or decreasing endorsement."
    )
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--paired-scores", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    features = build_features(args.release, args.chunks, args.membership, args.labels)
    pairs = pd.read_csv(args.paired_scores).drop(
        columns="conversation_hash", errors="ignore"
    )
    frame = pairs.merge(features, on="original_row_idx", validate="one_to_one")
    frame["assistant_words"] = frame["prefix_words"] - frame["user_words"]
    frame["assistant_delusion_density"] = (
        (frame["delusion_weighted"] - frame["user_delusion_weighted"])
        / (4 * frame["assistant_words"].replace(0, np.nan))
    ).fillna(0)
    frame = add_transition_labels(frame).merge(
        last_message_features(args.chunks, args.membership, args.labels),
        on="original_row_idx",
        validate="one_to_one",
    )

    regressions = pd.concat(
        [
            role_specific_regression(frame, "difference"),
            role_specific_regression(frame, "score_difference"),
        ],
        ignore_index=True,
    )
    transitions = transition_summary(frame)
    bins = last_assistant_bins(frame)
    transition_counts = (
        frame.groupby("transition", as_index=False)
        .size()
        .rename(columns={"size": "targets"})
    )
    negative_crossings = frame.loc[
        frame["transition"].eq("decrease_1_to_0"),
        [
            "original_row_idx",
            "conversation_hash",
            "message_hash",
            "target_only_score",
            "full_context_score",
        ],
    ]
    assistant_row = regressions[
        regressions["outcome"].eq("difference")
        & regressions["feature"].eq("assistant_delusion_density")
    ].iloc[0]
    summary = {
        "status": "post-result exploratory analysis",
        "targets": len(frame),
        "binary_transitions": dict(
            zip(
                transition_counts["transition"],
                transition_counts["targets"],
                strict=True,
            )
        ),
        "score_directions": frame["score_direction"].value_counts().to_dict(),
        "assistant_delusion_density_adjusted_coefficient": float(
            assistant_row["coefficient"]
        ),
        "assistant_delusion_density_adjusted_ci": [
            float(assistant_row["ci_low"]),
            float(assistant_row["ci_high"]),
        ],
        "assistant_delusion_density_adjusted_p": float(assistant_row["p_value"]),
        "interpretation_boundary": (
            "The full-history intervention is causal for fixed prompts. Role-specific "
            "feature associations are observational and were examined after outcomes."
        ),
        "manual_audit": (
            "All five negative threshold crossings were inspected privately; none "
            "contained clear reality testing or pushback."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    regressions.to_csv(args.out_dir / "role_specific_regressions.csv", index=False)
    transitions.to_csv(args.out_dir / "transition_feature_summary.csv", index=False)
    bins.to_csv(args.out_dir / "last_assistant_delusion_bins.csv", index=False)
    transition_counts.to_csv(args.out_dir / "transition_counts.csv", index=False)
    negative_crossings.to_csv(
        args.out_dir / "negative_crossings_no_text.csv", index=False
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
