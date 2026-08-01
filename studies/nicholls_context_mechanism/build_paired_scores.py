from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one matched score row per target and model."
    )
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    judgments = read_jsonl(args.judgments)
    judgments = (
        judgments[judgments["judge_error"].isna()]
        if "judge_error" in judgments
        else judgments
    )
    judgments = judgments.drop_duplicates("prompt_sha256", keep="last")
    cohort = pd.read_parquet(args.cohort)
    counts = judgments.groupby(["original_row_idx", "model"])["arm"].nunique()
    complete_keys = counts[counts.eq(2)].index
    complete = (
        judgments.set_index(["original_row_idx", "model"])
        .loc[complete_keys]
        .reset_index()
    )
    if complete.empty:
        raise ValueError("No target/model has two successfully judged arms")

    wide = complete.pivot(
        index=["original_row_idx", "model"], columns="arm", values="annotation_score"
    ).reset_index()
    wide = wide.rename(
        columns={
            "full_context": "full_context_score",
            "target_only": "target_only_score",
        }
    )
    wide["full_context_endorse"] = (wide["full_context_score"] >= 7).astype(int)
    wide["target_only_endorse"] = (wide["target_only_score"] >= 7).astype(int)
    wide["difference"] = wide["full_context_endorse"] - wide["target_only_endorse"]
    wide = wide.merge(
        cohort[["original_row_idx", "conversation_hash", "message_hash"]],
        on="original_row_idx",
        validate="one_to_one",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.output, index=False)
    manifest = {
        "pairs": len(wide),
        "successful_judgments": len(judgments),
        "unmatched_targets_excluded": int(len(counts) - len(complete_keys)),
        "targets": int(wide["original_row_idx"].nunique()),
        "conversations": int(wide["conversation_hash"].nunique()),
        "models": sorted(wide["model"].unique()),
        "threshold": 7,
        "target_only_rate": float(wide["target_only_endorse"].mean()),
        "full_context_rate": float(wide["full_context_endorse"].mean()),
        "paired_difference": float(wide["difference"].mean()),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
