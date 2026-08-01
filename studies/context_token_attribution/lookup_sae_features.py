from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from io_utils import read_jsonl, sha256_file


def json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-features", type=Path, required=True)
    parser.add_argument("--feature-metrics", type=Path, required=True)
    parser.add_argument("--feature-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    selected = read_jsonl(args.selected_features)[: args.limit]
    metrics = pd.read_parquet(args.feature_metrics).set_index("feature_idx")
    prompts = pd.read_parquet(args.feature_prompts)
    output = []
    for feature in selected:
        index = feature["feature_index"]
        metric = (
            {key: json_value(value) for key, value in metrics.loc[index].items()}
            if index in metrics.index
            else None
        )
        examples = prompts[prompts["feature_idx"] == index].sort_values(
            "activation", ascending=False
        )
        output.append(
            {
                **feature,
                "reverse_index_metrics": metric,
                "reverse_index_examples": [
                    {key: json_value(value) for key, value in row.items()}
                    for row in examples.to_dict("records")
                ],
            }
        )
    payload = {
        "selected_features_sha256": sha256_file(args.selected_features),
        "feature_metrics_sha256": sha256_file(args.feature_metrics),
        "feature_prompts_sha256": sha256_file(args.feature_prompts),
        "features": output,
        "interpretation_warning": (
            "Reverse-index examples are evidence for interpretation, not ground-truth "
            "feature labels or causal validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({**payload, "features": len(output)}, indent=2))


if __name__ == "__main__":
    main()
