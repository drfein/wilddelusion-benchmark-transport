#!/usr/bin/env python3
"""Audit completion, truncation, and paired context identity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import read_jsonl, write_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--conditions", nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        row
        for path in args.inputs
        for row in read_jsonl(path)
        if not row.get("generation_error")
    ]
    models = sorted({row["model"] for row in rows})
    diagnostics: list[dict[str, Any]] = []
    for model in models:
        selected = [row for row in rows if row["model"] == model]
        by_pair: dict[str, dict[str, dict[str, Any]]] = {}
        for row in selected:
            by_pair.setdefault(row["pair_id"], {})[row["condition"]] = row
        complete = [
            pair
            for pair in by_pair.values()
            if all(condition in pair for condition in args.conditions)
        ]
        mismatches = []
        for pair in complete:
            first, second = (pair[condition] for condition in args.conditions)
            if first["messages_used"][:-1] != second["messages_used"][:-1]:
                mismatches.append(first["pair_id"])
        input_tokens = np.array(
            [row["input_token_count"] for row in selected], dtype=float
        )
        diagnostics.append(
            {
                "model": model,
                "successful_rows": len(selected),
                "unique_targets": len(by_pair),
                "complete_pairs": len(complete),
                "rows_with_dropped_messages": sum(
                    int(row.get("dropped_message_count", 0) > 0)
                    for row in selected
                ),
                "pairs_with_prior_context_mismatch": len(mismatches),
                "prior_context_mismatch_pair_ids": mismatches,
                "input_tokens_median": float(np.median(input_tokens)),
                "input_tokens_p95": float(np.quantile(input_tokens, 0.95)),
                "input_tokens_max": int(input_tokens.max()),
                "response_chars_median": float(
                    np.median([len(row["response"]) for row in selected])
                ),
            }
        )
    write_manifest(
        args.output,
        {
            "conditions": args.conditions,
            "diagnostics": diagnostics,
        },
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
