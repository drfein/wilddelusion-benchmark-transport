#!/usr/bin/env python3
"""Validate and combine the 1,024-token primary regeneration."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from io_utils import read_jsonl, sha256_file, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--regenerated", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = {row["generation_id"]: row for row in read_jsonl(args.reference)}
    rows = [row for path in args.regenerated for row in read_jsonl(path)]
    regenerated = {row["generation_id"]: row for row in rows}
    if len(reference) != 960 or len(rows) != 960 or len(regenerated) != 960:
        raise ValueError("Expected 960 unique reference and regenerated rows")
    if set(reference) != set(regenerated):
        raise ValueError("Regenerated IDs do not exactly match the frozen primary cohort")

    prefix_matches = 0
    exact_matches = 0
    diverged_responses = 0
    for generation_id, row in regenerated.items():
        old = reference[generation_id]
        if row["prompt_token_ids_sha256"] != old["prompt_token_ids_sha256"]:
            raise ValueError(f"{generation_id}: prompt tokens changed")
        if row["request_seed"] != old["request_seed"]:
            raise ValueError(f"{generation_id}: request seed changed")
        if row["max_new_tokens"] != 1024:
            raise ValueError(f"{generation_id}: response cap is not 1024")
        if not row["response"]:
            raise ValueError(f"{generation_id}: empty response")
        if row["response"] == old["response"]:
            exact_matches += 1
        elif row["response"].startswith(old["response"]):
            prefix_matches += 1
        else:
            # vLLM's per-request seed is not bitwise invariant to changed batch
            # composition. Record this as a new stochastic draw, not an error.
            diverged_responses += 1

    rows.sort(key=lambda row: row["generation_id"])
    write_jsonl(args.output, rows)
    finish_reasons = Counter(row["finish_reason"] for row in rows)
    manifest = {
        "reference": str(args.reference),
        "reference_sha256": sha256_file(args.reference),
        "regenerated_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in args.regenerated
        ],
        "rows": len(rows),
        "candidate_count": len({row["candidate_id"] for row in rows}),
        "models": sorted({row["model"] for row in rows}),
        "replicate_seeds": sorted({row["replicate_seed"] for row in rows}),
        "exact_512_response_matches": exact_matches,
        "extended_responses_with_identical_512_prefix": prefix_matches,
        "responses_diverged_under_changed_batch_composition": diverged_responses,
        "sampling_interpretation": (
            "fresh pre-outcome stochastic draw; request seeds match but batching "
            "is not bitwise deterministic"
        ),
        "finish_reasons": finish_reasons,
        "output_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, default=dict) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, default=dict))


if __name__ == "__main__":
    main()
