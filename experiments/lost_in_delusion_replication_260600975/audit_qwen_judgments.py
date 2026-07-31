#!/usr/bin/env python3
"""Audit native-Qwen judgments and compare them with the prior API judge."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from io_utils import read_jsonl, write_manifest
from judge_responses_vllm import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    judge_protocol,
)


def metric_value(metric: str, value: Any) -> int:
    if metric in {"DCS", "HES"}:
        return {"N/A": 0, "1": 0, "2": 1, "3": 2}[str(value).upper()]
    if metric == "SIS":
        return int(value)
    raise KeyError(metric)


def load_unique(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    selected = {}
    for row in rows:
        generation_id = row["generation_id"]
        if generation_id in selected:
            raise ValueError(f"{path}: duplicate generation_id={generation_id}")
        selected[generation_id] = row
    return selected


def expected_ids(paths: list[Path]) -> set[str]:
    ids = set()
    for path in paths:
        for row in read_jsonl(path):
            if row.get("response") and not row.get("generation_error"):
                if row["generation_id"] in ids:
                    raise ValueError(
                        f"duplicate generation_id across expected inputs: "
                        f"{row['generation_id']}"
                    )
                ids.add(row["generation_id"])
    return ids


def validate(
    name: str,
    rows: dict[str, dict[str, Any]],
    expected: set[str],
    previous_exchanges: int,
) -> dict[str, Any]:
    protocol = judge_protocol(previous_exchanges)
    missing = sorted(expected - rows.keys())
    extra = sorted(rows.keys() - expected)
    errors = [
        generation_id
        for generation_id, row in rows.items()
        if row.get("judge_error")
    ]
    bad_protocol = [
        generation_id
        for generation_id, row in rows.items()
        if row.get("judge_protocol") != protocol
    ]
    bad_model = [
        generation_id
        for generation_id, row in rows.items()
        if row.get("judge_model_id") != DEFAULT_MODEL_ID
        or row.get("judge_model_revision") != DEFAULT_REVISION
        or row.get("judge_dtype") != "bfloat16"
    ]
    invalid_scores = []
    for generation_id, row in rows.items():
        try:
            metric_value("DCS", row["DCS"])
            metric_value("HES", row["HES"])
            metric_value("SIS", row["SIS"])
        except (KeyError, TypeError, ValueError):
            invalid_scores.append(generation_id)
    length_finished = [
        generation_id
        for generation_id, row in rows.items()
        if row.get("judge_finish_reason") == "length"
    ]
    audit = {
        "name": name,
        "expected_rows": len(expected),
        "actual_rows": len(rows),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "error_count": len(errors),
        "bad_protocol_count": len(bad_protocol),
        "bad_model_provenance_count": len(bad_model),
        "invalid_score_count": len(invalid_scores),
        "length_finished_count": len(length_finished),
        "missing_ids": missing[:25],
        "extra_ids": extra[:25],
        "error_ids": errors[:25],
        "bad_protocol_ids": bad_protocol[:25],
        "bad_model_provenance_ids": bad_model[:25],
        "invalid_score_ids": invalid_scores[:25],
        "length_finished_ids": length_finished[:25],
    }
    if any(
        audit[key]
        for key in (
            "missing_count",
            "extra_count",
            "error_count",
            "bad_protocol_count",
            "bad_model_provenance_count",
            "invalid_score_count",
            "length_finished_count",
        )
    ):
        raise RuntimeError(json.dumps(audit, indent=2))
    return audit


def weighted_kappa(old: list[int], new: list[int], levels: int) -> float:
    count = len(old)
    old_marginal = Counter(old)
    new_marginal = Counter(new)
    denominator = max(1, (levels - 1) ** 2)
    observed = sum(
        ((left - right) ** 2) / denominator
        for left, right in zip(old, new, strict=True)
    ) / count
    expected = sum(
        old_marginal[left]
        * new_marginal[right]
        / (count * count)
        * (((left - right) ** 2) / denominator)
        for left in range(levels)
        for right in range(levels)
    )
    return 1.0 - observed / expected if expected else math.nan


def compare(
    old: dict[str, dict[str, Any]],
    new: dict[str, dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    shared = sorted(old.keys() & new.keys())
    if set(shared) != set(new):
        raise ValueError("old and new judgment IDs do not match")
    summary = []
    confusion_rows = []
    model_rows: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    disagreements = []
    for metric in ("DCS", "HES", "SIS"):
        old_values = [metric_value(metric, old[key][metric]) for key in shared]
        new_values = [metric_value(metric, new[key][metric]) for key in shared]
        summary.append(
            {
                "metric": metric,
                "n": len(shared),
                "old_mean": sum(old_values) / len(old_values),
                "new_mean": sum(new_values) / len(new_values),
                "mean_shift_new_minus_old": (
                    sum(new_values) - sum(old_values)
                )
                / len(old_values),
                "exact_agreement": sum(
                    left == right
                    for left, right in zip(old_values, new_values, strict=True)
                )
                / len(old_values),
                "quadratic_weighted_kappa": weighted_kappa(
                    old_values,
                    new_values,
                    2 if metric == "SIS" else 3,
                ),
            }
        )
        confusion = Counter(zip(old_values, new_values, strict=True))
        for (old_score, new_score), count in sorted(confusion.items()):
            confusion_rows.append(
                {
                    "metric": metric,
                    "old_score": old_score,
                    "new_score": new_score,
                    "count": count,
                }
            )
        for key, old_value, new_value in zip(
            shared, old_values, new_values, strict=True
        ):
            row = new[key]
            model_rows[(row["model"], row["condition"], metric)].append(
                (old_value, new_value)
            )
            if old_value != new_value:
                disagreements.append(
                    {
                        "generation_id": key,
                        "model": row["model"],
                        "condition": row["condition"],
                        "metric": metric,
                        "old_score": old_value,
                        "new_score": new_value,
                        "absolute_change": abs(new_value - old_value),
                        "target_text": row.get("target_text"),
                        "response": row.get("response"),
                    }
                )

    by_model = []
    for (model, condition, metric), values in sorted(model_rows.items()):
        by_model.append(
            {
                "model": model,
                "condition": condition,
                "metric": metric,
                "n": len(values),
                "old_mean": sum(value[0] for value in values) / len(values),
                "new_mean": sum(value[1] for value in values) / len(values),
                "mean_shift_new_minus_old": sum(
                    value[1] - value[0] for value in values
                )
                / len(values),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "judge_comparison_summary.csv", summary)
    write_csv(output_dir / "judge_confusion_matrices.csv", confusion_rows)
    write_csv(output_dir / "judge_comparison_by_model.csv", by_model)
    disagreements.sort(
        key=lambda row: (
            -row["absolute_change"],
            row["metric"],
            row["model"],
            row["generation_id"],
        )
    )
    with (output_dir / "judge_disagreements.jsonl").open("w") as handle:
        for row in disagreements:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def promote(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".qwen.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    source_manifest = source.with_suffix(source.suffix + ".manifest.json")
    destination_manifest = destination.with_suffix(
        destination.suffix + ".manifest.json"
    )
    temporary_manifest = destination_manifest.with_suffix(
        destination_manifest.suffix + ".qwen.tmp"
    )
    shutil.copy2(source_manifest, temporary_manifest)
    temporary_manifest.replace(destination_manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-main", type=Path, required=True)
    parser.add_argument("--new-context", type=Path, required=True)
    parser.add_argument("--old-main", type=Path, required=True)
    parser.add_argument("--old-context", type=Path, required=True)
    parser.add_argument("--main-generations", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--context-generations", type=Path, nargs="+", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--promote-main", type=Path)
    parser.add_argument("--promote-context", type=Path)
    args = parser.parse_args()

    new_main = load_unique(args.new_main)
    new_context = load_unique(args.new_context)
    audits = [
        validate(
            "main",
            new_main,
            expected_ids(args.main_generations),
            previous_exchanges=3,
        ),
        validate(
            "context_ablation",
            new_context,
            expected_ids(args.context_generations),
            previous_exchanges=0,
        ),
    ]
    comparisons = {
        "main": compare(
            load_unique(args.old_main),
            new_main,
            args.output_dir / "main",
        ),
        "context_ablation": compare(
            load_unique(args.old_context),
            new_context,
            args.output_dir / "context_ablation",
        ),
    }
    manifest = {"audits": audits, "comparisons": comparisons}
    write_manifest(args.output_dir / "audit_summary.json", manifest)
    if bool(args.promote_main) != bool(args.promote_context):
        raise ValueError("provide both promotion destinations or neither")
    if args.promote_main:
        promote(args.new_main, args.promote_main)
        promote(args.new_context, args.promote_context)
        manifest["promoted"] = {
            "main": str(args.promote_main),
            "context": str(args.promote_context),
        }
        write_manifest(args.output_dir / "audit_summary.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
