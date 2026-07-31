#!/usr/bin/env python3
"""Reparse cached classifier text without altering any model generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prepare_inputs import read_jsonl, sha256_file
from recognition_prompts import parse_final_answer


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def reparse_output(path: Path) -> dict[str, Any]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        latest[str(row["generation_id"])] = row
    rows = list(latest.values())
    for row in rows:
        if row.get("raw_response") is not None:
            row.update(parse_final_answer(str(row["raw_response"])))
    write_jsonl(path, rows)

    manifests = list(path.parent.glob(path.name + ".*.manifest.json"))
    if len(manifests) != 1:
        raise ValueError(
            f"Expected one model manifest for {path}, found {len(manifests)}"
        )
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_rows = [row for row in rows if row.get("model") == manifest["model"]]
    manifest.update(
        {
            "rows_for_model": len(model_rows),
            "successful_rows": sum(
                row.get("raw_response") is not None
                and not row.get("generation_error")
                for row in model_rows
            ),
            "generation_error_rows": sum(
                bool(row.get("generation_error")) for row in model_rows
            ),
            "parse_valid_rows": sum(
                bool(row.get("parse_valid")) for row in model_rows
            ),
            "parse_exact_rows": sum(
                row.get("parse_format") == "exact_json" for row in model_rows
            ),
            "parse_repaired_rows": sum(
                row.get("parse_format") == "unquoted_scalar_repair"
                and bool(row.get("parse_valid"))
                for row in model_rows
            ),
            "parse_unresolved_rows": sum(
                row.get("raw_response") is not None
                and not bool(row.get("parse_valid"))
                for row in model_rows
            ),
            "parser_policy": (
                "Exact published labels and confidence values; accepts only "
                "omitted quotation marks around otherwise permitted scalar "
                "values; no semantic label or confidence mapping."
            ),
            "output_sha256": sha256_file(path),
        }
    )
    manifest["parse_protocol_complete"] = (
        manifest["parse_valid_rows"] == manifest["expected_rows"]
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        key: manifest[key]
        for key in (
            "model",
            "expected_rows",
            "successful_rows",
            "parse_valid_rows",
            "parse_exact_rows",
            "parse_repaired_rows",
            "parse_unresolved_rows",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", nargs="+", type=Path)
    args = parser.parse_args()
    summaries = [reparse_output(path) for path in args.outputs]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
