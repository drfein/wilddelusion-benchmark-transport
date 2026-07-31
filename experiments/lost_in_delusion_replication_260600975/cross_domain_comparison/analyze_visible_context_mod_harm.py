#!/usr/bin/env python3
"""Audit Mod+Harm coverage over the exact assistant-visible context."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(EXPERIMENT))

from io_utils import read_jsonl, write_manifest  # noqa: E402
from full_history_522.label_mod_harm_visible_context import (  # noqa: E402
    DEFINITION as VISIBLE_DEFINITION,
    MODEL as VISIBLE_LABEL_MODEL,
    PROTOCOL as VISIBLE_PROTOCOL,
)
from analyze_lost_exact_claims import (  # noqa: E402
    EXPECTED_COHORT_SHA256,
    EXPECTED_MAIN_JUDGE_GENERATION_FILES,
    EXPECTED_POSSIBLE_MOD_HARM_COUNT,
    LEXICAL_MOD_HARM_PATTERNS,
    audit_generation_protocol,
    audit_judgment_manifest,
    audit_judgments,
    audit_mod_harm_superset,
    bootstrap_ratio_intervals,
    sha256_file,
    verify_revision_attestation,
)


EXPECTED_INPUT_SHA256 = (
    "18aff72537297c5d2d6bd23a9d95617bab4e08d442906ed3a886b5b5fb24b38a"
)
EXPECTED_VISIBLE_UNION_COUNT = 161
PRIMARY_MODELS = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "allenai/Olmo-3-7B-Instruct",
)


def messages_sha256(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pair_ids_sha256(pair_ids: set[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(pair_ids)) + "\n").encode("utf-8")
    ).hexdigest()


def load_visible_inputs(path: Path) -> dict[str, dict[str, Any]]:
    rows = [
        row for row in read_jsonl(path) if row.get("condition") == "delusion"
    ]
    by_id = {row["pair_id"]: row for row in rows}
    checks = {
        "frozen_input_hash": sha256_file(path) == EXPECTED_INPUT_SHA256,
        "rows": len(rows) == 522,
        "unique_pair_ids": len(by_id) == 522,
        "context_scope": all(
            row.get("context_scope") == "final_4_user_turns" for row in rows
        ),
        "ends_in_user": all(
            row.get("messages")
            and row["messages"][-1].get("role") == "user"
            for row in rows
        ),
        "at_most_four_user_turns": all(
            sum(message.get("role") == "user" for message in row["messages"])
            <= 4
            for row in rows
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Visible input audit failed: {checks}")
    return by_id


def audit_visible_labels(
    path: Path,
    rubric: str,
    inputs: dict[str, dict[str, Any]],
    input_path: Path,
) -> tuple[set[str], dict[str, Any]]:
    rows = read_jsonl(path)
    by_id = {row["pair_id"]: row for row in rows}
    input_ids = set(inputs)
    counts = {
        label: sum(row.get("label") == label for row in rows)
        for label in ("yes", "no", "unclear")
    }
    evidence_ok = True
    for row in rows:
        source = inputs.get(row.get("pair_id"))
        if source is None:
            evidence_ok = False
            continue
        quote = str(row.get("evidence_quote") or "")
        index = int(row.get("evidence_message_index", -1))
        if row.get("label") == "yes":
            evidence_ok &= bool(
                0 <= index < len(source["messages"])
                and source["messages"][index]["role"] == "user"
                and quote
                and quote in source["messages"][index]["content"]
            )
        else:
            evidence_ok &= not quote and index == -1
    row_checks = {
        "rows": len(rows) == 522,
        "unique_complete_ids": set(by_id) == input_ids,
        "labels_valid": all(
            row.get("label") in {"yes", "no", "unclear"} for row in rows
        ),
        "no_errors": all(not row.get("label_error") for row in rows),
        "rubric": all(row.get("rubric_version") == rubric for row in rows),
        "protocol": all(row.get("protocol") == VISIBLE_PROTOCOL for row in rows),
        "definition": all(
            row.get("operational_definition") == VISIBLE_DEFINITION for row in rows
        ),
        "model": all(row.get("label_model") == VISIBLE_LABEL_MODEL for row in rows),
        "reasoning_effort": all(
            row.get("reasoning_effort") == "none" for row in rows
        ),
        "cluster_identity": all(
            row.get("cluster_id") == inputs[row["pair_id"]].get("cluster_id")
            for row in rows
        ),
        "input_id_identity": all(
            row.get("input_id") == inputs[row["pair_id"]].get("input_id")
            for row in rows
        ),
        "message_hash_identity": all(
            row.get("input_messages_sha256")
            == messages_sha256(inputs[row["pair_id"]]["messages"])
            for row in rows
        ),
        "evidence_verbatim_and_user_authored": evidence_ok,
    }
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_checks = {
        "protocol": manifest.get("protocol") == VISIBLE_PROTOCOL,
        "definition": manifest.get("operational_definition") == VISIBLE_DEFINITION,
        "rubric": manifest.get("rubric_version") == rubric,
        "model": manifest.get("model") == VISIBLE_LABEL_MODEL,
        "reasoning_effort": manifest.get("reasoning_effort") == "none",
        "input_hash": manifest.get("input_sha256") == sha256_file(input_path),
        "expected_rows": manifest.get("expected_rows") == 522,
        "successful_rows": manifest.get("successful_rows") == 522,
        "no_errors": manifest.get("new_errors") == 0,
        "label_counts": manifest.get("label_counts") == counts,
        "output_hash": manifest.get("output_sha256") == sha256_file(path),
    }
    if not all(row_checks.values()) or not all(manifest_checks.values()):
        raise ValueError(
            f"Visible-context rubric {rubric} audit failed: "
            + json.dumps(
                {"row_checks": row_checks, "manifest_checks": manifest_checks},
                sort_keys=True,
            )
        )
    possible = {
        pair_id for pair_id, row in by_id.items() if row["label"] != "no"
    }
    return possible, {
        "rubric": rubric,
        "path": str(path),
        "sha256": sha256_file(path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "row_checks": row_checks,
        "manifest_checks": manifest_checks,
        "label_counts": counts,
        "possible_count": len(possible),
        "quote_repairs": sum(
            row.get("quote_repaired_to_full_message") is True for row in rows
        ),
    }


def visible_lexical_ids(
    inputs: dict[str, dict[str, Any]],
) -> tuple[set[str], dict[str, int]]:
    compiled = {
        name: re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
        for name, pattern in LEXICAL_MOD_HARM_PATTERNS.items()
    }
    by_pattern: dict[str, set[str]] = {name: set() for name in compiled}
    for pair_id, row in inputs.items():
        user_texts = [
            message["content"]
            for message in row["messages"]
            if message["role"] == "user"
        ]
        for name, pattern in compiled.items():
            if any(pattern.search(text) for text in user_texts):
                by_pattern[name].add(pair_id)
    return set().union(*by_pattern.values()), {
        name: len(pair_ids) for name, pair_ids in by_pattern.items()
    }


def coverage_rows(
    eligible: pd.DataFrame,
    possible_ids: set[str],
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for offset, model in enumerate(PRIMARY_MODELS):
        target = eligible[
            (eligible["model"] == model)
            & (eligible["condition"] == "delusion")
        ].copy()
        if len(target) != 522:
            raise ValueError(f"{model}: expected 522 exact judged endpoints")
        target["possible_visible_mod_harm"] = target["pair_id"].isin(possible_ids)
        target["validate_amplify"] = (target["dcs"] == 2).astype(float)
        target["outside_validate_amplify"] = (
            target["validate_amplify"]
            * (~target["possible_visible_mod_harm"])
        ).astype(float)
        target["outside_non_grounding"] = (
            target["dcs_positive"]
            * (~target["possible_visible_mod_harm"])
        ).astype(float)
        target["outside_dcs_mass"] = (
            target["dcs"] * (~target["possible_visible_mod_harm"])
        ).astype(float)
        result: dict[str, Any] = {
            "model": model,
            "n_endpoints": len(target),
            "n_clusters": int(target["cluster_id"].nunique()),
            "possible_visible_mod_harm_n": int(
                target["possible_visible_mod_harm"].sum()
            ),
            "validate_amplify_n": int(target["validate_amplify"].sum()),
            "validate_amplify_inside_n": int(
                (
                    target["validate_amplify"]
                    * target["possible_visible_mod_harm"]
                ).sum()
            ),
        }
        metrics = (
            (
                "validate_amplify_outside_share",
                "outside_validate_amplify",
                "validate_amplify",
            ),
            (
                "non_grounding_outside_share",
                "outside_non_grounding",
                "dcs_positive",
            ),
            ("dcs_mass_outside_share", "outside_dcs_mass", "dcs"),
        )
        for metric_offset, (name, numerator, denominator) in enumerate(metrics):
            point, low, high, low99, high99 = bootstrap_ratio_intervals(
                target,
                numerator,
                denominator,
                draws,
                seed + offset * 100 + metric_offset,
            )
            result.update(
                {
                    name: point,
                    f"{name}_ci95_low": low,
                    f"{name}_ci95_high": high,
                    f"{name}_ci99_low": low99,
                    f"{name}_ci99_high": high99,
                }
            )
        result["validate_amplify_inside_share"] = (
            1.0 - result["validate_amplify_outside_share"]
        )
        rows.append(result)
    return rows


def gate_ready(rows: list[dict[str, Any]]) -> bool:
    return bool(
        len(rows) == len(PRIMARY_MODELS)
        and all(
            row["validate_amplify_outside_share_ci99_low"] > 0.5
            and row["non_grounding_outside_share_ci99_low"] > 0.5
            and row["dcs_mass_outside_share_ci99_low"] > 0.5
            for row in rows
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visible-inputs",
        type=Path,
        default=EXPERIMENT / "full_history_522/artifacts/model_inputs.jsonl",
    )
    parser.add_argument(
        "--visible-label-a",
        type=Path,
        default=EXPERIMENT
        / "full_history_522/artifacts/mod_harm_visible_context_a.jsonl",
    )
    parser.add_argument(
        "--visible-label-b",
        type=Path,
        default=EXPERIMENT
        / "full_history_522/artifacts/mod_harm_visible_context_b.jsonl",
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=EXPERIMENT
        / "full_dataset_522/artifacts/cohort_full522.jsonl",
    )
    parser.add_argument(
        "--target-label-a",
        type=Path,
        default=EXPERIMENT / "full_history_522/artifacts/mod_harm_labels_a.jsonl",
    )
    parser.add_argument(
        "--target-label-b",
        type=Path,
        default=EXPERIMENT / "full_history_522/artifacts/mod_harm_labels_b.jsonl",
    )
    parser.add_argument(
        "--target-label-c",
        type=Path,
        default=EXPERIMENT / "full_history_522/artifacts/mod_harm_labels_c.jsonl",
    )
    parser.add_argument(
        "--legacy-safety-labels",
        type=Path,
        default=EXPERIMENT
        / "psychogenic_machine_transport_250910970/full522/artifacts/safety_labels.jsonl",
    )
    parser.add_argument(
        "--judgments",
        type=Path,
        default=EXPERIMENT
        / "full_history_522/results/judgments.qwen3-30b-thinking.jsonl",
    )
    parser.add_argument(
        "--generations",
        type=Path,
        nargs="+",
        default=[
            EXPERIMENT / "full_history_522/results/generations_llama31_8b.jsonl",
            EXPERIMENT / "full_history_522/results/generations_llama33_70b.jsonl",
            EXPERIMENT / "full_history_522/results/generations_olmo.jsonl",
            EXPERIMENT / "full_history_522/results/generations_qwen3_30b.jsonl",
        ],
    )
    parser.add_argument(
        "--generation-attestation",
        type=Path,
        default=HERE / "generation_revision_attestation.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "visible_context_mod_harm",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=260600976)
    args = parser.parse_args()

    inputs = load_visible_inputs(args.visible_inputs)
    base_ids, base_audit = audit_mod_harm_superset(
        args.cohort,
        {
            "A": args.target_label_a,
            "B": args.target_label_b,
            "C": args.target_label_c,
        },
        args.legacy_safety_labels,
        expected_cohort_sha256=EXPECTED_COHORT_SHA256,
        expected_possible_count=EXPECTED_POSSIBLE_MOD_HARM_COUNT,
    )
    visible_a, audit_a = audit_visible_labels(
        args.visible_label_a, "A", inputs, args.visible_inputs
    )
    visible_b, audit_b = audit_visible_labels(
        args.visible_label_b, "B", inputs, args.visible_inputs
    )
    lexical_ids, lexical_counts = visible_lexical_ids(inputs)
    possible_ids = base_ids | visible_a | visible_b | lexical_ids
    if len(possible_ids) != EXPECTED_VISIBLE_UNION_COUNT:
        raise ValueError("Frozen visible-context screen count has drifted")

    generation_protocol = audit_generation_protocol(args.generations)
    attested_models, attestation_audit = verify_revision_attestation(
        args.generation_attestation, args.generations
    )
    completion, eligible = audit_judgments(
        args.judgments, args.generations, attested_models
    )
    judgment_manifest = audit_judgment_manifest(
        args.judgments,
        [
            EXPERIMENT / "full_history_522/results" / filename
            for filename in EXPECTED_MAIN_JUDGE_GENERATION_FILES
        ],
    )
    rows = coverage_rows(
        eligible[eligible["model"].isin(PRIMARY_MODELS)],
        possible_ids,
        args.bootstrap_draws,
        args.seed,
    )
    ready = gate_ready(rows)
    inside_shares = [row["validate_amplify_inside_share"] for row in rows]
    gate = {
        "claim_id": "lost_confirmation_failures_mostly_outside_mod_harm",
        "ready": ready,
        "required_evidence": (
            "Both official-BF16 revision-attested models must be complete. "
            "After unioning every target-level and exact-visible-context "
            "possible-Mod+Harm screen, each model's conversation-clustered "
            "99% lower bound must still place a majority of Validate/Amplify "
            "responses, all non-grounding responses, and ordinal DCS mass "
            "outside the screen."
        ),
        "allowed_claim": (
            "Even a deliberately overinclusive possible-Mod+Harm screen over "
            "the exact assistant-visible context captured only "
            f"{min(inside_shares) * 100:.1f}-{max(inside_shares) * 100:.1f}% "
            "of real responses scored Validate/Amplify."
        ),
        "causal_guardrail": (
            "This is a conservative corpus-composition result, not evidence "
            "that adding crisis language causes a safer response."
        ),
        "screen": {
            "possible_n": len(possible_ids),
            "outside_n": 522 - len(possible_ids),
            "possible_share": len(possible_ids) / 522,
            "pair_ids_sha256": pair_ids_sha256(possible_ids),
            "construction": (
                "Union of the frozen 94-endpoint target-level superset, both "
                "visible-context rubrics with UNCLEAR treated as possible, "
                "and deterministic harm-language matches in any exact-visible "
                "user message."
            ),
            "base_target_screen_n": len(base_ids),
            "visible_rubric_a_possible_n": len(visible_a),
            "visible_rubric_b_possible_n": len(visible_b),
            "visible_lexical_n": len(lexical_ids),
        },
        "evidence": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_dir / "claim_gate.json", gate)
    with (args.output_dir / "coverage_by_model.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    membership = []
    for pair_id in sorted(inputs):
        membership.append(
            {
                "pair_id": pair_id,
                "cluster_id": inputs[pair_id]["cluster_id"],
                "possible_visible_mod_harm": pair_id in possible_ids,
                "base_target_screen": pair_id in base_ids,
                "visible_rubric_a": pair_id in visible_a,
                "visible_rubric_b": pair_id in visible_b,
                "visible_lexical": pair_id in lexical_ids,
            }
        )
    with (args.output_dir / "screen_membership.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=membership[0].keys())
        writer.writeheader()
        writer.writerows(membership)
    audit = {
        "visible_inputs": {
            "path": str(args.visible_inputs),
            "sha256": sha256_file(args.visible_inputs),
            "rows": len(inputs),
        },
        "base_target_screen": base_audit,
        "visible_context_labels": [audit_a, audit_b],
        "visible_lexical_counts": lexical_counts,
        "generation_protocol": generation_protocol,
        "generation_revision_attestation": attestation_audit,
        "judgment_manifest": judgment_manifest,
        "model_completion": completion.to_dict(orient="records"),
    }
    write_manifest(args.output_dir / "audit.json", audit)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
