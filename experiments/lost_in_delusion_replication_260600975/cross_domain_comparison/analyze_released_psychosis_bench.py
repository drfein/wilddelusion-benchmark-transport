#!/usr/bin/env python3
"""Audit the Psychosis-Bench authors' released 128-experiment log."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
DEFAULT_SOURCE = (
    EXPERIMENT
    / "paper"
    / "psychosis-bench"
    / "outputs"
    / "20250905_000927_async_batch_8models_16cases.md"
)
EXPECTED_SOURCE_SHA256 = (
    "4909a447a73308b906aa13bade64200f9e4cb9b98e25b0f7a214aa46b9653a98"
)
PAPER_REPORTED_EFFECT = 0.31
EXPECTED_MODELS = {
    "anthropic/claude-sonnet-4",
    "deepseek/deepseek-chat-v3.1",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "meta-llama/llama-4-maverick",
    "openai/gpt-4o",
    "openai/gpt-5",
    "openai/o4-mini",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_match(pattern: str, text: str, field: str) -> re.Match[str]:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Missing {field} in released experiment block")
    return match


def parse_released_log(text: str) -> pd.DataFrame:
    blocks = re.split(r"(?=^## Experiment \d+:)", text, flags=re.MULTILINE)[1:]
    rows: list[dict[str, Any]] = []
    for block in blocks:
        heading = required_match(
            (
                r"^## Experiment (\d+): Case (\d+)\.(\d+): "
                r"(.+?) \[(EXPLICIT|IMPLICIT)\] × (.+)$"
            ),
            block,
            "experiment heading",
        )
        model = required_match(
            r"^- \*\*Model:\*\* (.+)$", block, "model"
        ).group(1)
        condition = required_match(
            r"^- \*\*Condition:\*\* (Explicit|Implicit)$",
            block,
            "condition",
        ).group(1).lower()
        heading_condition = heading.group(5).lower()
        if condition != heading_condition:
            raise ValueError("Heading and metadata conditions disagree")
        summary_mean = float(
            required_match(
                (
                    r"^- \*\*Average DCS \(Delusion Confirmation\):\*\* "
                    r"([0-9.]+)$"
                ),
                block,
                "average DCS",
            ).group(1)
        )
        scores = [
            int(value)
            for value in re.findall(
                r"^- DCS \(Delusion Confirmation Score\): ([012])$",
                block,
                flags=re.MULTILINE,
            )
        ]
        if not scores:
            raise ValueError("Released experiment has no turn-level DCS scores")
        raw_mean = float(np.mean(scores))
        if abs(raw_mean - summary_mean) > 0.0006:
            raise ValueError("Summary DCS is inconsistent with turn-level DCS")
        rows.append(
            {
                "experiment": int(heading.group(1)),
                "pair_id": int(heading.group(2)),
                "condition_index": int(heading.group(3)),
                "case_name": heading.group(4),
                "heading_condition": heading_condition,
                "model": model,
                "condition": condition,
                "theme": required_match(
                    r"^- \*\*Theme:\*\* (.+)$", block, "theme"
                ).group(1),
                "harm_type": required_match(
                    r"^- \*\*Harm Type:\*\* (.+)$", block, "harm type"
                ).group(1),
                "average_dcs_reported": summary_mean,
                "average_dcs_from_available_turns": raw_mean,
                "scored_turns": len(scores),
            }
        )
    return pd.DataFrame(rows)


def validate_release(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) != 128 or frame["experiment"].nunique() != 128:
        raise ValueError("Expected 128 unique released experiments")
    if set(frame["model"]) != EXPECTED_MODELS:
        raise ValueError("Released model set differs from the paper")
    if set(frame["pair_id"]) != set(range(1, 9)):
        raise ValueError("Expected eight scenario pairs")
    if set(frame["condition"]) != {"explicit", "implicit"}:
        raise ValueError("Expected explicit and implicit conditions")
    grouped = frame.groupby(["model", "pair_id"], sort=False)
    if any(
        len(group) != 2 or set(group["condition"]) != {"explicit", "implicit"}
        for _, group in grouped
    ):
        raise ValueError("Released explicit/implicit pairing is incomplete")
    missing_turn_rows = frame[frame["scored_turns"] != 9]
    if (
        int(frame["scored_turns"].sum()) != 1151
        or len(missing_turn_rows) != 1
        or int(missing_turn_rows.iloc[0]["scored_turns"]) != 8
        or int(missing_turn_rows.iloc[0]["experiment"]) != 91
    ):
        raise ValueError("Unexpected turn-level DCS completeness pattern")
    metadata_mismatch_pairs = sorted(
        {
            int(pair_id)
            for (_, pair_id), group in grouped
            if group["theme"].nunique() != 1
            or group["harm_type"].nunique() != 1
        }
    )
    return {
        "experiments": len(frame),
        "models": frame["model"].nunique(),
        "scenario_pairs": frame["pair_id"].nunique(),
        "expected_turn_level_scores": 128 * 9,
        "available_turn_level_scores": int(frame["scored_turns"].sum()),
        "missing_turn_level_scores": 1,
        "incomplete_experiment": 91,
        "scenario_pairs_with_condition_specific_theme_or_harm_metadata": (
            metadata_mismatch_pairs
        ),
        "missing_score_handling": (
            "Use the authors' reported experiment mean, which equals the mean "
            "of its eight available turn-level labels; do not impute."
        ),
    }


def fixed_model_scenario_bootstrap(
    scenario_effects: np.ndarray,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picks = rng.integers(
        0,
        len(scenario_effects),
        size=(repetitions, len(scenario_effects)),
    )
    return scenario_effects[picks].mean(axis=1)


def exact_sign_flip_p(effect_by_scenario: np.ndarray) -> float:
    observed = float(effect_by_scenario.mean())
    null = np.array(
        [
            np.mean(effect_by_scenario * np.asarray(signs))
            for signs in itertools.product((-1, 1), repeat=len(effect_by_scenario))
        ]
    )
    return float(np.mean(null >= observed - 1e-12))


def analyze(
    frame: pd.DataFrame,
    bootstrap_repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    wide = frame.pivot(
        index=["model", "pair_id"],
        columns="condition",
        values="average_dcs_reported",
    ).reset_index()
    wide["delta_implicit_minus_explicit"] = (
        wide["implicit"] - wide["explicit"]
    )
    model_rows = []
    for model, group in wide.groupby("model", sort=True):
        delta = group["delta_implicit_minus_explicit"]
        model_rows.append(
            {
                "model": model,
                "explicit_mean_dcs": float(group["explicit"].mean()),
                "implicit_mean_dcs": float(group["implicit"].mean()),
                "delta_implicit_minus_explicit": float(delta.mean()),
                "positive_scenario_pairs": int((delta > 0).sum()),
                "negative_scenario_pairs": int((delta < 0).sum()),
                "tied_scenario_pairs": int((delta == 0).sum()),
            }
        )
    model_effects = pd.DataFrame(model_rows)

    scenario_effects = (
        wide.groupby("pair_id", as_index=False)[
            "delta_implicit_minus_explicit"
        ]
        .mean()
        .rename(
            columns={
                "delta_implicit_minus_explicit": (
                    "fixed_model_delta_implicit_minus_explicit"
                )
            }
        )
    )
    pair_metadata = []
    for pair_id, group in frame.groupby("pair_id", sort=True):
        case_names = sorted(group["case_name"].unique())
        if len(case_names) != 1:
            raise ValueError(f"Scenario pair {pair_id} has inconsistent names")
        by_condition = group.drop_duplicates("condition").set_index(
            "condition"
        )
        pair_metadata.append(
            {
                "pair_id": int(pair_id),
                "case_name": case_names[0],
                "explicit_theme": by_condition.loc["explicit", "theme"],
                "implicit_theme": by_condition.loc["implicit", "theme"],
                "explicit_harm_type": by_condition.loc[
                    "explicit", "harm_type"
                ],
                "implicit_harm_type": by_condition.loc[
                    "implicit", "harm_type"
                ],
            }
        )
    scenario_effects = scenario_effects.merge(
        pd.DataFrame(pair_metadata),
        on="pair_id",
        validate="one_to_one",
    )
    values = scenario_effects[
        "fixed_model_delta_implicit_minus_explicit"
    ].to_numpy()
    draws = fixed_model_scenario_bootstrap(
        values, bootstrap_repetitions, seed
    )
    pooled = float(model_effects["delta_implicit_minus_explicit"].mean())
    gpt4o = model_effects.set_index("model").loc["openai/gpt-4o"]
    summary = {
        "paper_reported_effect": PAPER_REPORTED_EFFECT,
        "recomputed_from_released_summary_means": pooled,
        "absolute_reproduction_error": abs(pooled - PAPER_REPORTED_EFFECT),
        "fixed_model_scenario_cluster_bootstrap_ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "fixed_model_scenario_cluster_bootstrap_ci99": [
            float(np.quantile(draws, 0.005)),
            float(np.quantile(draws, 0.995)),
        ],
        "one_sided_exact_scenario_sign_flip_p": exact_sign_flip_p(values),
        "positive_model_effects": int(
            (model_effects["delta_implicit_minus_explicit"] > 0).sum()
        ),
        "negative_model_effects": int(
            (model_effects["delta_implicit_minus_explicit"] < 0).sum()
        ),
        "positive_fixed_model_scenario_effects": int((values > 0).sum()),
        "negative_fixed_model_scenario_effects": int((values < 0).sum()),
        "gpt4o": {
            "delta_implicit_minus_explicit": float(
                gpt4o["delta_implicit_minus_explicit"]
            ),
            "positive_scenario_pairs": int(
                gpt4o["positive_scenario_pairs"]
            ),
            "negative_scenario_pairs": int(
                gpt4o["negative_scenario_pairs"]
            ),
            "tied_scenario_pairs": int(gpt4o["tied_scenario_pairs"]),
        },
        "interpretation": (
            "The released synthetic effect is internally reproducible but "
            "heterogeneous: seven of eight model effects are positive, while "
            "one is negative. This artifact alone cannot establish transport "
            "to real conversations."
        ),
    }
    return model_effects, scenario_effects, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "released_psychosis_bench_audit",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=250910970)
    args = parser.parse_args()

    source_hash = sha256_file(args.source)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError("Released Psychosis-Bench log hash changed")
    frame = parse_released_log(args.source.read_text(encoding="utf-8"))
    integrity = validate_release(frame)
    model_effects, scenario_effects, summary = analyze(
        frame,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "released_experiments.csv", index=False)
    model_effects.to_csv(args.output_dir / "model_effects.csv", index=False)
    scenario_effects.to_csv(
        args.output_dir / "scenario_effects.csv", index=False
    )
    payload = {
        "source": str(args.source),
        "source_sha256": source_hash,
        "integrity": integrity,
        "analysis": summary,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
