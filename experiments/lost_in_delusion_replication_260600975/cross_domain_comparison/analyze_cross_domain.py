#!/usr/bin/env python3
"""Compare model rankings and corpus properties across synthetic and real data."""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

MODEL_LABELS = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "allenai/Olmo-3-7B-Instruct": "OLMo 3 7B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama 3.2 3B",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama 3.3 70B",
    "Qwen/Qwen3-4B": "Qwen 3 4B",
    "Qwen/Qwen3-14B": "Qwen 3 14B",
    "Qwen/Qwen3-30B-A3B": "Qwen 3 30B-A3B",
}
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
OLMO = "allenai/Olmo-3-7B-Instruct"
LLAMA70 = "meta-llama/Llama-3.3-70B-Instruct"
QWEN30 = "Qwen/Qwen3-30B-A3B"

LOST_SYNTHETIC = {
    LLAMA: {
        "DCS": {"control": 0.11, "target": 0.69, "effect": 0.58},
        "HES": {"control": 0.33, "target": 0.51, "effect": 0.18},
        "SIS": {"control": 0.59, "target": 0.49, "effect": -0.09},
    },
    OLMO: {
        "DCS": {"control": 0.12, "target": 1.08, "effect": 0.96},
        "HES": {"control": 0.48, "target": 1.09, "effect": 0.60},
        "SIS": {"control": 0.42, "target": 0.12, "effect": -0.30},
    },
    LLAMA70: {
        "DCS": {"control": 0.10, "target": 0.36, "effect": 0.26},
        "HES": {"control": 0.22, "target": 0.18, "effect": -0.04},
        "SIS": {"control": 0.63, "target": 0.71, "effect": 0.08},
    },
    QWEN30: {
        "DCS": {"control": 0.14, "target": 1.21, "effect": 1.06},
        "HES": {"control": 0.67, "target": 1.29, "effect": 0.62},
        "SIS": {"control": 0.29, "target": 0.06, "effect": -0.24},
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def dcs_ordinal(value: Any) -> int:
    return 0 if str(value) in {"N/A", "1"} else int(value) - 1


def clustered_ci(
    frame: pd.DataFrame,
    value: str,
    cluster: str = "cluster_id",
    draws: int = 50_000,
    seed: int = 260600975,
) -> tuple[float, float]:
    groups = [
        group[value].to_numpy(dtype=float)
        for _, group in frame.groupby(cluster, sort=False)
    ]
    if not groups:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for index in range(draws):
        selected = rng.integers(0, len(groups), len(groups))
        estimates[index] = np.concatenate(
            [groups[item] for item in selected]
        ).mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def model_gap(
    frame: pd.DataFrame,
    index: list[str],
    metric: str,
    sign: str,
) -> dict[str, Any]:
    wide = frame.pivot_table(
        index=index,
        columns="model",
        values=metric,
        aggfunc="first",
    ).dropna(subset=[LLAMA, OLMO])
    wide = wide.reset_index()
    if sign == "llama_minus_olmo":
        wide["model_gap"] = wide[LLAMA] - wide[OLMO]
    elif sign == "olmo_minus_llama":
        wide["model_gap"] = wide[OLMO] - wide[LLAMA]
    else:
        raise ValueError(sign)
    low, high = clustered_ci(wide, "model_gap")
    return {
        "n": len(wide),
        "cluster_n": wide["cluster_id"].nunique(),
        "gap": wide["model_gap"].mean(),
        "ci95_low": low,
        "ci95_high": high,
    }


def lost_rankings(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = pd.read_csv(
        root
        / "full_history_522/results/analysis_clustered/"
        "paired_effects_clustered.csv"
    )
    rates = pd.read_csv(
        root
        / "full_history_522/results/analysis_clustered/"
        "full_522_original_condition_rates.csv"
    )
    pair_level = pd.read_csv(
        root
        / "full_history_522/results/analysis_clustered/"
        "pair_level_effects.csv"
    )
    rows: list[dict[str, Any]] = []
    for model, model_reference in LOST_SYNTHETIC.items():
        for metric in ("DCS", "HES", "SIS"):
            reference = model_reference[metric]
            rows.append(
                {
                    "study": "Lost in Delusion",
                    "domain": "synthetic_paper",
                    "model": MODEL_LABELS[model],
                    "metric": metric,
                    "estimand": "target_mean",
                    "value": reference["target"],
                }
            )
            rows.append(
                {
                    "study": "Lost in Delusion",
                    "domain": "synthetic_paper",
                    "model": MODEL_LABELS[model],
                    "metric": metric,
                    "estimand": "target_minus_control",
                    "value": reference["effect"],
                }
            )

    real_models = [
        model
        for model in MODEL_LABELS
        if model in set(rates["model"]) and model in set(paired["model"])
    ]
    for model in real_models:
        model_rates = rates[rates["model"] == model]
        for metric, paper_metric in (
            ("DCS ordinal", "DCS"),
            ("HES ordinal", "HES"),
            ("SIS", "SIS"),
        ):
            selected = model_rates[model_rates["metric"] == metric].iloc[0]
            rows.append(
                {
                    "study": "Lost in Delusion",
                    "domain": "real",
                    "model": MODEL_LABELS[model],
                    "metric": paper_metric,
                    "estimand": "target_mean",
                    "value": selected["mean"],
                }
            )
        model_paired = paired[paired["model"] == model]
        for metric, paper_metric in (
            ("DCS ordinal", "DCS"),
            ("HES ordinal", "HES"),
            ("SIS", "SIS"),
        ):
            selected = model_paired[model_paired["metric"] == metric].iloc[0]
            rows.append(
                {
                    "study": "Lost in Delusion",
                    "domain": "real",
                    "model": MODEL_LABELS[model],
                    "metric": paper_metric,
                    "estimand": "target_minus_control",
                    "value": selected["paired_difference"],
                }
            )

    gap_rows: list[dict[str, Any]] = []
    for metric, paper_metric in (
        ("DCS ordinal", "DCS"),
        ("HES ordinal", "HES"),
        ("SIS", "SIS"),
    ):
        selected = pair_level[pair_level["metric"] == metric].copy()
        result = model_gap(
            selected,
            ["pair_id", "cluster_id"],
            "difference",
            "llama_minus_olmo",
        )
        gap_rows.append(
            {
                "study": "Lost in Delusion",
                "domain": "real",
                "metric": paper_metric,
                "estimand": "target_minus_control",
                "gap_definition": "Llama minus OLMo",
                **result,
            }
        )

    judgments = pd.DataFrame(
        read_jsonl(root / "full_history_522/results/judgments.jsonl")
    )
    judgments = judgments[judgments["condition"] == "delusion"].copy()
    judgments["DCS ordinal"] = judgments["DCS"].map(dcs_ordinal)
    judgments["HES ordinal"] = judgments["HES"].map(dcs_ordinal)
    for metric in ("DCS ordinal", "HES ordinal", "SIS"):
        selected = judgments
        if metric in {"HES ordinal", "SIS"}:
            if "high_stakes_action" not in selected:
                generation_meta = pd.concat(
                    [
                        pd.DataFrame(read_jsonl(path))
                        for path in sorted(
                            (
                                root / "full_history_522/results"
                            ).glob("generations_*.jsonl")
                        )
                    ],
                    ignore_index=True,
                )[["generation_id", "high_stakes_action"]].drop_duplicates(
                    "generation_id"
                )
                selected = selected.merge(
                    generation_meta, on="generation_id", validate="one_to_one"
                )
            selected = selected[selected["high_stakes_action"] == True]  # noqa: E712
        result = model_gap(
            selected,
            ["pair_id", "cluster_id"],
            metric,
            "llama_minus_olmo",
        )
        gap_rows.append(
            {
                "study": "Lost in Delusion",
                "domain": "real",
                "metric": metric.replace(" ordinal", ""),
                "estimand": "target_mean",
                "gap_definition": "Llama minus OLMo",
                **result,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(gap_rows)


def psychogenic_arm(
    judgments_path: Path,
    domain: str,
    phase_filtered: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(read_jsonl(judgments_path))
    if "judge_error" in frame:
        frame = frame[frame["judge_error"].isna()].copy()
    if phase_filtered and "turn_number" not in frame:
        frame["turn_number"] = frame["generation_id"].str.extract(
            r":turn(\d+)$", expand=False
        ).astype(int)
    rows: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for metric in ("DCS", "HES", "SIS"):
        eligible = frame.copy()
        if phase_filtered:
            minimum_turn = 4 if metric == "DCS" else 7
            eligible = eligible[eligible["turn_number"] >= minimum_turn]
        elif metric in {"HES", "SIS"}:
            eligible = eligible[eligible["harm_pair_valid"] == True]  # noqa: E712

        for model, model_frame in eligible.groupby("model", sort=False):
            condition_means = model_frame.groupby("condition")[metric].mean()
            paired = model_frame.pivot_table(
                index=["pair_id", "cluster_id"],
                columns="condition",
                values=metric,
                aggfunc="mean",
            ).dropna(subset=["explicit", "implicit"])
            rows.extend(
                [
                    {
                        "study": "Psychogenic Machine",
                        "domain": domain,
                        "model": MODEL_LABELS.get(model, model),
                        "metric": metric,
                        "estimand": "target_mean",
                        "value": model_frame[metric].mean(),
                    },
                    {
                        "study": "Psychogenic Machine",
                        "domain": domain,
                        "model": MODEL_LABELS.get(model, model),
                        "metric": metric,
                        "estimand": "implicit_minus_explicit",
                        "value": (
                            paired["implicit"] - paired["explicit"]
                        ).mean(),
                    },
                    {
                        "study": "Psychogenic Machine",
                        "domain": domain,
                        "model": MODEL_LABELS.get(model, model),
                        "metric": metric,
                        "estimand": "explicit_mean",
                        "value": condition_means.get("explicit", np.nan),
                    },
                    {
                        "study": "Psychogenic Machine",
                        "domain": domain,
                        "model": MODEL_LABELS.get(model, model),
                        "metric": metric,
                        "estimand": "implicit_mean",
                        "value": condition_means.get("implicit", np.nan),
                    },
                ]
            )

        absolute = model_gap(
            eligible,
            ["pair_id", "cluster_id", "condition"]
            + (["turn_number"] if phase_filtered else []),
            metric,
            "olmo_minus_llama",
        )
        gaps.append(
            {
                "study": "Psychogenic Machine",
                "domain": domain,
                "metric": metric,
                "estimand": "target_mean",
                "gap_definition": "OLMo minus Llama",
                **absolute,
            }
        )

        model_effect = (
            eligible.pivot_table(
                index=["pair_id", "cluster_id", "model"],
                columns="condition",
                values=metric,
                aggfunc="mean",
            )
            .dropna(subset=["explicit", "implicit"])
            .reset_index()
        )
        model_effect["condition_effect"] = (
            model_effect["implicit"] - model_effect["explicit"]
        )
        effect_gap = model_gap(
            model_effect,
            ["pair_id", "cluster_id"],
            "condition_effect",
            "olmo_minus_llama",
        )
        gaps.append(
            {
                "study": "Psychogenic Machine",
                "domain": domain,
                "metric": metric,
                "estimand": "implicit_minus_explicit",
                "gap_definition": "OLMo minus Llama",
                **effect_gap,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(gaps)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def jaccard(left: str, right: str) -> float:
    left_tokens, right_tokens = set(words(left)), set(words(right))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def sequence_ratio(left: str, right: str) -> float:
    return difflib.SequenceMatcher(
        None, " ".join(words(left)), " ".join(words(right))
    ).ratio()


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def corpus_characteristics(root: Path) -> dict[str, Any]:
    lost = read_jsonl(
        root / "full_history_522/artifacts/cohort_original.jsonl"
    )
    real_pairs = read_jsonl(
        root
        / "psychogenic_machine_transport_250910970/full522/artifacts/"
        "final_pairs.jsonl"
    )
    real_inputs = read_jsonl(
        root
        / "psychogenic_machine_transport_250910970/full522/artifacts/"
        "model_inputs.jsonl"
    )
    natural_inputs = read_jsonl(
        root
        / "psychogenic_machine_transport_250910970/full522_natural/"
        "artifacts/model_inputs.jsonl"
    )
    cases = json.loads(
        (
            root / "paper/psychosis-bench/data/test_cases.json"
        ).read_text()
    )["cases"]

    case_by_id = {case["id"]: case for case in cases}
    synthetic_pairs: list[tuple[str, str, int]] = []
    for case_id, explicit in case_by_id.items():
        if not case_id.endswith("_explicit"):
            continue
        implicit = case_by_id[case_id[: -len("_explicit")] + "_implicit"]
        synthetic_pairs.extend(
            (left, right, turn)
            for turn, (left, right) in enumerate(
                zip(explicit["prompts"], implicit["prompts"], strict=True),
                start=1,
            )
        )

    natural_by_pair: dict[str, dict[str, str]] = {}
    for row in natural_inputs:
        natural_by_pair.setdefault(row["pair_id"], {})[
            row["condition"]
        ] = row["messages"][-1]["content"]

    return {
        "lost_real": {
            "targets": len(lost),
            "source_clusters": len({row["cluster_id"] for row in lost}),
            "themes": dict(Counter(row["theme"] for row in lost)),
            "target_words": distribution(
                [len(words(row["target_text"])) for row in lost]
            ),
            "retained_messages": distribution(
                [row["retained_message_count"] for row in lost]
            ),
            "source_user_turns_through_target": distribution(
                [row["source_user_turn_count_through_target"] for row in lost]
            ),
            "single_message_histories": sum(
                row["retained_message_count"] == 1 for row in lost
            ),
        },
        "psychogenic_synthetic": {
            "scenario_pairs": len(cases) // 2,
            "conversations": len(cases),
            "turns_per_conversation": 12,
            "themes": dict(Counter(case["theme"] for case in cases)),
            "prompt_words": distribution(
                [
                    len(words(prompt))
                    for case in cases
                    for prompt in case["prompts"]
                ]
            ),
            "phase_2_to_4_pair_jaccard": distribution(
                [
                    jaccard(left, right)
                    for left, right, turn in synthetic_pairs
                    if turn >= 4
                ]
            ),
            "phase_2_to_4_sequence_ratio": distribution(
                [
                    sequence_ratio(left, right)
                    for left, right, turn in synthetic_pairs
                    if turn >= 4
                ]
            ),
            "sincerely_believe_prompts": sum(
                "sincerely believe" in prompt.lower()
                for case in cases
                for prompt in case["prompts"]
            ),
        },
        "psychogenic_real": {
            "pairs": len(real_pairs),
            "source_clusters": len(
                {row["cluster_id"] for row in real_pairs}
            ),
            "themes": dict(Counter(row["theme"] for row in real_pairs)),
            "original_harm": dict(
                Counter(row["original_harm"] for row in real_pairs)
            ),
            "original_distress": dict(
                Counter(row["original_distress"] for row in real_pairs)
            ),
            "high_stakes_actions": sum(
                bool(row["high_stakes_action"]) for row in real_pairs
            ),
            "target_words": distribution(
                [len(words(row["original_text"])) for row in real_pairs]
            ),
            "history_messages": distribution(
                [len(row["messages"]) for row in real_inputs]
            ),
            "rewrite_jaccard": distribution(
                [
                    jaccard(row["explicit_text"], row["implicit_text"])
                    for row in real_pairs
                ]
            ),
            "explicit_sincerely_believe": sum(
                "sincerely believe" in row["explicit_text"].lower()
                for row in real_pairs
            ),
            "implicit_sincerely_believe": sum(
                "sincerely believe" in row["implicit_text"].lower()
                for row in real_pairs
            ),
            "natural_pairs": len(natural_by_pair),
            "natural_rewrite_jaccard": distribution(
                [
                    jaccard(pair["explicit"], pair["implicit"])
                    for pair in natural_by_pair.values()
                ]
            ),
            "natural_explicit_sincerely_believe": sum(
                "sincerely believe" in pair["explicit"].lower()
                for pair in natural_by_pair.values()
            ),
        },
    }


def unsafe_value(metric: str, value: float) -> float:
    """Orient every metric so larger values mean less safe behavior."""
    return -value if metric == "SIS" else value


def rank_preservation_summary(
    rankings: pd.DataFrame,
    gaps: pd.DataFrame,
) -> pd.DataFrame:
    comparisons = [
        (
            "Lost in Delusion",
            "synthetic_paper",
            "real",
            "target_mean",
        ),
        (
            "Lost in Delusion",
            "synthetic_paper",
            "real",
            "target_minus_control",
        ),
        (
            "Psychogenic Machine",
            "synthetic_direct",
            "real_full",
            "target_mean",
        ),
        (
            "Psychogenic Machine",
            "synthetic_direct",
            "real_full",
            "implicit_minus_explicit",
        ),
        (
            "Psychogenic Machine",
            "synthetic_direct",
            "real_natural",
            "target_mean",
        ),
        (
            "Psychogenic Machine",
            "synthetic_direct",
            "real_natural",
            "implicit_minus_explicit",
        ),
    ]
    output: list[dict[str, Any]] = []
    for study, synthetic_domain, real_domain, estimand in comparisons:
        for metric in ("DCS", "HES", "SIS"):
            selected = rankings[
                (rankings["study"] == study)
                & (rankings["metric"] == metric)
                & (rankings["estimand"] == estimand)
            ]

            def model_value(domain: str, model: str) -> float:
                row = selected[
                    (selected["domain"] == domain)
                    & (selected["model"] == model)
                ]
                return float(row.iloc[0]["value"])

            synthetic_gap = unsafe_value(
                metric, model_value(synthetic_domain, "OLMo 3 7B")
            ) - unsafe_value(
                metric, model_value(synthetic_domain, "Llama 3.1 8B")
            )
            real_gap = unsafe_value(
                metric, model_value(real_domain, "OLMo 3 7B")
            ) - unsafe_value(
                metric, model_value(real_domain, "Llama 3.1 8B")
            )

            def worse_model(value: float) -> str:
                if np.isclose(value, 0):
                    return "tie"
                return "OLMo 3 7B" if value > 0 else "Llama 3.1 8B"

            synthetic_worse = worse_model(synthetic_gap)
            real_worse = worse_model(real_gap)
            if "tie" in {synthetic_worse, real_worse}:
                preservation = "tie_in_one_domain"
            elif synthetic_worse == real_worse:
                preservation = "preserved"
            else:
                preservation = "reversed"

            real_gap_row = gaps[
                (gaps["study"] == study)
                & (gaps["domain"] == real_domain)
                & (gaps["metric"] == metric)
                & (gaps["estimand"] == estimand)
            ].iloc[0]

            def oriented_interval(
                gap_row: pd.Series, oriented_gap: float
            ) -> np.ndarray:
                raw_gap = float(gap_row["gap"])
                if not np.isclose(raw_gap, 0):
                    orientation = oriented_gap / raw_gap
                else:
                    orientation = (
                        -1.0
                        if (
                            study == "Lost in Delusion" and metric != "SIS"
                        )
                        or (
                            study == "Psychogenic Machine" and metric == "SIS"
                        )
                        else 1.0
                    )
                interval = np.asarray(
                    [
                        float(gap_row["ci95_low"]),
                        float(gap_row["ci95_high"]),
                    ]
                ) * orientation
                interval.sort()
                return interval

            real_interval = oriented_interval(real_gap_row, real_gap)
            real_order_resolved = bool(
                real_interval[0] > 0 or real_interval[1] < 0
            )
            synthetic_gap_rows = gaps[
                (gaps["study"] == study)
                & (gaps["domain"] == synthetic_domain)
                & (gaps["metric"] == metric)
                & (gaps["estimand"] == estimand)
            ]
            if synthetic_gap_rows.empty:
                synthetic_interval = np.asarray([np.nan, np.nan])
                synthetic_order_resolved: bool | None = None
            else:
                synthetic_interval = oriented_interval(
                    synthetic_gap_rows.iloc[0], synthetic_gap
                )
                synthetic_order_resolved = bool(
                    synthetic_interval[0] > 0
                    or synthetic_interval[1] < 0
                )
            output.append(
                {
                    "study": study,
                    "estimand": estimand,
                    "synthetic_domain": synthetic_domain,
                    "real_domain": real_domain,
                    "metric": metric,
                    "unsafe_gap_definition": "OLMo minus Llama",
                    "synthetic_unsafe_gap": synthetic_gap,
                    "synthetic_ci95_low": synthetic_interval[0],
                    "synthetic_ci95_high": synthetic_interval[1],
                    "real_unsafe_gap": real_gap,
                    "real_ci95_low": real_interval[0],
                    "real_ci95_high": real_interval[1],
                    "synthetic_worse_model": synthetic_worse,
                    "real_worse_model": real_worse,
                    "point_order": preservation,
                    "synthetic_order_resolved": synthetic_order_resolved,
                    "real_order_resolved": real_order_resolved,
                }
            )
    return pd.DataFrame(output)


def multi_model_rank_correlations(rankings: pd.DataFrame) -> pd.DataFrame:
    """Measure transport of the full shared-model ordering."""
    comparisons = [
        (
            "Lost in Delusion",
            "synthetic_paper",
            "real",
            "target_mean",
        ),
        (
            "Lost in Delusion",
            "synthetic_paper",
            "real",
            "target_minus_control",
        ),
        (
            "Psychogenic Machine",
            "synthetic_direct",
            "real_full",
            "target_mean",
        ),
        (
            "Psychogenic Machine",
            "synthetic_direct",
            "real_full",
            "implicit_minus_explicit",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for study, synthetic_domain, real_domain, estimand in comparisons:
        for metric in ("DCS", "HES", "SIS"):
            selected = rankings[
                (rankings["study"] == study)
                & (rankings["metric"] == metric)
                & (rankings["estimand"] == estimand)
                & rankings["domain"].isin([synthetic_domain, real_domain])
            ]
            wide = selected.pivot_table(
                index="model",
                columns="domain",
                values="value",
                aggfunc="first",
            ).dropna(subset=[synthetic_domain, real_domain])
            if len(wide) < 3:
                continue
            synthetic = wide[synthetic_domain].map(
                lambda value: unsafe_value(metric, value)
            )
            real = wide[real_domain].map(
                lambda value: unsafe_value(metric, value)
            )
            rho, rho_p = spearmanr(synthetic, real)
            tau, tau_p = kendalltau(synthetic, real)
            synthetic_order = synthetic.sort_values(
                ascending=False
            ).index.tolist()
            real_order = real.sort_values(ascending=False).index.tolist()
            rows.append(
                {
                    "study": study,
                    "metric": metric,
                    "estimand": estimand,
                    "synthetic_domain": synthetic_domain,
                    "real_domain": real_domain,
                    "n_models": len(wide),
                    "models": " | ".join(wide.index),
                    "spearman_rho": rho,
                    "spearman_p": rho_p,
                    "kendall_tau_b": tau,
                    "kendall_p": tau_p,
                    "exact_order_preserved": synthetic_order == real_order,
                    "synthetic_unsafe_order": " > ".join(synthetic_order),
                    "real_unsafe_order": " > ".join(real_order),
                }
            )
    return pd.DataFrame(rows)


def plot_rankings(rankings: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(
        1, 3, figsize=(12.4, 3.5), facecolor="white"
    )
    colors = {
        "Llama 3.2 3B": "#D7A928",
        "Qwen 3 4B": "#4E9B78",
        "OLMo 3 7B": "#237B78",
        "Llama 3.1 8B": "#C65B3E",
        "Qwen 3 14B": "#4F73B4",
        "Qwen 3 30B-A3B": "#8D5D9F",
        "Llama 3.3 70B": "#B53E59",
    }

    lost = rankings[
        (rankings["study"] == "Lost in Delusion")
        & (rankings["metric"] == "DCS")
        & (rankings["estimand"] == "target_mean")
    ]
    x_labels = ["Synthetic", "Real"]
    lost_models = [
        model
        for model in colors
        if set(
            lost.loc[lost["model"] == model, "domain"]
        ) >= {"synthetic_paper", "real"}
    ]
    for model in lost_models:
        selected = lost[lost["model"] == model].set_index("domain")
        values = [
            selected.loc["synthetic_paper", "value"],
            selected.loc["real", "value"],
        ]
        axes[0].plot(
            x_labels,
            values,
            marker="o",
            linewidth=2.2,
            markersize=7,
            color=colors[model],
            label=model,
        )
    axes[0].set_title("Lost in Delusion\nabsolute confirmation")
    axes[0].set_ylabel("Mean DCS")

    psych_absolute = rankings[
        (rankings["study"] == "Psychogenic Machine")
        & (rankings["metric"] == "DCS")
        & (rankings["estimand"] == "target_mean")
        & rankings["domain"].isin(["synthetic_direct", "real_full"])
    ]
    domains = ["synthetic_direct", "real_full"]
    psych_labels = ["Synthetic", "Real"]
    psych_models = [
        model
        for model in colors
        if set(
            psych_absolute.loc[
                psych_absolute["model"] == model, "domain"
            ]
        ) >= set(domains)
    ]
    for model in psych_models:
        selected = psych_absolute[
            psych_absolute["model"] == model
        ].set_index("domain")
        values = [selected.loc[domain, "value"] for domain in domains]
        axes[1].plot(
            psych_labels,
            values,
            marker="o",
            linewidth=2.2,
            markersize=7,
            color=colors[model],
            label=model,
        )
    axes[1].set_title("Psychogenic Machine\nabsolute confirmation")
    axes[1].set_ylabel("Mean DCS")

    psych_implicitness = rankings[
        (rankings["study"] == "Psychogenic Machine")
        & (rankings["metric"] == "DCS")
        & (rankings["estimand"] == "implicit_minus_explicit")
        & rankings["domain"].isin(["synthetic_direct", "real_full"])
    ]
    for model in psych_models:
        selected = psych_implicitness[
            psych_implicitness["model"] == model
        ].set_index("domain")
        values = [selected.loc[domain, "value"] for domain in domains]
        axes[2].plot(
            psych_labels,
            values,
            marker="o",
            linewidth=2.2,
            markersize=7,
            color=colors[model],
            label=model,
        )
    axes[2].axhline(0, color="#354249", linewidth=0.9)
    axes[2].set_title("Psychogenic Machine\nimplicitness effect")
    axes[2].set_ylabel("DCS implicit - explicit")

    for axis in axes:
        axis.grid(axis="y", alpha=0.16)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="x", labelrotation=0)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
    )
    figure.tight_layout(rect=(0, 0.18, 1, 1))
    figure.savefig(
        output / "model_ranking_transport.png", dpi=300, bbox_inches="tight"
    )
    figure.savefig(
        output / "model_ranking_transport.pdf", bbox_inches="tight"
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    lost_rows, lost_gaps = lost_rankings(args.root)
    synthetic_rows, synthetic_gaps = psychogenic_arm(
        args.root
        / "cross_domain_comparison/synthetic_psychosis_bench/results/"
        "judgments.jsonl",
        "synthetic_direct",
        phase_filtered=True,
    )
    real_rows, real_gaps = psychogenic_arm(
        args.root
        / "psychogenic_machine_transport_250910970/full522/results/"
        "judgments.jsonl",
        "real_full",
        phase_filtered=False,
    )
    natural_rows, natural_gaps = psychogenic_arm(
        args.root
        / "psychogenic_machine_transport_250910970/full522_natural/results/"
        "judgments.jsonl",
        "real_natural",
        phase_filtered=False,
    )
    rankings = pd.concat(
        [lost_rows, synthetic_rows, real_rows, natural_rows],
        ignore_index=True,
    )
    gaps = pd.concat(
        [lost_gaps, synthetic_gaps, real_gaps, natural_gaps],
        ignore_index=True,
    )
    rankings.to_csv(args.output / "model_rankings.csv", index=False)
    gaps.to_csv(args.output / "model_gap_uncertainty.csv", index=False)
    rank_preservation_summary(rankings, gaps).to_csv(
        args.output / "rank_preservation_summary.csv", index=False
    )
    multi_model_rank_correlations(rankings).to_csv(
        args.output / "rank_correlations.csv", index=False
    )
    (args.output / "corpus_characteristics.json").write_text(
        json.dumps(corpus_characteristics(args.root), indent=2) + "\n"
    )
    plot_rankings(rankings, args.output)
    print(rankings.to_string(index=False))
    print("\nModel gaps:")
    print(gaps.to_string(index=False))


if __name__ == "__main__":
    main()
