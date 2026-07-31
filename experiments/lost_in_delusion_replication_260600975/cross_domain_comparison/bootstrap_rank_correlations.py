#!/usr/bin/env python3
"""Cluster-bootstrap multi-model ranking transport across synthetic and real data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from analyze_cross_domain import LOST_SYNTHETIC, MODEL_LABELS, dcs_ordinal


METRICS = ("DCS", "HES", "SIS")
QWEN_SCALE_MODELS = (
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-30B-A3B",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def scored_frame(path: Path, lost_protocol: bool) -> pd.DataFrame:
    frame = pd.DataFrame(read_jsonl(path))
    if "judge_error" in frame:
        frame = frame[frame["judge_error"].isna()].copy()
    if lost_protocol:
        frame["DCS"] = frame["DCS"].map(dcs_ordinal)
        frame["HES"] = frame["HES"].map(dcs_ordinal)
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric])
    return frame


def eligible_lost(
    frame: pd.DataFrame, metric: str, estimand: str
) -> pd.DataFrame:
    if estimand == "target_mean":
        selected = frame[frame["condition"] == "delusion"].copy()
        if metric != "DCS":
            selected = selected[selected["high_stakes_action"] == True]  # noqa: E712
        return selected
    if metric == "DCS":
        return frame.copy()
    return frame[frame["harm_pair_valid"] == True].copy()  # noqa: E712


def eligible_psych(
    frame: pd.DataFrame,
    metric: str,
    synthetic: bool,
) -> pd.DataFrame:
    selected = frame.copy()
    if synthetic:
        if "turn_number" not in selected:
            selected["turn_number"] = selected["generation_id"].str.extract(
                r":turn(\d+)$", expand=False
            ).astype(int)
        minimum_turn = 4 if metric == "DCS" else 7
        return selected[selected["turn_number"] >= minimum_turn]
    if metric != "DCS":
        return selected[selected["harm_pair_valid"] == True].copy()  # noqa: E712
    return selected


def units(
    frame: pd.DataFrame,
    metric: str,
    estimand: str,
    condition_names: tuple[str, str],
) -> pd.DataFrame:
    if estimand == "target_mean":
        return frame[["cluster_id", "model", metric]].rename(
            columns={metric: "value"}
        )
    first, second = condition_names
    paired = (
        frame.pivot_table(
            index=["pair_id", "cluster_id", "model"],
            columns="condition",
            values=metric,
            aggfunc="mean",
        )
        .dropna(subset=[first, second])
        .reset_index()
    )
    paired["value"] = paired[first] - paired[second]
    return paired[["cluster_id", "model", "value"]]


def cluster_tables(
    frame: pd.DataFrame, models: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    aggregate = (
        frame.groupby(["cluster_id", "model"])["value"]
        .agg(["sum", "count"])
        .reset_index()
    )
    sums = (
        aggregate.pivot(
            index="cluster_id", columns="model", values="sum"
        )
        .reindex(columns=models)
        .fillna(0.0)
    )
    counts = (
        aggregate.pivot(
            index="cluster_id", columns="model", values="count"
        )
        .reindex(index=sums.index, columns=models)
        .fillna(0.0)
    )
    if (counts.sum(axis=0) == 0).any():
        missing = counts.columns[counts.sum(axis=0) == 0].tolist()
        raise ValueError(f"Models lack eligible observations: {missing}")
    return (
        sums.to_numpy(dtype=float),
        counts.to_numpy(dtype=float),
        sums.index.astype(str).tolist(),
    )


def means(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return sums.sum(axis=0) / counts.sum(axis=0)


def bootstrap_means(
    sums: np.ndarray,
    counts: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    cluster_n = len(sums)
    output = np.empty((draws, sums.shape[1]), dtype=float)
    for draw in range(draws):
        selected = rng.integers(0, cluster_n, cluster_n)
        output[draw] = (
            sums[selected].sum(axis=0) / counts[selected].sum(axis=0)
        )
    return output


def oriented(metric: str, values: np.ndarray) -> np.ndarray:
    return -values if metric == "SIS" else values


def correlations(
    synthetic: np.ndarray,
    real: np.ndarray,
) -> tuple[float, float]:
    if np.ptp(synthetic) == 0 or np.ptp(real) == 0:
        return np.nan, np.nan
    return (
        float(spearmanr(synthetic, real).statistic),
        float(kendalltau(synthetic, real).statistic),
    )


def summarize(
    *,
    study: str,
    metric: str,
    estimand: str,
    models: list[str],
    synthetic_point: np.ndarray,
    real_point: np.ndarray,
    synthetic_draws: np.ndarray,
    real_draws: np.ndarray,
) -> dict[str, Any]:
    synthetic_point = oriented(metric, synthetic_point)
    real_point = oriented(metric, real_point)
    synthetic_draws = oriented(metric, synthetic_draws)
    real_draws = oriented(metric, real_draws)
    rho, tau = correlations(synthetic_point, real_point)
    draw_rho = np.empty(len(real_draws), dtype=float)
    draw_tau = np.empty(len(real_draws), dtype=float)
    exact_order = np.empty(len(real_draws), dtype=bool)
    for index, (left, right) in enumerate(
        zip(synthetic_draws, real_draws, strict=True)
    ):
        draw_rho[index], draw_tau[index] = correlations(left, right)
        exact_order[index] = np.array_equal(
            np.argsort(-left, kind="stable"),
            np.argsort(-right, kind="stable"),
        )
    finite_rho = draw_rho[np.isfinite(draw_rho)]
    finite_tau = draw_tau[np.isfinite(draw_tau)]
    synthetic_order = np.argsort(-synthetic_point, kind="stable")
    real_order = np.argsort(-real_point, kind="stable")
    return {
        "study": study,
        "metric": metric,
        "estimand": estimand,
        "n_models": len(models),
        "models": " | ".join(MODEL_LABELS.get(model, model) for model in models),
        "spearman_rho": rho,
        "spearman_ci95_low": np.quantile(finite_rho, 0.025),
        "spearman_ci95_high": np.quantile(finite_rho, 0.975),
        "kendall_tau_b": tau,
        "kendall_ci95_low": np.quantile(finite_tau, 0.025),
        "kendall_ci95_high": np.quantile(finite_tau, 0.975),
        "bootstrap_valid_draws": min(len(finite_rho), len(finite_tau)),
        "bootstrap_exact_order_probability": exact_order.mean(),
        "point_exact_order_preserved": np.array_equal(
            synthetic_order, real_order
        ),
        "synthetic_unsafe_order": " > ".join(
            MODEL_LABELS.get(models[index], models[index])
            for index in synthetic_order
        ),
        "real_unsafe_order": " > ".join(
            MODEL_LABELS.get(models[index], models[index])
            for index in real_order
        ),
    }


def scale_contrasts(
    *,
    study: str,
    domain: str,
    estimand: str,
    models: list[str],
    point: np.ndarray,
    draws: np.ndarray,
    cluster_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for smaller_index in range(len(models)):
        for larger_index in range(smaller_index + 1, len(models)):
            differences = (
                draws[:, smaller_index] - draws[:, larger_index]
            )
            rows.append(
                {
                    "study": study,
                    "domain": domain,
                    "metric": "DCS",
                    "estimand": estimand,
                    "smaller_model": MODEL_LABELS.get(
                        models[smaller_index], models[smaller_index]
                    ),
                    "larger_model": MODEL_LABELS.get(
                        models[larger_index], models[larger_index]
                    ),
                    "smaller_minus_larger": (
                        point[smaller_index] - point[larger_index]
                    ),
                    "cluster_bootstrap_ci95_low": np.quantile(
                        differences, 0.025
                    ),
                    "cluster_bootstrap_ci95_high": np.quantile(
                        differences, 0.975
                    ),
                    "bootstrap_probability_smaller_is_worse": (
                        differences > 0
                    ).mean(),
                    "cluster_n": cluster_n,
                }
            )
    return rows


def qwen_scale_results(
    lost: pd.DataFrame,
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    draws: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    models = [
        model
        for model in QWEN_SCALE_MODELS
        if model in set(lost["model"])
        and model in set(synthetic["model"])
        and model in set(real["model"])
    ]
    if len(models) < 2:
        return []

    rows: list[dict[str, Any]] = []
    for estimand in ("target_mean", "target_minus_control"):
        selected = eligible_lost(lost, "DCS", estimand)
        condition_names = (
            ("delusion", "grounded_control")
            if estimand == "target_minus_control"
            else ("", "")
        )
        unit_frame = units(selected, "DCS", estimand, condition_names)
        sums, counts, clusters = cluster_tables(unit_frame, models)
        point = means(sums, counts)
        bootstrap = bootstrap_means(sums, counts, draws, rng)
        rows.extend(
            scale_contrasts(
                study="Lost in Delusion",
                domain="real",
                estimand=estimand,
                models=models,
                point=point,
                draws=bootstrap,
                cluster_n=len(clusters),
            )
        )

    for domain, frame, is_synthetic in (
        ("synthetic_direct", synthetic, True),
        ("real", real, False),
    ):
        for estimand in ("target_mean", "implicit_minus_explicit"):
            selected = eligible_psych(frame, "DCS", is_synthetic)
            unit_frame = units(
                selected,
                "DCS",
                estimand,
                ("implicit", "explicit"),
            )
            sums, counts, clusters = cluster_tables(unit_frame, models)
            point = means(sums, counts)
            bootstrap = bootstrap_means(sums, counts, draws, rng)
            rows.extend(
                scale_contrasts(
                    study="Psychogenic Machine",
                    domain=domain,
                    estimand=estimand,
                    models=models,
                    point=point,
                    draws=bootstrap,
                    cluster_n=len(clusters),
                )
            )
    return rows


def lost_absolute_results(
    lost: pd.DataFrame,
    draws: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models = [
        model for model in LOST_SYNTHETIC if model in set(lost["model"])
    ]
    selected = eligible_lost(lost, "DCS", "target_mean")
    unit_frame = units(selected, "DCS", "target_mean", ("", ""))
    sums, counts, clusters = cluster_tables(unit_frame, models)
    point = means(sums, counts)
    bootstrap = bootstrap_means(sums, counts, draws, rng)

    model_rows: list[dict[str, Any]] = []
    for index, model in enumerate(models):
        model_rows.append(
            {
                "model": MODEL_LABELS.get(model, model),
                "synthetic_paper_mean": LOST_SYNTHETIC[model]["DCS"][
                    "target"
                ],
                "real_mean": point[index],
                "real_cluster_bootstrap_ci95_low": np.quantile(
                    bootstrap[:, index], 0.025
                ),
                "real_cluster_bootstrap_ci95_high": np.quantile(
                    bootstrap[:, index], 0.975
                ),
                "real_cluster_n": len(clusters),
            }
        )

    contrast_rows: list[dict[str, Any]] = []
    for first_index in range(len(models)):
        for second_index in range(first_index + 1, len(models)):
            differences = (
                bootstrap[:, first_index] - bootstrap[:, second_index]
            )
            first = models[first_index]
            second = models[second_index]
            synthetic_difference = (
                LOST_SYNTHETIC[first]["DCS"]["target"]
                - LOST_SYNTHETIC[second]["DCS"]["target"]
            )
            real_difference = point[first_index] - point[second_index]
            contrast_rows.append(
                {
                    "model_a": MODEL_LABELS.get(first, first),
                    "model_b": MODEL_LABELS.get(second, second),
                    "synthetic_paper_a_minus_b": synthetic_difference,
                    "real_a_minus_b": real_difference,
                    "real_cluster_bootstrap_ci95_low": np.quantile(
                        differences, 0.025
                    ),
                    "real_cluster_bootstrap_ci95_high": np.quantile(
                        differences, 0.975
                    ),
                    "real_probability_a_exceeds_b": (
                        differences > 0
                    ).mean(),
                    "point_order_preserved": (
                        np.sign(synthetic_difference)
                        == np.sign(real_difference)
                    ),
                    "real_cluster_n": len(clusters),
                }
            )
    return model_rows, contrast_rows


def lost_results(
    frame: pd.DataFrame,
    draws: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    available = set(frame["model"])
    models = [
        model for model in LOST_SYNTHETIC if model in available
    ]
    if len(models) < 3:
        return [], []
    results: list[dict[str, Any]] = []
    values: list[dict[str, Any]] = []
    for estimand in ("target_mean", "target_minus_control"):
        reference_key = "target" if estimand == "target_mean" else "effect"
        for metric in METRICS:
            selected = eligible_lost(frame, metric, estimand)
            condition_names = (
                ("delusion", "grounded_control")
                if estimand == "target_minus_control"
                else ("", "")
            )
            unit_frame = units(
                selected, metric, estimand, condition_names
            )
            sums, counts, clusters = cluster_tables(unit_frame, models)
            real_point = means(sums, counts)
            real_draws = bootstrap_means(sums, counts, draws, rng)
            synthetic_point = np.asarray(
                [
                    LOST_SYNTHETIC[model][metric][reference_key]
                    for model in models
                ],
                dtype=float,
            )
            synthetic_draws = np.broadcast_to(
                synthetic_point, (draws, len(models))
            )
            results.append(
                summarize(
                    study="Lost in Delusion",
                    metric=metric,
                    estimand=estimand,
                    models=models,
                    synthetic_point=synthetic_point,
                    real_point=real_point,
                    synthetic_draws=synthetic_draws,
                    real_draws=real_draws,
                )
                | {"synthetic_cluster_n": np.nan, "real_cluster_n": len(clusters)}
            )
            for index, model in enumerate(models):
                values.append(
                    {
                        "study": "Lost in Delusion",
                        "metric": metric,
                        "estimand": estimand,
                        "model": MODEL_LABELS.get(model, model),
                        "synthetic_value": synthetic_point[index],
                        "real_value": real_point[index],
                    }
                )
    return results, values


def psych_results(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    draws: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models = [
        model
        for model in MODEL_LABELS
        if model in set(synthetic["model"]) and model in set(real["model"])
    ]
    if len(models) < 3:
        return [], []
    results: list[dict[str, Any]] = []
    values: list[dict[str, Any]] = []
    for estimand in ("target_mean", "implicit_minus_explicit"):
        for metric in METRICS:
            synthetic_selected = eligible_psych(
                synthetic, metric, synthetic=True
            )
            real_selected = eligible_psych(real, metric, synthetic=False)
            synthetic_units = units(
                synthetic_selected,
                metric,
                estimand,
                ("implicit", "explicit"),
            )
            real_units = units(
                real_selected,
                metric,
                estimand,
                ("implicit", "explicit"),
            )
            synthetic_sums, synthetic_counts, synthetic_clusters = (
                cluster_tables(synthetic_units, models)
            )
            real_sums, real_counts, real_clusters = cluster_tables(
                real_units, models
            )
            synthetic_point = means(synthetic_sums, synthetic_counts)
            real_point = means(real_sums, real_counts)
            synthetic_draws = bootstrap_means(
                synthetic_sums, synthetic_counts, draws, rng
            )
            real_draws = bootstrap_means(
                real_sums, real_counts, draws, rng
            )
            results.append(
                summarize(
                    study="Psychogenic Machine",
                    metric=metric,
                    estimand=estimand,
                    models=models,
                    synthetic_point=synthetic_point,
                    real_point=real_point,
                    synthetic_draws=synthetic_draws,
                    real_draws=real_draws,
                )
                | {
                    "synthetic_cluster_n": len(synthetic_clusters),
                    "real_cluster_n": len(real_clusters),
                }
            )
            for index, model in enumerate(models):
                values.append(
                    {
                        "study": "Psychogenic Machine",
                        "metric": metric,
                        "estimand": estimand,
                        "model": MODEL_LABELS.get(model, model),
                        "synthetic_value": synthetic_point[index],
                        "real_value": real_point[index],
                    }
                )
    return results, values


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
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=260600975)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    lost = scored_frame(
        args.root / "full_history_522/results/judgments.jsonl",
        lost_protocol=True,
    )
    synthetic = scored_frame(
        args.root
        / "cross_domain_comparison/synthetic_psychosis_bench/results/"
        "judgments.jsonl",
        lost_protocol=False,
    )
    real = scored_frame(
        args.root
        / "psychogenic_machine_transport_250910970/full522/results/"
        "judgments.jsonl",
        lost_protocol=False,
    )
    lost_rows, lost_values = lost_results(lost, args.draws, rng)
    psych_rows, psych_values = psych_results(
        synthetic, real, args.draws, rng
    )
    scale_rows = qwen_scale_results(
        lost, synthetic, real, args.draws, rng
    )
    lost_absolute_means, lost_absolute_contrasts = lost_absolute_results(
        lost,
        args.draws,
        np.random.default_rng(args.seed + 1),
    )
    result = pd.DataFrame(lost_rows + psych_rows)
    values = pd.DataFrame(lost_values + psych_values)
    scale = pd.DataFrame(scale_rows)
    result.to_csv(
        args.output / "rank_correlations_cluster_bootstrap.csv", index=False
    )
    values.to_csv(
        args.output / "rank_correlation_model_values.csv", index=False
    )
    scale.to_csv(
        args.output / "qwen_scale_contrasts_cluster_bootstrap.csv",
        index=False,
    )
    pd.DataFrame(lost_absolute_means).to_csv(
        args.output / "lost_absolute_model_means_cluster_bootstrap.csv",
        index=False,
    )
    pd.DataFrame(lost_absolute_contrasts).to_csv(
        args.output / "lost_absolute_model_contrasts_cluster_bootstrap.csv",
        index=False,
    )
    print(result.to_string(index=False))
    print("\nQwen scale contrasts:")
    print(scale.to_string(index=False))


if __name__ == "__main__":
    main()
