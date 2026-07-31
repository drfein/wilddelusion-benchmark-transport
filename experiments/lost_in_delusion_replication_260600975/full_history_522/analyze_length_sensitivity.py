#!/usr/bin/env python3
"""Test whether paired DCS effects can be explained by text length."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import read_jsonl, write_manifest  # noqa: E402

from analyze_full_history import (  # noqa: E402
    cluster_bootstrap,
    cluster_randomization_p,
    ordinal_score,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--judgments",
        type=Path,
        default=Path("full_history_522/results/judgments.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("full_history_522/results/analysis_clustered"),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--permutation-draws", type=int, default=100_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generations = pd.DataFrame(
        row
        for path in args.generations
        for row in read_jsonl(path)
        if row.get("response") and not row.get("generation_error")
    )
    judgments = pd.DataFrame(
        row for row in read_jsonl(args.judgments) if not row.get("judge_error")
    )
    judgments["dcs_ordinal"] = judgments["DCS"].map(ordinal_score)
    frame = generations[
        [
            "generation_id",
            "model",
            "pair_id",
            "cluster_id",
            "condition",
            "input_token_count",
            "response",
        ]
    ].merge(
        judgments[["generation_id", "dcs_ordinal"]],
        on="generation_id",
        validate="one_to_one",
    )
    frame["response_chars"] = frame["response"].str.len()

    summaries: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for model, model_frame in frame.groupby("model", sort=False):
        metadata = (
            model_frame.groupby("pair_id", as_index=False)
            .agg(cluster_id=("cluster_id", "first"))
            .set_index("pair_id")
        )
        wide = model_frame.pivot_table(
            index="pair_id",
            columns="condition",
            values=["input_token_count", "response_chars", "dcs_ordinal"],
            aggfunc="first",
        ).dropna()
        wide.columns = ["_".join(column) for column in wide.columns]
        wide = wide.join(metadata).reset_index()
        wide["input_delta"] = (
            wide["input_token_count_delusion"]
            - wide["input_token_count_grounded_control"]
        )
        wide["input_ratio"] = (
            wide["input_token_count_grounded_control"]
            / wide["input_token_count_delusion"]
        )
        wide["response_delta"] = (
            wide["response_chars_delusion"]
            - wide["response_chars_grounded_control"]
        )
        wide["difference"] = (
            wide["dcs_ordinal_delusion"]
            - wide["dcs_ordinal_grounded_control"]
        )

        input_corr = spearmanr(wide["difference"], wide["input_delta"])
        response_corr = spearmanr(wide["difference"], wide["response_delta"])
        diagnostics.append(
            {
                "model": model,
                "paired_n": len(wide),
                "median_delusion_input_tokens": float(
                    wide["input_token_count_delusion"].median()
                ),
                "median_control_input_tokens": float(
                    wide["input_token_count_grounded_control"].median()
                ),
                "median_control_to_delusion_input_ratio": float(
                    wide["input_ratio"].median()
                ),
                "input_delta_spearman_rho": float(input_corr.statistic),
                "input_delta_spearman_p": float(input_corr.pvalue),
                "response_delta_spearman_rho": float(
                    response_corr.statistic
                ),
                "response_delta_spearman_p": float(response_corr.pvalue),
            }
        )

        for label, low_ratio, high_ratio in (
            ("all", 0.0, float("inf")),
            ("within_25_percent", 0.8, 1.25),
            ("within_10_percent", 0.9, 1.1),
        ):
            selected = wide[
                wide["input_ratio"].between(low_ratio, high_ratio)
            ].copy()
            low, high, equal_low, equal_high = cluster_bootstrap(
                selected, 260600975, args.bootstrap_draws
            )
            cluster_means = selected.groupby("cluster_id")["difference"].mean()
            summaries.append(
                {
                    "model": model,
                    "length_subset": label,
                    "paired_n": len(selected),
                    "cluster_n": selected["cluster_id"].nunique(),
                    "paired_dcs_difference": selected["difference"].mean(),
                    "cluster_bootstrap_ci95_low": low,
                    "cluster_bootstrap_ci95_high": high,
                    "cluster_randomization_p": cluster_randomization_p(
                        selected, 260600975, args.permutation_draws
                    ),
                    "cluster_equal_difference": cluster_means.mean(),
                    "cluster_equal_ci95_low": equal_low,
                    "cluster_equal_ci95_high": equal_high,
                }
            )

    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "length_sensitivity.csv", index=False)
    diagnostics_path = args.output_dir / "length_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(
        args.output_dir / "length_sensitivity.manifest.json",
        {
            "generations": [str(path) for path in args.generations],
            "judgments": str(args.judgments),
            "models": sorted(frame["model"].unique()),
            "bootstrap_draws": args.bootstrap_draws,
            "permutation_draws": args.permutation_draws,
            "analysis": "paired DCS sensitivity to input and response length",
        },
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
