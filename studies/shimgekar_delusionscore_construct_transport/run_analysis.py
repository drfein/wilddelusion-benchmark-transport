#!/usr/bin/env python3
"""Run the paper-style MiniLM classifier and real-trajectory transport tests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import torch
from scipy.special import expit
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from design import MODEL_NAME, MODEL_REVISION, N_BOOT, N_FOLDS, SEED, fold_for_group


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def cluster_bootstrap(values: pd.DataFrame, value_col: str, cluster_col: str) -> dict:
    grouped = values.groupby(cluster_col)[value_col].mean().dropna().to_numpy()
    rng = np.random.default_rng(SEED)
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        means[i] = rng.choice(grouped, size=len(grouped), replace=True).mean()
    return {
        "estimate": float(grouped.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "n_clusters": int(len(grouped)),
    }


def classifier_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p >= 0.5
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
    }


def cluster_bootstrap_auc(
    positive: pd.DataFrame, negative: pd.DataFrame, n_boot: int = N_BOOT
) -> list[float]:
    rng = np.random.default_rng(SEED)
    pos_groups = [g.score.to_numpy() for _, g in positive.groupby("conversation_id")]
    neg_groups = [g.score.to_numpy() for _, g in negative.groupby("conversation_id")]
    values = []
    for _ in range(n_boot):
        pos = np.concatenate([pos_groups[i] for i in rng.integers(0, len(pos_groups), len(pos_groups))])
        neg = np.concatenate([neg_groups[i] for i in rng.integers(0, len(neg_groups), len(neg_groups))])
        values.append(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def external_control_metrics(positive: pd.DataFrame, controls: pd.DataFrame) -> dict:
    y = np.r_[np.ones(len(positive)), np.zeros(len(controls))]
    p = np.r_[positive.score.to_numpy(), controls.score.to_numpy()]
    result = classifier_metrics(y, p)
    result["auc_cluster_bootstrap_ci95"] = cluster_bootstrap_auc(positive, controls)
    result["false_positive_rate_at_0_5"] = float((controls.score >= 0.5).mean())
    result["by_exclusion"] = {
        name: {
            "n": int(len(group)),
            "mean_score": float(group.score.mean()),
            "false_positive_rate_at_0_5": float((group.score >= 0.5).mean()),
            "auc_vs_real_delusion_endpoints": float(
                roc_auc_score(
                    np.r_[np.ones(len(positive)), np.zeros(len(group))],
                    np.r_[positive.score.to_numpy(), group.score.to_numpy()],
                )
            ),
        }
        for name, group in controls.groupby("exclusion")
    }
    within_source = {}
    common_sources = sorted(set(positive.source) & set(controls.source))
    for source in common_sources:
        pos = positive[positive.source == source]
        neg = controls[controls.source == source]
        if len(pos) and len(neg):
            auc = roc_auc_score(
                np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos.score, neg.score]
            )
            within_source[source] = {"positive_n": len(pos), "negative_n": len(neg), "auc": float(auc)}
    result["within_source_auc"] = within_source
    return result


def fit_logistic(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> LogisticRegression:
    model = LogisticRegression(C=c, max_iter=5000, random_state=SEED, solver="lbfgs")
    return model.fit(x, y)


def paired_effect(endpoint_df: pd.DataFrame) -> dict:
    wide = endpoint_df.pivot(index="pair_id", columns="condition", values="score")
    wide["difference"] = wide["delusion"] - wide["grounded_control"]
    joined = endpoint_df[["pair_id", "conversation_id"]].drop_duplicates("pair_id").merge(
        wide["difference"], left_on="pair_id", right_index=True
    )
    result = cluster_bootstrap(joined, "difference", "conversation_id")
    result["n_pairs"] = int(len(wide))
    result["positive_fraction"] = float((wide["difference"] > 0).mean())
    return result


def trajectory_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for pair_id, group in df.sort_values("user_turn_index").groupby("pair_id"):
        group = group.sort_values("user_turn_index")
        target = group[group["is_target"]]
        if target.empty:
            continue
        prior = group[~group["is_target"]]
        all_slope = np.nan
        prior_slope = np.nan
        if len(group) >= 2:
            x = np.linspace(0, 1, len(group))
            all_slope = np.polyfit(x, group["score"], 1)[0]
        if len(prior) >= 2:
            x = np.linspace(0, 1, len(prior))
            prior_slope = np.polyfit(x, prior["score"], 1)[0]
        rows.append(
            {
                "pair_id": pair_id,
                "conversation_id": group.iloc[0]["conversation_id"],
                "source": group.iloc[0]["source"],
                "theme": group.iloc[0]["theme"],
                "n_user_turns": len(group),
                "all_slope": all_slope,
                "prior_slope": prior_slope,
                "first_to_target": float(target.iloc[-1]["score"] - group.iloc[0]["score"]),
                "target_jump": float(target.iloc[-1]["score"] - prior.iloc[-1]["score"]) if len(prior) else np.nan,
                "first_to_last_prior": float(prior.iloc[-1]["score"] - prior.iloc[0]["score"])
                if len(prior) >= 2
                else np.nan,
            }
        )
    summaries = pd.DataFrame(rows)
    metrics = {
        column: cluster_bootstrap(summaries.dropna(subset=[column]), column, "conversation_id")
        for column in ["all_slope", "prior_slope", "first_to_target", "target_jump", "first_to_last_prior"]
    }
    metrics["minimum_history_sensitivity"] = {}
    for minimum in (3, 5, 10):
        subset = summaries[summaries.n_user_turns >= minimum]
        metrics["minimum_history_sensitivity"][str(minimum)] = {
            column: cluster_bootstrap(subset.dropna(subset=[column]), column, "conversation_id")
            for column in ["all_slope", "prior_slope", "target_jump", "first_to_last_prior"]
        }
    return summaries, metrics


def regression_results(df: pd.DataFrame) -> dict:
    work = df.copy()
    work["relative_position"] = np.where(
        work["n_user_turns"] > 1,
        work["user_turn_index"] / (work["n_user_turns"] - 1),
        1.0,
    )
    work["log_chars"] = np.log1p(work["text"].str.len())
    results = {}
    formulas = {
        "all_turns_unadjusted": "score ~ relative_position",
        "all_turns_adjusted": "score ~ relative_position + log_chars + C(source) + C(theme)",
        "all_turns_conversation_fe": "score ~ relative_position + log_chars + C(conversation_id)",
        "prior_only_unadjusted": "score ~ relative_position",
        "prior_only_adjusted": "score ~ relative_position + log_chars + C(source) + C(theme)",
        "prior_only_conversation_fe": "score ~ relative_position + log_chars + C(conversation_id)",
    }
    for name, formula in formulas.items():
        subset = work if name.startswith("all_turns") else work[~work["is_target"]]
        fitted = smf.ols(formula, data=subset).fit(
            cov_type="cluster", cov_kwds={"groups": subset["conversation_id"]}
        )
        results[name] = {
            "n_rows": int(fitted.nobs),
            "n_conversations": int(subset["conversation_id"].nunique()),
            "relative_position_beta": float(fitted.params["relative_position"]),
            "relative_position_se": float(fitted.bse["relative_position"]),
            "relative_position_p": float(fitted.pvalues["relative_position"]),
            "r_squared": float(fitted.rsquared),
        }
    return results


def plot_results(endpoint_df, natural_df, trajectory_df, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    palette = {"grounded_control": "#4C78A8", "delusion": "#E45756"}
    sns.violinplot(
        data=endpoint_df, x="condition", y="score", hue="condition", palette=palette,
        legend=False, inner="quart", cut=0, ax=axes[0]
    )
    axes[0].set(xlabel="Matched endpoint", ylabel="Classifier score", title="Held-out paired endpoints")

    natural_plot = pd.concat(
        [
            endpoint_df[endpoint_df.condition == "delusion"].assign(group="Real delusion endpoints"),
            natural_df.assign(group="Natural near-miss controls"),
        ],
        ignore_index=True,
    )
    sns.ecdfplot(data=natural_plot, x="score", hue="group", palette=["#E45756", "#54A24B"], ax=axes[1])
    axes[1].axvline(0.5, color="#444444", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Classifier score", ylabel="Cumulative share", title="External natural controls")

    work = trajectory_df.copy()
    work["relative_position"] = np.where(
        work.n_user_turns > 1, work.user_turn_index / (work.n_user_turns - 1), 1.0
    )
    work["position_bin"] = pd.cut(
        work.relative_position, bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.001], labels=False
    )
    grouped = work.groupby("position_bin", observed=True).agg(score=("score", "mean"), n=("score", "size")).reset_index()
    grouped["position"] = (grouped.position_bin + 0.5) / 5
    axes[2].plot(grouped.position, grouped.score, marker="o", color="#F58518", linewidth=2)
    axes[2].set(xlabel="Relative user-turn position", ylabel="Mean classifier score", title="Real conversation trajectories", xlim=(0, 1))
    sns.despine(fig=fig)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    endpoints = read_jsonl(args.input_dir / "endpoints.jsonl")
    trajectories = read_jsonl(args.input_dir / "trajectories.jsonl")
    natural = read_jsonl(args.input_dir / "natural_controls.jsonl")
    broad = read_jsonl(args.input_dir / "broad_controls.jsonl")

    # The paper's source corpus is English. Keep trajectory rows only for cases whose
    # target is English, but retain every prior turn in those cases to avoid cherry-picking.
    english_pairs = {
        row["pair_id"] for row in endpoints if row["condition"] == "delusion" and row["language"] == "ENGLISH"
    }
    endpoint_main = [row for row in endpoints if row["pair_id"] in english_pairs]
    trajectory_main = [row for row in trajectories if row["pair_id"] in english_pairs]
    natural_main = [row for row in natural if row["language"] == "ENGLISH"]
    broad_main = [row for row in broad if row["language"] == "ENGLISH"]

    all_rows = endpoint_main + trajectory_main + natural_main + broad_main
    unique_texts = sorted({row["text_hash"]: row["text"] for row in all_rows}.items())
    hashes = [item[0] for item in unique_texts]
    texts = [item[1] for item in unique_texts]
    model = SentenceTransformer(
        args.model,
        revision=args.model_revision,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embedding_by_hash = {h: embeddings[i] for i, h in enumerate(hashes)}

    endpoint_df = pd.DataFrame(endpoint_main)
    endpoint_df["fold"] = endpoint_df.conversation_id.map(fold_for_group)
    endpoint_df["score"] = np.nan
    trajectory_df = pd.DataFrame(trajectory_main)
    trajectory_df["fold"] = trajectory_df.conversation_id.map(fold_for_group)
    trajectory_df["score"] = np.nan
    natural_df = pd.DataFrame(natural_main)
    broad_df = pd.DataFrame(broad_main)
    natural_fold_scores = []
    broad_fold_scores = []
    fold_models = []

    for fold in range(N_FOLDS):
        train = endpoint_df[endpoint_df.fold != fold]
        test = endpoint_df[endpoint_df.fold == fold]
        x_train = np.stack([embedding_by_hash[h] for h in train.text_hash])
        x_test = np.stack([embedding_by_hash[h] for h in test.text_hash])
        clf = fit_logistic(x_train, train.label.to_numpy())
        fold_models.append(clf)
        endpoint_df.loc[test.index, "score"] = clf.predict_proba(x_test)[:, 1]

        trajectory_test = trajectory_df[trajectory_df.fold == fold]
        x_trajectory = np.stack([embedding_by_hash[h] for h in trajectory_test.text_hash])
        trajectory_df.loc[trajectory_test.index, "score"] = clf.predict_proba(x_trajectory)[:, 1]

        x_natural = np.stack([embedding_by_hash[h] for h in natural_df.text_hash])
        natural_fold_scores.append(clf.predict_proba(x_natural)[:, 1])
        x_broad = np.stack([embedding_by_hash[h] for h in broad_df.text_hash])
        broad_fold_scores.append(clf.predict_proba(x_broad)[:, 1])

    if endpoint_df.score.isna().any() or trajectory_df.score.isna().any():
        raise RuntimeError("Missing out-of-fold scores")
    natural_df["score"] = np.mean(natural_fold_scores, axis=0)
    broad_df["score"] = np.mean(broad_fold_scores, axis=0)

    endpoint_metrics = classifier_metrics(endpoint_df.label.to_numpy(), endpoint_df.score.to_numpy())
    pair_effect = paired_effect(endpoint_df)
    positive_oof = endpoint_df[endpoint_df.condition == "delusion"]
    natural_metrics = external_control_metrics(positive_oof, natural_df)
    broad_metrics = external_control_metrics(positive_oof, broad_df)

    trajectory_summary_df, trajectory_metrics = trajectory_summaries(trajectory_df)
    regressions = regression_results(trajectory_df)

    # Regularization sensitivity uses the same folds and embeddings.
    sensitivity = {}
    for c in (0.1, 1.0, 10.0):
        scores = np.full(len(endpoint_df), np.nan)
        for fold in range(N_FOLDS):
            train = endpoint_df[endpoint_df.fold != fold]
            test = endpoint_df[endpoint_df.fold == fold]
            clf = fit_logistic(
                np.stack([embedding_by_hash[h] for h in train.text_hash]), train.label.to_numpy(), c=c
            )
            scores[test.index] = clf.predict_proba(
                np.stack([embedding_by_hash[h] for h in test.text_hash])
            )[:, 1]
        sensitivity[str(c)] = classifier_metrics(endpoint_df.label.to_numpy(), scores)

    results = {
        "design": {
            "paper_construct": "MiniLM-L6-v2 embeddings plus regularized logistic regression",
            "exact_replication": False,
            "reason_not_exact": "The paper does not release its fitted classifier or 3,000 Reddit training posts.",
            "embedding_model": args.model,
            "embedding_revision": args.model_revision,
            "embedding_dimension": int(embeddings.shape[1]),
            "logistic_C": 1.0,
            "folds": N_FOLDS,
            "split_unit": "conversation_id",
            "language_rule": "Lingua top language is ENGLISH for the real delusion endpoint",
            "threshold": 0.5,
        },
        "cohort": {
            "endpoint_rows": int(len(endpoint_df)),
            "endpoint_pairs": int(endpoint_df.pair_id.nunique()),
            "endpoint_conversations": int(endpoint_df.conversation_id.nunique()),
            "trajectory_rows": int(len(trajectory_df)),
            "trajectory_cases": int(trajectory_df.pair_id.nunique()),
            "trajectory_conversations": int(trajectory_df.conversation_id.nunique()),
            "natural_controls": int(len(natural_df)),
            "broad_controls": int(len(broad_df)),
        },
        "held_out_endpoint_metrics": endpoint_metrics,
        "held_out_paired_effect": pair_effect,
        "natural_control_external_metrics": natural_metrics,
        "broad_control_external_metrics": broad_metrics,
        "trajectory_cluster_bootstrap": trajectory_metrics,
        "trajectory_cluster_robust_regressions": regressions,
        "regularization_sensitivity": sensitivity,
    }
    write_json(args.output_dir / "results.json", results)
    endpoint_df.to_csv(args.output_dir / "endpoint_scores.csv", index=False)
    natural_df.to_csv(args.output_dir / "natural_control_scores.csv", index=False)
    broad_df.to_csv(args.output_dir / "broad_control_scores.csv", index=False)
    trajectory_df.to_csv(args.output_dir / "trajectory_scores.csv", index=False)
    trajectory_summary_df.to_csv(args.output_dir / "trajectory_case_summaries.csv", index=False)
    plot_results(endpoint_df, natural_df, trajectory_df, args.output_dir / "transport_summary.png")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
