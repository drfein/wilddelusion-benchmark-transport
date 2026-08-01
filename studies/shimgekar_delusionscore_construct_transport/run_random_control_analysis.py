#!/usr/bin/env python3
"""Fit the paper-analogous broad-control classifier and test hard transport."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from design import MODEL_NAME, MODEL_REVISION, N_FOLDS, SEED, fold_for_group
from run_analysis import (
    classifier_metrics,
    external_control_metrics,
    regression_results,
    trajectory_summaries,
    write_json,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fit(x, y):
    return LogisticRegression(C=1.0, max_iter=5000, random_state=SEED).fit(x, y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--random-controls", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    endpoints = read_jsonl(args.input_dir / "endpoints.jsonl")
    positives = [r for r in endpoints if r["condition"] == "delusion" and r["language"] == "ENGLISH"]
    positive_pairs = {r["pair_id"] for r in positives}
    trajectories = [r for r in read_jsonl(args.input_dir / "trajectories.jsonl") if r["pair_id"] in positive_pairs]
    natural = [r for r in read_jsonl(args.input_dir / "natural_controls.jsonl") if r["language"] == "ENGLISH"]
    random_controls = read_jsonl(args.random_controls)
    for row in random_controls:
        row["label"] = 0
        row["condition"] = "random_real_control"
        row["text_hash"] = text_hash(row["text"])
    for row in positives:
        row["label"] = 1
    for row in natural:
        row["text_hash"] = text_hash(row["text"])

    all_rows = positives + random_controls + trajectories + natural
    unique = sorted({r["text_hash"]: r["text"] for r in all_rows}.items())
    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    embeddings = model.encode(
        [t for _, t in unique], batch_size=args.batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False
    )
    embedding_by_hash = {h: embeddings[i] for i, (h, _) in enumerate(unique)}

    train_df = pd.DataFrame(positives + random_controls)
    train_df["fold"] = train_df.conversation_id.map(fold_for_group)
    train_df["score"] = np.nan
    train_df["length_only_score"] = np.nan
    trajectory_df = pd.DataFrame(trajectories)
    trajectory_df["fold"] = trajectory_df.conversation_id.map(fold_for_group)
    trajectory_df["score"] = np.nan
    natural_df = pd.DataFrame(natural)
    natural_scores = []
    natural_length_scores = []

    for fold in range(N_FOLDS):
        train = train_df[train_df.fold != fold]
        test = train_df[train_df.fold == fold]
        clf = fit(
            np.stack([embedding_by_hash[h] for h in train.text_hash]), train.label.to_numpy()
        )
        length_clf = fit(
            np.log1p(train.text.str.len().to_numpy())[:, None], train.label.to_numpy()
        )
        train_df.loc[test.index, "score"] = clf.predict_proba(
            np.stack([embedding_by_hash[h] for h in test.text_hash])
        )[:, 1]
        train_df.loc[test.index, "length_only_score"] = length_clf.predict_proba(
            np.log1p(test.text.str.len().to_numpy())[:, None]
        )[:, 1]
        trajectory_test = trajectory_df[trajectory_df.fold == fold]
        trajectory_df.loc[trajectory_test.index, "score"] = clf.predict_proba(
            np.stack([embedding_by_hash[h] for h in trajectory_test.text_hash])
        )[:, 1]
        natural_scores.append(
            clf.predict_proba(np.stack([embedding_by_hash[h] for h in natural_df.text_hash]))[:, 1]
        )
        natural_length_scores.append(
            length_clf.predict_proba(np.log1p(natural_df.text.str.len().to_numpy())[:, None])[:, 1]
        )

    natural_df["score"] = np.mean(natural_scores, axis=0)
    natural_df["length_only_score"] = np.mean(natural_length_scores, axis=0)
    positive_oof = train_df[train_df.label == 1]
    trajectory_summary, trajectory_metrics = trajectory_summaries(trajectory_df)
    results = {
        "design": {
            "description": "Paper-analogous source-matched broad-control classifier",
            "random_controls_are_real": True,
            "random_control_sampling": "Deterministic reservoir sample over max(10,000, 100 times requested n) eligible English user messages per source",
            "hard_controls_reserved_from_training": True,
            "split_unit": "conversation_id",
        },
        "cohort": {
            "positive_endpoints": len(positive_oof),
            "random_real_controls": len(random_controls),
            "natural_hard_controls": len(natural_df),
        },
        "broad_control_oof_metrics": classifier_metrics(train_df.label.to_numpy(), train_df.score.to_numpy()),
        "length_only_broad_control_oof_metrics": classifier_metrics(
            train_df.label.to_numpy(), train_df.length_only_score.to_numpy()
        ),
        "natural_hard_control_transport": external_control_metrics(positive_oof, natural_df),
        "length_only_natural_hard_control_auc": classifier_metrics(
            np.r_[np.ones(len(positive_oof)), np.zeros(len(natural_df))],
            np.r_[
                positive_oof.length_only_score.to_numpy(),
                natural_df.length_only_score.to_numpy(),
            ],
        ),
        "trajectory_cluster_bootstrap": trajectory_metrics,
        "trajectory_cluster_robust_regressions": regression_results(trajectory_df),
    }
    write_json(args.output_dir / "results.json", results)
    train_df.to_csv(args.output_dir / "oof_scores.csv", index=False)
    natural_df.to_csv(args.output_dir / "natural_hard_control_scores.csv", index=False)
    trajectory_df.to_csv(args.output_dir / "trajectory_scores.csv", index=False)
    trajectory_summary.to_csv(args.output_dir / "trajectory_case_summaries.csv", index=False)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
