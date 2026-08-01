from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_table(responses: Path, judgments: Path, activation_dir: Path) -> pd.DataFrame:
    response_rows = [row for row in read_jsonl(responses) if not row.get("error")]
    judgment_rows = [
        row for row in read_jsonl(judgments) if row.get("score") in (0, 1, 2)
    ]
    judgment_by_key = {
        (row["trajectory_id"], int(row["turn_number"])): row for row in judgment_rows
    }
    records = []
    for row in response_rows:
        key = (row["trajectory_id"], int(row["turn_number"]))
        if key not in judgment_by_key:
            continue
        archive = np.load(activation_dir / row["activation_file"])
        layer_names = sorted(archive.files, key=lambda value: int(value.split("_")[1]))
        midpoint_name = layer_names[(len(layer_names) - 1) // 2]
        records.append(
            {
                **row,
                "dcs": int(judgment_by_key[key]["score"]),
                "endorses": int(judgment_by_key[key]["score"] == 2),
                "midpoint_layer": int(midpoint_name.split("_")[1]),
                "activation": archive[midpoint_name].astype(np.float32),
            }
        )
    frame = pd.DataFrame(records)
    if len(frame) != len(response_rows):
        raise ValueError(f"Matched {len(frame)} of {len(response_rows)} response rows")
    frame = frame.sort_values(["trajectory_id", "turn_number"]).reset_index(drop=True)
    grouped = frame.groupby("trajectory_id", sort=False)
    frame["prior_dcs"] = grouped["dcs"].shift(1).fillna(1.0)
    prior_count = grouped.cumcount()
    cumulative = grouped["endorses"].cumsum() - frame["endorses"]
    frame["prior_endorsement_rate"] = np.divide(
        cumulative,
        prior_count,
        out=np.zeros(len(frame), dtype=float),
        where=prior_count.to_numpy() > 0,
    )
    return frame


def add_horizon_target(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    targets = {
        (row.trajectory_id, int(row.turn_number)): int(row.endorses)
        for row in frame.itertuples()
    }
    subset = frame[frame["turn_number"] >= 3].copy()
    subset["target_turn"] = subset["turn_number"] + horizon
    subset["target"] = [
        targets.get((row.trajectory_id, int(row.target_turn)))
        for row in subset.itertuples()
    ]
    return subset[subset["target"].notna()].copy()


def activation_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_x = np.stack(train["activation"].to_numpy())
    test_x = np.stack(test["activation"].to_numpy())
    scaler = StandardScaler()
    return scaler.fit_transform(train_x), scaler.transform(test_x)


def schedule_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    def encode(frame: pd.DataFrame) -> np.ndarray:
        turn = frame["turn_number"].to_numpy(dtype=float) / 12.0
        explicit = (frame["condition"].str.lower() == "explicit").to_numpy(dtype=float)
        return np.column_stack([turn, turn**2, explicit, turn * explicit])

    return encode(train), encode(test)


def history_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    def encode(frame: pd.DataFrame) -> np.ndarray:
        turn = frame["turn_number"].to_numpy(dtype=float) / 12.0
        explicit = (frame["condition"].str.lower() == "explicit").to_numpy(dtype=float)
        return np.column_stack(
            [
                turn,
                turn**2,
                explicit,
                frame["prior_dcs"].to_numpy(dtype=float) / 2.0,
                frame["prior_endorsement_rate"].to_numpy(dtype=float),
            ]
        )

    return encode(train), encode(test)


def text_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=20_000,
        sublinear_tf=True,
    )
    return vectorizer.fit_transform(train["user_text"]), vectorizer.transform(test["user_text"])


def grouped_oof(
    frame: pd.DataFrame,
    feature_fn: Callable,
    *,
    c_value: float,
) -> np.ndarray:
    predictions = np.full(len(frame), np.nan)
    groups = frame["scenario_pair"].to_numpy()
    labels = frame["target"].to_numpy(dtype=int)
    for held_out in sorted(set(groups)):
        train_mask = groups != held_out
        test_mask = ~train_mask
        if len(np.unique(labels[train_mask])) != 2:
            raise ValueError(f"Training labels have one class with {held_out} held out")
        train = frame.loc[train_mask]
        test = frame.loc[test_mask]
        train_x, test_x = feature_fn(train, test)
        classifier = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2_000,
            solver="liblinear",
            random_state=20260801,
        )
        classifier.fit(train_x, labels[train_mask])
        predictions[test_mask] = classifier.predict_proba(test_x)[:, 1]
    if np.isnan(predictions).any():
        raise RuntimeError("Missing held-out predictions")
    return predictions


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, predictions)),
        "average_precision": float(average_precision_score(labels, predictions)),
        "brier": float(brier_score_loss(labels, predictions)),
    }


def cluster_bootstrap(
    frame: pd.DataFrame,
    prediction_columns: list[str],
    *,
    iterations: int = 2_000,
) -> dict:
    rng = np.random.default_rng(20260801)
    groups = sorted(frame["scenario_pair"].unique())
    draws = {column: [] for column in prediction_columns}
    draws["probe_minus_text"] = []
    draws["probe_minus_history"] = []
    for _ in range(iterations):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(frame["scenario_pair"].to_numpy() == group) for group in sampled]
        )
        labels = frame["target"].to_numpy(dtype=int)[indices]
        if len(np.unique(labels)) < 2:
            continue
        scores = {}
        for column in prediction_columns:
            score = roc_auc_score(labels, frame[column].to_numpy()[indices])
            draws[column].append(score)
            scores[column] = score
        draws["probe_minus_text"].append(scores["probe_prediction"] - scores["text_prediction"])
        draws["probe_minus_history"].append(
            scores["probe_prediction"] - scores["history_prediction"]
        )
    return {
        key: {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        }
        for key, values in draws.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--max-horizon", type=int, default=4)
    args = parser.parse_args()

    frame = load_table(args.responses, args.judgments, args.activation_dir)
    if frame["scenario_pair"].nunique() != 8:
        raise ValueError("Expected eight independent Psychosis-Bench scenario pairs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_oof = []
    summaries = []
    for horizon in range(args.max_horizon + 1):
        subset = add_horizon_target(frame, horizon)
        labels = subset["target"].to_numpy(dtype=int)
        subset["probe_prediction"] = grouped_oof(
            subset, activation_features, c_value=0.01
        )
        subset["schedule_prediction"] = grouped_oof(
            subset, schedule_features, c_value=1.0
        )
        subset["history_prediction"] = grouped_oof(
            subset, history_features, c_value=1.0
        )
        subset["text_prediction"] = grouped_oof(subset, text_features, c_value=1.0)
        interval = cluster_bootstrap(
            subset,
            [
                "probe_prediction",
                "schedule_prediction",
                "history_prediction",
                "text_prediction",
            ],
        )
        summary = {
            "model_key": args.model_key,
            "horizon": horizon,
            "midpoint_layer": int(subset["midpoint_layer"].iloc[0]),
            "scenario_pairs": int(subset["scenario_pair"].nunique()),
            "probe": metrics(labels, subset["probe_prediction"].to_numpy()),
            "schedule_baseline": metrics(
                labels, subset["schedule_prediction"].to_numpy()
            ),
            "history_baseline": metrics(
                labels, subset["history_prediction"].to_numpy()
            ),
            "user_text_baseline": metrics(labels, subset["text_prediction"].to_numpy()),
            "cluster_bootstrap_95": interval,
        }
        summaries.append(summary)
        output = subset.drop(columns=["activation"]).copy()
        output["horizon"] = horizon
        all_oof.append(output)

    pd.concat(all_oof, ignore_index=True).to_json(
        args.output_dir / "oof_predictions.jsonl", orient="records", lines=True
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
