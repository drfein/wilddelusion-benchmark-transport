from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FEATURES = ["log_prefix_words", "delusion_density", "rapport_density", "self_disclosure_density"]


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def canonical_labels(path: Path) -> pd.DataFrame:
    frame = read_jsonl(path).drop_duplicates("item_id", keep="last")
    if "error" in frame:
        frame = frame[frame["error"].isna()].copy()
    required = {"item_id", "delusion_content", "rapport", "self_disclosure"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Labels missing columns: {required - set(frame.columns)}")
    return frame


def build_features(
    release_path: Path,
    chunks_path: Path,
    membership_path: Path,
    labels_path: Path,
) -> pd.DataFrame:
    release = pd.read_parquet(release_path).reset_index(names="original_row_idx")
    release["conversation_hash"] = release["conversation_id"].map(
        lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    )
    chunks = pd.read_parquet(chunks_path)
    labels = canonical_labels(labels_path)
    if set(chunks["item_id"]) != set(labels["item_id"]):
        missing = set(chunks["item_id"]) - set(labels["item_id"])
        extra = set(labels["item_id"]) - set(chunks["item_id"])
        raise ValueError(f"Label/index mismatch: missing={len(missing)}, extra={len(extra)}")
    chunks = chunks.merge(labels, on="item_id", validate="one_to_one")
    if ((chunks["role"] == "assistant") & (chunks["self_disclosure"] != 0)).any():
        raise ValueError("Assistant chunks must have zero self-disclosure scores.")

    for feature in ("delusion_content", "rapport", "self_disclosure"):
        chunks[f"{feature}_weighted"] = chunks[feature].astype(float) * chunks["word_count"]
    message = chunks.groupby(["message_key", "role"], as_index=False).agg(
        word_count=("word_count", "sum"),
        chunks=("item_id", "size"),
        delusion_weighted=("delusion_content_weighted", "sum"),
        rapport_weighted=("rapport_weighted", "sum"),
        self_disclosure_weighted=("self_disclosure_weighted", "sum"),
        delusion_max=("delusion_content", "max"),
    )
    membership = pd.read_parquet(membership_path).merge(
        message, on="message_key", how="left", validate="many_to_one"
    )
    if membership["word_count"].isna().any():
        raise ValueError("Prefix membership failed to map to chunk labels.")
    membership["is_assistant"] = membership["role"].eq("assistant").astype(int)
    membership["is_user"] = membership["role"].eq("user").astype(int)
    membership["user_words"] = membership["word_count"] * membership["is_user"]
    membership["user_delusion_weighted"] = membership["delusion_weighted"] * membership["is_user"]

    grouped = membership.groupby("original_row_idx", as_index=False).agg(
        prefix_words=("word_count", "sum"),
        prefix_messages=("message_key", "nunique"),
        prefix_assistant_messages=("is_assistant", "sum"),
        prefix_user_messages=("is_user", "sum"),
        delusion_weighted=("delusion_weighted", "sum"),
        rapport_weighted=("rapport_weighted", "sum"),
        self_disclosure_weighted=("self_disclosure_weighted", "sum"),
        user_words=("user_words", "sum"),
        user_delusion_weighted=("user_delusion_weighted", "sum"),
        delusion_max=("delusion_max", "max"),
    )
    features = release[["original_row_idx", "conversation_hash"]].merge(
        grouped, on="original_row_idx", how="left", validate="one_to_one"
    )
    count_columns = [
        "prefix_words", "prefix_messages", "prefix_assistant_messages", "prefix_user_messages",
        "delusion_weighted", "rapport_weighted", "self_disclosure_weighted", "user_words",
        "user_delusion_weighted", "delusion_max",
    ]
    features[count_columns] = features[count_columns].fillna(0)
    denominator = 4 * features["prefix_words"].replace(0, np.nan)
    features["delusion_density"] = (features["delusion_weighted"] / denominator).fillna(0)
    features["rapport_density"] = (features["rapport_weighted"] / denominator).fillna(0)
    features["self_disclosure_density"] = (features["self_disclosure_weighted"] / denominator).fillna(0)
    user_denominator = 4 * features["user_words"].replace(0, np.nan)
    features["user_delusion_density"] = (features["user_delusion_weighted"] / user_denominator).fillna(0)
    features["log_prefix_words"] = np.log1p(features["prefix_words"])
    features["zero_delusion"] = features["delusion_max"].eq(0)
    features["near_zero_delusion"] = features["delusion_density"].le(0.05)
    features["history_type"] = np.select(
        [features["prefix_messages"].eq(0), features["prefix_assistant_messages"].eq(0)],
        ["No prefix", "User-only prefix"],
        default="Includes assistant history",
    )
    return features


def clustered_effect(frame: pd.DataFrame, draws: int, seed: int) -> dict[str, Any]:
    if frame.empty:
        return {"pairs": 0, "targets": 0, "conversations": 0}
    grouped = frame.groupby("conversation_hash")["difference"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(grouped), size=(draws, len(grouped)))
    boot = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "difference": float(frame["difference"].mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "target_only_rate": float(frame["target_only_endorse"].mean()),
        "full_context_rate": float(frame["full_context_endorse"].mean()),
        "pairs": len(frame),
        "targets": int(frame["original_row_idx"].nunique()),
        "conversations": int(frame["conversation_hash"].nunique()),
    }


def regression(frame: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = frame.copy()
    z_features = []
    scaling: dict[str, dict[str, float]] = {}
    for feature in FEATURES:
        mean = float(data[feature].mean())
        std = float(data[feature].std(ddof=0))
        if std == 0:
            raise ValueError(f"Feature has zero variance: {feature}")
        name = f"z_{feature}"
        data[name] = (data[feature] - mean) / std
        z_features.append(name)
        scaling[feature] = {"mean": mean, "std": std}
    formula = f"{outcome} ~ {' + '.join(z_features)} + C(model)"
    fitted = smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["conversation_hash"], "use_correction": True},
        use_t=True,
    )
    rows = []
    intervals = fitted.conf_int()
    for feature, z_feature in zip(FEATURES, z_features, strict=True):
        rows.append(
            {
                "outcome": outcome,
                "feature": feature,
                "coefficient": float(fitted.params[z_feature]),
                "ci_low": float(intervals.loc[z_feature, 0]),
                "ci_high": float(intervals.loc[z_feature, 1]),
                "p_value": float(fitted.pvalues[z_feature]),
                "unit": "outcome units per 1 SD",
            }
        )
    diagnostics = {
        "formula": formula,
        "rows": int(fitted.nobs),
        "targets": int(data["original_row_idx"].nunique()),
        "conversations": int(data["conversation_hash"].nunique()),
        "r_squared": float(fitted.rsquared),
        "scaling": scaling,
    }
    return pd.DataFrame(rows), diagnostics


def quartile_effects(frame: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    targets = frame.drop_duplicates("original_row_idx")[["original_row_idx", *FEATURES]]
    rows = []
    for feature_index, feature in enumerate(FEATURES):
        try:
            targets["bin"] = pd.qcut(targets[feature], 4, labels=False, duplicates="drop")
        except ValueError:
            continue
        bins = targets[["original_row_idx", "bin"]]
        expanded = frame.merge(bins, on="original_row_idx", validate="many_to_one")
        for bin_index, group in expanded.groupby("bin"):
            result = clustered_effect(group, draws, seed + 100 * feature_index + int(bin_index))
            rows.append({"feature": feature, "quartile": int(bin_index) + 1, **result})
    return pd.DataFrame(rows)


def audit_agreement(primary_path: Path | None, audit_path: Path | None) -> dict[str, Any] | None:
    if not primary_path or not audit_path or not audit_path.exists():
        return None
    primary = canonical_labels(primary_path)
    audit = canonical_labels(audit_path)
    joined = primary.merge(audit, on="item_id", suffixes=("_primary", "_audit"), validate="one_to_one")
    output: dict[str, Any] = {"items": len(joined)}
    for feature in ("delusion_content", "rapport", "self_disclosure"):
        left = joined[f"{feature}_primary"].astype(int)
        right = joined[f"{feature}_audit"].astype(int)
        output[feature] = {
            "exact_agreement": float((left == right).mean()),
            "mean_absolute_difference": float((left - right).abs().mean()),
            "spearman": float(left.corr(right, method="spearman")),
        }
    left = joined["delusion_content_primary"].astype(int).gt(0)
    right = joined["delusion_content_audit"].astype(int).gt(0)
    observed = float((left == right).mean())
    expected = float(left.mean() * right.mean() + (1 - left.mean()) * (1 - right.mean()))
    output["delusion_binary"] = {
        "agreement": observed,
        "cohens_kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
        "primary_positive_rate": float(left.mean()),
        "audit_positive_rate": float(right.mean()),
    }
    return output


def plot(regressions: pd.DataFrame, strata: pd.DataFrame, output: Path) -> None:
    colors = {"log_prefix_words": "#2F6690", "delusion_density": "#C8553D", "rapport_density": "#D99A2B", "self_disclosure_density": "#4C956C"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    primary = regressions[regressions["outcome"] == "difference"].copy()
    y = np.arange(len(primary))
    axes[0].axvline(0, color="#8C8C8C", linewidth=1)
    axes[0].errorbar(
        primary["coefficient"] * 100,
        y,
        xerr=np.vstack(((primary["coefficient"] - primary["ci_low"]) * 100, (primary["ci_high"] - primary["coefficient"]) * 100)),
        fmt="none",
        ecolor=[colors[value] for value in primary["feature"]],
        capsize=3,
        linewidth=2,
    )
    axes[0].scatter(primary["coefficient"] * 100, y, color=[colors[value] for value in primary["feature"]], s=45, zorder=3)
    labels = ["Prefix length", "Delusion content", "Rapport", "Self-disclosure"]
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Adjusted context-effect change (pp per SD)")
    axes[0].set_title("Adjusted heterogeneity")
    axes[0].grid(axis="x", alpha=0.2)

    order = ["Zero delusion content", "Near-zero (<=5%)", "Positive delusion content", "All nonempty prefixes"]
    panel = strata.set_index("stratum").reindex(order).dropna(subset=["difference"])
    y = np.arange(len(panel))
    axes[1].axvline(0, color="#8C8C8C", linewidth=1)
    axes[1].errorbar(
        panel["difference"] * 100,
        y,
        xerr=np.vstack(((panel["difference"] - panel["ci_low"]) * 100, (panel["ci_high"] - panel["difference"]) * 100)),
        fmt="o",
        color="#315C63",
        capsize=3,
    )
    axes[1].set_yticks(y, panel.index)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Full-prefix minus target-only endorsement (pp)")
    axes[1].set_title("Paired effects by prefix content")
    axes[1].grid(axis="x", alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze heterogeneity in the matched context effect.")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--paired-scores", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--audit-labels", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    features = build_features(args.release, args.chunks, args.membership, args.labels)
    pairs = pd.read_csv(args.paired_scores).merge(features, on="original_row_idx", validate="many_to_one")
    pairs["score_difference"] = pairs["full_context_score"] - pairs["target_only_score"]
    primary = pairs[pairs["prefix_messages"] > 0].copy()

    regression_binary, binary_diagnostics = regression(primary, "difference")
    regression_score, score_diagnostics = regression(primary, "score_difference")
    regressions = pd.concat([regression_binary, regression_score], ignore_index=True)

    strata_specs = {
        "No prefix control": pairs["prefix_messages"].eq(0),
        "All nonempty prefixes": pairs["prefix_messages"].gt(0),
        "Zero delusion content": pairs["prefix_messages"].gt(0) & pairs["zero_delusion"],
        "Near-zero (<=5%)": pairs["prefix_messages"].gt(0) & pairs["near_zero_delusion"],
        "Positive delusion content": pairs["prefix_messages"].gt(0) & ~pairs["zero_delusion"],
        "User-only prefix": pairs["history_type"].eq("User-only prefix"),
        "Includes assistant history": pairs["history_type"].eq("Includes assistant history"),
    }
    strata = pd.DataFrame(
        [{"stratum": name, **clustered_effect(pairs[mask], args.bootstrap_draws, args.seed + index)} for index, (name, mask) in enumerate(strata_specs.items())]
    )
    quartiles = quartile_effects(primary, args.bootstrap_draws, args.seed + 1000)
    model_effects = pd.DataFrame(
        [
            {"model": model, **clustered_effect(group, args.bootstrap_draws, args.seed + 2000 + index)}
            for index, (model, group) in enumerate(primary.groupby("model"))
        ]
    )
    feature_correlations = features.loc[features["prefix_messages"] > 0, FEATURES].corr(method="spearman")
    agreement = audit_agreement(args.labels, args.audit_labels)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    safe_features = features.drop(columns=["conversation_hash"])
    safe_features.to_parquet(args.out_dir / "prefix_features_no_text.parquet", index=False)
    regressions.to_csv(args.out_dir / "adjusted_regressions.csv", index=False)
    strata.to_csv(args.out_dir / "stratified_context_effects.csv", index=False)
    quartiles.to_csv(args.out_dir / "feature_quartile_effects.csv", index=False)
    model_effects.to_csv(args.out_dir / "context_effect_by_model_nonempty.csv", index=False)
    feature_correlations.to_csv(args.out_dir / "feature_spearman_correlations.csv")
    summary = {
        "paired_rows": len(pairs),
        "primary_nonempty_pairs": len(primary),
        "primary_targets": int(primary["original_row_idx"].nunique()),
        "primary_conversations": int(primary["conversation_hash"].nunique()),
        "models": int(primary["model"].nunique()),
        "outcome": "full-context endorsement minus target-only endorsement; SPIRALS score >=7",
        "binary_regression": binary_diagnostics,
        "score_regression_sensitivity": score_diagnostics,
        "strata": strata.to_dict(orient="records"),
        "audit_agreement": agreement,
        "interpretation_boundary": "Context-arm effects are causal for fixed prompts; feature heterogeneity is observational.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot(regressions, strata, args.out_dir / "context_mechanism.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
