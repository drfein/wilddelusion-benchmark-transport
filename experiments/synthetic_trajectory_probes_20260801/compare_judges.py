from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    keys = ["trajectory_id", "turn_number"]
    primary = read_jsonl(args.primary)[keys + ["score"]].rename(
        columns={"score": "primary_score"}
    )
    secondary = read_jsonl(args.secondary)[keys + ["score"]].rename(
        columns={"score": "secondary_score"}
    )
    joined = primary.merge(secondary, on=keys, validate="one_to_one")
    if len(joined) != len(primary) or len(joined) != len(secondary):
        raise ValueError("Judge files do not have identical row keys")
    primary_binary = (joined["primary_score"] == 2).astype(int)
    secondary_binary = (joined["secondary_score"] == 2).astype(int)
    summary = {
        "n": len(joined),
        "three_class_exact_agreement": float(
            (joined["primary_score"] == joined["secondary_score"]).mean()
        ),
        "three_class_cohen_kappa": float(
            cohen_kappa_score(joined["primary_score"], joined["secondary_score"])
        ),
        "endorsement_exact_agreement": float((primary_binary == secondary_binary).mean()),
        "endorsement_cohen_kappa": float(
            cohen_kappa_score(primary_binary, secondary_binary)
        ),
        "primary_endorsement_rate": float(primary_binary.mean()),
        "secondary_endorsement_rate": float(secondary_binary.mean()),
        "confusion_primary_rows_secondary_columns": pd.crosstab(
            joined["primary_score"], joined["secondary_score"]
        ).to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
