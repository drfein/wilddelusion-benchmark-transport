from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Select an outcome-blind stratified recoding sample.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    labels = read_jsonl(args.labels).drop_duplicates("item_id", keep="last")
    labels = labels[labels["error"].isna()] if "error" in labels else labels
    rng = np.random.default_rng(args.seed)
    selected: list[str] = []
    per_level = max(1, args.n // 5)
    for level in range(5):
        candidates = labels.loc[labels["delusion_content"] == level, "item_id"].to_numpy()
        take = min(per_level, len(candidates))
        if take:
            selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    remaining = args.n - len(selected)
    if remaining > 0:
        candidates = labels.loc[~labels["item_id"].isin(selected), "item_id"].to_numpy()
        selected.extend(rng.choice(candidates, size=min(remaining, len(candidates)), replace=False).tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": len(selected), "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
