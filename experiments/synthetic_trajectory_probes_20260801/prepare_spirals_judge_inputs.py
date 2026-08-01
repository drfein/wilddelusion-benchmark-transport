from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    output = []
    for row in rows:
        if row.get("error") or not row.get("response"):
            continue
        prepared = {
            **row,
            "target_text": row["user_text"],
            "generation_sha256": stable_hash(
                {
                    "trajectory_id": row["trajectory_id"],
                    "turn_number": row["turn_number"],
                    "response_sha256": row["response_sha256"],
                }
            ),
        }
        output.append(prepared)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"input_rows": len(rows), "output_rows": len(output)}, indent=2))


if __name__ == "__main__":
    main()
