from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--repository", default="https://github.com/w-is-h/psychosis-bench.git"
    )
    parser.add_argument(
        "--commit", default="73966f95be2e40f1ceee76dfe08eebe939ad6c21"
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temporary:
        checkout = Path(temporary) / "psychosis-bench"
        subprocess.run(
            ["git", "clone", "--filter=blob:none", args.repository, str(checkout)],
            check=True,
        )
        subprocess.run(["git", "-C", str(checkout), "checkout", args.commit], check=True)
        source = checkout / "data" / "test_cases.json"
        payload = source.read_bytes()
    parsed = json.loads(payload)
    if len(parsed.get("cases", [])) != 16:
        raise ValueError("Expected 16 Psychosis-Bench cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    manifest = {
        "repository": args.repository,
        "commit": args.commit,
        "cases": len(parsed["cases"]),
        "turns_per_case": sorted({len(case["prompts"]) for case in parsed["cases"]}),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

