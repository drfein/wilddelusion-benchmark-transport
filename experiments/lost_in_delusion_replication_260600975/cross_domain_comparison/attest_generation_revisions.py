#!/usr/bin/env python3
"""Attest the revision resolved by unpinned Hugging Face generation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
SPECS = {
    "allenai/Olmo-3-7B-Instruct": {
        "revision": "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
        "generations": [
            EXPERIMENT
            / "full_history_522"
            / "results"
            / "generations_olmo.jsonl",
            EXPERIMENT
            / "full_history_522"
            / "results"
            / "generations_olmo_last_user_only.jsonl",
        ],
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "generations": [
            EXPERIMENT
            / "full_history_522"
            / "results"
            / "generations_llama31_8b.jsonl",
            EXPERIMENT
            / "full_history_522"
            / "results"
            / "generations_llama31_8b_last_user_only.jsonl",
        ],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "generation_revision_attestation.json",
    )
    args = parser.parse_args()

    api = HfApi()
    models: dict[str, Any] = {}
    for model, spec in SPECS.items():
        generations = [Path(path) for path in spec["generations"]]
        generation = generations[0]
        manifest = generation.with_suffix(generation.suffix + ".manifest.json")
        commits = api.list_repo_commits(model, repo_type="model")
        latest = commits[0]
        artifacts = []
        for artifact in generations:
            artifact_manifest = artifact.with_suffix(
                artifact.suffix + ".manifest.json"
            )
            artifact_mtime = datetime.fromtimestamp(
                artifact.stat().st_mtime, tz=timezone.utc
            )
            artifacts.append(
                {
                    "generation_file": str(artifact),
                    "generation_file_mtime": iso(artifact_mtime),
                    "generation_sha256": sha256_file(artifact),
                    "generation_manifest_sha256": sha256_file(
                        artifact_manifest
                    ),
                }
            )
        generation_mtime = datetime.fromtimestamp(
            generation.stat().st_mtime, tz=timezone.utc
        )
        attested = bool(
            latest.commit_id == spec["revision"]
            and all(
                latest.created_at
                <= datetime.fromisoformat(row["generation_file_mtime"])
                for row in artifacts
            )
        )
        models[model] = {
            "expected_revision": spec["revision"],
            "latest_revision_at_audit": latest.commit_id,
            "latest_revision_created_at": iso(latest.created_at),
            "latest_revision_title": latest.title,
            "generation_file": str(generation),
            "generation_file_mtime": iso(generation_mtime),
            "generation_sha256": sha256_file(generation),
            "generation_manifest_sha256": sha256_file(manifest),
            "generation_artifacts": artifacts,
            "attested": attested,
            "attestation_logic": (
                "The repository's latest commit at audit is the expected "
                "revision and predates every completed generation artifact; "
                "therefore an unpinned main-branch resolution during generation "
                "could not have resolved to a later revision."
            ),
        }
    if not all(row["attested"] for row in models.values()):
        raise ValueError("At least one generation revision could not be attested")

    output = {
        "protocol": "huggingface_main_temporal_revision_attestation_v1",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
    }
    args.output.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
