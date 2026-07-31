#!/usr/bin/env python3
"""Tests for independent paper-theme consensus construction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analyze_recognition import (
    EXPECTED_INPUT_SHA256,
    EXPECTED_THEME_ADJUDICATION_MODEL,
    paper_theme_consensus,
)


def write_rubric(path: Path, version: str) -> set[str]:
    identifiers = {f"id-{index:03d}" for index in range(522)}
    with path.open("w", encoding="utf-8") as handle:
        for index, input_id in enumerate(sorted(identifiers)):
            theme = "outside-ontology"
            if index == 0:
                theme = "sentient-ai"
            elif index == 1:
                theme = "spiritual-messianic"
            elif index == 2:
                theme = (
                    "sentient-ai"
                    if version == "A"
                    else "outside-ontology"
                )
            handle.write(
                json.dumps(
                    {
                        "input_id": input_id,
                        "cluster_id": f"cluster-{index // 2}",
                        "paper_theme": theme,
                        "confidence": "high",
                        "necessary_condition_present": (
                            theme != "outside-ontology"
                        ),
                        "rubric_version": version,
                        "adjudication_model": (
                            EXPECTED_THEME_ADJUDICATION_MODEL
                        ),
                    }
                )
                + "\n"
            )
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(
            {
                "model": EXPECTED_THEME_ADJUDICATION_MODEL,
                "rubric_version": version,
                "reasoning_effort": "none",
                "input_sha256": EXPECTED_INPUT_SHA256,
                "expected_rows": 522,
                "successful_rows": 522,
                "new_errors": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return identifiers


class PaperThemeConsensusTests(unittest.TestCase):
    def test_requires_exact_agreement_and_necessary_condition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path_a = root / "a.jsonl"
            path_b = root / "b.jsonl"
            identifiers = write_rubric(path_a, "A")
            write_rubric(path_b, "B")
            consensus, audit = paper_theme_consensus(
                path_a, path_b, identifiers
            )
            self.assertEqual(
                int(consensus["is_paper_theme_consensus"].sum()), 2
            )
            self.assertEqual(
                int(consensus["is_outside_ontology_consensus"].sum()), 519
            )
            self.assertEqual(audit["strict_consensus_rows"], 2)
            self.assertEqual(audit["strict_outside_ontology_rows"], 519)

    def test_partial_adjudication_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path_a = root / "a.jsonl"
            path_b = root / "b.jsonl"
            identifiers = write_rubric(path_a, "A")
            write_rubric(path_b, "B")
            lines = path_b.read_text(encoding="utf-8").splitlines()
            path_b.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity gate"):
                paper_theme_consensus(path_a, path_b, identifiers)

    def test_manifest_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path_a = root / "a.jsonl"
            path_b = root / "b.jsonl"
            identifiers = write_rubric(path_a, "A")
            write_rubric(path_b, "B")
            manifest_path = path_a.with_suffix(
                path_a.suffix + ".manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["reasoning_effort"] = "low"
            manifest_path.write_text(
                json.dumps(manifest) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "manifest failed"):
                paper_theme_consensus(path_a, path_b, identifiers)


if __name__ == "__main__":
    unittest.main()
