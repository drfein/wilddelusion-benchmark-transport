from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from wilddelusion_repro.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_is_acyclic_and_all_reaches_five_analyses():
    pipeline = Pipeline.load(ROOT, ROOT / "configs" / "pipeline.toml")
    order = pipeline.order(["all"])
    expected = {
        "analyze_visible_harm",
        "analyze_control_realism",
        "analyze_psychogenic",
        "analyze_theme_coverage",
        "analyze_recognition_gap",
    }
    assert expected <= set(order)
    assert order[-1] == "all"


def test_every_costly_stage_is_declared():
    pipeline = Pipeline.load(ROOT, ROOT / "configs" / "pipeline.toml")
    assert pipeline.stages["classify_recognition"].kind == "gpu"
    assert pipeline.stages["judge_psychogenic"].kind == "api"
    assert pipeline.stages["label_visible_harm"].kind == "api"


def test_repository_input_fingerprint_changes_with_code():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "configs").mkdir()
        (root / "scripts").mkdir()
        config = root / "configs" / "pipeline.toml"
        config.write_text('[stages.one]\ncommand = "true"\n', encoding="utf-8")
        script = root / "scripts" / "one.py"
        script.write_text("value = 1\n", encoding="utf-8")
        before = Pipeline.load(root, config).repository_input_hash()
        script.write_text("value = 2\n", encoding="utf-8")
        after = Pipeline.load(root, config).repository_input_hash()
        assert before != after


def test_deterministic_hash_contract_fails_closed():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "configs").mkdir()
        config = root / "configs" / "pipeline.toml"
        config.write_text(
            """
[stages.one]
command = "printf wrong > output.txt"
outputs = ["output.txt"]
expected_sha256 = { "output.txt" = "0000000000000000000000000000000000000000000000000000000000000000" }
""".strip()
            + "\n",
            encoding="utf-8",
        )
        pipeline = Pipeline.load(root, config)
        with pytest.raises(RuntimeError, match="deterministic output mismatch"):
            pipeline.run(
                ["one"],
                dry_run=False,
                resume=True,
                allow_api=False,
                allow_gpu=False,
            )
