from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analysis = load("analyze")
validation = load("validate_rewrites")


def test_strict_trajectory_requires_clean_large_stance_change() -> None:
    row = {
        "target_escalates": True,
        "target_delusion": 4,
        "previous_user_delusion": 1,
        "original_agreement": 3,
        "paraphrase_agreement": 3,
        "neutral_agreement": 1,
        "paraphrase_preserves_core_content": True,
        "neutral_preserves_core_content": True,
        "neutral_adds_overt_pushback": False,
        "paraphrase_adds_new_claim": False,
    }
    assert validation.strict_trajectory(row)
    row["neutral_adds_overt_pushback"] = True
    assert not validation.strict_trajectory(row)


def test_cluster_interval_averages_target_values() -> None:
    frame = pd.DataFrame(
        {
            "conversation_hash": ["a", "a", "b"],
            "effect": [1.0, 0.0, -1.0],
        }
    )
    result = analysis.cluster_interval(frame, "effect", draws=200, seed=7)
    assert result["estimate"] == 0
