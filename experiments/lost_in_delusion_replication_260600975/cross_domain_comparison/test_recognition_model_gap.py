from __future__ import annotations

import pandas as pd

from analyze_recognition_model_gap import (
    classification_error_bounds,
    clustered_partial_interval,
)


def test_classification_error_bounds_are_adversarial() -> None:
    assert classification_error_bounds(
        pd.Series({"parse_valid": False, "predicted_delusion": None})
    ) == (0.0, 1.0)
    assert classification_error_bounds(
        pd.Series({"parse_valid": True, "predicted_delusion": "none"})
    ) == (1.0, 1.0)
    assert classification_error_bounds(
        pd.Series(
            {
                "parse_valid": True,
                "predicted_delusion": "spiritual-messianic",
            }
        )
    ) == (0.0, 0.0)


def test_clustered_partial_interval_preserves_order() -> None:
    frame = pd.DataFrame(
        {
            "cluster_id": ["a", "a", "b", "c"],
            "gap_lower": [0.0, 0.0, 0.25, -0.25],
            "gap_upper": [0.5, 0.5, 0.75, 0.25],
        }
    )
    result = clustered_partial_interval(frame, draws=2_000, seed=7)
    assert result["gap_lower"] <= result["gap_upper"]
    assert result["partial_ci99_low"] <= result["partial_ci99_high"]
    assert result["gap_lower"] == 0.0
    assert result["gap_upper"] == 0.5
