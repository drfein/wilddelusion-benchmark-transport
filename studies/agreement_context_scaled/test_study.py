from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("scaled_analysis", HERE / "analyze.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    raw = {"a": 0.01, "b": 0.04, "c": 0.03}
    adjusted = MODULE.holm_adjust(raw)
    ordered = sorted(raw, key=raw.get)
    values = [adjusted[name] for name in ordered]
    assert values == sorted(values)
    assert adjusted["a"] == 0.03
    assert adjusted["c"] == 0.06
    assert adjusted["b"] == 0.06


def test_sign_flip_detects_consistent_positive_effect() -> None:
    values = np.ones(30)
    pvalue = MODULE.sign_flip_pvalue(values, np.random.default_rng(7))
    assert pvalue < 0.001


def test_estimate_keeps_effect_sign() -> None:
    result = MODULE.estimate(np.array([0.0, 1.0, 1.0]), np.random.default_rng(9))
    assert result["estimate"] > 0
    assert result["ci_high"] >= result["estimate"]
