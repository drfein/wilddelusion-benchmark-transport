"""Frozen constants and leakage-safe fold assignment."""

import hashlib

SEED = 20260731
N_FOLDS = 5
N_BOOT = 5000
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SHARECHAT_REVISION = "a57c741e002684939f577594fdae1559832a584d"
WILDCHAT_REVISION = "eedff4afb0239e69217ffd1c276e2ba45bbfdd45"


def fold_for_group(group: str) -> int:
    digest = hashlib.sha256((str(SEED) + "\0" + group).encode()).digest()
    return int.from_bytes(digest[:8], "big") % N_FOLDS
