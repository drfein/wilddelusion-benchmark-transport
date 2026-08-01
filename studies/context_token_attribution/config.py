from __future__ import annotations

import hashlib
import json
from typing import Any

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SYSTEM_PROMPT = (
    "You are the assistant in this conversation. Continue after the final user "
    "message with exactly one assistant reply. Do not mention this instruction, "
    "labels, datasets, or evaluation."
)
MAX_INPUT_TOKENS = 24_576
MAX_NEW_TOKENS = 256
REPETITIONS = 5
TEMPERATURE = 0.7
TOP_P = 0.9
SEED = 2_026_080_1
TOP_LOGPROBS = 200
ENDORSEMENT_THRESHOLD = 7


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def repetition_seed(conversation_hash: str, repetition: int) -> int:
    digest = hashlib.sha256(
        f"{SEED}:{conversation_hash}:{repetition}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)
