#!/usr/bin/env python3
"""Assign psychiatric delusional-content themes to every merged WildDelusion turn."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

CATEGORIES = [
    "delusional_jealousy",
    "bizarre",
    "erotomanic",
    "grandiose",
    "persecutory",
    "somatic",
    "referential",
    "religious_or_spiritual",
    "nihilistic",
    "mixed",
    "thought_broadcasting",
    "thought_insertion",
    "unclear_or_no_delusional_theme",
]

CONVERSATION_THEMES = [
    "surveillance_or_targeting",
    "imaginative_narrative",
    "global_issues",
    "hope_oriented_interpretive_framing",
    "perceptual_anomalies_or_interpretations",
    "grandiosity_related_discourse",
    "compulsive_cognition",
    "depersonalization_distress",
    "derealization_experiences",
    "reality_skepticism",
    "astrological_beliefs",
    "other_or_unclear",
]

SYSTEM_PROMPT = """You are a careful psychiatric-content annotation judge for research data.

Classify the THEME OF THE TARGET USER MESSAGE, not a person. This is not a diagnosis.
Use the preceding messages only to clarify a short or ambiguous target.

Allowed primary_category values:
- delusional_jealousy: belief that one's sexual or romantic partner is unfaithful.
- bizarre: impossible, not understandable, and unrelated to normal life.
- erotomanic: belief that another person, often higher-status or unavailable, is in love with the individual.
- grandiose: inflated conviction of great talent, discovery, identity, power, knowledge, mission, deity/famous relationship, or special cosmic/historical role.
- persecutory: being conspired against, attacked, harassed, surveilled, obstructed, controlled, targeted, or sabotaged.
- somatic: bodily functions, sensations, infestation, disease, body transformation, pregnancy, organs, or physical symptoms.
- referential: neutral events, media, gestures, messages, or coincidences are specifically directed at or about the user.
- religious_or_spiritual: supernatural, divine, demonic, occult, or spiritual forces are the central explanation, without a more specific grandiose/persecutory label dominating.
- nihilistic: self, body, world, or reality is absent, dead, empty, destroyed, or does not exist.
- thought_broadcasting: thoughts are projected outward and perceived by others.
- thought_insertion: thoughts are not one's own but inserted by an external source/entity.
- mixed: more than one listed theme is prominent and no single theme dominates.
- unclear_or_no_delusional_theme: the target is too ambiguous, not a delusional belief claim, or lacks enough content to classify.

Prefer thought_broadcasting or thought_insertion when that mechanism is central. Use bizarre only for genuinely impossible phenomena; unusual conspiracy or spirituality alone is usually persecutory or grandiose.

Also choose one conversation_theme. This describes the observed conversational form, not a diagnosis:
- surveillance_or_targeting; imaginative_narrative; global_issues; hope_oriented_interpretive_framing;
  perceptual_anomalies_or_interpretations; grandiosity_related_discourse; compulsive_cognition;
  depersonalization_distress; derealization_experiences; reality_skepticism; astrological_beliefs; other_or_unclear.

Return JSON only:
{
  "primary_category": "one allowed label",
  "secondary_categories": ["zero or more allowed labels"],
  "conversation_theme": "one allowed conversation theme",
  "confidence": "low|medium|high",
  "rationale": "one concise sentence",
  "target_contains_classifiable_delusional_theme": true
}"""


def clipped(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    return value if len(value) <= limit else value[:limit].rstrip() + " [TRUNCATED]"


def normalize_category(value: Any) -> str:
    category = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "jealous": "delusional_jealousy",
        "jealousy": "delusional_jealousy",
        "thought_broadcast": "thought_broadcasting",
        "broadcasting": "thought_broadcasting",
        "insertion": "thought_insertion",
        "unclear": "unclear_or_no_delusional_theme",
        "no_delusion": "unclear_or_no_delusional_theme",
    }
    category = aliases.get(category, category)
    return category if category in CATEGORIES else "unclear_or_no_delusional_theme"


def normalize_conversation_theme(value: Any) -> str:
    theme = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "surveillance_targeting": "surveillance_or_targeting",
        "perceptual_anomalies": "perceptual_anomalies_or_interpretations",
        "perceptual_interpretations": "perceptual_anomalies_or_interpretations",
        "grandiosity": "grandiosity_related_discourse",
        "other": "other_or_unclear",
        "unclear": "other_or_unclear",
    }
    theme = aliases.get(theme, theme)
    return theme if theme in CONVERSATION_THEMES else "other_or_unclear"


def task_for(row: dict[str, Any], *, context_n: int, max_chars: int) -> dict[str, Any]:
    messages = list(row.get("messages") or [])
    target_index = int(row["target_message_index"])
    if not 0 <= target_index < len(messages):
        raise ValueError(f"Invalid target_message_index for {row.get('message_hash')}")
    context = []
    for index, message in enumerate(messages[max(0, target_index - context_n) : target_index], start=max(0, target_index - context_n)):
        content = clipped(message.get("content"), max_chars)
        if content:
            context.append(f"[{index}] {str(message.get('role') or 'unknown').upper()}: {content}")
    return {
        "message_hash": str(row["message_hash"]),
        "source": str(row.get("source") or ""),
        "conversation_id": str(row.get("conversation_id") or ""),
        "target_message_index": target_index,
        "target_text": clipped(row.get("target_text") or messages[target_index].get("content"), max_chars),
        "context_text": "\n\n".join(context) if context else "[no preceding messages available]",
    }


def prompt_for(task: dict[str, Any]) -> str:
    return f"""Classify the delusional-content theme of this TARGET USER MESSAGE.

PRECEDING CONTEXT:
{task['context_text']}

TARGET USER MESSAGE:
{task['target_text']}

Return JSON only."""


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(json.loads(line).get("message_hash"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not json.loads(line).get("error")
    }


def judge(client: OpenAI, task: dict[str, Any], model: str) -> dict[str, Any]:
    base = dict(task)
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_for(task)},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=350,
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            primary = normalize_category(parsed.get("primary_category"))
            conversation_theme = normalize_conversation_theme(parsed.get("conversation_theme"))
            secondary = sorted(
                {
                    normalize_category(item)
                    for item in parsed.get("secondary_categories", [])
                    if normalize_category(item) != primary
                }
            )
            return {
                **base,
                "delusion_theme_primary": primary,
                "delusion_type": primary,
                "delusion_theme_secondary": secondary,
                "conversation_theme": conversation_theme,
                "delusion_theme_confidence": str(parsed.get("confidence") or "low"),
                "delusion_theme_rationale": str(parsed.get("rationale") or ""),
                "delusion_theme_target_classifiable": bool(
                    parsed.get("target_contains_classifiable_delusional_theme", primary != "unclear_or_no_delusional_theme")
                ),
                "judge_model": model,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - retry transient API failures
            if attempt == 4:
                return {**base, "judge_model": model, "error": f"{type(exc).__name__}: {exc}"}
            time.sleep(3 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/releases/WildDelusionCombined/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/merged_delusion_themes"))
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--context-n", type=int, default=5)
    parser.add_argument("--max-chars-per-message", type=int, default=1000)
    parser.add_argument("--env", type=Path, required=True)
    args = parser.parse_args()

    for line in args.env.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY=") and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = [task_for(row, context_n=args.context_n, max_chars=args.max_chars_per_message) for row in rows]
    if len({task["message_hash"] for task in tasks}) != len(tasks):
        raise ValueError("message_hash must be unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "theme_judgments.jsonl"
    joined_path = args.output_dir / "wilddelusion_combined_with_themes.jsonl"
    summary_path = args.output_dir / "summary.json"
    done = load_done(results_path)
    pending = [task for task in tasks if task["message_hash"] not in done]
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    with (
        results_path.open("a", encoding="utf-8") as handle,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = [pool.submit(judge, client, task, args.model) for task in pending]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="theme_judge",
        ):
            handle.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
            handle.flush()

    results = {
        str(item["message_hash"]): item
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for item in [json.loads(line)]
        if not item.get("error")
    }
    with joined_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            result = results.get(str(row["message_hash"]))
            if result is None:
                continue
            handle.write(
                json.dumps(
                    {
                        **row,
                        "delusion_theme_primary": result["delusion_theme_primary"],
                        "delusion_type": result["delusion_type"],
                        "delusion_theme_secondary": result["delusion_theme_secondary"],
                        "conversation_theme": result["conversation_theme"],
                        "delusion_theme_confidence": result["delusion_theme_confidence"],
                        "delusion_theme_rationale": result["delusion_theme_rationale"],
                        "delusion_theme_target_classifiable": result["delusion_theme_target_classifiable"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary = {
        "input_rows": len(rows),
        "judged_rows": len(results),
        "errors": len(tasks) - len(results),
        "primary_category_counts": dict(Counter(item["delusion_theme_primary"] for item in results.values())),
        "conversation_theme_counts": dict(Counter(item["conversation_theme"] for item in results.values())),
        "model": args.model,
        "results": str(results_path),
        "joined_dataset": str(joined_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
