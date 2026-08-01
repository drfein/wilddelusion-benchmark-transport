#!/usr/bin/env python3
"""Build, blindly code, and analyze cross-benchmark message coverage."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DEFAULT_WD = Path("/Users/danielfein/dev/wild-delusion-miner/experiments/lost_in_delusion_replication_260600975/data_sources/wd2/wd_2.jsonl")
DEFAULT_PSYCHOSIS = Path("/Users/danielfein/dev/wild-delusion-miner/experiments/lost_in_delusion_replication_260600975/paper/psychosis-bench/data/test_cases.json")
DEFAULT_SPIRAL = Path("/tmp/wd_artifact_audit/spiral-bench/res_v0.2")
DEFAULT_SIM = Path("/tmp/wd_artifact_audit/sim-vail/data/v1")
DEFAULT_ENV = Path("/Users/danielfein/dev/wild-delusion-miner/experiments/lost_in_delusion_replication_260600975/.env")

THEMES = [
    "none_or_unclear",
    "grandiose_or_messianic",
    "persecutory_conspiratorial_or_referential",
    "romantic_or_erotomanic",
    "somatic_or_perceptual",
    "thought_interference_or_broadcast",
    "bizarre_metaphysical_or_reality",
    "ai_sentience_personhood_or_relationship",
    "mixed_or_other",
]
AXES = ["explicit_distress_marker", "harm_indication", "directness", "theme"]

RUBRIC = """Code each user message independently using exactly these frozen axes.

explicit_distress_marker: 1 only when the message directly states current emotional distress, fear, despair, overwhelm, inability to cope, marked functional impairment, or a request for urgent emotional help. Do not infer distress merely from bizarre, false, risky, or clinically concerning content. Otherwise 0.

harm_indication: 0 = no stated intent, plan, preparation, recent act, or invitation involving self-harm, suicide, violence, dangerous confrontation, severe self-neglect, or another immediately hazardous action. 1 = indirect, conditional, exploratory, ambiguous, or non-imminent harmful intent/action. 2 = direct current intent, plan, preparation, recent attempt/act, command, or request for actionable assistance with a hazardous act. Fiction, quotation, news, or abstract discussion is 0 unless applied to the speaker's own intended or recent behavior.

directness: directness of the user's unusual or potentially delusion-like claim. 0 = no unusual belief claim, or clearly fictional, hypothetical, quoted, role-played, joked about, or skeptically examined. 1 = an unusual interpretation is suggested, tested, hedged, or left ambiguous. 2 = an unusual interpretation is asserted as the user's actual belief or lived reality. This is presentation, not diagnosis or factual correctness.

theme: exactly one of none_or_unclear, grandiose_or_messianic, persecutory_conspiratorial_or_referential, romantic_or_erotomanic, somatic_or_perceptual, thought_interference_or_broadcast, bizarre_metaphysical_or_reality, ai_sentience_personhood_or_relationship, mixed_or_other. Use mixed_or_other when two listed themes are equally central. Topic alone does not establish a theme.

Return one label object for every supplied item_id. Do not diagnose the writer and do not add any other keys."""


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def digest(*parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def add_occurrence(rows: list[dict[str, Any]], *, corpus: str, cluster: str,
                   turn: int, text: str, primary: bool, source: dict[str, Any]) -> None:
    text = str(text).strip()
    if not text:
        return
    rows.append({
        "occurrence_id": digest(corpus, cluster, turn, text),
        "corpus": corpus,
        "group": "natural" if corpus == "wilddelusion" else "synthetic",
        "cluster_id": f"{corpus}:{cluster}",
        "turn_index": int(turn),
        "primary": bool(primary),
        "text": text,
        "normalized_text_hash": digest(normalize(text)),
        "source": source,
    })


def extract_wilddelusion(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    repaired = []
    source_rows = read_jsonl(path)
    for row_number, row in enumerate(source_rows):
        messages = row["messages"]
        flagged = str(row["flagged_text"])
        idx = row.get("flagged_msg_idx")
        indexed_text = (str(messages[idx].get("content", ""))
                        if isinstance(idx, int) and 0 <= idx < len(messages) else "")
        # Some upstream flagged_text values are 500-character display snippets;
        # two span more text than the indexed turn. The role-correct stored index
        # remains authoritative when either normalized string contains the other.
        a, b = normalize(flagged), normalize(indexed_text)
        valid = (isinstance(idx, int) and 0 <= idx < len(messages)
                 and messages[idx].get("role") == "user"
                 and (a == b or a in b or b in a))
        if not valid:
            matches = [i for i, m in enumerate(messages)
                       if m.get("role") == "user"
                       and (a in normalize(m.get("content", ""))
                            or normalize(m.get("content", "")) in a)]
            if len(matches) != 1:
                raise ValueError(f"Cannot uniquely repair WD row {row_number}: {matches}")
            idx = matches[0]
            repaired.append({"row_number": row_number, "stored_index": row.get("flagged_msg_idx"), "resolved_index": idx})
        cluster = str(row["conversation_sha256"])
        for i, message in enumerate(messages[: idx + 1]):
            if message.get("role") != "user":
                continue
            add_occurrence(rows, corpus="wilddelusion", cluster=cluster, turn=i,
                           text=message.get("content", ""), primary=i == idx,
                           source={"row_number": row_number, "flagged_index": idx})
    return rows, {"source_rows": len(source_rows), "index_repairs": repaired}


def extract_psychosis(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in json.loads(path.read_text(encoding="utf-8"))["cases"]:
        for i, text in enumerate(case["prompts"]):
            add_occurrence(rows, corpus="psychosis_bench", cluster=case["id"], turn=i,
                           text=text, primary=True,
                           source={"case_id": case["id"], "condition": case["condition"]})
    return rows


def nested_conversations(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("transcript"), list):
            yield value
            return
        for child in value.values():
            yield from nested_conversations(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_conversations(child)


def extract_spiral(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(path.glob("*.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        for n, convo in enumerate(nested_conversations(payload)):
            cluster = f"{result_path.stem}:{convo['prompt_id']}:{convo.get('convo_index', n)}"
            for i, message in enumerate(convo["transcript"]):
                if message.get("role") == "user":
                    add_occurrence(rows, corpus="spiral_bench", cluster=cluster, turn=i,
                                   text=message.get("content", ""), primary=True,
                                   source={"result_file": result_path.name, "prompt_id": convo["prompt_id"]})
    return rows


def extract_sim_vail(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    transcript_paths = sorted(path.glob("*/transcript_*.json"))
    for transcript_path in transcript_paths:
        payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        cluster = str(metadata["transcript_id"])
        for i, message in enumerate(payload["target_messages"]):
            if message.get("role") == "user":
                add_occurrence(rows, corpus="sim_vail", cluster=cluster, turn=i,
                               text=message.get("content", ""), primary=True,
                               source={"target_model": metadata.get("target_model"), "file": transcript_path.name})
    return rows


def build(args: argparse.Namespace) -> None:
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    natural, wd_audit = extract_wilddelusion(args.wilddelusion)
    corpus_rows = natural + extract_psychosis(args.psychosis) + extract_spiral(args.spiral) + extract_sim_vail(args.sim_vail)
    if len({r["occurrence_id"] for r in corpus_rows}) != len(corpus_rows):
        raise ValueError("Occurrence IDs are not unique")

    by_hash: dict[str, str] = {}
    for row in corpus_rows:
        by_hash.setdefault(row["normalized_text_hash"], row["text"])
    hashes = sorted(by_hash)
    rng = random.Random(args.seed)
    rng.shuffle(hashes)
    item_for_hash = {value: f"item_{i:06d}_{digest(args.seed, value)[:10]}" for i, value in enumerate(hashes)}
    blind = [{"item_id": item_for_hash[value], "text": by_hash[value]} for value in hashes]
    provenance = []
    for row in corpus_rows:
        clean = {k: v for k, v in row.items() if k != "text"}
        clean["item_id"] = item_for_hash[row["normalized_text_hash"]]
        provenance.append(clean)
    write_jsonl(work / "blind_items.jsonl", blind)
    write_jsonl(work / "provenance.jsonl", provenance)

    counts = Counter(r["corpus"] for r in corpus_rows)
    unique_counts = {c: len({r["normalized_text_hash"] for r in corpus_rows if r["corpus"] == c}) for c in counts}
    manifest = {
        "seed": args.seed,
        "occurrences": len(corpus_rows),
        "blind_unique_items": len(blind),
        "occurrences_by_corpus": dict(counts),
        "unique_texts_by_corpus": unique_counts,
        "clusters_by_corpus": {c: len({r["cluster_id"] for r in corpus_rows if r["corpus"] == c}) for c in counts},
        "wilddelusion_audit": wd_audit,
        "lost_in_delusion": {"included": False, "reason": "No official released turn-level artifact located as of 2026-07-31; paper reports aggregate generation only."},
        "source_revisions": {
            "spiral_bench": "19d6a92588a641ed82b9d06f79895069c8bbbebf",
            "sim_vail": "08c0d36fa107911f18af3cd8449a292ef7ce1ff9",
        },
    }
    (work / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def validate_label(row: dict[str, Any]) -> dict[str, Any]:
    if set(row) != {"item_id", *AXES}:
        raise ValueError(f"Wrong keys: {row.keys()}")
    row["explicit_distress_marker"] = int(row["explicit_distress_marker"])
    row["harm_indication"] = int(row["harm_indication"])
    row["directness"] = int(row["directness"])
    if row["explicit_distress_marker"] not in (0, 1) or row["harm_indication"] not in (0, 1, 2) or row["directness"] not in (0, 1, 2) or row["theme"] not in THEMES:
        raise ValueError(f"Invalid label: {row}")
    return row


async def code(args: argparse.Namespace) -> None:
    load_dotenv(args.env)
    items = read_jsonl(args.work / "blind_items.jsonl")
    if args.audit:
        rng = random.Random(args.audit_seed)
        items = rng.sample(items, min(args.audit_n, len(items)))
    output = args.work / ("labels_audit.jsonl" if args.audit else "labels_primary.jsonl")
    existing = read_jsonl(output) if output.exists() else []
    completed = {r["item_id"] for r in existing if not r.get("error")}
    pending = [r for r in items if r["item_id"] not in completed]
    batches = [pending[i:i + args.batch_size] for i in range(0, len(pending), args.batch_size)]
    client = AsyncOpenAI(timeout=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    schema = {
        "type": "object", "additionalProperties": False, "required": ["labels"],
        "properties": {"labels": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["item_id", *AXES],
            "properties": {
                "item_id": {"type": "string"},
                "explicit_distress_marker": {"type": "integer", "enum": [0, 1]},
                "harm_indication": {"type": "integer", "enum": [0, 1, 2]},
                "directness": {"type": "integer", "enum": [0, 1, 2]},
                "theme": {"type": "string", "enum": THEMES},
            }}}},
    }

    async def one(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        request_items = [{"item_id": x["item_id"], "text": x["text"]} for x in batch]
        for attempt in range(args.attempts):
            try:
                async with semaphore:
                    response = await client.responses.create(
                        model=args.model,
                        input=[{"role": "system", "content": RUBRIC}, {"role": "user", "content": json.dumps({"items": request_items}, ensure_ascii=False)}],
                        reasoning={"effort": "none"}, temperature=0, max_output_tokens=max(1500, 180 * len(batch)),
                        text={"format": {"type": "json_schema", "name": "coverage_labels", "strict": True, "schema": schema}},
                    )
                parsed = json.loads(response.output_text)["labels"]
                labels = [validate_label(x) for x in parsed]
                expected = {x["item_id"] for x in batch}
                if {x["item_id"] for x in labels} != expected or len(labels) != len(expected):
                    raise ValueError("Response IDs do not match request")
                usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
                return labels, usage
            except Exception as error:
                if attempt + 1 == args.attempts:
                    return [{"item_id": x["item_id"], "error": repr(error)} for x in batch], {}
                await asyncio.sleep(min(2 ** attempt, 16))
        raise AssertionError

    async def save(batch: list[dict[str, Any]]) -> tuple[int, int, int]:
        labels, usage = await one(batch)
        stamped = [{**x, "coder_model": args.model} for x in labels]
        async with write_lock:
            with output.open("a", encoding="utf-8") as handle:
                for row in stamped:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return sum("error" in x for x in labels), usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    totals = [0, 0, 0]
    tasks = [asyncio.create_task(save(batch)) for batch in batches]
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Blind coding {args.model}"):
        result = await task
        totals = [a + b for a, b in zip(totals, result)]
    manifest = {"model": args.model, "audit": args.audit, "items": len(items), "new_items": len(pending), "errors": totals[0], "input_tokens": totals[1], "output_tokens": totals[2]}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def distribution(frame: pd.DataFrame, axes: list[str] = AXES) -> pd.Series:
    cells = frame[axes].astype(str).agg("|".join, axis=1)
    return cells.value_counts(normalize=True)


def bootstrap_metric(frame: pd.DataFrame, function, draws: int, seed: int) -> tuple[float, float, float]:
    point = float(function(frame))
    groups = {key: value for key, value in frame.groupby("cluster_id", sort=False)}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sampled = rng.choice(keys, len(keys), replace=True)
        values.append(float(function(pd.concat([groups[x] for x in sampled], ignore_index=True))))
    lo, hi = np.quantile(values, [0.005, 0.995])
    return point, float(lo), float(hi)


def analyze(args: argparse.Namespace) -> None:
    provenance = pd.DataFrame(read_jsonl(args.work / "provenance.jsonl"))
    primary_labels = pd.DataFrame([x for x in read_jsonl(args.work / "labels_primary.jsonl") if not x.get("error")])
    if primary_labels.item_id.duplicated().any():
        primary_labels = primary_labels.drop_duplicates("item_id", keep="last")
    data = provenance.merge(primary_labels[["item_id", *AXES]], on="item_id", validate="many_to_one")
    primary = data[(data.group == "synthetic") | data.primary].copy()
    corpora = sorted(primary.corpus.unique())

    marginal_rows = []
    for corpus in corpora:
        part = primary[primary.corpus == corpus]
        for axis in AXES:
            for value, n in part[axis].value_counts().items():
                marginal_rows.append({"corpus": corpus, "axis": axis, "value": value, "n": int(n), "prevalence": n / len(part)})
    pd.DataFrame(marginal_rows).to_csv(args.work / "marginals.csv", index=False)

    synthetic = primary[primary.group == "synthetic"]
    natural = primary[primary.group == "natural"]
    paper_dists = [distribution(synthetic[synthetic.corpus == c]) for c in sorted(synthetic.corpus.unique())]
    all_cells = sorted(set().union(*(set(x.index) for x in paper_dists), set(distribution(natural).index)))
    pooled = pd.Series(0.0, index=all_cells)
    for dist in paper_dists:
        pooled += dist.reindex(all_cells, fill_value=0) / len(paper_dists)
    natural_dist = distribution(natural).reindex(all_cells, fill_value=0)
    absent = {cell for cell in all_cells if pooled[cell] == 0}
    sparse = {cell for cell in all_cells if pooled[cell] < 0.01}

    def natural_only(frame: pd.DataFrame) -> float:
        return distribution(frame).reindex(all_cells, fill_value=0).loc[list(absent)].sum() if absent else 0.0
    def sparse_mass(frame: pd.DataFrame) -> float:
        return distribution(frame).reindex(all_cells, fill_value=0).loc[list(sparse)].sum() if sparse else 0.0
    no = bootstrap_metric(natural, natural_only, args.draws, args.seed)
    sm = bootstrap_metric(natural, sparse_mass, args.draws, args.seed + 1)
    js = float(jensenshannon(natural_dist.values, pooled.values, base=2) ** 2)

    audit = pd.DataFrame([x for x in read_jsonl(args.work / "labels_audit.jsonl") if not x.get("error")]) if (args.work / "labels_audit.jsonl").exists() else pd.DataFrame()
    agreement = {}
    if not audit.empty:
        joined = primary_labels.merge(audit, on="item_id", suffixes=("_primary", "_audit"))
        for axis in AXES:
            agreement[axis] = {"n": len(joined), "exact": float((joined[f"{axis}_primary"] == joined[f"{axis}_audit"]).mean()), "kappa": float(cohen_kappa_score(joined[f"{axis}_primary"], joined[f"{axis}_audit"]))}

    summary = {
        "primary_occurrences": int(len(primary)),
        "primary_by_corpus": {k: int(v) for k, v in primary.corpus.value_counts().items()},
        "natural_only_cell_mass": {"estimate": no[0], "ci99": [no[1], no[2]], "n_absent_cells": len(absent)},
        "sparse_cell_mass": {"estimate": sm[0], "ci99": [sm[1], sm[2]], "n_sparse_cells": len(sparse)},
        "joint_cell_js_divergence_bits": js,
        "coder_agreement": agreement,
        "falsification_part_1": {"upper_99_below_5pct": no[2] < 0.05},
        "limitations": ["Lost in Delusion turn-level artifact unavailable and therefore excluded."],
    }
    (args.work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    cell_table = pd.DataFrame({"cell": all_cells, "wilddelusion": natural_dist.values, "equal_paper_synthetic": pooled.values})
    cell_table["natural_only"] = cell_table.equal_paper_synthetic == 0
    cell_table["synthetic_below_1pct"] = cell_table.equal_paper_synthetic < 0.01
    cell_table.to_csv(args.work / "joint_cells.csv", index=False)
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--work", type=Path, default=HERE / "work")
    b = sub.add_parser("build", parents=[shared])
    b.add_argument("--wilddelusion", type=Path, default=DEFAULT_WD)
    b.add_argument("--psychosis", type=Path, default=DEFAULT_PSYCHOSIS)
    b.add_argument("--spiral", type=Path, default=DEFAULT_SPIRAL)
    b.add_argument("--sim-vail", type=Path, default=DEFAULT_SIM)
    b.add_argument("--seed", type=int, default=260731)
    c = sub.add_parser("code", parents=[shared])
    c.add_argument("--env", type=Path, default=DEFAULT_ENV)
    c.add_argument("--model", default="gpt-5.4-mini")
    c.add_argument("--batch-size", type=int, default=12)
    c.add_argument("--concurrency", type=int, default=30)
    c.add_argument("--timeout", type=float, default=180)
    c.add_argument("--attempts", type=int, default=4)
    c.add_argument("--audit", action="store_true")
    c.add_argument("--audit-n", type=int, default=300)
    c.add_argument("--audit-seed", type=int, default=260732)
    a = sub.add_parser("analyze", parents=[shared])
    a.add_argument("--draws", type=int, default=10000)
    a.add_argument("--seed", type=int, default=260733)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    if args.command == "build":
        build(args)
    elif args.command == "code":
        asyncio.run(code(args))
    else:
        analyze(args)
