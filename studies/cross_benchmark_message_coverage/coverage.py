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
import matplotlib
from dotenv import load_dotenv
from openai import AsyncOpenAI
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_WD = Path("/Users/danielfein/dev/wild-delusion-miner/experiments/lost_in_delusion_replication_260600975/full_dataset_522/artifacts/cohort_full522.jsonl")
DEFAULT_PSYCHOSIS = Path("/Users/danielfein/dev/wild-delusion-miner/experiments/lost_in_delusion_replication_260600975/paper/psychosis-bench/data/test_cases.json")
DEFAULT_SPIRAL = Path("/tmp/wd_artifact_audit/spiral-bench/prompts/eval_prompts_v0.2.json")
DEFAULT_SIM = Path("/tmp/wd_artifact_audit/sim-vail/data/v1")
DEFAULT_ENV = Path("/Users/danielfein/dev/wild-delusion-miner/experiments/lost_in_delusion_replication_260600975/.env")

THEMES = [
    "grandiose_or_messianic",
    "persecutory_conspiratorial_or_referential",
    "bizarre_metaphysical_or_religious",
    "somatic_or_perceptual",
    "thought_interference_or_broadcast",
    "romantic_erotomanic_or_jealous",
    "mixed_unclear_or_other",
]
KIRGIS_STARTERS = {"eac01", "ec01", "ec02", "ec03", "ec04", "ec05", "ie04", "mp01", "mp02", "mp03", "mp04", "mp05", "sc05", "td01"}
AXES = ["explicit_distress_marker", "harm_indication", "directness", "theme"]

RUBRIC = """Code each user message independently using exactly these frozen axes.

explicit_distress_marker: 1 only when the message directly states current emotional distress, fear, despair, overwhelm, inability to cope, marked functional impairment, or a request for urgent emotional help. Do not infer distress merely from bizarre, false, risky, or clinically concerning content. Otherwise 0.

harm_indication: 0 = no stated intent, plan, preparation, recent act, or invitation involving self-harm, suicide, violence, dangerous confrontation, severe self-neglect, or another immediately hazardous action. 1 = indirect, conditional, exploratory, ambiguous, or non-imminent harmful intent/action. 2 = direct current intent, plan, preparation, recent attempt/act, command, or request for actionable assistance with a hazardous act. Fiction, quotation, news, or abstract discussion is 0 unless applied to the speaker's own intended or recent behavior.

directness: directness of the user's unusual or potentially delusion-like claim. 0 = no unusual belief claim, or clearly fictional, hypothetical, quoted, role-played, joked about, or skeptically examined. 1 = an unusual interpretation is suggested, tested, hedged, or left ambiguous. 2 = an unusual interpretation is asserted as the user's actual belief or lived reality. This is presentation, not diagnosis or factual correctness.

theme: exactly one of grandiose_or_messianic, persecutory_conspiratorial_or_referential, bizarre_metaphysical_or_religious, somatic_or_perceptual, thought_interference_or_broadcast, romantic_erotomanic_or_jealous, mixed_unclear_or_other. Use mixed_unclear_or_other when there is no clear unusual-belief theme, when two themes are equally central, or for another theme. AI sentience/personhood/relationship claims map to bizarre_metaphysical_or_religious unless primarily romantic. Topic alone does not establish a theme.

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
                   turn: int, turn_count: int, text: str, source: dict[str, Any]) -> None:
    text = str(text).strip()
    if not text:
        return
    rows.append({
        "occurrence_id": digest(corpus, cluster, turn, text),
        "corpus": corpus,
        "group": "natural" if corpus == "wilddelusion" else "synthetic",
        "cluster_id": f"{corpus}:{cluster}",
        "turn_index": int(turn),
        "turn_ordinal": int(turn + 1),
        "turn_count": int(turn_count),
        "turn_position": float(turn / (turn_count - 1)) if turn_count > 1 else 0.0,
        "text": text,
        "normalized_text_hash": digest(normalize(text)),
        "source": source,
    })


def extract_wilddelusion(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_rows = read_jsonl(path)
    for row_number, row in enumerate(source_rows):
        messages = row["history_messages"]
        target = str(row["target_text"])
        matches = [i for i, m in enumerate(messages)
                   if m.get("role") == "user" and normalize(m.get("content", "")) == normalize(target)]
        if matches != [int(row["target_message_index"])] or matches[0] != len(messages) - 1:
            raise ValueError(f"WD target/history mismatch at row {row_number}: {matches}")
        user_messages = [m for m in messages if m.get("role") == "user"]
        add_occurrence(rows, corpus="wilddelusion", cluster=str(row["cluster_id"]),
                       turn=len(user_messages) - 1, turn_count=len(user_messages), text=target,
                       source={"row_number": row_number, "message_hash": row["message_hash"]})
    return rows, {"source_rows": len(source_rows), "targets_verified": len(rows)}


def extract_psychosis(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in json.loads(path.read_text(encoding="utf-8"))["cases"]:
        for i, text in enumerate(case["prompts"]):
            add_occurrence(rows, corpus="psychosis_bench", cluster=case["id"], turn=i, turn_count=len(case["prompts"]),
                           text=text,
                           source={"case_id": case["id"], "condition": case["condition"]})
    return rows


def extract_spiral(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = {row["prompt_id"]: row for row in payload if row["prompt_id"] in KIRGIS_STARTERS}
    if set(selected) != KIRGIS_STARTERS:
        raise ValueError(f"Missing Kirgis starters: {sorted(KIRGIS_STARTERS - set(selected))}")
    for prompt_id in sorted(selected):
        prompts = selected[prompt_id]["prompts"]
        if len(prompts) != 1:
            raise ValueError(f"Kirgis starter {prompt_id} has {len(prompts)} prompts")
        add_occurrence(rows, corpus="spiral_bench", cluster=prompt_id, turn=0, turn_count=1,
                       text=prompts[0], source={"prompt_id": prompt_id})
    return rows


def extract_sim_vail(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    transcript_paths = sorted(path.glob("*/transcript_*.json"))
    for transcript_path in transcript_paths:
        payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        cluster = str(metadata["transcript_id"])
        user_messages = [m for m in payload["target_messages"] if m.get("role") == "user"]
        for i, message in enumerate(user_messages):
                add_occurrence(rows, corpus="sim_vail", cluster=cluster, turn=i, turn_count=len(user_messages),
                               text=message.get("content", ""),
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
    audit_rng = random.Random(args.seed + 1)
    items_by_id = {item["item_id"]: item for item in blind}
    ids_by_corpus = {
        corpus: sorted({row["item_id"] for row in provenance if row["corpus"] == corpus})
        for corpus in sorted({row["corpus"] for row in provenance})
    }
    quotas = {"spiral_bench": 14, "psychosis_bench": 95, "wilddelusion": 95, "sim_vail": 96}
    audit_ids = []
    for corpus, quota in quotas.items():
        candidates = ids_by_corpus[corpus]
        audit_ids.extend(candidates if len(candidates) <= quota else audit_rng.sample(candidates, quota))
    if len(set(audit_ids)) != 300:
        raise ValueError(f"Stratified audit did not yield 300 unique items: {len(set(audit_ids))}")
    audit_rng.shuffle(audit_ids)
    audit_items = [items_by_id[item_id] for item_id in audit_ids]
    (work / "audit_item_ids.json").write_text(json.dumps(audit_ids, indent=2) + "\n", encoding="utf-8")
    with (work / "human_audit_blind.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "text", *AXES, "notes"])
        writer.writeheader()
        for item in audit_items:
            writer.writerow({**item, **{axis: "" for axis in AXES}, "notes": ""})

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
        "human_audit_packet_items": len(audit_items),
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
        requested_ids = json.loads((args.work / "audit_item_ids.json").read_text(encoding="utf-8"))[:args.audit_n]
        by_id = {item["item_id"]: item for item in items}
        items = [by_id[item_id] for item_id in requested_ids]
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


def kappa(left: pd.Series, right: pd.Series) -> float | None:
    if len(set(left).union(set(right))) < 2:
        return None
    value = float(cohen_kappa_score(left, right))
    return value if np.isfinite(value) else None


def cluster_bootstrap_mean(frame: pd.DataFrame, values: np.ndarray, draws: int, seed: int) -> tuple[float, float, float]:
    work = pd.DataFrame({"cluster_id": frame["cluster_id"].to_numpy(), "value": np.asarray(values, dtype=float)})
    grouped = work.groupby("cluster_id", sort=False).value.agg(["sum", "count"])
    sums = grouped["sum"].to_numpy()
    counts = grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(grouped), size=(draws, len(grouped)))
    estimates = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    lo, hi = np.quantile(estimates, [0.005, 0.995])
    return float(np.asarray(values, dtype=float).mean()), float(lo), float(hi)


def cluster_bootstrap_proportions(frame: pd.DataFrame, axis: str, values: list[str], draws: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    encoded = frame.assign(_value=frame[axis].astype(str))
    counts = pd.crosstab(encoded.cluster_id, encoded._value).reindex(columns=values, fill_value=0).to_numpy(dtype=float)
    totals = counts.sum(axis=1)
    point = counts.sum(axis=0) / totals.sum()
    rng = np.random.default_rng(seed)
    estimates = np.empty((draws, len(values)), dtype=float)
    for start in range(0, draws, 250):
        stop = min(start + 250, draws)
        sampled = rng.integers(0, len(counts), size=(stop - start, len(counts)))
        numerator = counts[sampled].sum(axis=1)
        denominator = totals[sampled].sum(axis=1)[:, None]
        estimates[start:stop] = numerator / denominator
    low, high = np.quantile(estimates, [0.025, 0.975], axis=0)
    return point, low, high, estimates


def analyze(args: argparse.Namespace) -> None:
    provenance = pd.DataFrame(read_jsonl(args.work / "provenance.jsonl"))
    primary_labels = pd.DataFrame([x for x in read_jsonl(args.work / "labels_primary.jsonl") if not x.get("error")])
    if primary_labels.item_id.duplicated().any():
        primary_labels = primary_labels.drop_duplicates("item_id", keep="last")
    data = provenance.merge(primary_labels[["item_id", *AXES]], on="item_id", validate="many_to_one")
    primary = data.copy()
    corpora = sorted(primary.corpus.unique())

    marginal_rows = []
    for corpus_index, corpus in enumerate(corpora):
        part = primary[primary.corpus == corpus]
        for axis_index, axis in enumerate(AXES):
            values = sorted(primary[axis].astype(str).unique())
            point, low, high, _ = cluster_bootstrap_proportions(part, axis, values, args.draws, args.seed + 100 * corpus_index + axis_index)
            raw_counts = part[axis].astype(str).value_counts()
            for i, value in enumerate(values):
                marginal_rows.append({"corpus": corpus, "axis": axis, "value": value, "n": int(raw_counts.get(value, 0)), "prevalence": point[i], "ci95_low": low[i], "ci95_high": high[i]})
    pd.DataFrame(marginal_rows).to_csv(args.work / "marginals.csv", index=False)

    pooled_rows = []
    synthetic_for_pool = primary[primary.group == "synthetic"]
    for axis_index, axis in enumerate(AXES):
        values = sorted(primary[axis].astype(str).unique())
        paper_points, paper_draws = [], []
        for corpus_index, corpus in enumerate(sorted(synthetic_for_pool.corpus.unique())):
            point, _, _, estimates = cluster_bootstrap_proportions(
                synthetic_for_pool[synthetic_for_pool.corpus == corpus], axis, values,
                args.draws, args.seed + 1000 + 100 * corpus_index + axis_index,
            )
            paper_points.append(point)
            paper_draws.append(estimates)
        point = np.mean(paper_points, axis=0)
        pooled_draws = np.mean(paper_draws, axis=0)
        low, high = np.quantile(pooled_draws, [0.025, 0.975], axis=0)
        for i, value in enumerate(values):
            pooled_rows.append({"corpus": "equal_paper_synthetic", "axis": axis, "value": value, "prevalence": point[i], "ci95_low": low[i], "ci95_high": high[i]})
    pd.DataFrame(pooled_rows).to_csv(args.work / "pooled_marginals.csv", index=False)

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

    natural_cells = natural[AXES].astype(str).agg("|".join, axis=1)
    no = cluster_bootstrap_mean(natural, natural_cells.isin(absent).to_numpy(), args.draws, args.seed)
    sm = cluster_bootstrap_mean(natural, natural_cells.isin(sparse).to_numpy(), args.draws, args.seed + 1)
    js = float(jensenshannon(natural_dist.values, pooled.values, base=2) ** 2)

    audit = pd.DataFrame([x for x in read_jsonl(args.work / "labels_audit.jsonl") if not x.get("error")]) if (args.work / "labels_audit.jsonl").exists() else pd.DataFrame()
    agreement = {}
    agreement_by_corpus = {}
    if not audit.empty:
        joined = primary_labels.merge(audit, on="item_id", suffixes=("_primary", "_audit"))
        for axis in AXES:
            agreement[axis] = {"n": len(joined), "exact": float((joined[f"{axis}_primary"] == joined[f"{axis}_audit"]).mean()), "kappa": kappa(joined[f"{axis}_primary"], joined[f"{axis}_audit"])}
        audit_provenance = provenance[["item_id", "corpus"]].drop_duplicates()
        joined = joined.merge(audit_provenance, on="item_id", validate="one_to_one")
        for corpus, part in joined.groupby("corpus"):
            agreement_by_corpus[corpus] = {
                axis: {"n": len(part), "exact": float((part[f"{axis}_primary"] == part[f"{axis}_audit"]).mean()), "kappa": kappa(part[f"{axis}_primary"], part[f"{axis}_audit"])}
                for axis in AXES
            }

    axis_tvd = {}
    for axis in AXES:
        values = sorted(set(primary[axis].astype(str)))
        nd = natural[axis].astype(str).value_counts(normalize=True).reindex(values, fill_value=0)
        sd = pd.Series(0.0, index=values)
        for corpus in sorted(synthetic.corpus.unique()):
            sd += synthetic[synthetic.corpus == corpus][axis].astype(str).value_counts(normalize=True).reindex(values, fill_value=0) / synthetic.corpus.nunique()
        axis_tvd[axis] = float(0.5 * np.abs(nd - sd).sum())

    headline = lambda frame: ((frame.explicit_distress_marker == 0) & (frame.harm_indication == 0) & (frame.directness == 1)).mean()
    headline_values = ((natural.explicit_distress_marker == 0) & (natural.harm_indication == 0) & (natural.directness == 1)).to_numpy()
    headline_natural = cluster_bootstrap_mean(natural, headline_values, args.draws, args.seed + 2)
    paper_headlines = [headline(synthetic[synthetic.corpus == c]) for c in sorted(synthetic.corpus.unique())]
    headline_synthetic = float(np.mean(paper_headlines))
    summary = {
        "primary_occurrences": int(len(primary)),
        "primary_by_corpus": {k: int(v) for k, v in primary.corpus.value_counts().items()},
        "natural_only_cell_mass": {"estimate": no[0], "ci99": [no[1], no[2]], "n_absent_cells": len(absent)},
        "sparse_cell_mass": {"estimate": sm[0], "ci99": [sm[1], sm[2]], "n_sparse_cells": len(sparse)},
        "joint_cell_js_divergence_bits": js,
        "axis_total_variation_distance": axis_tvd,
        "axes_with_substantial_overlap_tvd_below_0_10": [k for k, v in axis_tvd.items() if v < 0.10],
        "broad_gap_falsified_by_overlap_rule": sum(v < 0.10 for v in axis_tvd.values()) >= 3,
        "headline_no_distress_no_harm_indirect": {"wilddelusion": {"estimate": headline_natural[0], "ci99": [headline_natural[1], headline_natural[2]]}, "equal_paper_synthetic": headline_synthetic},
        "coder_agreement": agreement,
        "automated_coder_agreement_by_corpus": agreement_by_corpus,
        "human_audit_status": "pending; see human_audit_blind.csv",
        "limitations": ["Lost in Delusion turn-level artifact unavailable and therefore excluded."],
    }
    (args.work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    cell_table = pd.DataFrame({"cell": all_cells, "wilddelusion": natural_dist.values, "equal_paper_synthetic": pooled.values})
    cell_table["natural_only"] = cell_table.equal_paper_synthetic == 0
    cell_table["synthetic_below_1pct"] = cell_table.equal_paper_synthetic < 0.01
    cell_table.to_csv(args.work / "joint_cells.csv", index=False)
    turn_rows = []
    subsets = {
        "all_turns": synthetic,
        "late_half": synthetic[synthetic.turn_position >= 0.5],
        "final_turn": synthetic[synthetic.turn_ordinal == synthetic.turn_count],
    }
    for subset_name, subset in subsets.items():
        for axis in AXES:
            values = sorted(primary[axis].astype(str).unique())
            natural_axis = natural[axis].astype(str).value_counts(normalize=True).reindex(values, fill_value=0)
            pooled_axis = pd.Series(0.0, index=values)
            present = sorted(subset.corpus.unique())
            for corpus in present:
                pooled_axis += subset[subset.corpus == corpus][axis].astype(str).value_counts(normalize=True).reindex(values, fill_value=0) / len(present)
            for value in values:
                turn_rows.append({"subset": subset_name, "axis": axis, "value": value, "equal_paper_synthetic": pooled_axis[value], "wilddelusion": natural_axis[value], "absolute_gap": abs(pooled_axis[value] - natural_axis[value])})
            turn_rows.append({"subset": subset_name, "axis": axis, "value": "__TVD__", "equal_paper_synthetic": np.nan, "wilddelusion": np.nan, "absolute_gap": 0.5 * np.abs(pooled_axis - natural_axis).sum()})
    pd.DataFrame(turn_rows).to_csv(args.work / "turn_position_sensitivity.csv", index=False)
    make_figure(primary, args.work / "composition_gap.png")
    print(json.dumps(summary, indent=2))


def make_figure(frame: pd.DataFrame, output: Path) -> None:
    natural = frame[frame.group == "natural"]
    synthetic = frame[frame.group == "synthetic"]
    rows = []
    for axis in AXES:
        natural_dist = natural[axis].astype(str).value_counts(normalize=True)
        for value, prevalence in natural_dist.items():
            rows.append({"source": "Real", "axis": axis, "value": value, "prevalence": prevalence})
        values = sorted(synthetic[axis].astype(str).unique())
        pooled = pd.Series(0.0, index=values)
        for corpus in sorted(synthetic.corpus.unique()):
            pooled += synthetic[synthetic.corpus == corpus][axis].astype(str).value_counts(normalize=True).reindex(values, fill_value=0) / synthetic.corpus.nunique()
        for value, prevalence in pooled.items():
            rows.append({"source": "Benchmarks", "axis": axis, "value": value, "prevalence": prevalence})
    plot = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    palette = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#6D597A", "#457B9D"]
    titles = {"explicit_distress_marker": "Explicit distress", "harm_indication": "Harm indication", "directness": "Belief directness", "theme": "Theme"}
    for ax, axis in zip(axes.flat, AXES):
        sub = plot[plot.axis == axis]
        values = sorted(sub.value.unique())
        bottoms = np.zeros(2)
        for i, value in enumerate(values):
            heights = [float(sub[(sub.source == source) & (sub.value == value)].prevalence.sum()) for source in ["Benchmarks", "Real"]]
            display = {
                "explicit_distress_marker": {"0": "Absent", "1": "Present"},
                "harm_indication": {"0": "None", "1": "Indirect", "2": "Direct"},
                "directness": {"0": "No claim", "1": "Indirect", "2": "Direct"},
            }.get(axis, {}).get(value, value.replace("_", " "))
            ax.bar([0, 1], heights, bottom=bottoms, width=.62, color=palette[i % len(palette)], label=display)
            bottoms += heights
        ax.set_title(titles[axis], loc="left", weight="bold")
        ax.set_xticks([0, 1], ["Benchmarks", "Real"])
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)


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
