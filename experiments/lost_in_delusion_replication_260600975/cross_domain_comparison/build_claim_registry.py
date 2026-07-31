#!/usr/bin/env python3
"""Build a fail-closed registry of non-overlapping abstract claims."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent


@dataclass(frozen=True)
class Candidate:
    claim_id: str
    evidence_family: str
    priority: int
    study: str
    headline: str
    synthetic_estimand: str
    real_estimand: str
    guardrail: str
    gate_source: str
    gate_key: str = "ready"
    headline_eligible: bool = True


CANDIDATES = [
    Candidate(
        "lost_confirmed_endpoint_underestimation",
        "lost_confirmation_severity",
        9,
        "Lost in Delusion (arXiv:2606.00975)",
        (
            "Handcrafted harm trajectories understate observed "
            "delusion-perpetuation/confirmation severity at confirmed "
            "real-world endpoints."
        ),
        "Published Mod+Harm DCS for matched synthetic trajectories.",
        (
            "DCS on 522 confirmed endpoints, clustered by 321 source "
            "conversations, replicated in two official unquantized models."
        ),
        (
            "This is benchmark transport across case compositions, not a "
            "common-population prevalence estimate. The paper does not publish "
            "the exact baseline system-prompt text, so only its disclosed "
            "concise/natural semantics are matched."
        ),
        "lost",
        gate_key="descriptive_ready",
    ),
    Candidate(
        "paired_context_increases_non_grounding_severity",
        "context_conditioned_localization",
        4,
        "WildDelusion prompt-deduplicated context ablation",
        (
            "Retaining conversational history increases the ordinal severity "
            "of non-grounding responses even when binary prevalence changes "
            "little."
        ),
        "Greedy response to the exact final user message alone.",
        "Greedy response to the same message with retained prior turns.",
        (
            "The effect was discovered with the pinned mini judge and must "
            "replicate at 99% with the previously unseen native Qwen judge in "
            "both official-BF16 models."
        ),
        "context",
        gate_key="severity_ready",
    ),
    Candidate(
        "paired_context_relocalizes_non_grounding_failures",
        "context_conditioned_localization",
        3,
        "WildDelusion prompt-deduplicated context ablation",
        (
            "Single-turn evaluation hides context-conditioned non-grounding: "
            "aggregate rates can remain stable while the failing cases change."
        ),
        (
            "Prompt-hash-cached greedy responses to the exact final user "
            "message with all prior conversation turns removed."
        ),
        (
            "Prompt-hash-cached greedy responses to the same final message "
            "with retained prior conversation turns."
        ),
        (
            "This covers 449 endpoints per model where the exact tokenized "
            "prompts differ. It is cancellation of heterogeneous context "
            "effects, not evidence that either arm is ground truth."
        ),
        "context",
    ),
    Candidate(
        "lost_primary_model_gap_overestimation",
        "primary_model_comparison",
        8,
        "Lost in Delusion (arXiv:2606.00975)",
        (
            "The reported 0.39-point OLMo-Llama gap on handcrafted "
            "trajectories does not transport to confirmed real endpoints."
        ),
        (
            "Published OLMo-minus-Llama Mod+Harm DCS gap of 0.39 points."
        ),
        (
            "Paired OLMo-minus-Llama DCS gap on 522 confirmed endpoints, "
            "clustered by 321 source conversations."
        ),
        (
            "The source paper does not release row-level classifier scores, "
            "so this compares its published descriptive gap with the real-data "
            "99% interval rather than estimating a cross-study interaction; "
            "the exact original baseline system-prompt text is also unavailable."
        ),
        "lost",
        gate_key="descriptive_ready",
    ),
    Candidate(
        "lost_confirmation_failures_mostly_outside_mod_harm",
        "lost_case_composition",
        1,
        "Lost in Delusion (arXiv:2606.00975)",
        (
            "Handcrafted all-harm trajectories miss most real validation "
            "failures: even an intentionally overinclusive screen of the "
            "exact assistant-visible context captures only 26.5-30.0% of "
            "responses scored Validate/Amplify."
        ),
        "Mod+Harm trajectories in which every case reaches a harm phase.",
        (
            "DCS=3 (Validate/Amplify) responses at 522 confirmed real endpoints, "
            "split by an audited 161-endpoint possible-Mod+Harm superset over "
            "the exact context shown to the assistant."
        ),
        (
            "This is a conservative corpus-composition result, not evidence "
            "that crisis wording causally improves model behavior."
        ),
        "visible_mod_harm",
    ),
    Candidate(
        "psychogenic_implicitness_domain_drop",
        "implicitness_transport",
        3,
        "Psychogenic Machine (arXiv:2509.10970)",
        (
            "Synthetic trajectories overestimate the effect of implicit rather "
            "than explicit wording in matched real histories."
        ),
        (
            "Paper-style full trajectories plus a matched-local synthetic arm: "
            "three-seed, temperature-1 implicit-minus-explicit DCS on three "
            "official-BF16 models."
        ),
        (
            "Matched-local implicit-minus-explicit DCS on 327 real histories "
            "for the same three models."
        ),
        (
            "The claim is disabled unless the full-trajectory reproduction and "
            "matched-local synthetic effect are positive at 99%, all three "
            "models and seeds agree, and the matched domain drop is resolved."
        ),
        "psychogenic",
        gate_key="abstract_claim_ready",
    ),
    Candidate(
        "psychogenic_implicitness_model_generalization",
        "implicitness_transport",
        4,
        "Psychogenic Machine (arXiv:2509.10970)",
        (
            "A synthetic study's reported +0.31 implicit-wording penalty "
            "does not generalize: it reverses to -0.289 across three current "
            "open instruct models and all eight released scenarios (exact "
            "sign-flip sensitivity p=0.008), with the same direction in 327 "
            "matched real histories."
        ),
        (
            "Published eight-model implicit-minus-explicit DCS direction and "
            "the released 16-case synthetic benchmark."
        ),
        (
            "Three-seed exact-prompt DCS on three current open instruct "
            "models, with matched-local synthetic and 327 matched real-history "
            "counterfactual pairs."
        ),
        (
            "This is a model-generalization failure, not a same-model "
            "contradiction. The gate requires negative pooled 99% intervals "
            "in all three designs, seed/model directional consistency, and "
            "eight leave-one-scenario-out checks. The scenario sign-flip is "
            "a symmetry sensitivity analysis, not a randomized-experiment "
            "p-value; the matched-local synthetic sensitivity is weaker "
            "(two-sided p=0.031). The paper's OpenRouter judge alias did not "
            "disclose its historical snapshot; this reproduction pins "
            "gpt-4o-mini-2024-07-18."
        ),
        "psychogenic",
        gate_key="model_generalization_claim_ready",
    ),
    Candidate(
        "synthetic_overestimates_direct_recognition_sensitivity",
        "recognition_sensitivity",
        6,
        "Lost in Delusion (arXiv:2606.00975)",
        (
            "Synthetic scripts overestimate direct delusion-recognition "
            "sensitivity on confirmed real messages."
        ),
        "Published FNR under the paper's exact disclosed classifier prompt.",
        (
            "FNR on 98 independently dual-adjudicated in-ontology endpoints "
            "from 63 source conversations under the identical published "
            "prompt text and three official-BF16 shared models."
        ),
        (
            "The prompt admits only three synthetic themes; headline wording "
            "must separate in-ontology transport from broader ontology "
            "coverage. The paper does not disclose the assessment pass's API "
            "message role; this reproduction records a user-role "
            "operationalization."
        ),
        "recognition",
    ),
    Candidate(
        "synthetic_recognition_model_gap_overestimation",
        "recognition_model_separation",
        6,
        "Lost in Delusion (arXiv:2606.00975)",
        (
            "Synthetic scripts overstate model separation in direct "
            "delusion recognition: a reported 71-point OLMo-Llama FNR gap is "
            "bounded to 1.0-24.5 points on strict in-ontology real endpoints "
            "with a 36.8-point clustered 99% upper bound."
        ),
        (
            "Published OLMo-minus-Llama false-negative-rate gap under the "
            "paper's exact disclosed classifier prompt."
        ),
        (
            "Adversarial partial-identification bound for the paired FNR gap "
            "on 98 dual-rubric in-ontology endpoints across 63 source "
            "conversations."
        ),
        (
            "Every malformed output is assigned to maximize the real gap and "
            "uncertainty is clustered by source conversation. This is "
            "descriptive transport, not a cross-study population interaction, "
            "because synthetic row-level predictions are unavailable. The two "
            "ontology rubrics are independently worded calls to the same "
            "pinned mini model, not independent human adjudicators."
        ),
        "recognition_gap",
    ),
    Candidate(
        "synthetic_controls_underestimate_near_miss_false_positives",
        "recognition_specificity",
        2,
        "Lost in Delusion (arXiv:2606.00975)",
        (
            "Synthetic distress controls underestimate false alarms on natural "
            "near-misses such as fiction, role-play, and quoted beliefs."
        ),
        "Published FPR on synthetic distress-only trajectories.",
        (
            "FPR on 66 explicit-evidence retained-context natural near-misses "
            "across 65 conversations under the identical published prompt text "
            "and three official-BF16 shared models; the 100-case consensus "
            "cohort is a sensitivity analysis."
        ),
        (
            "This is a hard-negative stress test, not a population false-positive "
            "rate. The paper does not disclose the assessment pass's API "
            "message role; this reproduction records a user-role "
            "operationalization."
        ),
        "recognition",
    ),
    Candidate(
        "lost_matched_effect_underestimation",
        "lost_confirmation_severity",
        11,
        "Lost in Delusion (arXiv:2606.00975)",
        (
            "Handcrafted trajectories underestimate models' sensitivity to "
            "delusional rather than grounded framing."
        ),
        "Synthetic delusion-minus-distress-control DCS.",
        (
            "Original-minus-validated-grounded-counterfactual DCS on 101 "
            "matched real histories."
        ),
        (
            "The synthetic and real controls preserve different aspects of the "
            "underlying interaction; this is an alternate, not an independent "
            "claim."
        ),
        "lost",
    ),
    Candidate(
        "natural_near_misses_are_harder_than_generated_controls",
        "recognition_specificity",
        2,
        "WildDelusion control-realism audit",
        (
            "Generated non-delusion controls make direct recognition look too "
            "specific: natural near-misses retain 29.6-38.7-point observed "
            "false-positive gaps after adversarial parse assignments, with "
            "99% lower bounds of 11.9-20.2 points."
        ),
        "Generated matched non-delusion controls.",
        (
            "Within-model natural-minus-generated FPR for the exact published "
            "classifier on 66 two-pass fixed-model-adjudicated non-delusion "
            "near-misses with verbatim retained-context exclusion evidence and "
            "101 generated non-delusion controls."
        ),
        (
            "Negative denotes non-delusion ground truth, not assistant safety. "
            "This is a hard non-delusion stress test, not population FPR. All "
            "unparsable outputs are assigned adversarially against the result, "
            "and uncertainty is clustered by source conversation. The two "
            "adjudication passes are not independent human labels."
        ),
        "recognition",
    ),
    Candidate(
        "lost_synthetic_leaderboard_separation_collapses",
        "model_ranking_transport",
        10,
        "Lost in Delusion (arXiv:2606.00975)",
        (
            "A wide synthetic safety leaderboard compresses to an unresolved "
            "near-tie on confirmed real endpoints."
        ),
        "Published absolute DCS ordering across four shared model families.",
        (
            "Paired endpoint DCS contrasts on 522 confirmed real endpoints per "
            "model."
        ),
        (
            "Two large-model checkpoints are FP8/AWQ sensitivity variants, so "
            "this cannot support a resolved ranking reversal."
        ),
        "lost",
    ),
    Candidate(
        "synthetic_three_theme_ontology_undercoverage",
        "theme_coverage",
        5,
        "Lost in Delusion (arXiv:2606.00975)",
        (
            "A three-theme handcrafted ontology misses much of real content: "
            "52.5% of confirmed endpoints are strict outside-ontology "
            "consensus (99% CI 44.0-61.0%)."
        ),
        (
            "Thirty personas restricted to emotional dependence, sentient AI, "
            "and spiritual/messianic themes."
        ),
        (
            "Dual-rubric closed-ontology adjudication of 522 confirmed real "
            "endpoints across 321 source conversations."
        ),
        (
            "Both rubrics use the same fixed mini-model. Report the 99% "
            "conversation-cluster interval and do not claim a population "
            "majority because it crosses 50%."
        ),
        "theme_coverage",
    ),
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def index_gates(
    lost: dict[str, Any] | None,
    context: dict[str, Any] | None,
    psychogenic: dict[str, Any] | None,
    recognition: dict[str, Any] | None,
    control_realism: dict[str, Any] | None,
    theme_coverage: dict[str, Any] | None,
    recognition_gap: dict[str, Any] | None,
    visible_mod_harm: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    if lost:
        indexed.update(
            {
                row["claim_id"]: row
                for row in lost.get("claim_gates", [])
            }
        )
    if psychogenic:
        indexed["psychogenic_implicitness_domain_drop"] = psychogenic
        indexed["psychogenic_implicitness_model_generalization"] = (
            psychogenic
        )
    if context:
        indexed["paired_context_relocalizes_non_grounding_failures"] = context
        indexed["paired_context_increases_non_grounding_severity"] = context
    if recognition:
        indexed.update(
            {
                row["claim_id"]: row
                for row in recognition.get("claim_gates", [])
            }
        )
    if control_realism:
        indexed[
            "natural_near_misses_are_harder_than_generated_controls"
        ] = control_realism
    if theme_coverage:
        indexed["synthetic_three_theme_ontology_undercoverage"] = (
            theme_coverage
        )
    if recognition_gap:
        indexed["synthetic_recognition_model_gap_overestimation"] = (
            recognition_gap
        )
    if visible_mod_harm:
        indexed["lost_confirmation_failures_mostly_outside_mod_harm"] = (
            visible_mod_harm
        )
    return indexed


def build_registry(
    gate_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    selected_families: set[str] = set()
    for candidate in sorted(CANDIDATES, key=lambda item: item.priority):
        gate = gate_index.get(candidate.claim_id)
        ready = bool(gate and gate.get(candidate.gate_key) is True)
        duplicate = candidate.evidence_family in selected_families
        selected = (
            ready
            and candidate.headline_eligible
            and not duplicate
            and len(selected_families) < 5
        )
        if selected:
            selected_families.add(candidate.evidence_family)
        status = (
            "selected"
            if selected
            else "excluded_exploratory"
            if not candidate.headline_eligible
            else "ready_but_duplicate"
            if ready and duplicate
            else "ready_below_top_five"
            if ready and len(selected_families) >= 5
            else "ready_secondary"
            if ready
            else "awaiting_evidence"
            if gate is None
            else "gate_closed"
        )
        rows.append(
            {
                **asdict(candidate),
                "gate_present": gate is not None,
                "gate_ready": ready,
                "status": status,
                "selected_for_abstract": selected,
                "gate": gate,
            }
        )
    return rows


def markdown_report(rows: list[dict[str, Any]]) -> str:
    selected = [row for row in rows if row["selected_for_abstract"]]
    lines = [
        "# Abstract claim registry",
        "",
        (
            f"**{len(selected)}/5 non-overlapping claims currently pass all "
            "locked gates.**"
        ),
        "",
        (
            "Headline gates use published prompts or rubrics, complete model "
            "runs, source-conversation clustering, disclosed judge families "
            "with pinned revisions, canonical unquantized primary checkpoints "
            "with pinned revisions where recorded, and 99% intervals for "
            "searched claim families. The context ablation uses official BF16 "
            "revision-pinned checkpoints and generates each unique exact "
            "tokenized prompt once, eliminating identical-input decoder drift. "
            "For Lost "
            "in Delusion, we match the active rendered reasoning rubric; a stale "
            "commented source block mentions an unpublished output-only main "
            "pass, so no claim assumes access to that undisclosed template."
        ),
        "",
        (
            "These are fail-closed discovery gates, not a preregistration. "
            "Candidate families and some thresholds were finalized during "
            "exploratory analysis after preliminary mini-judge or partial "
            "results; confirmatory wording requires a held-out corpus or "
            "held-out model set."
        ),
        "",
        "| Priority | Candidate | Family | Status |",
        "|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['priority']} | {row['headline']} | "
            f"`{row['evidence_family']}` | `{row['status']}` |"
        )
    lines.extend(["", "## Selected claims", ""])
    if not selected:
        lines.append("None yet. Partial or cross-protocol results do not count.")
    for index, row in enumerate(selected, 1):
        lines.extend(
            [
                f"{index}. **{row['headline']}**",
                f"   Guardrail: {row['guardrail']}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lost",
        type=Path,
        default=HERE / "lost_exact_claim_audit" / "claim_audit.json",
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=(
            EXPERIMENT
            / "full_history_522"
            / "results"
            / "context_ablation_paired"
            / "analysis_shared_context"
            / "claim_gate.json"
        ),
    )
    parser.add_argument(
        "--psychogenic",
        type=Path,
        default=HERE / "psychogenic_robustness" / "claim_gate.json",
    )
    parser.add_argument(
        "--recognition",
        type=Path,
        default=(
            EXPERIMENT
            / "delusion_recognition_transport"
            / "results"
            / "analysis"
            / "analysis_protocol.json"
        ),
    )
    parser.add_argument(
        "--theme-coverage",
        type=Path,
        default=HERE / "theme_coverage" / "claim_gate.json",
    )
    parser.add_argument(
        "--control-realism",
        type=Path,
        default=HERE / "control_realism" / "claim_gate.json",
    )
    parser.add_argument(
        "--recognition-gap",
        type=Path,
        default=HERE / "recognition_model_gap" / "claim_gate.json",
    )
    parser.add_argument(
        "--visible-mod-harm",
        type=Path,
        default=HERE / "visible_context_mod_harm" / "claim_gate.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "abstract_claim_registry",
    )
    args = parser.parse_args()

    sources = {
        "lost": read_json(args.lost),
        "context": read_json(args.context),
        "psychogenic": read_json(args.psychogenic),
        "recognition": read_json(args.recognition),
        "control_realism": read_json(args.control_realism),
        "theme_coverage": read_json(args.theme_coverage),
        "recognition_gap": read_json(args.recognition_gap),
        "visible_mod_harm": read_json(args.visible_mod_harm),
    }
    gates = index_gates(**sources)
    rows = build_registry(gates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "claim_registry.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    scalar_rows = [
        {key: value for key, value in row.items() if key != "gate"}
        for row in rows
    ]
    with (args.output_dir / "claim_registry.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_rows[0].keys())
        writer.writeheader()
        writer.writerows(scalar_rows)
    report = markdown_report(rows)
    (args.output_dir / "README.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
