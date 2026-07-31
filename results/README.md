# Abstract claim registry

**5/5 non-overlapping claims currently pass all locked gates.**

Headline gates use published prompts or rubrics, complete model runs, source-conversation clustering, disclosed judge families with pinned revisions, canonical unquantized primary checkpoints with pinned revisions where recorded, and 99% intervals for searched claim families. The context ablation uses official BF16 revision-pinned checkpoints and generates each unique exact tokenized prompt once, eliminating identical-input decoder drift. For Lost in Delusion, we match the active rendered reasoning rubric; a stale commented source block mentions an unpublished output-only main pass, so no claim assumes access to that undisclosed template.

These are fail-closed discovery gates, not a preregistration. Candidate families and some thresholds were finalized during exploratory analysis after preliminary mini-judge or partial results; confirmatory wording requires a held-out corpus or held-out model set.

| Priority | Candidate | Family | Status |
|---:|---|---|---|
| 1 | Handcrafted all-harm trajectories miss most real validation failures: even an intentionally overinclusive screen of the exact assistant-visible context captures only 26.5-30.0% of responses scored Validate/Amplify. | `lost_case_composition` | `selected` |
| 2 | Synthetic distress controls underestimate false alarms on natural near-misses such as fiction, role-play, and quoted beliefs. | `recognition_specificity` | `awaiting_evidence` |
| 2 | Generated non-delusion controls make direct recognition look too specific: natural near-misses retain 29.6-38.7-point observed false-positive gaps after adversarial parse assignments, with 99% lower bounds of 11.9-20.2 points. | `recognition_specificity` | `selected` |
| 3 | Single-turn evaluation hides context-conditioned non-grounding: aggregate rates can remain stable while the failing cases change. | `context_conditioned_localization` | `gate_closed` |
| 3 | Synthetic trajectories overestimate the effect of implicit rather than explicit wording in matched real histories. | `implicitness_transport` | `gate_closed` |
| 4 | Retaining conversational history increases the ordinal severity of non-grounding responses even when binary prevalence changes little. | `context_conditioned_localization` | `gate_closed` |
| 4 | A synthetic study's reported +0.31 implicit-wording penalty does not generalize: it reverses to -0.289 across three current open instruct models and all eight released scenarios (exact sign-flip sensitivity p=0.008), with the same direction in 327 matched real histories. | `implicitness_transport` | `selected` |
| 5 | A three-theme handcrafted ontology misses much of real content: 52.5% of confirmed endpoints are strict outside-ontology consensus (99% CI 44.0-61.0%). | `theme_coverage` | `selected` |
| 6 | Synthetic scripts overestimate direct delusion-recognition sensitivity on confirmed real messages. | `recognition_sensitivity` | `awaiting_evidence` |
| 6 | Synthetic scripts overstate model separation in direct delusion recognition: a reported 71-point OLMo-Llama FNR gap is bounded to 1.0-24.5 points on strict in-ontology real endpoints with a 36.8-point clustered 99% upper bound. | `recognition_model_separation` | `selected` |
| 8 | The reported 0.39-point OLMo-Llama gap on handcrafted trajectories does not transport to confirmed real endpoints. | `primary_model_comparison` | `ready_below_top_five` |
| 9 | Handcrafted harm trajectories understate observed delusion-perpetuation/confirmation severity at confirmed real-world endpoints. | `lost_confirmation_severity` | `ready_below_top_five` |
| 10 | A wide synthetic safety leaderboard compresses to an unresolved near-tie on confirmed real endpoints. | `model_ranking_transport` | `gate_closed` |
| 11 | Handcrafted trajectories underestimate models' sensitivity to delusional rather than grounded framing. | `lost_confirmation_severity` | `gate_closed` |

## Selected claims

1. **Handcrafted all-harm trajectories miss most real validation failures: even an intentionally overinclusive screen of the exact assistant-visible context captures only 26.5-30.0% of responses scored Validate/Amplify.**
   Guardrail: This is a conservative corpus-composition result, not evidence that crisis wording causally improves model behavior.
2. **Generated non-delusion controls make direct recognition look too specific: natural near-misses retain 29.6-38.7-point observed false-positive gaps after adversarial parse assignments, with 99% lower bounds of 11.9-20.2 points.**
   Guardrail: Negative denotes non-delusion ground truth, not assistant safety. This is a hard non-delusion stress test, not population FPR. All unparsable outputs are assigned adversarially against the result, and uncertainty is clustered by source conversation. The two adjudication passes are not independent human labels.
3. **A synthetic study's reported +0.31 implicit-wording penalty does not generalize: it reverses to -0.289 across three current open instruct models and all eight released scenarios (exact sign-flip sensitivity p=0.008), with the same direction in 327 matched real histories.**
   Guardrail: This is a model-generalization failure, not a same-model contradiction. The gate requires negative pooled 99% intervals in all three designs, seed/model directional consistency, and eight leave-one-scenario-out checks. The scenario sign-flip is a symmetry sensitivity analysis, not a randomized-experiment p-value; the matched-local synthetic sensitivity is weaker (two-sided p=0.031). The paper's OpenRouter judge alias did not disclose its historical snapshot; this reproduction pins gpt-4o-mini-2024-07-18.
4. **A three-theme handcrafted ontology misses much of real content: 52.5% of confirmed endpoints are strict outside-ontology consensus (99% CI 44.0-61.0%).**
   Guardrail: Both rubrics use the same fixed mini-model. Report the 99% conversation-cluster interval and do not claim a population majority because it crosses 50%.
5. **Synthetic scripts overstate model separation in direct delusion recognition: a reported 71-point OLMo-Llama FNR gap is bounded to 1.0-24.5 points on strict in-ontology real endpoints with a 36.8-point clustered 99% upper bound.**
   Guardrail: Every malformed output is assigned to maximize the real gap and uncertainty is clustered by source conversation. This is descriptive transport, not a cross-study population interaction, because synthetic row-level predictions are unavailable. The two ontology rubrics are independently worded calls to the same pinned mini model, not independent human adjudicators.
