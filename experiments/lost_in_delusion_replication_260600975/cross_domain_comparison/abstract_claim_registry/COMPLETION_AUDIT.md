# Abstract claim completion audit

**PASS: exactly five non-overlapping claims satisfy every claim-specific gate.**

## 1. Handcrafted all-harm trajectories miss most real validation failures: even an intentionally overinclusive screen of the exact assistant-visible context captures only 26.5-30.0% of responses scored Validate/Amplify.

- Evidence family: `lost_case_composition`
- Source gate SHA-256: `a5e01cd7124cbfe6203383dcb47ccbc5320737bffec172770283d84246fe9c4c`
- Verified evidence: `{"endpoints_per_model": 522, "guardrail": "This is a conservative corpus-composition result, not evidence that adding crisis language causes a safer response.", "models": 2, "possible_mod_harm_endpoints": 161, "validate_amplify_captured_range": [0.26530612244897955, 0.3003300330033003]}`
- Guardrail: This is a conservative corpus-composition result, not evidence that crisis wording causally improves model behavior.

## 2. Generated non-delusion controls make direct recognition look too specific: natural near-misses retain 29.6-38.7-point observed false-positive gaps after adversarial parse assignments, with 99% lower bounds of 11.9-20.2 points.

- Evidence family: `recognition_specificity`
- Source gate SHA-256: `2f386b6a53cb02c876a3eb31f4802b0e610ad14f20066f417ed3081429bcd955`
- Verified evidence: `{"ci99_lower_bound_range": [0.11899529042386187, 0.20154279228026062], "guardrail": "Negative means non-delusion ground truth for the recognition classifier; it does not refer to assistant validation or safety.", "models": 2, "worst_case_gap_range": [0.29552955295529554, 0.38703870387038697]}`
- Guardrail: Negative denotes non-delusion ground truth, not assistant safety. This is a hard non-delusion stress test, not population FPR. All unparsable outputs are assigned adversarially against the result, and uncertainty is clustered by source conversation. The two adjudication passes are not independent human labels.

## 3. A synthetic study's reported +0.31 implicit-wording penalty does not generalize: it reverses to -0.289 across three current open instruct models and all eight released scenarios (exact sign-flip sensitivity p=0.008), with the same direction in 327 matched real histories.

- Evidence family: `implicitness_transport`
- Source gate SHA-256: `f0bd0f85c52cf1bd9d977d021e19d26b043a6e8d0a406990601421548a8e7dd9`
- Verified evidence: `{"guardrail": "Exact finite sign-flip sensitivity over authored scenario-pair effects; not a randomized-experiment p-value because script explicitness was not randomly assigned.", "matched_real_pairs": 327, "models": 3, "released_scenarios": 8, "scenario_sign_flip_two_sided": 0.0078125, "synthetic_full_ci99": [-0.4753086419753087, -0.11265432098765434], "synthetic_full_delta": -0.28858024691358025}`
- Guardrail: This is a model-generalization failure, not a same-model contradiction. The gate requires negative pooled 99% intervals in all three designs, seed/model directional consistency, and eight leave-one-scenario-out checks. The scenario sign-flip is a symmetry sensitivity analysis, not a randomized-experiment p-value; the matched-local synthetic sensitivity is weaker (two-sided p=0.031). The paper's OpenRouter judge alias did not disclose its historical snapshot; this reproduction pins gpt-4o-mini-2024-07-18.

## 4. A three-theme handcrafted ontology misses much of real content: 52.5% of confirmed endpoints are strict outside-ontology consensus (99% CI 44.0-61.0%).

- Evidence family: `theme_coverage`
- Source gate SHA-256: `bdeb69ee4de02477359f02bb2df9140153adbb321e7f0e91fb59e449c509680d`
- Verified evidence: `{"adjudicator": "gpt-5.4-mini-2026-03-17", "ci99": [0.4400825891867118, 0.6099011305385859], "guardrail": ["Do not call the two rubrics independent human adjudicators.", "Do not claim a population majority because the 99% interval crosses 50%.", "Do not treat unresolved rubric disagreements as in-ontology."], "outside_rows": 274, "outside_share": 0.524904214559387, "total_rows": 522}`
- Guardrail: Both rubrics use the same fixed mini-model. Report the 99% conversation-cluster interval and do not claim a population majority because it crosses 50%.

## 5. Synthetic scripts overstate model separation in direct delusion recognition: a reported 71-point OLMo-Llama FNR gap is bounded to 1.0-24.5 points on strict in-ontology real endpoints with a 36.8-point clustered 99% upper bound.

- Evidence family: `recognition_model_separation`
- Source gate SHA-256: `14d2b344628f30dfd3c6932dd28f0842fb28185d4d6a52be146e9b0793819660`
- Verified evidence: `{"cohort": "98 dual-rubric strict in-ontology real endpoints across 63 source conversations", "guardrail": "This is a descriptive benchmark-transport result, not a cross-study population interaction: the paper releases rounded aggregate classifier rates but not row-level outputs. The paper also does not disclose whether its assessment prompt was sent as a user or system message; this reproduction records the user-role operationalization.", "paper_gap": 0.7100000000000001, "real_clustered_ci99_upper": 0.3684210526315789, "real_point_bounds": [0.01020408163265306, 0.24489795918367346]}`
- Guardrail: Every malformed output is assigned to maximize the real gap and uncertainty is clustered by source conversation. This is descriptive transport, not a cross-study population interaction, because synthetic row-level predictions are unavailable. The two ontology rubrics are independently worded calls to the same pinned mini model, not independent human adjudicators.
