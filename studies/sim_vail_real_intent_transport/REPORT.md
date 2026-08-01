# SIM-VAIL intent mechanisms transport to real delusion contexts

## Result

Three user intents identified as high-risk in the synthetic SIM-VAIL study
causally increased their matching response-risk dimensions when inserted into
validated real delusion conversations. This held for both Qwen3-4B and
Qwen3-14B, for all three independently written cue templates per intent, and
under the full predeclared claim gate.

| Model | Paired manipulation | Control mean | Target mean | Paired effect | 99% CI | Holm p |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B | dependence request -> boundary/dependence risk | 1.79 | 3.31 | +1.53 | [1.06, 2.03] | 0.000030 |
| Qwen3-14B | dependence request -> boundary/dependence risk | 1.86 | 3.36 | +1.50 | [0.99, 2.05] | 0.000030 |
| Qwen3-4B | special-insight request -> glorification/romanticization | 4.29 | 5.70 | +1.41 | [0.84, 1.99] | 0.000030 |
| Qwen3-14B | special-insight request -> glorification/romanticization | 4.48 | 5.63 | +1.15 | [0.64, 1.68] | 0.000030 |
| Qwen3-4B | action request -> risky-action enablement | 2.21 | 4.09 | +1.88 | [1.30, 2.48] | 0.000030 |
| Qwen3-14B | action request -> risky-action enablement | 2.41 | 4.09 | +1.68 | [1.09, 2.29] | 0.000030 |

Scores use SIM-VAIL's 1-10 response-risk rubrics. Intervals are 99%
source-conversation bootstrap intervals. P-values are two-sided source-level
sign-flip tests with Holm correction across the six confirmatory tests. The
Monte Carlo p-value floor with 200,000 draws is 0.000005 before correction.

## Interpretation

This validates a specific central mechanism from SIM-VAIL outside its fully
synthetic interaction loop: within real delusion-like conversational contexts,
what the user asks the model to do changes the kind of safety failure that the
model produces. Dependence language selectively increases boundary/dependence
risk, requests to recognize special insight increase romanticization, and
requests for next steps increase risky-action enablement.

The experiment does **not** establish that these responses cause downstream
human harm, that all SIM-VAIL vulnerabilities or intents transport, or that the
paper's model rankings transport. It also does not estimate prevalence in a
random sample of chatbot traffic: WildDelusion was selected for delusion-like
content. The causal estimand is the response-risk change produced by adding an
intent cue to an already selected real context.

There was no detectable difference in treatment-effect size between the 4B and
14B checkpoints. The 4B-minus-14B interaction estimates were +0.03 for
dependence (99% CI [-0.51, 0.58]), +0.26 for glorification ([-0.45, 0.96]), and
+0.20 for risky action ([-0.46, 0.86]). This is evidence of similar effects in
these two checkpoints, not evidence that model scale is generally irrelevant.

## Design

- Source pool: 522 validated WildDelusion endpoints from 321 source
  conversations.
- Primary cohort: 80 independently language-filtered English endpoints, with
  one endpoint per source conversation and inclusion fixed before outcome
  scoring.
- Intervention: the original conversation was preserved verbatim and one
  target-intent or word-count-matched control sentence was appended to the last
  user message.
- Robustness: three independently written cue pairs per intent were assigned by
  fixed SHA-256 hash. All 18 model-by-intent-by-template effects were positive.
- Models: pinned `Qwen/Qwen3-4B-AWQ` and `Qwen/Qwen3-14B-AWQ` revisions,
  temperature 1.0, thinking disabled, paired request seed 17011.
- Outcomes: the released SIM-VAIL XML judge prompt and exact five relevant
  rubric texts, scored blind to condition with `gpt-5.2-2025-12-11` at high
  reasoning.
- Manual audit: 18 blinded pairs, stratified over model, intent, and completion
  status; all passed coherence and leakage checks.

## Judge validation

Two cheaper shortened prompts failed the frozen calibration gate and were not
used for claims. The final exact scorer passed on a third, disjoint set of 200
released SIM-VAIL turns:

| Primary dimension | Spearman rho | ICC(3,1) | MAE |
|---|---:|---:|---:|
| boundary/dependence risk | 0.740 | 0.831 | 0.940 |
| glorification/romanticization | 0.842 | 0.872 | 0.825 |
| risky-action enablement | 0.765 | 0.767 | 1.300 |

The exact outcome run produced 960/960 valid judgments with no failed API calls
or parses. Input and output artifacts are content-hashed in the manifests.

## Sensitivity and limitations

- The three cue-template effects were positive in every confirmatory cell,
  ranging from +0.57 to +3.24 points.
- Dependence and glorification effects remained positive in every paired
  finish-status stratum. Risky-action effects were strong when either response
  reached the 1,024-token window but small among the few stop/stop pairs. This
  makes response length an important limitation for the risky-action estimate,
  although target cues were not simply more likely to truncate than controls.
- A blinded pre-outcome audit found that the original 512-token run truncated
  too often. The fixed cohort was regenerated at 1,024 tokens before any
  outcome was viewed. The estimand is therefore risk within the first 1,024
  generated tokens.
- Qwen3-4B and Qwen3-14B are only two open checkpoints and differ in more than
  nominal size. Replication across model families is needed before making a
  scale claim.
- The manipulation adds an explicit intent sentence. It does not estimate the
  effect of subtler, naturally occurring intent differences.

## Reproducibility artifacts

- Complete raw prompts/responses/judgments: `raw/*.jsonl.gz`
- Raw byte-level manifest: `raw/manifest.json`
- Primary statistics: `results/primary_exact/primary_results.json`
- Conversation-level paired data: `results/primary_exact/conversation_pair_deltas.csv`
- Model interactions: `results/primary_exact/model_interactions.json`
- Source sensitivity: `results/primary_exact/source_sensitivity.json`
- Largest paired examples: `results/primary_exact/top_pairs.json`
- Main figure: `results/primary_exact/primary_effects.png` and `.pdf`
- Outcome manifest: `results/primary_exact/manifest.json`
- Judge calibration: `results/judge_calibration_exact/summary.json`
- Full frozen design and amendments: `README.md`
- Exact zero-cost reproduction: `python reproduce.py analyze`
