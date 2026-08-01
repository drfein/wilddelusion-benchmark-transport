# SIM-VAIL Real-Intent Transport

## Question

Do the three high-risk psychosis intents highlighted by SIM-VAIL causally
increase the corresponding response risks when introduced into validated real
delusion conversations?

The tested intents and primary outcomes are copied from Weilnhammer et al.
(arXiv:2602.01347v2):

| User intent | Primary response outcome |
|---|---|
| Dependence and anthropomorphism | Boundary / dependence risk |
| Glorification and romanticization | Glorification / romanticization |
| Risky-action permission | Risky-action enablement |

Overall risk and belief reinforcement are secondary outcomes. The experiment
does not test whether an intent harms a human user; it tests whether changing a
user request changes a model response on the paper's response-risk rubric.

## Frozen design

- Source: 522 validated WildDelusion endpoints in 321 source conversations.
- Unit: one endpoint per source conversation, selected before generation.
- Cohort: 100 English endpoints with no clear pre-existing dependence,
  glorification, or risky-action request under a separate screening pass.
- Manipulation: append one target-intent or matched control sentence to the
  final user message while preserving every original source token.
- Lexical robustness: three independently written, word-count-matched cue pairs
  per intent, assigned to conversations by a fixed SHA-256 hash.
- Targets: `Qwen/Qwen3-4B-AWQ` and `Qwen/Qwen3-14B-AWQ`, with repository
  revisions pinned in run manifests.
- Sampling: three paired random seeds per target, intent, arm, and model at
  temperature 1.0; thinking disabled.
- Primary scoring: the released SIM-VAIL XML judge prompt and exact text of the
  five relevant 1-10 dimensions, scored blind to arm with GPT-5.2 at high
  reasoning as in the paper's independent-judge sensitivity analysis.
- Inference: paired target-level effects with source-conversation clustered
  bootstrap intervals; multiplicity correction across the six model-by-intent
  primary tests. Model-size interactions are secondary because the two
  checkpoints differ in more than parameter count.

## Pre-outcome scoring amendment

This amendment was frozen before any experimental response was judged. A
shortened five-score prompt failed a preregistered calibration gate twice on
two disjoint sets of 200 released SIM-VAIL turn labels, so it is not used for
outcome claims. The final scorer uses the paper's released XML prompt and exact
rubric text and must pass the same gate on a third disjoint 200-turn set.

Exact high-reasoning scoring is materially more expensive than estimated from
the shortened prompt. To remain within the study budget without substituting
an unvalidated cheap proxy, the confirmatory analysis uses the first seed in
the already frozen seed tuple (`17011`) and 80 of the 92 endpoints independently
detected as English, selected by a fixed SHA-256 hash, for every target/control
arm, model, and intent: 960
blind judgments total. All three seeds and all 100 conversations are still
generated and retained, but the remainder is not part of the primary claim.
These choices are independent of response text and outcome scores.

The local language check was added before outcome scoring after the blinded
generation-quality audit found that the original nano-model screen had labeled
8 clearly non-English endpoints as English. It is applied only to the original
endpoint text with `lingua-language-detector`; model responses and outcomes are
not inputs to the filter.

The same pre-outcome audit found that the original 512-token response cap
truncated roughly 82% of primary-cohort responses. The frozen 80-conversation,
one-seed cohort is therefore regenerated with a 1,024-token response cap before
judging to reduce premature cutoff. The estimand is response risk within the
first 1,024 generated tokens, and results include sensitivity estimates by
paired finish status. Prompts, models, sampling parameters, request seeds, and
inclusion are unchanged; however, vLLM was not bitwise invariant to the smaller batch
composition, so this is treated as a fresh pre-outcome stochastic draw rather
than a deterministic continuation. The 512-token files remain archived as a
superseded generation run.

## Claim gate

An intent mechanism transports only if:

1. the preregistered matching response dimension increases;
2. its 99% source-conversation bootstrap interval excludes zero after the
   frozen family of six model-by-intent primary tests is considered;
3. the direction is not driven by one cue template; and
4. blinded manual audit finds no condition leakage or malformed generations in
   a fixed sample.

No effect on a non-matching secondary dimension is relabeled as success.

## Published package

The repository contains the complete final confirmatory evidence in
`raw/*.jsonl.gz`, including all 960 generated responses and every unabridged
judge output. `raw/manifest.json` records hashes of both compressed and
uncompressed bytes. Superseded 512-token exploratory generations are excluded
from the final package rather than mixed with confirmatory evidence; their run
manifests remain in `provenance/` where relevant.

Verify all raw archives without API or GPU work:

```bash
python studies/sim_vail_real_intent_transport/reproduce.py verify
```

Recreate every numeric calibration and primary result exactly from the archived
raw evidence:

```bash
python -m pip install -r studies/sim_vail_real_intent_transport/requirements.txt
python studies/sim_vail_real_intent_transport/reproduce.py analyze
```

Outputs are written under the ignored `.repro/` directory and byte-compared
against the published JSON and CSV tables. Plot bytes are not compared because
font rendering varies by platform.

To run a new stochastic generation and judge replication, inspect and invoke
`fresh_reproduction.sh`. It requires both pinned AWQ checkpoints, a CUDA/vLLM
environment, and an OpenAI API key. The script refuses to begin unless both
`ALLOW_GPU=1` and `ALLOW_API=1` are set. A fresh run tests robustness but cannot
be bitwise identical to sampled model or API outputs.

## Directory map

| Path | Contents |
|---|---|
| `raw/` | Compressed raw inputs, responses, judge outputs, and byte-level manifest |
| `results/primary_exact/` | Final paired effects, sensitivity tables, examples, and figure |
| `results/judge_calibration_exact/` | Released-label judge calibration |
| `audit/` | Blinded manual generation audit and condition key |
| `provenance/` | Candidate IDs and pinned generation manifests |
| `upstream/` | Exact released SIM-VAIL prompt, commit, and license |
| `reproduce.py` | Zero-cost verification and exact numeric reproduction |
| `fresh_reproduction.sh` | Guarded API/GPU replication runner |
