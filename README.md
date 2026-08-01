# WildDelusion benchmark transport

An auditable reproduction package for testing whether safety conclusions from
handcrafted delusion benchmarks transport to 522 natural, context-dependent
WildDelusion endpoints.

The repository contains:

- the exact scripts and prompts used to construct, validate, generate, judge,
  analyze, and gate the experiments;
- immutable validated counterfactual inputs needed for exact cohort reuse;
- pinned dataset, upstream-code, model, and judge revisions;
- compact tables and fail-closed gates for every reported result; and
- one resumable runner that can rebuild all five result families.

Here, `all` means all five finalized, revision-audited result families in the
locked claim registry. The source snapshot also retains an earlier exploratory
six-model report. It is clearly marked historical because several secondary
checkpoint revisions were not preserved; this repository does not fabricate
those revisions or present that pass as an exact reproduction target.

## Headline results

All five claims pass their locked gates. These are discovery results, not a
preregistration.

| Result | Evidence |
|---|---|
| Handcrafted all-harm trajectories miss most real validation failures. | An intentionally overinclusive possible-Mod+Harm screen captured only **26.5-30.0%** of real responses scored Validate/Amplify. |
| Generated controls make recognition look too specific. | Natural non-delusion near-misses retained **29.6-38.7 percentage-point** observed FPR gaps after adversarial parse assignments; clustered 99% lower bounds were **11.9-20.2 points**. |
| The published implicit-wording effect did not generalize. | The reported +0.31 penalty reversed to **-0.289** across three current open models and all eight released scenarios; exact scenario sign-flip sensitivity **p=0.008**. |
| A three-theme synthetic ontology has limited coverage. | **52.5%** of real endpoints were strict outside-ontology consensus, with clustered **99% CI 44.0-61.0%**. The interval crosses 50%, so this is not a population-majority claim. |
| Synthetic scripts overstate recognition-model separation. | A published 71-point OLMo-Llama FNR gap was bounded to **1.0-24.5 points** on strict in-ontology real endpoints; clustered 99% upper bound **36.8 points**. |

The authoritative claim wording, guardrails, and gate evidence are in
[results/README.md](results/README.md) and
[results/COMPLETION_AUDIT.md](results/COMPLETION_AUDIT.md).

## Additional completed transport study

[`studies/sim_vail_real_intent_transport`](studies/sim_vail_real_intent_transport)
contains the complete SIM-VAIL real-context experiment. On 80 fixed real
WildDelusion endpoints, dependence, special-insight, and action-seeking cues
causally increased their matching response-risk dimensions in both Qwen3-4B
and Qwen3-14B. All six 99% source-conversation intervals exclude zero after
Holm correction, and all 18 cue-template effects agree in direction.

The study directory includes the full compressed raw evidence: selected
conversation contexts, all 960 final responses, every raw judge rationale,
calibration data, manifests, analysis code, and a zero-cost exact numeric
reproduction command:

```bash
python studies/sim_vail_real_intent_transport/reproduce.py analyze
```

This study is additive to the original five-family locked claim registry; it
does not silently alter that earlier selection procedure.

[`studies/shimgekar_delusionscore_construct_transport`](studies/shimgekar_delusionscore_construct_transport)
tests whether the MiniLM-plus-logistic DelusionScore construct transports from
easy broad controls to natural context-resolved near misses. A source-matched
broad-control model reached **0.979** out-of-fold AUC, then fell to **0.670**
(clustered 95% CI **0.617-0.720**) with **84.5%** false positives on 207 held-out
natural near misses. The apparent longitudinal rise was not significant before
the retrieval-selected endpoint after length and conversation fixed effects
(`p=0.162`). This is a construct stress test, not an exact rerun of the
unreleased fitted classifier.

## Quick verification

This verifies the packaged gates and hashes. It performs no network, API, or GPU
work.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
wd-reproduce verify
pytest
```

## Complete reproduction

Inspect every command first:

```bash
wd-reproduce list
wd-reproduce run all --dry-run --no-resume --allow-api --allow-gpu
```

Then configure credentials and licensed inputs:

```bash
cp .env.example .env
set -a
source .env
set +a

wd-reproduce doctor
wd-reproduce run all --allow-api --allow-gpu
```

The runner refuses API or GPU stages unless the matching flag is supplied.
Every finished stage writes its rendered command, timestamps, and output hashes
under `.repro/state/`. Interrupted JSONL jobs resume from durable rows, while
the runner skips only outputs whose recorded hashes still match.

## What “from scratch” means

The primary reproduction starts from two immutable research inputs:

1. the public 522-row
   [WildDelusionCombined](https://huggingface.co/datasets/danielfein/WildDelusionCombined)
   release at revision `f547346...`; and
2. the frozen, validated counterfactual example sets in [frozen](frozen).

The counterfactuals are frozen because they were generated and adjudicated by
LLMs. Reissuing the same API request is not bitwise deterministic and can
change which examples pass. The exact generation, repair, and validation code
is retained under
[experiments/lost_in_delusion_replication_260600975](experiments/lost_in_delusion_replication_260600975);
[docs/REBUILDING_EXAMPLES.md](docs/REBUILDING_EXAMPLES.md) maps those scripts
to each frozen artifact. Rebuilding them tests robustness to a new draw;
installing them reproduces the reported estimands.

The public 522-row source release predates its finalized delusion-theme fields.
[`frozen/theme_labels.jsonl`](frozen/theme_labels.jsonl) is a text-free,
message-hash-keyed overlay that restores those exact labels without duplicating
conversation text. The fetch stage requires a one-to-one match before building
the cohort.

Five headline natural controls use LMSYS-Chat-1M source text. Its license does
not permit republishing those conversations. The aggregate result is included,
but an exact raw rerun requires a locally authorized control bundle via
`WD_NATURAL_CONTROL_DIR`. No restricted text or API key is committed.

## Repository map

| Path | Purpose |
|---|---|
| [configs/pipeline.toml](configs/pipeline.toml) | Complete declarative stage graph and exact commands |
| [src/wilddelusion_repro](src/wilddelusion_repro) | Small generic runner, hashing, and integrity checks |
| [experiments/lost_in_delusion_replication_260600975](experiments/lost_in_delusion_replication_260600975) | Exact study scripts, prompts, parsers, tests, and result gates |
| [frozen](frozen) | Immutable validated generated examples, separately licensed |
| [results](results) | Compact final claim registry and completion audit |
| [provenance](provenance) | File hashes, upstream commits, model revisions, and source map |
| [docs/METHODS.md](docs/METHODS.md) | Estimands and claim-specific workflows |
| [DATA_LICENSE.md](DATA_LICENSE.md) | Dataset restrictions and sensitive-data boundary |
| [studies/sim_vail_real_intent_transport](studies/sim_vail_real_intent_transport) | Complete SIM-VAIL intent-mechanism transport study and raw evidence |
| [studies/shimgekar_delusionscore_construct_transport](studies/shimgekar_delusionscore_construct_transport) | Natural-control and longitudinal stress test of the DelusionScore construct |

## Reproducibility boundary

- Open-model generations use official BF16 checkpoints at pinned revisions.
- Llama access requires acceptance of its Hugging Face terms.
- API labels use pinned snapshots where the provider exposes one.
- The original Psychosis-Bench OpenRouter judge alias did not expose its
  historical snapshot; this reproduction pins `gpt-4o-mini-2024-07-18`.
- Bootstrap uncertainty resamples source-conversation clusters, not target rows.
- Generated and natural negatives are non-delusion ground truth for a
  recognition classifier. “Negative” does not mean unsafe assistant behavior.

See [docs/REPRODUCING.md](docs/REPRODUCING.md) for hardware, stage-specific
commands, and expected artifact counts.
