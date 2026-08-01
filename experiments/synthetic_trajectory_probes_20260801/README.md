# Model-specific synthetic trajectory probes

This experiment asks whether a model's pre-response residual-stream activation
predicts its own imminent or future delusion endorsement on conversations it
has not seen during probe training.

## Design

- Stimuli: Psychosis-Bench commit
  `73966f95be2e40f1ceee76dfe08eebe939ad6c21`, 16 scripted 12-turn cases.
- Target models: Llama-3.1-8B-Instruct, Qwen3-8B, and Gemma-3-12B-IT at pinned
  revisions in `experiment.json`. The gated Llama and Gemma repositories are
  loaded from pinned full-precision Unsloth mirrors, also recorded there.
- Each target model generates every assistant message, so both context and
  activation trajectories are model-specific.
- Activations are captured immediately before each assistant response at four
  depth fractions.
- Independent trajectories are batched at the same turn. Responses may use up
  to 1,536 tokens, and cap hits are recorded rather than silently truncated.
- Explicit and implicit variants of one underlying scenario remain in the same
  outer and inner folds. Repetitions never cross folds independently.
- Outcomes: endorsement on the current turn and at fixed future horizons.
- Labeling: the exact released Psychosis-Bench 0/1/2 DCS rubric, evaluated from
  the complete generated trajectory through the target response.
- Judges: Qwen3-8B is the primary judge and Gemma-3-12B-IT is an independent
  sensitivity judge. Three-class and binary endorsement agreement are saved;
  Qwen labels remain primary for Gemma, avoiding a self-judge primary endpoint.
- Baselines: turn/condition metadata and user-text features. A probe is useful
  only if it improves on these scripted-trajectory baselines out of group.

Psychosis-Bench has only eight independent scenario pairs. Results are a
mechanistic pilot and require replication on the larger Lost in Delusion
persona set before a broad synthetic-data claim.

## Run

Fetch the exact stimuli:

```bash
python fetch_psychosis_bench.py \
  --output artifacts/private/test_cases.json \
  --manifest artifacts/test_cases_manifest.json
```

Set `ROOT`, `QWEN`, `LLAMA`, and `GEMMA` if the defaults do not match the GPU
machine, then run:

```bash
./run_gpu_pipeline.sh
```

Each model directory contains raw responses, one compressed activation file per
turn, two independent DCS judgment files, judge agreement, out-of-fold
predictions, and current/future-horizon summaries. The unit of generalization
is `scenario_pair`; individual turns and repeated trajectories never define a
fold.

If `REAL_COHORT` points to the frozen 187-history WildDelusion cohort, the same
runner also generates five continuations per history for Llama and Gemma and
caches one deterministic pre-response activation per history. These outputs are
used to train real-domain probes before frozen evaluation on Psychosis-Bench.

After copying GPU outputs under `artifacts/private/transfer/gpu_export`, run
`run_spirals_judging_and_transfer.sh` with `ENV_FILE` set. It uses the same
released `bot-endorses-delusion` rubric and GPT-5.4-mini judge as the original
Qwen real-data probe, fits Llama/Gemma probes using only real histories, and
applies all three frozen probe ensembles to synthetic activations without
recalibration.
