# Runbook

All commands run from this directory. GPU outputs are copied back under the
gitignored `artifacts/private/` directory after every expensive stage.

```bash
python prepare_inputs.py \
  --input ../agreement_context_scaled/artifacts/run/private/executable_cohort.jsonl \
  --private-output artifacts/private/cohort.jsonl \
  --manifest artifacts/cohort_manifest.json

python generate_open_model.py \
  --input artifacts/private/cohort.jsonl \
  --output artifacts/private/generations.jsonl \
  --manifest artifacts/generation_manifest.json \
  --activation-dir artifacts/private/activations \
  --activation-index artifacts/private/activation_index.jsonl

python judge_endorsement.py \
  --input artifacts/private/generations.jsonl \
  --output artifacts/private/judgments.jsonl \
  --manifest artifacts/judgment_manifest.json \
  --env-file /path/to/private/.env

python fit_activation_probe.py \
  --cohort artifacts/private/cohort.jsonl \
  --judgments artifacts/private/judgments.jsonl \
  --activation-dir artifacts/private/activations \
  --checkpoint-dir artifacts/checkpoints/probes \
  --predictions artifacts/private/probe_oof_predictions.jsonl \
  --summary artifacts/probe_summary.json

python analyze_uncertainty.py \
  --cohort artifacts/private/cohort.jsonl \
  --judgments artifacts/private/judgments.jsonl \
  --predictions artifacts/private/uncertainty_oof_predictions.jsonl \
  --summary artifacts/uncertainty_summary.json
```

Subsequent GPU stages use `attribute_probe_tokens.py`,
`attribute_probe_leave_one_out.py`, and `attribute_fixed_response.py`. The
behavioral intervention is materialized with
`prepare_behavior_interventions.py` and passed back through the same generator
and judge.

