# Agreement-ratchet causal study

This study changes only the immediately preceding assistant message in 45
known GPT-4.1-mini context failures. It compares an agreement-preserving
paraphrase against an epistemically neutral rewrite while holding the later
user target and all other history fixed.

The protocol was frozen before counterfactual generation. Raw conversations,
rewrites, model responses, and judgment rationales are stored under gitignored
`artifacts/**/private/`; publishable outputs contain no message text.

See `PREREGISTRATION.md` for the estimand and interpretation boundary.

The study is complete. Agreement preservation produced 64.9% endorsement versus
59.6% after neutralization, a paired +5.33-point effect whose 95% interval
crossed zero. See `RESULTS.md` for the full result.

## Reproduce

The scripts are intentionally stage-separated so the pre-outcome artifacts can
be inspected before model generation. Paths below assume the completed parent
context study.

```bash
BASE=studies/agreement_ratchet_causal
PARENT=studies/nicholls_context_mechanism
RUN=$BASE/artifacts/run

python $BASE/prepare_cohort.py \
  --release $PARENT/artifacts/hf_rehydrated/train-00000-of-00001.parquet \
  --paired-scores $PARENT/published_results/gpt41mini/paired_scores.csv \
  --chunks $PARENT/artifacts/context_mechanism_run/prefixes/chunk_index.parquet \
  --membership $PARENT/artifacts/context_mechanism_run/prefixes/private/prefix_membership.parquet \
  --labels $PARENT/artifacts/context_mechanism_run/prefixes/private/prefix_labels_nano.jsonl \
  --transition-module $PARENT/analyze_transition_contexts.py --out-dir $RUN
python $BASE/generate_rewrites.py \
  --input $RUN/private/cohort.jsonl --output $RUN/private/rewrites.jsonl \
  --env-file "$OPENAI_ENV"
python $BASE/validate_rewrites.py \
  --cohort $RUN/private/cohort.jsonl --rewrites $RUN/private/rewrites.jsonl \
  --output $RUN/private/rewrite_validation.jsonl \
  --public-output $RUN/rewrite_validation.csv --env-file "$OPENAI_ENV"
python $BASE/prepare_intervention_inputs.py \
  --cohort $RUN/private/cohort.jsonl --rewrites $RUN/private/rewrites.jsonl \
  --out-dir $RUN
python $PARENT/generate_openai_responses.py \
  --input $RUN/private/generation_inputs.jsonl \
  --output $RUN/private/generations.jsonl --env-file "$OPENAI_ENV"
python $PARENT/judge_endorsement.py \
  --input $RUN/private/generations.jsonl \
  --output $RUN/private/judgments.jsonl --env-file "$OPENAI_ENV"
python $BASE/analyze.py --judgments $RUN/private/judgments.jsonl \
  --validation $RUN/rewrite_validation.csv --out-dir $RUN/results
```
