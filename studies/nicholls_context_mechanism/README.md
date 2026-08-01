# Complete-history context mechanism study

This directory tests whether a matched context effect is associated with
delusion-content density or with more general conversational accumulation.

## Integrity correction

The original 433-row ablation used histories embedded in
`WildDelusionVerified`. Those are not complete conversations for most ShareChat
rows: only 24/433 targets had a prior assistant message. No feature-outcome
analysis from those histories is valid for a dialogue-context claim. Labels
created before this discovery are quarantined and ignored.

The final repair does not stop at the 480-transcript canonical archive.
`rehydrate_hf_release.py` reconstructs primary rows from their retained upstream
offsets and pinned source revisions, then uses the archive for the legacy rows.
It fails closed unless every target resolves to an exact user turn and the
recovered transcript contains assistant messages.

Current release integrity result:

- 522 target rows across 321 source conversations are structurally complete;
- 450 targets have at least one prior assistant message;
- 521 rows have text available for every source node; and
- one Grok transcript retains four explicit placeholders for image-only
  assistant nodes whose source text field is null.

No transcript is silently clipped. Model runs must use the prefix through
`target_message_index`, declare a context limit, and exclude rather than
truncate prompts that do not fit.

## Rebuild the release histories

```bash
python studies/nicholls_context_mechanism/rehydrate_hf_release.py \
  --release /path/to/WildDelusionCombined/train.parquet \
  --canonical-conversations /path/to/wilddelusion_combined_conversations.jsonl \
  --out-dir studies/nicholls_context_mechanism/artifacts/hf_rehydrated

python studies/nicholls_context_mechanism/validate_rehydrated_release.py \
  --original /path/to/WildDelusionCombined/train.parquet \
  --rehydrated studies/nicholls_context_mechanism/artifacts/hf_rehydrated/train-00000-of-00001.parquet \
  --report studies/nicholls_context_mechanism/artifacts/hf_rehydrated/integrity_report.json
```

Raw fetch caches and rebuilt artifacts are gitignored. The corrected release is
hosted at `danielfein/WildDelusionCombined`.

## Analysis status

The complete-history experiment and a post-result GPT-4.1-mini replication are
finished. GPT-5.4 mini increased from 0/445 target-only endorsements to 6/445
with complete history, but its sparse outcome did not resolve the mechanism.
On the exact same targets, GPT-4.1 mini increased from 21/445 to 186/445, and
the context effect was strongly associated with prior delusion-content density.
See `RESULTS.md` for estimates, uncertainty, coder sensitivity, and
interpretation boundaries. The frozen specification and dated amendments are
in `PREREGISTRATION.md`.

## Reproduce the GPT-4.1-mini replication

After rebuilding the release and the frozen prefix labels, the following
commands recreate the exact common-cohort run. Set `OPENAI_ENV` to a local env
file containing `OPENAI_API_KEY`; private text and responses remain gitignored.

```bash
STUDY=studies/nicholls_context_mechanism
RUN=$STUDY/artifacts/context_mechanism_gpt41mini_common445
PREFIX=$STUDY/artifacts/context_mechanism_run/prefixes
RELEASE=$STUDY/artifacts/hf_rehydrated/train-00000-of-00001.parquet
REFERENCE=$STUDY/published_results/paired_scores.csv

python $STUDY/prepare_complete_context_inputs.py \
  --release "$RELEASE" --out-dir "$RUN" \
  --model gpt-4.1-mini-2025-04-14 --context-window 1047576 \
  --row-indices-from "$REFERENCE"
python $STUDY/generate_openai_responses.py \
  --input "$RUN/private/generation_inputs.jsonl" \
  --output "$RUN/private/generations.jsonl" \
  --env-file "$OPENAI_ENV" --concurrency 16
python $STUDY/judge_endorsement.py \
  --input "$RUN/private/generations.jsonl" \
  --output "$RUN/private/endorsement_judgments.jsonl" \
  --env-file "$OPENAI_ENV" --concurrency 30
python $STUDY/build_paired_scores.py \
  --judgments "$RUN/private/endorsement_judgments.jsonl" \
  --cohort "$RUN/generation_cohort.parquet" \
  --output "$RUN/paired_scores.csv"
python $STUDY/analyze.py \
  --release "$RELEASE" --paired-scores "$RUN/paired_scores.csv" \
  --chunks "$PREFIX/chunk_index.parquet" \
  --membership "$PREFIX/private/prefix_membership.parquet" \
  --labels "$PREFIX/private/prefix_labels_nano.jsonl" \
  --audit-labels "$PREFIX/private/prefix_labels_mini_audit.jsonl" \
  --out-dir "$RUN/results"
python $STUDY/compare_context_models.py \
  --reference "$REFERENCE" --comparison "$RUN/paired_scores.csv" \
  --out-dir "$RUN/model_comparison"
python $STUDY/analyze_transition_contexts.py \
  --release "$RELEASE" --paired-scores "$RUN/paired_scores.csv" \
  --chunks "$PREFIX/chunk_index.parquet" \
  --membership "$PREFIX/private/prefix_membership.parquet" \
  --labels "$PREFIX/private/prefix_labels_nano.jsonl" \
  --out-dir "$RUN/transition_contexts"
```
