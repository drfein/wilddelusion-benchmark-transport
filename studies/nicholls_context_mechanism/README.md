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

The complete-history experiment is finished. For the fixed mini snapshot,
strict endorsement increased from 0/445 target-only responses to 6/445
full-history responses. No adjusted prefix-property coefficient excluded zero,
so the causal context effect is established for these prompts but its mechanism
is unresolved. See `RESULTS.md` for estimates, uncertainty, coder sensitivity,
and interpretation boundaries. The frozen specification and amendments are in
`PREREGISTRATION.md`.
