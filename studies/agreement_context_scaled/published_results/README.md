# Public scaled-intervention artifacts

This directory intentionally contains no conversation, rewrite, generated
response, judge rationale, or credential text.

- `results/summary.json`: primary and sensitivity estimates.
- `results/target_results_no_text.csv`: target-level arm rates and validation
  flags keyed only by hashes and row IDs.
- `results/source_results.json`: source-stratified sensitivity estimates.
- `results/repetition_transitions.csv`: paired binary transition counts.
- `results/causal_effect.{png,pdf}`: main result figure.
- `generation_execution_manifest.json` and `judge_execution_manifest.json`:
  model snapshots, token counts, hashes, and completion counts.

The frozen cohort, operational attrition, intervention manifest, and blinded
pre-outcome validation labels are in `../frozen_design/`.
