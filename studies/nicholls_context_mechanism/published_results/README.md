# Published no-text results

These are the aggregate, no-text outputs supporting `../RESULTS.md`.

- `paired_scores.csv` contains one matched result per target. It includes only
  row indices, hashes, scores, and binary outcomes, not conversation text.
- `primary/` uses the frozen nano prefix coder.
- `mini_coder_sensitivity/` independently recodes all prefix chunks with mini.
- Raw prompts, responses, judgments, prefix chunks, and coding rationales are
  intentionally excluded and remain in gitignored `artifacts/**/private/`.

Re-run `analyze.py` with regenerated private labels to rebuild these files.
