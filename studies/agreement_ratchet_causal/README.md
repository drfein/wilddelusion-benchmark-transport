# Agreement-ratchet causal study

This study changes only the immediately preceding assistant message in 45
known GPT-4.1-mini context failures. It compares an agreement-preserving
paraphrase against an epistemically neutral rewrite while holding the later
user target and all other history fixed.

The protocol was frozen before counterfactual generation. Raw conversations,
rewrites, model responses, and judgment rationales are stored under gitignored
`artifacts/**/private/`; publishable outputs contain no message text.

See `PREREGISTRATION.md` for the estimand and interpretation boundary.
