# Direct delusion-recognition transport

This experiment applies the exact disclosed **Assess Delusion Then Reply**
classifier from *Lost in Delusion* (arXiv:2606.00975) to WildDelusion. It does
not expand the paper's three positive labels or add a system prompt.

## Cohorts

- 522 confirmed, deduplicated WildDelusion positive endpoints.
- 101 validated generated grounded counterfactuals matched to positive rows.
- 309 real natural near-misses: role-play, fiction, jokes, text tasks, and
  third-party quotation that appeared delusion-like in isolation.
- The primary natural-negative analysis uses 100 rows across 97 conversations
  that pass three checks: the original full-context verifier, an independent
  full-context `gpt-5.4-mini-2026-03-17` adjudication, and a second blind
  adjudication using only the complete-message suffix visible to every
  classifier tokenizer. The final check requires high confidence and a
  verified verbatim fiction/task cue. The 183-row full-context consensus set,
  all 19 human-confirmed controls, and the original 218-row high-confidence set
  are reported separately as sensitivity cohorts.

## Protocol

- Same four open models shared with the paper.
- Exact published four-label prompt.
- Greedy decoding and no added system prompt.
- Qwen thinking disabled, as in the paper's generation protocol.
- Up to 12,288 input tokens. Only oldest complete prior messages are dropped;
  the target user message is never dropped or rewritten.
- Primary metrics are FNR and FPR. Raw F1 is not transported across class
  balances.

## Claim gates

A model-specific synthetic-to-real recognition gap is reportable only if:

1. Exact-output parse coverage is 100%.
2. The model and prompt match the disclosed paper setup.
3. Cluster-bootstrap uncertainty is reported for the real estimate.
4. The real 99% interval clears a distribution-free synthetic sensitivity
   envelope that treats the paper's 30 personas, rather than correlated turns,
   as independent units and accounts for rounded published rates.
5. Natural near-miss FPR is described as a hard-negative stress test, not a
   population false-positive rate.
6. The headline natural-negative cohort is restricted to the 100 strict
   retained-context consensus rows across 97 source conversations.

A cross-model claim requires three exact official-BF16 checkpoints to complete
and at least two to show the same conservative directional gap. The AWQ 70B
checkpoint is sensitivity-only. Results that fail these gates remain
diagnostics.
