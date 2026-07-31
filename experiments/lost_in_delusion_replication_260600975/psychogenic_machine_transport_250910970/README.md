# Psychogenic Machine transport test

This directory tests the main findings of arXiv:2509.10970 on real
WildDelusion conversations rather than its 16 synthetic scenarios.

## Primary estimand

The paper reports that implicit scenarios produce higher DCS and HES and lower
SIS than matched explicit scenarios. For each human-saved WildDelusion target,
this experiment:

1. creates matched explicit and implicit versions of the same final user turn;
2. preserves every preceding real user and assistant message;
3. independently validates belief, distress, harm, style, and explicitness;
4. generates greedy responses from OLMo-3-7B-Instruct, Llama-3.1-8B-Instruct,
   Llama-3.3-70B-Instruct, Qwen3-4B, Qwen3-14B, and Qwen3-30B-A3B with no
   system prompt, matching Psychosis-Bench;
5. applies the repository's separate DCS, HES, and SIS judge rubrics using
   `gpt-4o-mini-2024-07-18`.

The completed real transport uses 327 strictly validated matched pairs from
229 source conversations. HES and SIS use the 28-pair strict high-stakes
subset. An earlier pilot was rejected before analysis because its
implicit/explicit manipulation was too weak.

The estimand is target-level explicitness in a fixed real history. It is a
transport test, not an exact recreation of the paper's 12-turn synthetic
trajectory manipulation.

## Secondary checks

- Spearman association between DCS and HES, compared with the paper's
  `rho = .77`.
- Theme-level descriptive results, particularly whether grandiose content is
  the highest-risk category.

## Reproduction

```bash
set -a
source ../.env
set +a

python generate_pairs.py
python validate_pairs.py
python prepare_inputs.py

# Local-model generation is run through ../run_local_models.py on the GPU host.

python judge.py \
  --inputs results/generations_olmo.jsonl \
           results/generations_llama31_8b.jsonl
python analyze.py
```

## Results

The full arm's explicit rewrite has formulaic wording contamination, so the
cleanest causal check remains the separate 27-pair natural-wording sensitivity.

The synthetic DCS effect (`implicit - explicit = +0.31`) does not reliably
transport:

| Model | DCS difference | 95% CI |
|---|---:|---:|
| Llama-3.1-8B | -0.015 | [-0.096, +0.069] |
| OLMo-3-7B | 0.000 | [-0.064, +0.064] |
| Llama-3.3-70B | -0.003 | [-0.070, +0.070] |
| Qwen3-4B | +0.037 | [-0.035, +0.104] |
| Qwen3-14B | -0.003 | [-0.068, +0.059] |
| Qwen3-30B-A3B | -0.052 | [-0.131, +0.024] |

No model reproduces a reliable positive implicitness penalty. The DCS-HES
association does transport: broad-harm pooled `rho=0.722` and strict
high-stakes pooled `rho=0.847`, versus the paper's `0.77`.

See [`../RESULTS.md`](../RESULTS.md) for the joint interpretation with the Lost
in Delusion transport.
