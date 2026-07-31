# Methods

## Shared real cohort

The real cohort contains 522 verified target user turns from 321 source
conversations. The target is always the final user message. Target-model
contexts retain at most the final four user turns, preserve complete messages,
start with a user message, and end with the exact target. Uncertainty is
clustered by `(source, conversation_id)`.

## 1. Visible-context Mod+Harm coverage

Two official BF16 models generate greedy responses to each endpoint. The exact
native Qwen rubric assigns DCS, where the headline failure is DCS=3 in the
paper's labels, represented internally as `dcs == 2` (Validate/Amplify).

The possible-Mod+Harm screen is deliberately overinclusive: it unions three
independent target-level rubrics, two rubrics over every user message visible
to the assistant, and deterministic harm-language matches. “Unclear” counts as
possible. For each model, the estimand is

```text
captured share = Validate/Amplify responses inside the screen
                 / all Validate/Amplify responses
```

Cluster bootstrap intervals are computed for the complementary outside share.

## 2. Natural versus generated recognition controls

The exact published direct-delusion classifier is applied to:

- 101 generated, validated non-delusion counterfactuals; and
- 66 natural non-delusion near-misses with two fixed-model adjudications and
  verbatim retained-context exclusion evidence.

Malformed classifier outputs are not discarded. The lower-bound gap assigns
unknown natural outputs as negative and unknown generated outputs as positive,
the assignment least favorable to the claim. The estimand is the within-model
natural-minus-generated false-positive-rate gap.

## 3. Implicit wording

Three official BF16 models are evaluated with three seeded stochastic draws in
three designs:

- all 12 turns of each of eight released explicit/implicit scenario pairs;
- matched-local rewrites of the current turn in the same synthetic history; and
- 327 validated explicit/implicit rewrites in fixed natural histories.

The score is implicit-minus-explicit DCS. Analysis includes model, seed, and
scenario effects; clustered 99% intervals; all eight leave-one-scenario-out
checks; and exact finite sign flips over the eight authored scenario effects.
The sign-flip calculation is a symmetry sensitivity analysis, not a randomized
experiment p-value.

## 4. Ontology coverage

Two independently worded prompts map each confirmed endpoint to the synthetic
benchmark's three themes or outside ontology. Both calls use the same pinned
mini-model and are not independent human adjudicators. Strict outside consensus
requires both rubrics to assign outside. The interval resamples source
conversations.

## 5. Recognition model gap

The primary cohort contains 98 endpoints across 63 conversations that both
ontology rubrics place inside the paper's three themes. Every malformed model
output is assigned to maximize the real OLMo-minus-Llama FNR gap. Point and
cluster-bootstrap bounds are compared descriptively with the paper's rounded
aggregate gap; row-level synthetic predictions were not released, so this is
not a cross-study population interaction.

## Gate discipline

[build_claim_registry.py](../experiments/lost_in_delusion_replication_260600975/cross_domain_comparison/build_claim_registry.py)
selects at most one claim per evidence family and at most five total.
[verify_abstract_claims.py](../experiments/lost_in_delusion_replication_260600975/cross_domain_comparison/verify_abstract_claims.py)
then independently checks claim-specific sample sizes, model completeness,
interval direction, adversarial parse handling, and wording guardrails.
