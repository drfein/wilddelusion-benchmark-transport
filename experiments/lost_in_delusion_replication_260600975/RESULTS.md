# Six-model real-data transport results

> **Historical exploratory report.** This predates the locked five-claim
> registry at the repository root. Several secondary-model checkpoint
> revisions were not preserved, so the report and its code are retained for
> audit but are not part of the exact `wd-reproduce run all` target. Use the
> [completion audit](../../results/COMPLETION_AUDIT.md) for finalized
> reportable findings.

This experiment tests core results from
[Lost in Delusion](https://arxiv.org/abs/2606.00975) and
[Psychogenic Machine](https://arxiv.org/abs/2509.10970) on frozen,
human-confirmed WildDelusion histories.

## Scope

Six open-weight instruct models were evaluated:

- Llama-3.1-8B
- Llama-3.3-70B
- OLMo-3-7B
- Qwen3-4B
- Qwen3-14B
- Qwen3-30B-A3B

Each model produced:

- 623 Lost-in-Delusion real responses: 522 original targets and 101 matched
  grounded controls.
- 654 Psychogenic-Machine real responses: 327 matched explicit/implicit pairs.
- 192 Psychosis-Bench synthetic responses: 16 conversations with 12 turns.

This is 8,814 model responses and 8,814 completed judgments, with no generation
or judging failures. Generation is greedy with at most 512 new tokens. The
Llama-70B and Qwen-14B runs use AWQ checkpoints; Qwen-30B-A3B uses an FP8
checkpoint. Fifty-eight long Psychogenic-Machine inputs per model drop complete
oldest messages to fit the 12,288-token cap; no message text is cut.

## Main findings

1. **Grounding has a large causal effect in every model.** On 101 matched real
   histories, replacing delusion-like framing with a close grounded version
   lowers Delusion Confirmation Scale (DCS) scores in all six models. Every
   clustered 95% interval excludes zero after Holm correction.
2. **The Lost-in-Delusion absolute ranking collapses to a real-data tie.**
   Among the four models shared with the paper, synthetic absolute DCS spans
   `0.36–1.21`, whereas real means compress to `1.303–1.337`. The point rank
   correlation is `rho=-0.800`, but every real pairwise model-difference
   interval crosses zero. This is failure to reproduce the synthetic
   leaderboard, not evidence for a resolved reversal.
3. **Psychogenic Machine's absolute susceptibility ranking does transport.**
   Across all six models, synthetic and real absolute DCS rankings are the same
   up to a synthetic Qwen-4B/Qwen-14B tie: `rho=0.986`, clustered bootstrap 95%
   interval `[0.486, 1.000]`.
4. **Its implicitness result does not transport.** The same-model synthetic
   bridge and real minimal rewrites produce different condition rankings
   (`rho=-0.754`, 95% interval `[-0.943, +0.522]`), and no real model has a
   reliable positive implicit-minus-explicit DCS effect.
5. **Confirmation and harm enablement remain coupled.** Across six models,
   pooled DCS-HES Spearman correlation is `0.722` on the broad-harm cohort and
   `0.847` on the strict high-stakes cohort, compared with `0.77` in
   Psychogenic Machine.

## Lost in Delusion

### Absolute model ordering

Absolute DCS on original target conversations is the primary cross-domain
model-order comparison. Unlike the matched effects, it does not assume that
the paper's synthetic controls and the WildDelusion grounded rewrites define
the same baseline.

| Model | Synthetic DCS | Real DCS | Real cluster 95% CI |
|---|---:|---:|---:|
| Qwen3-30B-A3B | 1.210 | 1.316 | [1.234, 1.394] |
| OLMo-3-7B | 1.080 | 1.303 | [1.212, 1.391] |
| Llama-3.1-8B | 0.690 | 1.322 | [1.225, 1.419] |
| Llama-3.3-70B | 0.360 | 1.337 | [1.240, 1.436] |

The synthetic order is
`Qwen3-30B > OLMo-3-7B > Llama-3.1-8B > Llama-3.3-70B`; the real point order
is `Llama-3.3-70B > Llama-3.1-8B > Qwen3-30B > OLMo-3-7B`. The point
correlation is `rho=-0.800`, with clustered bootstrap interval
`[-1.000, +0.949]`.

The ordering is not statistically resolved: all six real pairwise
model-difference intervals include zero. Exact synthetic-order preservation
occurs in 2.4% of clustered bootstrap draws. The defensible conclusion is that
the large synthetic separation collapses to a near-tie on real data.

Absolute values should still not be treated as calibrated cross-domain effect
sizes because the paper's published synthetic aggregates and this real
evaluation use different judges. Their ordering is more defensible than their
numerical difference.

### Within-real grounding intervention

The matched estimand is original delusion-like history minus its jointly
grounded, distress- and request-matched rewrite. It is a causal test within the
real cohort, not the primary cross-domain leaderboard comparison. DCS and HES
use the paper's ordinal mapping: `N/A, 1 -> 0`, `2 -> 1`, `3 -> 2`.

| Model | Original DCS | Grounded DCS | Difference | Cluster 95% CI |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 1.277 | 0.366 | +0.911 | [+0.726, +1.094] |
| Llama-3.3-70B | 1.257 | 0.416 | +0.842 | [+0.624, +1.058] |
| Qwen3-4B | 1.198 | 0.446 | +0.752 | [+0.570, +0.935] |
| OLMo-3-7B | 1.168 | 0.446 | +0.723 | [+0.545, +0.905] |
| Qwen3-30B-A3B | 1.149 | 0.465 | +0.683 | [+0.490, +0.870] |
| Qwen3-14B | 1.059 | 0.416 | +0.644 | [+0.447, +0.837] |

All six Holm-adjusted p-values are `0.00018`. Thus grounding reliably reduces
confirmation even though the model ordering is unstable. The strict
harm-preserved subset has only eight pairs, so its HES and SIS model rankings
are not interpretable.

### Qwen snapshot comparison

Qwen3-4B has higher absolute DCS than Qwen3-14B by
`+0.199 [0.138, 0.260]` and Qwen3-30B-A3B by
`+0.146 [0.090, 0.204]`. However, the matched framing effects are
`0.752`, `0.644`, and `0.683`; every pairwise scale-contrast interval crosses
zero. Scale therefore predicts baseline confirmation here better than
sensitivity to the delusional-versus-grounded intervention.

## Psychogenic Machine

The same six models were run directly on the original 16 Psychosis-Bench
conversations and on 327 real matched explicit/implicit pairs. Higher DCS is
less safe.

| Model | Synthetic DCS | Real DCS | Synthetic implicit-explicit | Real implicit-explicit |
|---|---:|---:|---:|---:|
| Qwen3-4B | 1.715 | 1.480 | -0.514 | +0.037 |
| Qwen3-14B | 1.715 | 1.396 | -0.403 | -0.003 |
| Qwen3-30B-A3B | 1.646 | 1.310 | -0.181 | -0.052 |
| OLMo-3-7B | 1.319 | 1.284 | -0.472 | 0.000 |
| Llama-3.3-70B | 1.201 | 1.173 | -0.069 | -0.003 |
| Llama-3.1-8B | 1.146 | 1.054 | -0.264 | -0.015 |

Absolute DCS produces almost identical rankings across domains:
`rho=0.986 [0.486, 1.000]`. The lower bound remains positive despite only eight
independent synthetic scenarios.

By contrast, every real implicit-minus-explicit interval includes zero:

- Llama-3.1-8B: `-0.015 [-0.096, +0.069]`
- Llama-3.3-70B: `-0.003 [-0.070, +0.070]`
- OLMo-3-7B: `0.000 [-0.064, +0.064]`
- Qwen3-4B: `+0.037 [-0.035, +0.104]`
- Qwen3-14B: `-0.003 [-0.068, +0.059]`
- Qwen3-30B-A3B: `-0.052 [-0.131, +0.024]`

The original paper pooled different models and reported a positive
implicitness effect. In this direct bridge, all six models instead have lower
synthetic DCS under the implicit scripts. This shows that the pooled finding is
model- and protocol-dependent even before moving to real data.

### Qwen snapshot comparison

Absolute real DCS decreases across the three Qwen snapshots:
`4B 1.480 > 14B 1.396 > 30B-A3B 1.310`. The paired differences are:

- 4B minus 14B: `+0.084 [0.018, 0.143]`
- 4B minus 30B-A3B: `+0.170 [0.104, 0.234]`
- 14B minus 30B-A3B: `+0.086 [0.029, 0.145]`

This is a cohort-specific snapshot trend, not a clean parameter-scaling law:
30B-A3B is a sparse mixture-of-experts model with roughly 3B active
parameters.

## Why real and synthetic results differ

### Lost in Delusion

The synthetic benchmark contains fixed 16-turn trajectories in which the
evaluated assistant helps create later context and every trajectory reaches a
moderate-plus-harm phase. WildDelusion contains frozen natural histories and
only 42/522 targets (8.0%) are high stakes. Real failures are often subtler:
epistemic legitimization, theory-building, identity reinforcement, or
bureaucratic/procedural formalization. This is materially different from
dramatic narrative co-construction in the synthetic trajectories.

### Psychogenic Machine

Synthetic implicitness is discourse-level concealment across a 12-turn script.
The real intervention is a close rewrite of the final user message in a fixed
history. Synthetic turn pairs have median token-set Jaccard `0.119`; real final
turn pairs have median `0.633`. In addition, 278/327 real explicit rewrites use
formulaic wording such as "sincerely believe", versus one implicit rewrite.
These interventions share a label but do not isolate the same construct.

## Interpretation

The result is not one universal model leaderboard. Three claims separate:

- Delusion-like rather than grounded framing robustly increases confirmation
  on real histories.
- A model's baseline tendency to confirm can transport across synthetic and
  real cohorts when the task operationalization is closely matched.
- Sensitivity to a particular intervention, and therefore model ordering under
  that intervention, may fail to transport when real and synthetic tasks encode
  different mechanisms.

## Limitations

- Lost synthetic values are published aggregates rather than regenerated
  outputs; only four evaluated models overlap the paper.
- Psychogenic Machine has only eight independent synthetic scenario pairs, so
  rank intervals remain broad.
- Each model/input has one deterministic response.
- Counterfactual controls and explicitness rewrites are generated and
  validated, not naturally randomized.
- The Lost rubric is judged by `gpt-5.4-mini`; Psychosis-Bench uses
  `gpt-4o-mini-2024-07-18`. These match each experiment's locked protocol but
  introduce judge dependence.
- The exact last-user-only context ablation and natural-wording sensitivity
  remain two-model analyses.

## Primary artifacts

- Lost paired effects:
  `full_history_522/results/analysis_clustered/paired_effects_clustered.csv`
- Psychogenic paired effects:
  `psychogenic_machine_transport_250910970/full522/results/analysis_clustered/paired_effects_clustered.csv`
- Cross-domain rank bootstrap:
  `cross_domain_comparison/results/rank_correlations_cluster_bootstrap.csv`
- Lost absolute model means:
  `cross_domain_comparison/results/lost_absolute_model_means_cluster_bootstrap.csv`
- Lost absolute pairwise contrasts:
  `cross_domain_comparison/results/lost_absolute_model_contrasts_cluster_bootstrap.csv`
- Qwen scale contrasts:
  `cross_domain_comparison/results/qwen_scale_contrasts_cluster_bootstrap.csv`
- Ranking figure:
  `cross_domain_comparison/results/model_ranking_transport.png`
- All-model real Lost figure:
  `full_history_522/results/analysis_clustered/grounded_real_all_models_clustered.png`
