# Model-ranking transport across synthetic and real data

This analysis compares model orderings for two studies:

- [Lost in Delusion](https://arxiv.org/abs/2606.00975)
- [Psychogenic Machine](https://arxiv.org/abs/2509.10970)

Higher DCS and HES are less safe; lower SIS is less safe. Rank uncertainty is
estimated by resampling source-conversation clusters for real data and scenario
clusters for the direct synthetic bridge.

## DCS ranking results

| Study and estimand | Models | Spearman rho | Cluster-bootstrap 95% CI | Exact point order |
|---|---:|---:|---:|---|
| Lost, target mean | 4 | -0.800 | [-1.000, +0.949] | Not preserved |
| Lost, target-control sensitivity | 4 | -0.800 | [-1.000, +0.400] | Not preserved |
| Psychogenic, target mean | 6 | +0.986 | [+0.486, +1.000] | Preserved up to a synthetic tie |
| Psychogenic, implicit-explicit | 6 | -0.754 | [-0.943, +0.522] | Not preserved |

The key distinction is between **absolute susceptibility** and **intervention
sensitivity**:

- Psychogenic Machine's absolute DCS ordering transfers almost exactly from its
  synthetic scripts to real histories.
- The ordering of sensitivity to implicit versus explicit wording does not.
- Lost in Delusion's synthetic absolute ordering collapses to an unresolved
  near-tie on real histories.

## Lost in Delusion

The primary comparison uses absolute target DCS because the paper's synthetic
controls and the real grounded rewrites are not equivalent baselines.

| Model | Synthetic DCS | Real DCS | Real cluster 95% CI |
|---|---:|---:|---:|
| Qwen3-30B-A3B | 1.210 | 1.316 | [1.234, 1.394] |
| OLMo-3-7B | 1.080 | 1.303 | [1.212, 1.391] |
| Llama-3.1-8B | 0.690 | 1.322 | [1.225, 1.419] |
| Llama-3.3-70B | 0.360 | 1.337 | [1.240, 1.436] |

The synthetic order is
`Qwen3-30B-A3B > OLMo-3-7B > Llama-3.1-8B > Llama-3.3-70B`; the real point
order is
`Llama-3.3-70B > Llama-3.1-8B > Qwen3-30B-A3B > OLMo-3-7B`.
The point `rho=-0.800` is based on only four models and has a broad interval.
More importantly, all six pairwise real model-difference intervals cross zero.
Exact synthetic-order preservation occurs in 2.4% of bootstrap draws.

This should be reported as **synthetic ranking collapse**, not a resolved
ranking reversal. Absolute scores also come from different judges across
domains, so only their ordering, not their numerical separation, is compared.

As a separate within-real causal result, all six models show a large positive
target-minus-grounded DCS effect:

| Real model | DCS effect | Cluster 95% CI |
|---|---:|---:|
| Llama-3.1-8B | +0.911 | [+0.726, +1.094] |
| Llama-3.3-70B | +0.842 | [+0.624, +1.058] |
| Qwen3-4B | +0.752 | [+0.570, +0.935] |
| OLMo-3-7B | +0.723 | [+0.545, +0.905] |
| Qwen3-30B-A3B | +0.683 | [+0.490, +0.870] |
| Qwen3-14B | +0.644 | [+0.447, +0.837] |

Thus the grounding effect is robust even though it should not define the
cross-domain leaderboard.

## Psychogenic Machine

Absolute DCS order is the same on the direct synthetic bridge and real data,
apart from an exact synthetic tie between Qwen3-4B and Qwen3-14B:

`Qwen3-4B > Qwen3-14B > Qwen3-30B-A3B > OLMo-3-7B >
Llama-3.3-70B > Llama-3.1-8B`

Its rank correlation is `rho=0.986 [0.486, 1.000]`. The interval remains wide
because the synthetic bridge has eight independent scenario pairs, but its
lower bound is positive.

The implicit-minus-explicit condition effect behaves differently:

- Synthetic unsafe order:
  `Llama-3.3-70B > Qwen3-30B-A3B > Llama-3.1-8B > Qwen3-14B >
  OLMo-3-7B > Qwen3-4B`
- Real unsafe order:
  `Qwen3-4B > OLMo-3-7B > Llama-3.3-70B > Qwen3-14B >
  Llama-3.1-8B > Qwen3-30B-A3B`

Here `rho=-0.754`, but its clustered interval crosses zero. More importantly,
every real model's implicit-minus-explicit DCS confidence interval contains
zero.

## Qwen snapshot contrasts

On real Psychogenic Machine histories, absolute DCS decreases monotonically:

| Contrast | Difference | Cluster 95% CI |
|---|---:|---:|
| 4B minus 14B | +0.084 | [+0.018, +0.143] |
| 4B minus 30B-A3B | +0.170 | [+0.104, +0.234] |
| 14B minus 30B-A3B | +0.086 | [+0.029, +0.145] |

On Lost's target-minus-grounded effect, all three Qwen pairwise confidence
intervals cross zero. The snapshot trend therefore describes absolute
confirmation on one task, not a universal scaling law. Qwen3-30B-A3B is also a
sparse MoE with roughly 3B active parameters, so total parameter count is not a
controlled scale axis.

## Qualitative domain differences

### Lost in Delusion

The synthetic study uses fixed 16-turn trajectories where the evaluated model
helps generate future context and every trajectory reaches a harm phase. The
real study freezes natural histories and evaluates one continuation; only
42/522 targets are high stakes. Synthetic failures are often dramatic
narrative co-construction. Real failures more often involve epistemic
legitimization, systematizing elaboration, identity reinforcement, or
bureaucratic/procedural formalization.

### Psychogenic Machine

The synthetic manipulation is discourse-level concealment across separate
12-turn scripts. The real manipulation is a close final-turn rewrite in a fixed
history. Synthetic pairs have median token-set Jaccard `0.119`, versus `0.633`
for real pairs. Formulaic explicit wording also appears in 278/327 real pairs.
The two settings therefore operationalize "implicitness" differently.

## Artifacts

- `results/rank_correlations_cluster_bootstrap.csv`
- `results/rank_correlation_model_values.csv`
- `results/lost_absolute_model_means_cluster_bootstrap.csv`
- `results/lost_absolute_model_contrasts_cluster_bootstrap.csv`
- `results/qwen_scale_contrasts_cluster_bootstrap.csv`
- `results/model_rankings.csv`
- `results/model_ranking_transport.png`
- `results/corpus_characteristics.json`
