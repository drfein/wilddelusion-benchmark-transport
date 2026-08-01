# Complete-history context mechanism result

## Result in one paragraph

For `gpt-5.4-mini-2026-03-17`, supplying the complete source history rather
than the identical flagged user turn alone increased strict SPIRALS
`bot-endorses-delusion >= 7` from 0/445 to 6/445 responses, a paired increase
of 1.35 percentage points (conversation-cluster bootstrap 95% CI 0.22 to 2.82
points). All six flips had prefixes with positive delusion-content codes, and
none occurred in the 60 prefixes whose primary coded delusion density was at
most 5%. This is directionally consistent with worldview inheritance, but it
does not identify the mechanism: after jointly adjusting for prefix length,
delusion density, rapport, and self-disclosure, no coefficient excluded zero.
The largest adjusted coefficient was prefix length, not delusion density. A
complete independent mini-model recode produced the same conclusion. The
defensible result is therefore that accumulated real context can causally
induce endorsement for a small subset of prompts, but these data do not
reliably distinguish delusional-content inheritance from correlated context
length or conversational commitment.

## Design

- Unit: one WildDelusion flagged user turn with a complete source history.
- Model: `gpt-5.4-mini-2026-03-17`, reasoning `none`, temperature 0.
- Full arm: every source message through the fixed target user turn.
- Target-only arm: the exact same target user turn with all preceding messages
  removed.
- Outcome: fixed SPIRALS endorsement judge, score threshold 7, given the same
  target-user context for both generated responses.
- Primary prefix coding: blinded `gpt-5.4-nano-2026-03-17` ordinal chunk codes.
- Reliability: 200 stratified chunks independently recoded by mini; all 13,874
  chunks were also recoded by mini as a post-result sensitivity analysis.
- Uncertainty: conversation-clustered regression and conversation-cluster
  bootstrap for paired effects. Wilson intervals are included for individual
  arm rates because a nonparametric bootstrap is degenerate in all-zero strata.

The prepared cohort contained 447 targets. Two additional complete-context
requests were rejected by the API as over-window despite passing the local
token estimate; both targets were excluded pairwise without truncation. The
final analysis contains 445 targets from 256 source conversations.

## Primary estimates

| Contrast | Estimate | 95% interval |
|---|---:|---:|
| Target-only endorsement | 0/445 (0.00%) | Wilson 0.00% to 0.86% |
| Full-history endorsement | 6/445 (1.35%) | Wilson 0.62% to 2.91% |
| Paired full minus target-only | +1.35 pp | Cluster bootstrap +0.22 to +2.82 pp |
| Near-zero delusion prefix | 0/60 (0.00%) | Wilson 0.00% to 6.02% |
| Positive delusion prefix | 6/414 (1.45%) | Wilson 0.67% to 3.13% |

The zero-event near-zero stratum is not proof of no mundane-context effect:
its upper 95% Wilson bound is 6.02%. The six observed flips are too sparse to
separate correlated mechanisms with precision.

## Adjusted heterogeneity

Primary nano-coded coefficients are changes in the paired endorsement risk per
one standard deviation, jointly adjusted for all four features.

| Prefix feature | Coefficient | 95% CI | p |
|---|---:|---:|---:|
| Log prefix words | +1.12 pp | -0.44 to +2.67 | 0.159 |
| Delusion density | +0.92 pp | -1.41 to +3.26 | 0.436 |
| Rapport density | +0.20 pp | -1.18 to +1.58 | 0.775 |
| Self-disclosure density | -0.08 pp | -1.25 to +1.09 | 0.892 |

The continuous 0-10 judge-score sensitivity was also null: delusion density
was `+0.125` score points per SD (95% CI `-0.114` to `+0.363`, p=0.304).

## Coding robustness

On the 200-chunk stratified recode, nano versus mini agreement for the binary
presence of delusion content was 87.5% (Cohen's kappa 0.674). Exact ordinal
agreement was 45.0%, with Spearman correlation 0.655. Because ordinal
calibration differed, the entire 13,874-chunk corpus was recoded with mini.

The full mini-coded sensitivity retained 0 flips in its 41 near-zero prefixes
and 6 flips in 416 positive-content prefixes. Its adjusted coefficients were:

| Prefix feature | Coefficient | 95% CI | p |
|---|---:|---:|---:|
| Log prefix words | +1.17 pp | -0.41 to +2.74 | 0.145 |
| Delusion density | +0.59 pp | -1.65 to +2.82 | 0.606 |
| Rapport density | +0.37 pp | -1.47 to +2.20 | 0.696 |
| Self-disclosure density | +0.19 pp | -0.94 to +1.32 | 0.742 |

Thus the qualitative stratum pattern survives an independent coder, but the
adjusted mechanism result remains null.

## What this does and does not show

The full-versus-target-only intervention is causal for these fixed prompts:
the preceding dialogue changed six model outputs across the endorsement
threshold. Prefix-feature associations are observational because natural
histories did not randomize delusion density, length, rapport, or disclosure.
All six flips had relatively delusion-dense histories, but those histories were
also generally long, and neither feature independently predicted the sparse
outcome with adequate precision.

This result does not validate the old 2.9% to 6.7% estimate. That estimate
pooled ten snapshots and used reduced histories for most rows. It motivated the
question but is not comparable to this corrected, single-snapshot,
complete-history experiment.

No mechanism conclusion should be called robust before a human audit of the
prefix codes. The next decisive experiment is a randomized minimal-edit or
prefix-splice intervention that varies delusional-content density while holding
length and conversational structure fixed, with repeated generations to
estimate low event rates.

## Public artifacts

- `results/summary.json`: primary preregistered analysis.
- `results/adjusted_regressions.csv`: primary coefficients.
- `results/stratified_context_effects.csv`: primary subgroup estimates.
- `results/context_mechanism.png`: compact result figure.
- `results_mini_sensitivity/`: full mini-coder sensitivity analysis.
- `paired_scores.csv`: no-text matched outcome table.

Raw prompts, model responses, judgments, prefix text, and coder rationales are
stored under gitignored `private/` directories.
