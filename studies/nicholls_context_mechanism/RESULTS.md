# Complete-history context mechanism results

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

## Post-result GPT-4.1-mini replication

An exact same-target replication changed only the response model to
`gpt-4.1-mini-2025-04-14`. On the 445 targets shared with GPT-5.4 mini, strict
endorsement rose from 21/445 (4.72%) in the target-only arm to 186/445 (41.80%)
with complete history: a paired increase of 37.08 percentage points
(conversation-cluster bootstrap 95% CI 30.20 to 42.91 points). There were 170
positive threshold crossings and five negative crossings. The context effect
was 35.73 points larger than GPT-5.4 mini's effect on the same targets (95% CI
28.78 to 41.79 points). Only three of GPT-5.4 mini's six positive flips were
also positive flips for GPT-4.1 mini, so the result is not driven solely by a
tiny universally difficult subset.

Unlike the sparse GPT-5.4-mini outcome, the GPT-4.1-mini run resolves the
candidate mechanism. The adjusted context-effect increase was +16.32 points
per SD of prefix delusion density (95% CI +11.06 to +21.58, p<0.001). Prefix
length was +4.67 points (95% CI -0.14 to +9.48, p=0.057), while rapport and
self-disclosure intervals included zero. The full independent mini-coder
sensitivity again identified delusion density (+18.15 points per SD, 95% CI
+13.06 to +23.24) and gave the same qualitative conclusion.

The primary coder estimated a +10.00-point context effect among 60 near-zero
delusion prefixes (95% CI +3.33 to +18.33), compared with +39.37 points among
414 prefixes with positive delusion content (95% CI +32.41 to +45.13). Across
delusion-density quartiles, the effect increased monotonically from 12.50% to
27.03%, 49.55%, and 59.46%. Thus mundane context can matter for this model, but
accumulated delusion-like content is the dominant measured driver.

A private spot audit of 12 randomly selected full-context positives found
substantive endorsements rather than mild safety caveats. Examples included
affirming activated 12-strand DNA, multidimensional portals, spirits awaiting
the user, and special cosmic identities as real. This was not a blinded human
validation study and is not included as inferential evidence.

This replication was requested after the GPT-5.4-mini result and is therefore
post-result. It supplies strong model-heterogeneity evidence under a frozen
protocol, but it is not a preregistered independent confirmation. Each arm also
uses one temperature-zero API response; temperature zero does not guarantee
bitwise repeatability.

### Exploratory increase-versus-decrease analysis

Separating prior user and assistant content after observing the main result
suggests that GPT-4.1 mini primarily inherits the preceding assistant's stance.
In a joint exploratory regression, assistant delusion-content density predicted
a +21.46-point context effect per SD (95% CI +15.72 to +27.19), adjusting for
prefix length, user delusion-content density, rapport, and self-disclosure.
Prefix length, rapport, and self-disclosure intervals included zero. The
conditional user-density coefficient was slightly negative, but user and
assistant densities are correlated and that post-result suppression coefficient
should not be interpreted causally.

Among the 424 targets that did not elicit target-only endorsement, full-history
endorsement rose monotonically with delusion content in the immediately
preceding assistant message: 19.3%, 26.2%, 49.4%, 58.0%, and 70.1% across
density bins of 0, >0-0.25, >0.25-0.50, >0.50-0.75, and >0.75-1. This is
consistent with stance or persona inheritance, not merely sensitivity to the
user's unusual claim.

Only five targets crossed downward from endorsement to non-endorsement. All
five were manually inspected. None produced clear pushback or reality testing;
instead, the full-history answer shifted into premise-marking, speculative,
summary, ceremonial, or task-execution framing that fell below the strict judge
threshold. Therefore, these five decreases should not be described as safety
improvements. A randomized assistant-message removal or neutral-rewrite
ablation is required to establish the assistant-stance mechanism causally.

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
- `gpt41mini/`: no-text GPT-4.1-mini replication artifacts.
- `gpt41mini/model_comparison/model_comparison.json`: exact common-cohort model
  contrast and conversation-cluster uncertainty.
- `gpt41mini/model_comparison/model_comparison.png`: compact model comparison.
- `gpt41mini/transition_contexts/`: exploratory role-separated context analysis.

Raw prompts, model responses, judgments, prefix text, and coder rationales are
stored under gitignored `private/` directories.
