# Scaled agreement-context causal result

## Bottom line

The outcome-selected pilot's apparent immediate-agreement effect did not
replicate in an independent, outcome-blind cohort. Across 216 source
conversations and three calls per arm, GPT-4.1-mini endorsed 123/648 responses
(19.0%) after an agreement-preserving rewrite and 129/648 (19.9%) after local
epistemic neutralization. The target-averaged immediate-stance effect was
**-0.93 percentage points** (agreement preserved minus local neutral; 95%
conversation-bootstrap CI -3.24 to +1.39; two-sided sign-flip p=.510,
Holm-adjusted p=1.0).

This rules out the preregistered positive one-turn agreement-ratchet effect at
the scale suggested by the pilot. The pilot's +5.33-point estimate came from
45 cases selected because GPT-4.1-mini had previously changed from
non-endorsement to endorsement. The scaled study excluded those conversations,
selected no row using a model outcome, and produced a near-zero estimate in
the opposite direction. Outcome selection and regression to the mean are the
most plausible explanation for the pilot difference.

## Design

- Source: the frozen 445-row complete-history WildDelusion input cohort.
- Selection: one structurally eligible target per non-pilot source
  conversation, selected without model responses, endorsement outcomes, or
  prefix labels.
- Final paired cohort: 216 conversations. Five extreme histories were excluded
  before target generation because cumulative counterfactual construction did
  not complete; attrition is fully enumerated and biases coverage away from the
  longest tail.
- Arms: agreement-preserving immediate rewrite; identical-content local
  neutral rewrite; and local neutral plus neutralization of earlier
  stance-carrying assistant messages.
- Target model: `gpt-4.1-mini-2025-04-14`, temperature 0, three repeated calls,
  no truncation, 512-token output cap.
- Outcome: unchanged target-only SPIRALS `bot-endorses-delusion >= 7` judge,
  using `gpt-5.4-mini-2026-03-17` with reasoning disabled.
- Inference: target-averaged paired effects, conversation bootstrap intervals,
  sign-flip tests, and Holm correction across three endorsement contrasts.

## Primary results

| Arm or contrast | Estimate | 95% CI |
|---|---:|---:|
| Agreement-preserving endorsement | 18.98% | 14.35% to 23.92% |
| Local-neutral endorsement | 19.91% | 15.43% to 24.69% |
| Cumulative-neutral endorsement | 19.44% | 14.97% to 24.07% |
| Immediate stance: agreement - local | -0.93 pp | -3.24 to +1.39 pp |
| Earlier stance: local - cumulative | +0.46 pp | -2.16 to +3.09 pp |
| Total stance: agreement - cumulative | -0.46 pp | -3.40 to +2.31 pp |

All three primary endorsement contrasts had Holm-adjusted p=1.0. Immediate
stance increased target-level endorsement in 17 conversations, decreased it
in 21, and made no difference in 178. At the repetition level, 33 responses
endorsed only under agreement preservation versus 39 only under local
neutralization.

The frozen valid-local sensitivity subset contained 149 targets. Its immediate
effect was also null: -0.67 points (95% CI -3.58 to +2.24). This subset had
perfect neutral content preservation, no overt neutral pushback, and retained
the intended local agreement contrast.

## Cumulative stance

The cumulative intervention changed 2,639 earlier assistant messages and was
active in 195/216 conversations, but the independent validator judged only 21
as strictly valid: 154 retained some earlier agreement, 14 added overt
pushback, and one validation exceeded the context window. The all-cohort
cumulative package therefore cannot be interpreted as complete removal of
accumulated stance.

Within the 21 strict valid-cumulative cases, local-neutral endorsement was
33.3% and cumulative-neutral endorsement was 27.0%, a directionally protective
+6.35-point contrast with a wide 95% CI (-3.17 to +15.87). The corresponding
0-10 score contrast was +0.71 (95% CI +0.14 to +1.35). This small,
manipulation-selected sensitivity is suggestive that accumulated stance may
matter when it is successfully removed, but it is not a confirmatory binary
result.

## Repeatability

Temperature zero was not deterministic. Of 648 target-condition cells, 620
produced three unique responses, 17 produced two, and 11 produced one. Repeated
calls were therefore necessary even though the primary target-level effects
were near zero.

## Interpretation

The strongest defensible conclusion is negative and useful: **a single
immediately preceding assistant agreement does not robustly cause subsequent
GPT-4.1-mini endorsement in outcome-independent real conversations.** The
earlier observational association and outcome-selected pilot should not be
reported as a causal agreement ratchet.

This experiment does not establish that accumulated context is irrelevant.
The clean cumulative subset was too small because faithfully neutralizing long
real histories without changing content is difficult. A decisive cumulative
test needs a simpler intervention, such as a standardized epistemic reset,
rather than thousands of generative historical rewrites.

## Public artifacts

`published_results/` contains no conversation, rewrite, response, rationale,
or API credential text. It contains the no-text cohort and attrition records,
pre-outcome manipulation checks, execution manifests, target-averaged scores,
source sensitivities, machine-readable summaries, and the result figure.
