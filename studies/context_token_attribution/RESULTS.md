# Context as a persistent epistemic anchor

## Scope

These results concern 187 complete, naturally occurring conversations selected
without using Qwen outcomes. The evaluated model is `Qwen/Qwen3-8B` at revision
`b968826d9c46dd6066d109eabc6255188de91218`. Each prompt has five continuations
sampled at temperature 0.7 and top-p 0.9. Endorsement is a score of at least 7
under the frozen SPIRALS `bot-endorses-delusion` rubric, judged by
`gpt-5.4-mini-2026-03-17` with no reasoning. Inference resamples whole
conversations, not individual continuations.

This is a selected real-dialogue cohort, not an estimate of endorsement
prevalence in all chatbot use. Assistant-specific interventions are smaller
selected subsets and should be interpreted for those eligible conversations.

## Primary result

Complete history raised endorsement from 4.39% with the target user turn alone
to 27.70%, a paired increase of 23.32 percentage points (95% conversation-level
bootstrap CI 18.18 to 28.45; 187 conversations, 935 samples per condition).
The latest prior user-assistant exchange accounted for 15.29 points (10.70 to
20.21), while earlier history accounted for another 8.02 points (4.28 to
11.76).

The same progression is visible before generation in the held-out dense probe.
Its mean projected endorsement score rises from 2.21% on the target turn alone,
to 13.82% with the latest exchange, to 26.64% with full history. For the 37
conversations whose preceding assistant message was itself endorsing, the
corresponding trajectory is 2.99%, 26.76%, and 45.01%. These projected scores
are useful as a latent trajectory but are not calibrated causal probabilities
on ablated prompts; paired regeneration remains the causal endpoint.

The latest preceding assistant turn was itself endorsing in 37 conversations.
In this selected subgroup, deleting that turn reduced endorsement by 15.00
points (5.56 to 24.44), and by 10.00 points relative to deleting a matched
earlier assistant turn (1.11 to 19.44; 36 conversations). The matched deletion
estimate is imperfectly balanced for recency, so an exact-content position swap
was used as a stricter follow-up.

For 31 eligible conversations, swapping the complete latest endorsing assistant
message with an earlier non-endorsing assistant message preserved every token,
role, turn, and total prompt length. Merely moving the endorsing content earlier
had a small, statistically unresolved effect of -2.58 points (95% CI -11.61 to
6.45; paired p=0.690). Removing that same content instead of relocating it
reduced endorsement by 12.26 points relative to relocation (3.23 to 21.29;
paired p=0.0209). Thus, in this selected cohort, belief-specific assistant
content remained causally active after intervening dialogue; its presence was
more important than being the immediately preceding assistant message.

The most defensible interpretation is a **persistent epistemic anchor**:
assistant-generated, belief-specific commitments can remain in context and
increase later endorsement even when they are no longer the latest turn. This
does not establish that generic agreement, rapport, or assistant text has the
same effect.

## Localization methods

### Turn-by-turn dynamics

A second nested grouped probe was trained at the header-only assistant-generation
boundary, which is exactly shared by historical and final user turns. It
retained strong held-out prediction (AUROC 0.831, 95% CI 0.782 to 0.874;
average precision 0.640). Causal masking then allowed all 2,272 historical
assistant-decision states to be extracted from 187 full-conversation forwards
without future-token leakage.

In the fixed-composition subset with at least six decision states, conversations
whose next source-assistant response endorsed showed a Qwen susceptibility ramp
from 18.7% five states earlier to 46.7% immediately before that response
(+27.9 points, 95% CI +18.7 to +37.6; 24 conversations). The corresponding
ramp was +2.6 points (-1.1 to +6.6) when the next source-assistant response did
not endorse (83 conversations). The between-group ramp difference was +25.3
points (+15.4 to +35.6), and remained +25.9 points (+14.8 to +37.0) when
restricted to ShareChat conversations with ChatGPT as the source assistant.

The held-out Qwen state immediately before the source response predicted whether
that different source assistant would endorse (AUROC 0.757, 95% CI 0.660 to
0.843; ShareChat-ChatGPT sensitivity AUROC 0.716). By contrast, the final state
change after the latest assistant response and target user turn differed by only
+1.8 points between groups (95% CI -4.7 to +8.5). Thus endorsement occurs in a
cross-model conversational risk state that has often been building for several
turns; the source assistant's endorsement is more a marker and reinforcement of
that state than its sole point of origin. This trajectory is predictive, not a
randomized effect. The separate deletion and relocation experiments establish
that endorsing assistant content subsequently contributes causally and persists.

### Predictive representations

The final prompt-token residual stream predicted sampled endorsement out of
fold (dense linear probe AUROC 0.806, 95% CI 0.750 to 0.856; average precision
0.605). Projecting full and target-only prompts through the held-out direction
showed a mean context logit shift of +3.07, and the representational probability
shift correlated with the paired behavioral context effect (Spearman r=0.611).

A pretrained layer-20 TopK-64 SAE from the exact Qwen3-8B revision recovered a
smaller but real context signal. Sparse context-difference features predicted
endorsement at AUROC 0.659 (0.578 to 0.732), versus 0.482 for target-only
features. Reverse-index inspection found that the most stable selected features
were generic lexical or discourse features such as `story`, `should`,
`applications`, first-person `I`, and refusal language. The SAE result is useful
as evidence that context information is distributed across sparse features,
but it does not yield a clean human-readable endorsement circuit.

### Attribution faithfulness

Fine-grained attribution did not reliably identify the causal assistant turn.

- Gradient-times-input and 16-step integrated gradients selected the same top
  assistant message in only 3 of 23 overlapping conversations.
- Deleting the gradient-selected assistant message changed endorsement by
  +6.4 points relative to its matched control (95% CI -0.8 to 14.0), opposite
  the predicted direction. Integrated gradients gave +7.0 points (-2.6 to
  18.3), also opposite and unresolved.
- AttriCoT-style fixed-response leave-one-message-out attribution selected the
  same top message for paired endorsing and non-endorsing samples in 80% of ten
  conversations. Assistant support did not differ reliably between response
  classes.
- FlashTrace selected the same top message for both response classes in all ten
  paired conversations. Total assistant information-flow share differed by
  only +0.11 percentage points (95% CI -0.66 to 1.00).

These methods mostly localized contextual support shared by plausible
continuations, not the stochastic boundary between endorsing and
non-endorsing responses. Their internal scores should therefore be treated as
hypothesis generators, not causal rankings, unless validated by regeneration.

## Safety implication

Monitoring only the latest user turn misses a large contextual contribution,
and monitoring only the immediately preceding assistant turn can also miss the
causal content after it moves deeper into history. A practical safety mechanism
should track and, when needed, explicitly repair prior assistant-generated
epistemic commitments. Token saliency alone is not a reliable basis for deciding
which history to remove.

## Reproducibility

- Frozen design: `PREREGISTRATION.md`
- Method-to-paper map: `LITERATURE_METHOD_MAP.md`
- End-to-end commands: `RUNBOOK.md`
- Machine-readable aggregate results: `results_summary.json`
- Raw private artifacts: `artifacts/private/` in the local archival bundle
