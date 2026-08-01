# Agreement-ratchet causal result

## Bottom line

Neutralizing only the immediately preceding assistant message produced a small,
directionally protective but statistically imprecise change in GPT-4.1 mini.
Across 45 outcome-selected targets and five calls per condition, endorsement
was 146/225 (64.9%) after an agreement-preserving paraphrase and 134/225
(59.6%) after an epistemically neutral rewrite. The target-averaged paired
effect was +5.33 percentage points for agreement preserved minus neutral
(source-conversation bootstrap 95% CI -1.78 to +12.20 points). The corresponding
0-10 judge-score effect was +0.369 (95% CI -0.075 to +0.819).

This does not provide strong evidence that one immediately preceding agreement
is sufficient to cause the failure. It is consistent with a modest effect, but
the interval includes zero. The stronger interpretation of the earlier
observational result is accumulated assistant-stance or persona inheritance:
changing one local message often leaves many earlier demonstrations of the same
stance intact.

## Design

- Cohort: all 45 prior GPT-4.1-mini upward flips whose immediately preceding
  assistant message had blindly coded delusion-content density at most 0.25.
- Intervention: replace only that assistant message with either an
  agreement-preserving paraphrase or neutral rewrite.
- Active control: both conditions were rewritten by the same model; every other
  message and the target were identical.
- Rewriter: `gpt-5.4-mini-2026-03-17`, blind to the later target and outcomes.
- Target model: `gpt-4.1-mini-2025-04-14`, temperature 0, five calls per
  condition, no truncation.
- Outcome: unchanged SPIRALS `bot-endorses-delusion >= 7` judge and target-only
  judge context.
- Analysis: average repetitions within target, then bootstrap 35 source
  conversations.

The cohort was selected using the earlier target-model outcome. The experiment
therefore tests a mechanism within known failures and cannot estimate its
population prevalence.

## Primary results

| Estimate | Result | 95% cluster-bootstrap CI |
|---|---:|---:|
| Agreement-preserving endorsement | 64.9% | 55.6% to 74.0% |
| Neutral endorsement | 59.6% | 50.2% to 68.9% |
| Agreement minus neutral | +5.33 pp | -1.78 to +12.20 pp |
| Agreement minus neutral judge score | +0.369 | -0.075 to +0.819 |

At the target level, agreement preservation increased endorsement for 15
targets, made no difference for 23, and decreased it for seven. Across the 225
matched repetitions, 43 pairs endorsed only under agreement preservation and
31 endorsed only under neutralization; 103 endorsed in both and 48 in neither.

## Strict trajectory sensitivity

The preregistered manipulation check identified eight targets satisfying the
strict “simpler claim, assistant agreement, stronger target, successful neutral
rewrite” definition. In this small subset, endorsement was 75.0% after
agreement preservation and 65.0% after neutralization, a +10.0-point effect
(95% CI 0.0 to +22.9). The score difference was +0.75 points (95% CI +0.075 to
+1.77). This is directionally stronger, but eight targets from seven
conversations are insufficient for a stable binary-effect conclusion.

## Manipulation and repeatability

The final blinded manipulation check found no neutral rewrite adding overt
pushback, correction, warning, safety language, or redirection. Core content
was retained in 45/45 neutral rewrites and 44/45 agreement paraphrases. An
initial validator incorrectly counted intended hedging as pushback; that
definition was corrected and committed before target-model generation, while
the rewrites remained unchanged.

Temperature zero did not make the API deterministic. Eighty-nine of 90
target-condition cells produced five unique responses, and one produced three.
Repeated calls were therefore necessary. The original complete-history cohort
had 100% endorsement by selection, while the active agreement-control arm had
64.9%; that drop combines regression to the mean, API variation, and rewrite
perturbation and is not itself a causal contrast.

## Interpretation

The experiment weakens the strongest one-turn version of the agreement-ratchet
hypothesis. A single local neutralization is not a reliable safety intervention
once a long dialogue has established an assistant persona or worldview. The
point estimate suggests immediate stance may contribute, especially in the
strict trajectory subset, but most endorsement survives its removal.

The next decisive intervention would neutralize the cumulative assistant stance
across the prefix, or insert a stance reset immediately before the target, and
test that in a cohort selected without using the target-model outcome.

## Public artifacts

`published_results/` contains no conversation or response text. It includes the
cohort and execution manifests, blinded manipulation labels, target-averaged
and repetition-level scores, machine-readable summaries, and the result figure.
Raw text, rewrites, generations, judgments, and rationales remain gitignored.
